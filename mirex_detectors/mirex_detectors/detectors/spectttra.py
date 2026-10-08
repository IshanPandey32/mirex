"""SpecTTTra - Spectro-Temporal Tokens Transformer (SONICS, Rahman et al., arXiv 2408.14080).

Verified: the spectrogram is tokenised SEPARATELY along time and along frequency (temporal
tokens and spectral tokens), the two token streams are concatenated and attended to globally.
Variants alpha/beta/gamma trade token count (cost) for accuracy.

NOT verified: exact clip sizes per variant, embed dim, depth. The values below are my
recollection of the SONICS configs (alpha: f1/t3, beta: f3/t5, gamma: f5/t7) - check them
against the paper/official repo if you need a faithful reproduction. This is a from-scratch
implementation: no official weights are loaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from .common import BaseDetector

VARIANTS = {"alpha": (1, 3), "beta": (3, 5), "gamma": (5, 7)}   # (f_clip, t_clip)


class SpecTTTra(BaseDetector):
    name = "spectttra"
    sr = 16000
    seg_seconds = 5.0

    def __init__(self, variant="alpha", dim=384, depth=6, heads=6, n_mels=128, hop=512, drop=0.1):
        super().__init__()
        f_clip, t_clip = VARIANTS[variant]
        self.mel = torchaudio.transforms.MelSpectrogram(self.sr, n_fft=2048, hop_length=hop, n_mels=n_mels)
        n_frames = int(self.seg_seconds * self.sr) // hop + 1
        self.n_frames = n_frames // t_clip * t_clip            # crop to a multiple of the clip
        self.n_mels = n_mels // f_clip * f_clip
        self.temporal = nn.Conv1d(self.n_mels, dim, t_clip, stride=t_clip)     # mels as channels
        self.spectral = nn.Conv1d(self.n_frames, dim, f_clip, stride=f_clip)   # frames as channels
        n_tok = self.n_frames // t_clip + self.n_mels // f_clip
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, wave):
        m = torch.log(self.mel(wave.float()) + 1e-6)                      # [B,M,T]
        m = m[:, :self.n_mels, :self.n_frames]
        m = (m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-5)
        t_tok = self.temporal(m).transpose(1, 2)                           # [B,Nt,D]
        s_tok = self.spectral(m.transpose(1, 2)).transpose(1, 2)           # [B,Ns,D]
        x = torch.cat([t_tok, s_tok], 1) + self.pos
        return self.head(self.norm(self.enc(x).mean(1)))


def build(variant="alpha", **kw):
    return SpecTTTra(variant=variant, **kw)
