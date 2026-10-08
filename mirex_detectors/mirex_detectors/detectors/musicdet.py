"""MusicDET-style zero-shot detector (arXiv 2605.18072, ICML 2026).

Verified from the abstract: trained on REAL music only, a frequency-guided normalizing flow
models the distribution of real-music features; the likelihood of an input under that model is
the detection score (low likelihood => out-of-distribution => AI). Evaluated by EER.

NOT verified (full text not read): what "frequency-guided" means precisely, the feature
front-end and the flow depth. My interpretation: log-mel energy spectrogram -> RealNVP-style
affine couplings, each conditioned on the per-band mean-energy profile plus a learned frequency
embedding. Treat it as "MusicDET-style".

Zero-shot means the trainer feeds this model ONLY real tracks and picks the checkpoint by
real-validation likelihood, never by AUROC. Caveat inherent to flows: very simple/quiet inputs
can receive HIGH likelihood, so keep the confound gate honest and check silence/low-level audio.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio

from .common import BaseDetector

SQ = 4  # squeeze factor: [B,1,F,T] -> [B,16,F/4,T/4]


def squeeze(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H // SQ, SQ, W // SQ, SQ).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * SQ * SQ, H // SQ, W // SQ)


class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ls = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        return x * self.ls.exp() + self.b, self.ls.sum() * x.shape[2] * x.shape[3]


class Coupling(nn.Module):
    def __init__(self, c, cond_c, hidden=64, flip=False):
        super().__init__()
        self.flip = flip
        self.net = nn.Sequential(nn.Conv2d(c // 2 + cond_c, hidden, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(hidden, hidden, 1), nn.ReLU(True),
                                 nn.Conv2d(hidden, c, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, g):
        a, b = x.chunk(2, 1)
        if self.flip:
            a, b = b, a
        s, t = self.net(torch.cat([a, g], 1)).chunk(2, 1)
        s = 2.0 * torch.tanh(s / 2.0)                    # bounded log-scale => stable
        yb = b * s.exp() + t
        out = torch.cat([yb, a], 1) if self.flip else torch.cat([a, yb], 1)
        return out, s.flatten(1).sum(1)


class MusicDET(BaseDetector):
    name = "musicdet"
    sr = 16000
    seg_seconds = 4.0
    zero_shot = True
    use_amp = False                                      # flows want fp32

    def __init__(self, n_mels=128, K=8, hidden=64, cond_c=8):
        super().__init__()
        assert n_mels % SQ == 0
        self.mel = torchaudio.transforms.MelSpectrogram(self.sr, n_fft=1024, hop_length=256, n_mels=n_mels)
        self.Fq = n_mels // SQ
        C = SQ * SQ
        self.cond = nn.Sequential(nn.Conv2d(1, cond_c, 1), nn.ReLU(True), nn.Conv2d(cond_c, cond_c, 1))
        self.freq_emb = nn.Parameter(torch.randn(1, cond_c, self.Fq, 1) * 0.02)
        self.norms = nn.ModuleList([ActNorm(C) for _ in range(K)])
        self.cps = nn.ModuleList([Coupling(C, cond_c, hidden, flip=i % 2 == 1) for i in range(K)])

    def spec(self, wave):
        m = self.mel(wave.float())
        m = 10 * torch.log10(m + 1e-8).clamp(min=-80)
        m = (m + 40) / 20                                # fixed (not per-clip) normalisation
        T = m.shape[-1] // SQ * SQ
        return m[..., :T].unsqueeze(1)                   # [B,1,F,T]

    def nll(self, wave):
        x = self.spec(wave)
        if self.training:
            x = x + 0.01 * torch.randn_like(x)
        # per-band mean-energy profile at squeezed resolution: [B,1,Fq,1]
        band = x.mean(-1).view(x.shape[0], 1, self.Fq, SQ).mean(-1).unsqueeze(-1)
        z = squeeze(x)
        g = self.cond(band) + self.freq_emb                                # [B,cond_c,Fq,1]
        g = g.expand(-1, -1, -1, z.shape[-1])
        logdet = torch.zeros(z.shape[0], device=z.device)
        for an, cp in zip(self.norms, self.cps):
            z, ld = an(z)
            logdet = logdet + ld
            z, ld = cp(z, g)
            logdet = logdet + ld
        logp = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(1)
        return -(logp + logdet) / z[0].numel()                             # nats / dim

    def forward(self, wave):
        return self.nll(wave).unsqueeze(-1)

    def loss(self, x, y=None):
        return self.nll(x).mean()

    def score(self, x):
        return self.nll(x)


def build(**kw):
    return MusicDET(**kw)
