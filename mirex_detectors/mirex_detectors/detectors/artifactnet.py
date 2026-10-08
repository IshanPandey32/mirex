"""ArtifactNet-style detector (Oh, 2026, arXiv 2604.16254 - "forensic residual physics").

Published recipe, as I could verify it from the abstract and model card:
    STFT magnitude -> bounded-mask UNet extracts a codec residual
                   -> HPSS splits it into 7 forensic channels
                   -> compact CNN -> P(AI)           (~4M params total)
NOT verified (the paper's full text was not available to me): the exact composition of the
7 channels, the UNet depth/width, and any auxiliary losses. The channel set below is my own
reasonable choice - treat this as "ArtifactNet-style", and diff it against the paper/code
before you quote numbers. Codec-aware (WAV/MP3/AAC/Opus) augmentation, which the paper
credits for cross-codec robustness, is NOT included: do it at manifest level by adding
re-encoded copies of real tracks, or your confound gate will flag the difference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseDetector


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class ArtifactUNet(nn.Module):
    """Predicts a mask in (0,1) over the log-magnitude spectrogram ("bounded mask")."""

    def __init__(self, base=32, depth=3):
        super().__init__()
        self.depth = depth
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(1, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.dec = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        x = F.pad(x, (0, (-W) % m, 0, (-H) % m), mode="reflect")
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for blk in self.dec:
            s = skips.pop()
            x = blk(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="nearest"), s], 1))
        return torch.sigmoid(self.out(x))[..., :H, :W]


def _median(x, k, dim):
    """Median filter along dim 3 (time) or dim 2 (freq) of a [B,1,F,T] tensor."""
    p = k // 2
    xp = F.pad(x, (p, p, 0, 0) if dim == 3 else (0, 0, p, p), mode="reflect")
    return xp.unfold(dim, k, 1).median(-1).values


class ArtifactNet(BaseDetector):
    name = "artifactnet"
    sr = 44100
    seg_seconds = 4.0

    def __init__(self, n_fft=1024, hop=512, base=32, hpss_k=15):
        super().__init__()
        self.n_fft, self.hop, self.k = n_fft, hop, hpss_k
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.unet = ArtifactUNet(base)
        self.bn = nn.BatchNorm2d(7)
        c = [7, 32, 64, 96, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(c[i + 1]),
                       nn.ReLU(True), nn.MaxPool2d(2)]
        self.cnn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * c[-1], 1))

    def features(self, wave):
        wave = wave.float()
        spec = torch.stft(wave, self.n_fft, self.hop, window=self.win, return_complex=True)
        S = torch.log1p(spec.abs()).unsqueeze(1)                      # [B,1,F,T]
        M = self.unet(S)                                               # bounded mask
        R = M * S                                                      # codec residual
        Hm, Pm = _median(R, self.k, 3), _median(R, self.k, 2)         # HPSS (soft masks)
        w = Hm ** 2 / (Hm ** 2 + Pm ** 2 + 1e-8)
        Hh, Ph = w * R, (1 - w) * R
        ratio = (Hh / (R + 1e-6)).clamp(0, 1)
        return torch.cat([S, R, Hh, Ph, S - R, M, ratio], 1)          # 7 channels

    def forward(self, wave):
        z = self.cnn(self.bn(self.features(wave)))
        return self.head(torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1))


def build(**kw):
    return ArtifactNet(**kw)
