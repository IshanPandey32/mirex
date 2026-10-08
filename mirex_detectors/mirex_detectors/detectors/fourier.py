"""Fourier-artifact detector, after Afchar et al., "A Fourier explanation of AI-music
artifacts" (ISMIR 2025, arXiv 2506.19108).

What I could verify: the paper's thesis is that the upsampling/deconvolution layers inside
neural music decoders leave periodic (checkerboard-type) peaks in the Fourier spectrum, and that
this makes a simple, explainable detector possible. I did NOT read the full method, so this is
my own minimal version of that idea, not their exact detector:

    average log-magnitude spectrum over the clip
      -> subtract a median-filtered baseline (leaves only narrow peaks)
      -> standardise -> linear classifier  (optionally a small MLP)

Because the head is linear, `peak_report()` tells you WHICH frequencies drive the decision -
the main reason to keep this model in the zoo: it is a sanity check for the other five.
It is also the most codec-sensitive model here (MP3/AAC rewrite exactly the band it looks at),
so it is a good canary for the confound gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseDetector


class FourierPeaks(BaseDetector):
    name = "fourier"
    sr = 44100
    seg_seconds = 4.0

    def __init__(self, n_fft=4096, hop=2048, baseline_k=65, pool=2, hidden=0):
        super().__init__()
        self.n_fft, self.hop, self.k, self.pool = n_fft, hop, baseline_k, pool
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        d = (n_fft // 2) // pool                       # drop DC bin, then max-pool
        self.bn = nn.BatchNorm1d(d, affine=False)
        self.head = (nn.Linear(d, 1) if hidden == 0 else
                     nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1)))

    def peaks(self, wave):
        spec = torch.stft(wave.float(), self.n_fft, self.hop, window=self.win, return_complex=True).abs()
        avg = torch.log(spec.mean(-1) + 1e-8)[:, 1:]                       # [B, n_fft/2]
        p = self.k // 2
        base = F.pad(avg.unsqueeze(1), (p, p), mode="reflect").unfold(2, self.k, 1).median(-1).values
        return F.max_pool1d((avg.unsqueeze(1) - base), self.pool).squeeze(1)  # peaks only

    def forward(self, wave):
        return self.head(self.bn(self.peaks(wave)))

    @torch.no_grad()
    def peak_report(self, topk=15):
        """Frequencies (Hz) with the largest |weight| in the linear head."""
        if not isinstance(self.head, nn.Linear):
            return []
        w = self.head.weight.squeeze(0)
        hz = (torch.arange(len(w)) * self.pool + 1) * (self.sr / self.n_fft)
        idx = w.abs().topk(min(topk, len(w))).indices
        return [(round(float(hz[i]), 1), round(float(w[i]), 4)) for i in idx]


def build(**kw):
    return FourierPeaks(**kw)
