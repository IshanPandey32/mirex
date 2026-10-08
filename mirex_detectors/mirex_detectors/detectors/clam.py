"""CLAM-style detector - dual-stream contrastive learning (Batra et al., "Melody or Machine:
Detecting Synthetic Music with Dual-Stream Contrastive Learning", TMLR 2025).

Verified (title/abstract level): two pretrained audio streams + a contrastive objective.
The ArtifactNet paper lists CLAM at 194M parameters, so the original is bigger than this.
NOT verified: the exact streams, projection sizes and the form of the contrastive loss. My
reconstruction:

    stream A: frozen MERT-v1-330M        (music-oriented SSL)
    stream B: frozen wav2vec2 XLS-R 300M (speech/vocal-oriented SSL)
    each: learnable softmax-weighted mix of ALL hidden layers -> MLP projection
    loss = BCE(fused logit)  +  lam * consistency loss
           real  : pull the two streams together   (1 - cos)
           fake  : push them apart past a margin   (relu(cos - margin))

i.e. the bet is that real recordings look coherent to both a music model and a speech model,
while generated audio is where the two views disagree. Only ~1.5M parameters are trained.
Both backbones are the ones your Dockerfile already caches in hf_cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import load_mert, load_w2v
from .common import BaseDetector


class CLAM(BaseDetector):
    name = "clam"
    sr = 24000                       # MERT rate; wav2vec2 gets a 16 kHz resample
    seg_seconds = 10.0
    frozen_prefixes = ("mert.", "w2v.")

    def __init__(self, dummy=False, dim=256, lam=0.5, margin=0.2):
        super().__init__()
        self.mert, self.w2v = load_mert(dummy), load_w2v(dummy)
        self.lam, self.margin = lam, margin
        self.lw_m = nn.Parameter(torch.zeros(self.mert.n_layers))
        self.lw_w = nn.Parameter(torch.zeros(self.w2v.n_layers))
        self.pm = nn.Sequential(nn.Linear(self.mert.dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pw = nn.Sequential(nn.Linear(self.w2v.dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, 1))

    def train(self, mode=True):
        super().train(mode)
        self.mert.eval()
        self.w2v.eval()
        return self

    def embed(self, wave):
        import torchaudio
        hm = self.mert.pooled(wave)                                               # [B,L,d]
        hw = self.w2v.pooled(torchaudio.functional.resample(wave.float(), self.sr, 16000))
        zm = self.pm((F.softmax(self.lw_m, 0)[None, :, None] * hm.float()).sum(1))
        zw = self.pw((F.softmax(self.lw_w, 0)[None, :, None] * hw.float()).sum(1))
        return zm, zw

    def _logit(self, zm, zw):
        zm, zw = F.normalize(zm, dim=-1), F.normalize(zw, dim=-1)
        return self.head(torch.cat([zm, zw, zm * zw, (zm - zw).abs()], -1)), zm, zw

    def forward(self, wave):
        return self._logit(*self.embed(wave))[0]

    def loss(self, x, y):
        logit, zm, zw = self._logit(*self.embed(x))
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y.float())
        cos = (zm * zw).sum(-1)
        con = torch.where(y == 0, 1 - cos, F.relu(cos - self.margin)).mean()
        return bce + self.lam * con


def build(dummy=False, **kw):
    return CLAM(dummy=dummy, **kw)
