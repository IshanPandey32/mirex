"""Fusion Segment Transformer (Kim & Go, MIPPIA, ICASSP 2026, arXiv 2601.13647).

Follows the paper's two-stage design, which I read in full:

  Stage-1  segment embeddings. Track -> 4-bar segments (downbeats from Beat This!) -> frozen
           MERT embedding per segment. Done ONCE per track and cached (`embed` step), because
           running a 330M encoder on 48 segments inside every training step is not viable.
  Stage-2  this file's `FST` module (trained, ~4M params):
           * embedding stream : Transformer over the N segment embeddings
           * SSM stream       : SSM_ij = exp(-||e_i - e_j||^2 / d)  -> linear -> Transformer
           * bi-directional cross-attention between the streams (+ residual + norm)
           * gated fusion     : X = G*X_contents + (1-G)*X_structure, G = sigmoid(W[Xc; Xs] + b)
           * masked mean-pool -> BCE head.   Tracks padded/cropped to 48 segments.

Deviations from the paper, deliberately: (1) Stage-1 here is a FROZEN MERT with layer-averaged
mean pooling - the paper fine-tunes its extractor with its AudioCAT decoder (170M params);
(2) if `beat_this` is not installed, segments fall back to fixed 10 s windows, which the paper's
own ablation shows costs accuracy (0.9867 -> 0.966 on AIME). `pip install beat-this` to get the
real 4-bar segmentation. Expect lower numbers than the paper until Stage-1 is fine-tuned.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .backbones import load_mert
from .common import DATA_DIR, BaseDetector

EMB_SR = 24000
MAX_SEG = 48
SEG_CAP_S = 20.0           # never feed the encoder more than 20 s per segment
DEFAULT_CACHE = DATA_DIR / "fst_cache"


def cache_path(cache_dir: Path, audio_path: str) -> Path:
    return Path(cache_dir) / (hashlib.md5(audio_path.encode()).hexdigest()[:16] + ".pt")


# --------------------------------------------------------------------------
# Stage-1: segmentation + embedding
# --------------------------------------------------------------------------
_F2B = {}


def _downbeats(path: str, device: str):
    try:
        if device not in _F2B:
            from beat_this.inference import File2Beats
            _F2B[device] = File2Beats(device=device, dbn=False)
        _, downbeats = _F2B[device](path)
        return np.asarray(downbeats, dtype=float)
    except Exception:
        return None


def _bounds(path: str, dur: float, device: str, fallback_s=10.0):
    db = _downbeats(path, device)
    if db is not None and len(db) >= 5:
        starts = db[::4]                                   # 4 bars per segment
        ends = np.append(starts[1:], min(dur, starts[-1] + (starts[-1] - starts[-2])))
        return [(float(s), float(e)) for s, e in zip(starts, ends) if e - s > 1.0]
    return [(t, min(t + fallback_s, dur)) for t in np.arange(0.0, max(dur - 2.0, 0.1), fallback_s)]


@torch.no_grad()
def embed_rows(rows, cache_dir: Path, encoder, device: str, log=print) -> int:
    import soundfile as sf
    import torchaudio
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder.to(device).eval()
    done = 0
    for n, r in enumerate(rows):
        out = cache_path(cache_dir, r["path"])
        if out.exists():
            continue
        try:
            x, fs = sf.read(r["path"], dtype="float32", always_2d=True)
            x = torch.from_numpy(x.mean(1))
            if fs != EMB_SR:
                x = torchaudio.functional.resample(x, fs, EMB_SR)
            dur = len(x) / EMB_SR
            segs = _bounds(r["path"], dur, device)[:MAX_SEG]
            clips = [x[int(s * EMB_SR): int(min(e, s + SEG_CAP_S) * EMB_SR)] for s, e in segs]
            clips = [c for c in clips if len(c) > EMB_SR]
            L = max(len(c) for c in clips)
            batch = torch.stack([torch.nn.functional.pad(c, (0, L - len(c))) for c in clips])
            embs = []
            for i in range(0, len(batch), 8):
                embs.append(encoder.pooled(batch[i:i + 8].to(device)).mean(1).cpu())   # layer-avg
            torch.save(torch.cat(embs).half(), out)
            done += 1
        except Exception as e:
            log(f"[warn] embed failed {r['path']}: {e}")
        if n % 100 == 0:
            log(f"  embedded {n}/{len(rows)}")
    return done


class FSTEmbDataset(torch.utils.data.Dataset):
    """One item per track: x = [MAX_SEG, dim+1], last column is the validity mask."""

    def __init__(self, rows, cache_dir, dim):
        self.rows, self.dir, self.dim = rows, Path(cache_dir), dim
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = cache_path(self.dir, r["path"])
        x = torch.zeros(MAX_SEG, self.dim + 1)
        if not p.exists():
            return x, -1, i                                 # not embedded yet -> skipped
        e = torch.load(p).float()[:MAX_SEG]
        x[:len(e), :self.dim] = e
        x[:len(e), self.dim] = 1.0
        return x, r["label"], i


# --------------------------------------------------------------------------
# Stage-2
# --------------------------------------------------------------------------
class FST(BaseDetector):
    name = "fst"
    input_kind = "fst_emb"
    use_amp = False                                         # exp()/cdist want fp32

    def __init__(self, in_dim=1024, d=256, heads=8, layers=2, cache_dir=DEFAULT_CACHE):
        super().__init__()
        self.in_dim, self.cache_dir = in_dim, Path(cache_dir)
        self.proj_e = nn.Linear(in_dim, d)
        self.proj_s = nn.Linear(MAX_SEG, d)                 # each SSM row -> d
        self.pos = nn.Parameter(torch.randn(1, MAX_SEG, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu",
                                                   batch_first=True, norm_first=True)
        self.enc_e = nn.TransformerEncoder(layer(), layers, enable_nested_tensor=False)
        self.enc_s = nn.TransformerEncoder(layer(), layers, enable_nested_tensor=False)
        self.ca_e = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)   # emb  <- ssm
        self.ca_s = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)   # ssm  <- emb
        self.n_e, self.n_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))

    def make_dataset(self, rows, train):
        return FSTEmbDataset(rows, self.cache_dir, self.in_dim)

    def forward(self, x):
        emb, valid = x[..., :-1], x[..., -1] > 0.5
        valid = valid.clone()
        valid[:, 0] = True                                  # never an all-masked row (NaN guard)
        pad = ~valid
        ssm = torch.exp(-torch.cdist(emb, emb).pow(2) / self.in_dim)
        ssm = ssm * (valid[:, :, None] & valid[:, None, :])
        xe = self.enc_e(self.proj_e(emb) + self.pos, src_key_padding_mask=pad)
        xs = self.enc_s(self.proj_s(ssm) + self.pos, src_key_padding_mask=pad)
        x_c = self.n_e(xe + self.ca_e(xe, xs, xs, key_padding_mask=pad, need_weights=False)[0])
        x_s = self.n_s(xs + self.ca_s(xs, xe, xe, key_padding_mask=pad, need_weights=False)[0])
        g = torch.sigmoid(self.gate(torch.cat([x_c, x_s], -1)))
        fused = g * x_c + (1 - g) * x_s
        m = valid.unsqueeze(-1).float()
        return self.out((fused * m).sum(1) / m.sum(1))


def build(dummy=False, cache_dir=DEFAULT_CACHE, **kw):
    return FST(in_dim=64 if dummy else 1024, cache_dir=cache_dir, **kw)


def build_encoder(dummy=False):
    return load_mert(dummy)
