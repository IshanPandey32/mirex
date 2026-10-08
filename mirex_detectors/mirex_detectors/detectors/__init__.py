"""Detector zoo registry.

    from detectors import build, MODELS
    model = build("spectttra")

Every model is a `BaseDetector`: forward(x) -> logit / loss(x, y) / score(x) (higher = more AI).
"""
from __future__ import annotations

import importlib

MODELS = {
    "artifactnet": "detectors.artifactnet",   # residual-physics UNet + HPSS + CNN
    "musicdet":    "detectors.musicdet",      # zero-shot, flow on real music only
    "fst":         "detectors.fst",           # Fusion Segment Transformer (2-stage)
    "clam":        "detectors.clam",          # dual-stream MERT + wav2vec2, contrastive
    "spectttra":   "detectors.spectttra",     # SONICS spectro-temporal tokens transformer
    "fourier":     "detectors.fourier",       # baseline-removed Fourier-peak linear detector
}


def build(name: str, **kw):
    if name not in MODELS:
        raise SystemExit(f"unknown model '{name}'. choices: {', '.join(MODELS)}")
    return importlib.import_module(MODELS[name]).build(**kw)


def make_dataset(model, rows, train: bool):
    if model.input_kind == "wave":
        from .common import SegDataset
        return SegDataset(rows, model.sr, model.seg_seconds, train)
    return model.make_dataset(rows, train)
