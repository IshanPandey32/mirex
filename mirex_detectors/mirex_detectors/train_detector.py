#!/usr/bin/env python3
"""Train ONE detector on ONE leave-one-generator-out fold. One job owns one GPU
(run_detectors.py launches many of these in parallel).

    python train_detector.py --model spectttra --holdout suno
    python train_detector.py --model fourier   --holdout none          # the 'full' fold
    python train_detector.py --model clam --smoke                      # CPU, ~1 min, no downloads
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import build, MODELS                       # noqa: E402
from detectors.common import (DATA_DIR, FOLDS, fit, fold_dir, load_manifest,  # noqa: E402
                              make_split)

# per-model sensible defaults (override on the command line)
DEFAULTS = {
    "artifactnet": dict(lr=1e-3, bs=16),
    "musicdet":    dict(lr=5e-4, bs=16),
    "fst":         dict(lr=1e-4, bs=32),
    "clam":        dict(lr=3e-4, bs=8),
    "spectttra":   dict(lr=1e-4, bs=32),
    "fourier":     dict(lr=1e-3, bs=64),
}


def synth_manifest(root: Path, n_real=30, n_per_fam=15, seconds=12, sr=22050):
    """Tiny synthetic corpus so the plumbing can be exercised without any real data."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    rows = []
    for fam, n in [("real", n_real), ("suno", n_per_fam), ("udio", n_per_fam)]:
        for i in range(n):
            f0 = rng.uniform(100, 400)
            x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
            x = x + 0.05 * rng.standard_normal(len(t))
            if fam != "real":                              # periodic spectral comb = fake 'artifact'
                x = x + 0.03 * np.sin(2 * np.pi * (3000 + 10 * (fam == "udio")) * t)
            x = (x / np.abs(x).max() * 0.5).astype("float32")
            p = root / f"{fam}_{i}.wav"
            sf.write(p, x, sr)
            rows.append({"path": str(p), "label": int(fam != "real"),
                         "family": "real" if fam == "real" else fam})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--holdout", default="none", choices=FOLDS)
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.jsonl"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, CPU, dummy backbones")
    a = ap.parse_args(argv)

    cfg = {**DEFAULTS[a.model]}
    bs = a.bs or cfg["bs"]
    lr = a.lr or cfg["lr"]
    kw = {}
    tmp = None
    if a.smoke:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        rows = synth_manifest(root)
        a.device, a.epochs, a.workers, bs = "cpu", 2, 0, 4
        a.holdout = "suno" if a.holdout == "none" else a.holdout
        out = root / "ckpt"
        if a.model in ("clam", "fst"):
            kw["dummy"] = True
        if a.model == "fst":
            kw["cache_dir"] = root / "fst_cache"
    else:
        rows = load_manifest(a.manifest)
        out = Path(a.out) if a.out else fold_dir(a.model, a.holdout)

    model = build(a.model, **kw)
    if a.model == "fst":
        from detectors.fst import embed_rows, build_encoder, cache_path
        if a.smoke:
            embed_rows(rows, model.cache_dir, build_encoder(dummy=True), "cpu")
        else:
            missing = sum(1 for r in rows[:200] if not cache_path(model.cache_dir, r["path"]).exists())
            if missing > len(rows[:200]) // 2:
                sys.exit("FST needs Stage-1 embeddings first: python run_detectors.py embed")

    train, val, test = make_split(rows, a.holdout)
    print(f"[{a.model}] holdout={a.holdout}  train {len(train)}  val {len(val)}  test {len(test)}  "
          f"device={a.device}  bs={bs} lr={lr}", flush=True)
    if not train or not test:
        sys.exit("empty split - check the manifest")

    res = fit(model, train, val, test, out, epochs=a.epochs, bs=bs, lr=lr,
              workers=a.workers, device=a.device)
    if a.model == "fourier":
        print("  top Fourier bins (Hz, weight):", model.peak_report(), flush=True)
    if a.smoke:
        print("SMOKE OK", json.dumps({k: res[k] for k in ("auroc", "eer")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
