"""Shared plumbing for the detector zoo.

Manifest -> leave-one-generator-out split -> segment dataset -> trainer -> metrics.
Every model in this package exposes the same tiny contract (see detectors/__init__.py),
so the trainer never needs to know which architecture it is driving.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MIREX_DATA_DIR", ROOT / "data"))
CKPT_DIR = Path(os.environ.get("MIREX_CHECKPOINT_DIR", ROOT / "checkpoints"))
ZOO_DIR = CKPT_DIR / "zoo"

FOLDS = ["suno", "udio", "mureka", "minimax", "yue", "ace-step", "none"]


def fold_dir(model: str, fold: str) -> Path:
    return ZOO_DIR / model / ("full" if fold == "none" else f"logo_{fold}")


# --------------------------------------------------------------------------
# manifest + split
# --------------------------------------------------------------------------
def load_manifest(path: str | Path) -> list[dict]:
    """JSONL, one track per line: {"path": ..., "label": 0|1, "generator_family": ...}.
    label 1 = AI-generated. Real tracks get family "real"."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            label = int(r["label"])
            fam = r.get("generator_family") or r.get("family")
            rows.append({"path": r["path"], "label": label,
                         "family": "real" if label == 0 else str(fam or "unknown")})
    return rows


def _bucket(path: str) -> int:
    return int(hashlib.md5(path.encode()).hexdigest()[:8], 16) % 100


def make_split(rows: list[dict], holdout: str):
    """Deterministic, track-level split.

    bucket <10  -> test   (real tracks only matter for LOGO folds)
    bucket <20  -> val
    otherwise   -> train
    A held-out generator family goes ENTIRELY to test and never touches train/val.
    """
    train, val, test = [], [], []
    for r in rows:
        if holdout != "none" and r["label"] == 1 and r["family"] == holdout:
            test.append(r)
            continue
        b = _bucket(r["path"])
        (test if b < 10 else val if b < 20 else train).append(r)
    return train, val, test


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def load_segment(path: str, sr: int, seconds: float, frac: float | None) -> np.ndarray:
    """Read one mono segment at `sr`. frac=None -> random crop, else deterministic position."""
    import soundfile as sf
    info = sf.info(path)
    fs, n = info.samplerate, info.frames
    need = int(round(seconds * fs))
    if n <= need:
        start = 0
    elif frac is None:
        start = random.randint(0, n - need)
    else:
        start = int(frac * (n - need))
    x, _ = sf.read(path, start=start, frames=need, dtype="float32", always_2d=True)
    x = x.mean(1)
    if fs != sr:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), fs, sr).numpy()
    target = int(round(seconds * sr))
    if len(x) < target:
        x = np.pad(x, (0, target - len(x)))
    return x[:target]


class SegDataset(Dataset):
    """Train: one random crop per track per epoch. Eval: `n_eval` fixed crops per track.
    Returns (wave, label, track_index). label == -1 marks an unreadable file (skipped)."""

    def __init__(self, rows, sr, seconds, train, n_eval=3):
        self.rows, self.sr, self.sec, self.train, self.n_eval = rows, sr, seconds, train, n_eval
        self.items = ([(i, None) for i in range(len(rows))] if train else
                      [(i, (k + 0.5) / n_eval) for i in range(len(rows)) for k in range(n_eval)])
        self.labels = [rows[i]["label"] for i, _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i, frac = self.items[j]
        try:
            x = load_segment(self.rows[i]["path"], self.sr, self.sec, frac)
            return torch.from_numpy(x), self.rows[i]["label"], i
        except Exception as e:                       # corrupt file: never kill a 3-day job
            print(f"[warn] unreadable {self.rows[i]['path']}: {e}", flush=True)
            return torch.zeros(int(self.sr * self.sec)), -1, i


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def eer(y, s) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2)


# --------------------------------------------------------------------------
# prediction + training
# --------------------------------------------------------------------------
def _autocast(device: str, enabled: bool):
    if device.startswith("cuda") and enabled:
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, loader, device) -> dict[int, float]:
    """track_index -> median score over that track's segments (higher = more AI)."""
    model.eval()
    acc: dict[int, list[float]] = {}
    for x, y, idx in loader:
        with _autocast(device, model.use_amp):
            s = model.score(x.to(device)).float().cpu().numpy()
        s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
        for si, yi, ii in zip(s, y.tolist(), idx.tolist()):
            if yi >= 0:
                acc.setdefault(ii, []).append(float(si))
    return {i: float(np.median(v)) for i, v in acc.items()}


def _collect(rows, scores):
    ids = sorted(scores)
    return ([rows[i]["label"] for i in ids], [scores[i] for i in ids],
            [rows[i]["family"] for i in ids], [rows[i]["path"] for i in ids])


def _loader(model, rows, train, bs, workers, balanced=False):
    from . import make_dataset
    ds = make_dataset(model, rows, train)
    if train and balanced and len(set(ds.labels)) == 2:
        lab = np.array(ds.labels)
        w = np.where(lab == 1, 0.5 / max(lab.sum(), 1), 0.5 / max((1 - lab).sum(), 1))
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
        return DataLoader(ds, bs, sampler=sampler, num_workers=workers, drop_last=True,
                          persistent_workers=workers > 0)
    return DataLoader(ds, bs, shuffle=train, num_workers=workers, drop_last=train and len(ds) >= bs,
                      persistent_workers=train and workers > 0)


def fit(model, train_rows, val_rows, test_rows, out_dir: Path, *, epochs=10, bs=16, lr=3e-4,
        workers=4, device="cuda", patience=3, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    zero = model.zero_shot
    if zero:                                   # zero-shot detectors never see an AI track
        train_rows = [r for r in train_rows if r["label"] == 0]
    tr = _loader(model, train_rows, True, bs, workers, balanced=not zero)
    va = _loader(model, val_rows, False, bs, workers)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"  trainable params: {sum(p.numel() for p in params)/1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    total = max(1, epochs * len(tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    use_fp16 = device.startswith("cuda") and model.use_amp and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    best, bad, best_path = -1e18, 0, out_dir / "best.ckpt"
    for ep in range(1, epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y, _ in tr:
            keep = y >= 0
            if not keep.any():
                continue
            x, y = x[keep].to(device), y[keep].to(device)
            with _autocast(device, model.use_amp):
                loss = model.loss(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run, n = run + float(loss), n + 1
        sc = predict(model, va, device)
        y, s, _, _ = _collect(val_rows, sc)
        if zero:                               # strictly zero-shot selection: real-only likelihood
            metric = -float(np.mean([si for yi, si in zip(y, s) if yi == 0] or [0.0]))
        else:
            metric = auroc(y, s)
            metric = metric if metric == metric else -float(np.mean(s) if s else 0.0)
        if metric != metric:                   # empty/degenerate val set: still save something
            metric = -1e9
        log(f"  epoch {ep:2d}/{epochs}  loss {run/max(n,1):.4f}  val {metric:.4f}  "
            f"({time.time()-t0:.0f}s)")
        if metric > best:
            best, bad = metric, 0
            sd = {k: v for k, v in model.state_dict().items()
                  if not k.startswith(tuple(model.frozen_prefixes))}
            torch.save({"model": sd, "name": model.name, "val_metric": metric}, best_path)
        else:
            bad += 1
            if bad >= patience:
                log("  early stop")
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["model"], strict=False)
    return evaluate(model, val_rows, test_rows, out_dir, bs, workers, device, log)


def evaluate(model, val_rows, test_rows, out_dir: Path, bs, workers, device, log=print) -> dict:
    va_y, va_s, _, _ = _collect(val_rows, predict(model, _loader(model, val_rows, False, bs, workers), device))
    real_va = np.array([s for y, s in zip(va_y, va_s) if y == 0] or [0.0])
    mu, sd = float(real_va.mean()), float(real_va.std() + 1e-6)
    thr = float(np.quantile(real_va, 0.95))                  # 5 % FPR on validation real

    sc = predict(model, _loader(model, test_rows, False, bs, workers), device)
    y, s, fam, paths = _collect(test_rows, sc)
    y_, s_ = np.array(y), np.array(s)
    res = {"model": model.name, "n_test": len(y), "auroc": auroc(y, s),
           "eer": eer(y, s) if len(set(y)) == 2 else float("nan"),
           "thr_5pct_fpr": thr, "val_real_mu": mu, "val_real_sd": sd,
           "fpr_at_thr": float((s_[y_ == 0] > thr).mean()) if (y_ == 0).any() else float("nan"),
           "tpr_by_family": {f: float((s_[(np.array(fam) == f)] > thr).mean())
                             for f in sorted(set(fam)) if f != "real"}}
    with open(out_dir / "test_scores.jsonl", "w") as fh:
        for p, yi, f, si in zip(paths, y, fam, s):
            fh.write(json.dumps({"path": p, "label": yi, "family": f, "score": si}) + "\n")
    (out_dir / "test.json").write_text(json.dumps(res, indent=2))
    log(f"  TEST auroc {res['auroc']:.4f}  eer {res['eer']:.4f}  fpr@thr {res['fpr_at_thr']:.3f}")
    for f, v in res["tpr_by_family"].items():
        log(f"    TPR {f:10s} {v:.3f}")
    return res


class BaseDetector(nn.Module):
    """Contract every model follows. Subclasses set the class attributes and override
    forward() (a logit). loss()/score() default to BCE / the raw logit."""
    name = "base"
    sr = 16000
    seg_seconds = 5.0
    input_kind = "wave"          # "wave" -> SegDataset, anything else -> model.make_dataset()
    zero_shot = False            # True -> trained on real audio only, score = anomaly
    use_amp = True
    frozen_prefixes: tuple = ("__none__",)

    def loss(self, x, y):
        return nn.functional.binary_cross_entropy_with_logits(self(x).squeeze(-1), y.float())

    def score(self, x):
        return self(x).squeeze(-1)
