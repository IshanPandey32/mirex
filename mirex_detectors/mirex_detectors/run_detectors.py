#!/usr/bin/env python3
"""Detector-zoo driver - same shape as the MIREX run.py, stdlib only until it spawns jobs.

    python run_detectors.py preflight
    python run_detectors.py manifest            # build manifest.jsonl from your metadata DB
    python run_detectors.py smoke               # every model on synthetic data (CPU, no downloads)
    python run_detectors.py embed               # FST Stage-1 embeddings (once, needs a GPU)
    python run_detectors.py train               # 6 models x 7 LOGO folds across the GPUs
    python run_detectors.py oof                 # merge held-out scores for the fusion step

Run it with the same venv as run.py (it uses ./.venv if present, else the current python).
Every step is resumable: a fold with an existing checkpoint is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"
DATA_DIR = Path(os.environ.setdefault("MIREX_DATA_DIR", str(ROOT / "data")))
CKPT_DIR = Path(os.environ.setdefault("MIREX_CHECKPOINT_DIR", str(ROOT / "checkpoints")))
ZOO = CKPT_DIR / "zoo"
LOGS = ROOT / "logs"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
PY = str(VENV_PY if VENV_PY.exists() else sys.executable)

MODELS = ["artifactnet", "musicdet", "fst", "clam", "spectttra", "fourier"]
FOLDS = ["suno", "udio", "mureka", "minimax", "yue", "ace-step", "none"]
MANIFEST = DATA_DIR / "manifest.jsonl"


def _p(code, tag, msg):
    on = sys.stdout.isatty() and not WINDOWS
    print(f"\033[{code}m{tag}\033[0m {msg}" if on else f"{tag} {msg}", flush=True)

def say(m):  _p("1", "==>", m)
def ok(m):   _p("32", "  ok", m)
def warn(m): _p("33", " warn", m)
def die(m):  _p("31", " fail", m); sys.exit(1)


def fold_dir(model, fold):
    return ZOO / model / ("full" if fold == "none" else f"logo_{fold}")


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    g = []
    for line in out.strip().splitlines():
        p = [f.strip() for f in line.split(",")]
        if len(p) >= 3:
            g.append({"index": int(p[0]), "name": p[1], "mem_mb": int(float(p[2]))})
    return g


# --------------------------------------------------------------------------
def step_preflight(_a):
    say("GPUs")
    gpus = gpu_info()
    if not gpus:
        warn("no GPU visible - CLAM / FST / ArtifactNet need one for real runs")
    for g in gpus:
        print(f"  gpu{g['index']}  {g['name']}  {g['mem_mb']} MiB")
    say("Python deps")
    for mod in ("torch", "torchaudio", "soundfile", "sklearn", "numpy"):
        r = subprocess.run([PY, "-c", f"import {mod}"], capture_output=True)
        (ok if r.returncode == 0 else warn)(f"{mod}" + ("" if r.returncode == 0 else "  MISSING"))
    for mod, why in (("transformers", "CLAM / FST backbones"), ("beat_this", "FST 4-bar segmentation")):
        r = subprocess.run([PY, "-c", f"import {mod}"], capture_output=True)
        (ok if r.returncode == 0 else warn)(f"{mod}  ({why})" + ("" if r.returncode == 0 else "  MISSING"))
    say("Paths")
    print(f"  manifest   {MANIFEST}  ({'found' if MANIFEST.exists() else 'MISSING - run `manifest`'})")
    print(f"  checkpoints {ZOO}")
    ok("preflight done")


def step_manifest(a):
    """Dump (path, label, generator_family) from the pipeline's MetadataDatabase."""
    code = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from metadata_db import MetadataDatabase;"
        "rows = MetadataDatabase().fetch();"
        "keys = ('path','file_path','filepath','audio_path');"
        "n = 0;"
        f"fh = open({str(MANIFEST)!r}, 'w');"
        "miss = 0\n"
        "for r in rows:\n"
        "    p = next((r[k] for k in keys if k in r.keys() and r[k]), None)\n"
        "    if p is None or not __import__('os').path.exists(str(p)) "
        "or 'sdd' in str(p).lower(): miss += 1; continue\n"
        "    fh.write(json.dumps({'path': str(p), 'label': int(r['is_ai']), "
        "'generator_family': r['generator_family']}) + '\\n'); n += 1\n"
        "print(f'  wrote {n} rows, {miss} without a path column')\n"
        "if n == 0: raise SystemExit('no path column found - edit `keys` in step_manifest')"
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run([PY, "-c", code], cwd=a.repo).returncode
    if rc != 0:
        die(f"manifest build failed (run from the MIREX repo, or pass --repo): rc={rc}")
    ok(f"manifest -> {MANIFEST}")


def step_smoke(a):
    bad = []
    for m in a.models:
        say(f"smoke: {m}")
        rc = subprocess.run([PY, "train_detector.py", "--model", m, "--smoke"], cwd=ROOT).returncode
        (ok if rc == 0 else warn)(f"{m} {'passed' if rc == 0 else 'FAILED'}")
        if rc != 0:
            bad.append(m)
    if bad:
        die(f"smoke failed for: {', '.join(bad)} - fix before spending GPU time")
    ok("all smoke tests passed")


def step_embed(a):
    say("FST Stage-1 embeddings (frozen MERT, cached once)")
    if not MANIFEST.exists():
        die("no manifest - run `manifest` first")
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from detectors.common import load_manifest;"
        "from detectors.fst import embed_rows, build_encoder, DEFAULT_CACHE;"
        f"rows = load_manifest({str(MANIFEST)!r});"
        "print(len(rows), 'tracks ->', DEFAULT_CACHE);"
        f"embed_rows(rows, DEFAULT_CACHE, build_encoder(), {a.device!r})"
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "fst_embed.log", "ab") as fh:
        rc = subprocess.run([PY, "-c", code], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        die(f"embedding failed - see {LOGS / 'fst_embed.log'}")
    ok("embeddings cached (re-run is free: finished tracks are skipped)")


def step_train(a):
    if not MANIFEST.exists():
        die("no manifest - run `manifest` first")
    gpus = [g["index"] for g in gpu_info()] or [0]
    if a.gpus:
        gpus = gpus[:a.gpus]
    queue = []
    for m in a.models:
        for f in a.folds:
            d = fold_dir(m, f)
            if d.exists() and (d / "test.json").exists():
                ok(f"skip {m}/{f} (done)")
            else:
                queue.append((m, f))
    if not queue:
        ok("everything already trained")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    say(f"{len(queue)} job(s) across {len(gpus)} GPU(s), {a.epochs} epochs each")
    running, started, failed, t0 = {}, 0, 0, time.time()
    while queue or running:
        while queue and len(running) < len(gpus):
            gpu = next(g for g in gpus if g not in running)
            m, f = queue.pop(0)
            started += 1
            log = LOGS / f"zoo_{m}_{f}.log"
            print(f"  [{started:2d}] gpu{gpu}  {m:12s} holdout {f:<9s} -> {log}", flush=True)
            proc = subprocess.Popen(
                [PY, "train_detector.py", "--model", m, "--holdout", f, "--epochs", str(a.epochs),
                 "--workers", str(a.workers), "--manifest", str(MANIFEST)],
                cwd=ROOT, stdout=open(log, "wb"), stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[gpu] = (proc, m, f, log)
        time.sleep(5)
        for gpu in list(running):
            proc, m, f, log = running[gpu]
            if proc.poll() is not None:
                if proc.returncode == 0:
                    ok(f"done  {m}/{f}")
                else:
                    failed += 1
                    _p("31", "  FAIL", f"{m}/{f} - see {log}")
                del running[gpu]
    say(f"finished in {(time.time() - t0) / 60:.0f} min, {failed} failed")
    step_report(a)


def step_report(_a):
    say("held-out results (AUROC / EER; fold = generator the model never saw)")
    print(f"  {'model':12s} " + " ".join(f"{f:>9s}" for f in FOLDS))
    for m in MODELS:
        cells = []
        for f in FOLDS:
            p = fold_dir(m, f) / "test.json"
            cells.append(f"{json.loads(p.read_text())['auroc']:9.3f}" if p.exists() else f"{'-':>9s}")
        print(f"  {m:12s} " + " ".join(cells))
    warn("rank by WORST held-out fold, not the mean - a detector is only as good as the generator it misses")


def step_oof(a):
    """Merge each model's held-out scores into rows for the fusion step:
       {"scores": {model: p}, "label": 0|1, "fold": "suno", "path": ...}
       p = sigmoid of the raw score for classifiers; for zero-shot models the score is first
       z-scored against that fold's real-validation scores. NOTE: your StackedFusion expects
       branch keys a..e - map these model names to them (or edit the fusion script)."""
    import math
    out, n_rows = open(a.oof, "w"), 0
    for f in a.folds:
        if f == "none":
            continue
        per, missing = {}, []
        for m in a.models:
            d = fold_dir(m, f)
            if not (d / "test_scores.jsonl").exists():
                missing.append(m)
                continue
            meta = json.loads((d / "test.json").read_text())
            zero = m == "musicdet"
            for line in open(d / "test_scores.jsonl"):
                r = json.loads(line)
                s = (r["score"] - meta["val_real_mu"]) / meta["val_real_sd"] if zero else r["score"]
                p = 1 / (1 + math.exp(-max(min(s, 30), -30)))
                per.setdefault(r["path"], {"label": r["label"], "scores": {}})["scores"][m] = p
        if missing:
            warn(f"fold {f}: skipped - not trained yet for {missing}")
            continue
        for path, v in per.items():
            if len(v["scores"]) == len(a.models):          # scored by every model
                out.write(json.dumps({"scores": v["scores"], "label": v["label"],
                                      "fold": f, "path": path}) + "\n")
                n_rows += 1
    out.close()
    ok(f"{n_rows} out-of-fold rows -> {a.oof}")


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_detectors.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, folds=False, models=True):
        if models:
            p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
        if folds:
            p.add_argument("--folds", nargs="+", default=FOLDS, choices=FOLDS)

    sub.add_parser("preflight").set_defaults(fn=step_preflight)
    p = sub.add_parser("manifest")
    p.add_argument("--repo", default=".", help="path of the MIREX repo (has src/metadata_db.py)")
    p.set_defaults(fn=step_manifest)
    p = sub.add_parser("smoke"); common(p); p.set_defaults(fn=step_smoke)
    p = sub.add_parser("embed"); p.add_argument("--device", default="cuda"); p.set_defaults(fn=step_embed)
    p = sub.add_parser("train"); common(p, folds=True)
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)))
    p.add_argument("--workers", type=int, default=2 if WINDOWS else 6)
    p.add_argument("--gpus", type=int, default=None)
    p.set_defaults(fn=step_train)
    p = sub.add_parser("report"); p.set_defaults(fn=step_report)
    p = sub.add_parser("oof"); common(p, folds=True)
    p.add_argument("--oof", default="oof_scores.jsonl"); p.set_defaults(fn=step_oof)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        warn("interrupted - every step is resumable, run it again")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
