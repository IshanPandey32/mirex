# Detector zoo - six AI-music detectors on your MIREX pipeline

Drop this folder next to `run.py` (same venv). Uses the same env vars: MIREX_DATA_DIR, MIREX_CHECKPOINT_DIR.

    pip install -r requirements-detectors.txt
    python run_detectors.py preflight
    python run_detectors.py smoke --models fourier spectttra musicdet artifactnet   # no downloads
    python run_detectors.py smoke --models clam fst                                 # dummy backbones
    python run_detectors.py manifest --repo /path/to/mirex-repo
    python run_detectors.py embed          # FST Stage-1, once
    python run_detectors.py train          # 6 models x 7 LOGO folds, one job per GPU
    python run_detectors.py oof            # -> oof_scores.jsonl for your fusion step

Run `run.py quarantine` and `run.py confound` BEFORE `train`: these detectors will happily learn
codec/duration shortcuts. `fourier` is the canary - if it scores ~1.0 on a fold, suspect the data.

| model       | status vs. the paper                                                        |
|-------------|-----------------------------------------------------------------------------|
| spectttra   | architecture verified; variant clip sizes from memory                       |
| fst         | read in full; Stage-1 frozen MERT instead of fine-tuned AudioCAT            |
| artifactnet | pipeline verified from abstract; 7-channel set is my guess; no codec aug    |
| musicdet    | idea verified from abstract; "frequency guidance" is my interpretation      |
| clam        | dual-stream + contrastive verified from title; loss form is my guess        |
| fourier     | thesis verified; detector is my minimal version, not the paper's            |
| musicscope  | NOT INCLUDED - could not find any paper/repo by that name                   |

Outputs per job: checkpoints/zoo/<model>/<full|logo_X>/{best.ckpt, test.json, test_scores.jsonl}.
