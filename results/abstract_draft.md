# MusicScope-CL: Contrastive Learning for AI-Generated Music Detection

MusicScope-CL is a contrastive-learning-based classifier that distinguishes AI-generated music from human-composed music. Beyond binary detection, the project extends into a **learned latent space of "musical narrative"** — allowing per-track novelty/rarity scoring and 2D manifold visualization of how different generators occupy the embedding space relative to human music.

This README documents the model's evaluation history across multiple checkpoints and held-out benchmarks, including a mid-project pipeline bug and its fix.

---

## 1. Overview

| | |
|---|---|
| **Task** | Binary classification: Human-composed vs. AI-generated music |
| **Approach** | Contrastive learning (embedding space trained to separate human/AI representations) |
| **Training scale** | Up to 200k tracks (`MusicScope-CL 200k` checkpoint) |
| **Generators covered (in-domain)** | Suno v5, Udio, AudioLDM, MusicGen, Mustango, StableAudio, Echoes |
| **Generators covered (zero-shot / unseen)** | Mubert, StableAudio, Producer, ElevenLabs, Brev, AceStep, DiffRhythm, SongGen |
| **Extensions built on top of the base classifier** | Latent-space t-SNE manifold, per-track "narrative rarity" percentile scoring |

---

## 2. Evaluation Timeline

Results below are reported **in chronological/checkpoint order**, not cherry-picked — later numbers supersede earlier ones only where they measure the same thing. Different runs use different held-out sets and generator mixes, so they are not all directly comparable; each is labeled with what it measured.

### 2.1 Initial 5,000-track validation

Early sanity-check run on a smaller held-out set, split by generator family (FMA/human, Suno/Sonics, Echoes).

- **AUROC:** 0.9487 · **AP:** 0.9398
- **Overall accuracy at threshold 0.50:** 87.8%
- **Per-family accuracy:** Suno/Sonics 92.3% · Human (FMA) 88.6% · **Echoes 65.6%** (weakest family)
- Confusion matrix @0.50: 2,214 TN / 286 FP / 325 FN / 2,175 TP
- Peak F1 threshold ≈ 0.45, close to the default 0.50 operating point

**Takeaway:** strong separation overall, but Echoes (a harder/edge-case generator) was already flagged as the weak point this early.

### 2.2 28k held-out benchmark — before vs. after a padding-bug fix

A pipeline bug (audio padding handling) was discovered and patched. The same 28k held-out live/raw audio set was re-run before and after the fix:

| Generator | Before fix | After fix |
|---|---|---|
| AudioLDM | 0% | **100%** |
| MusicGen | 0% | **100%** |
| Mustango | 0% | **100%** |
| Udio | 67% | **100%** |
| Suno v5 | 100% | 100% |
| Echoes | 28% | **76%** |
| Human (Real) | 99% | 97% |

**Takeaway:** the padding bug was silently zeroing out detection on several generators (AudioLDM, MusicGen, Mustango went from complete misses to perfect detection). This is the single most important fix in the project's history — it's worth stating plainly in the README as a "lesson learned," since it shows the eval pipeline itself, not just the model, needed validation.

### 2.3 200k-model formal benchmark — 28,134 unseen tracks

Full 6-panel evaluation of the retrained 200k-parameter-scale checkpoint on a clean 28,134-track held-out set.

- **AUROC:** 0.9970 · **AP:** 0.9969
- **Optimal decision threshold (Youden's J):** 0.79
- **Confusion matrix @0.79:** Human correctly identified 99.2% (15,828) · AI correctly identified 98.2% (11,948)
- **Per-family accuracy:** fmc_audioldm 99.9% · sonics_udio 99.9% · fmc_mustango 99.9% · sonics_suno 99.7% · fmc_musicgen 99.5% · fmc_udio 99.4% · fma (human) 99.2% · **echoes 71.5%** (still the outlier)
- Calibration curve shows the model is reasonably well-calibrated at the extremes but noisy in the mid-probability range — expected given how few tracks sit near the decision boundary.

**Takeaway:** at 200k scale, the model is near-ceiling on every generator family except Echoes, which remains a consistent, reproducible weak point across every eval run.

### 2.4 130,000-track out-of-domain interim scorecard

A larger, more adversarial out-of-domain test mixing Suno v5 with an "Unknown" generator bucket (likely undisclosed/novel models).

- Suno v5 detected: **72.1%** (115k tracks)
- Unknown-generator bucket detected: **13.4%** (13k tracks)
- Score distribution is strongly bimodal (near 0 or near 1), but the "Unknown" bucket's CDF shows a much flatter climb through low-probability scores — consistent with out-of-distribution generators producing audio the model is less confident about.

**Takeaway:** this is the most honest stress test in the set — performance clearly degrades on genuinely unfamiliar generators, which is the expected and correct behavior to report (rather than a failure to hide).

### 2.5 Comprehensive multi-generator benchmark — 47,374 tracks

A broad benchmark across all seven trained-on generators simultaneously.

- Human: 99.4% · Suno: 100.0% · Udio: 100.0% · AudioLDM: 100.0% · MusicGen: 100.0% · Mustango: 100.0%
- **Echoes: 91.9%** — notably higher here than in 2.3/2.2, suggesting Echoes detection is sensitive to which specific held-out subset is sampled (smaller/harder sub-splits pull the number down).

### 2.6 Zero-shot generalization — 8 completely unseen generators

The most rigorous test: generators **never seen during training** at all (Mubert, StableAudio, Producer, ElevenLabs, Brev, AceStep, DiffRhythm, SongGen), evaluated against real human music (FMA).

- **Macro-AUROC: 0.9933 · Macro-F1: 0.9094**
- Human (FMA) correctly classified: **100.0%**
- Best zero-shot generalization: SongGen 94.5%, DiffRhythm 94.1%
- Mid-tier: AceStep 86.1%, Brev 84.6%, ElevenLabs 82.3%
- Weaker zero-shot generalization: Producer 79.7%, StableAudio 73.7%
- **Mubert (loop-assembler style generator): 11.4%** — a clear failure case, consistent with Mubert's non-neural, loop-based synthesis method producing audio that doesn't resemble the diffusion/autoregressive artifacts the model learned to detect.

**Takeaway:** strong zero-shot transfer to most unseen neural generators, but near-total failure on Mubert. This is a meaningful, explainable limitation — Mubert's loop-assembly approach is architecturally different from every generator family the model trained on, so this is a domain-coverage gap rather than random noise.

---

## 3. Latent Space Analysis (added independently)

Two additional analyses were built directly on the contrastive embedding space, beyond binary classification.

### 3.1 t-SNE 2D manifold of the embedding space

Projecting the learned embeddings to 2D reveals clear, mostly-separated clusters per generator:

- **Udio** forms a tight, fully isolated cluster (top of the plot) — the most distinctive embedding signature of any generator.
- **Human music (FMA)** occupies a large, coherent region on the right, overlapping only modestly with Suno v5 and StableAudio/Echoes.
- **MusicGen, AudioLDM, and Mustango** cluster together in the lower-left, showing embedding-space similarity between these three — plausibly because they share more similar underlying architectures/training data than the commercial systems do.
- **Suno v5** sits in two sub-clusters near the center, overlapping partially with human music — consistent with Suno v5 being the hardest generator to separate from human tracks in every accuracy table above.

### 3.2 Musical narrative rarity (novelty scoring)

A rarity/novelty percentile was computed per track (position of a track's embedding relative to the train+val distribution). Violin plot across generators:

| Source | Mean rarity percentile | Median |
|---|---|---|
| Human | ~0.67 | ~0.68 |
| StableAudio | ~0.72 | ~0.75 |
| MusicGen | ~0.68 | ~0.70 |
| AudioLDM | ~0.46 | ~0.43 |
| Udio | ~0.41 | ~0.36 |
| Mustango | ~0.37 | ~0.32 |
| Suno | ~0.18 | ~0.13 |

**Interpretation:** Human compositions skew toward the high-rarity/idiosyncratic end of the distribution, while several generators (especially **Suno**) show strong "mode collapse" — their outputs cluster densely around a narrow, average region of the latent space (~0.13–0.18 percentile), meaning Suno tracks tend to sound structurally similar to one another. StableAudio and MusicGen are the exceptions, producing comparably rare/varied structures to human music — this is a useful, non-obvious finding: detection accuracy and structural diversity are not the same thing (StableAudio/Echoes is *harder to detect* but *more structurally varied*, while Suno is *easy to detect* but *structurally repetitive*).

---

## 4. Summary of Key Findings

1. **A pipeline bug (audio padding) was silently causing 0% detection** on three full generators (AudioLDM, MusicGen, Mustango) before being caught and fixed — after the fix, all three jumped to 100% on the same benchmark. This is documented as a cautionary/process finding, not just a model result.
2. **AUROC on in-domain, unseen tracks reaches 0.997**, with near-perfect (98–100%) per-generator accuracy across six of seven trained generator families.
3. **Echoes is the consistent hard case** across every in-domain eval (28–92% depending on the specific held-out split), suggesting it produces audio closer to the human/AI decision boundary than other generators.
4. **Zero-shot generalization to unseen generators is strong overall (Macro-AUROC 0.9933)**, with the notable exception of Mubert (11.4%), a non-neural loop-assembly system architecturally unlike anything in training.
5. **The learned latent space is interpretable beyond classification**: t-SNE shows generator-specific clustering (Udio fully separable; MusicGen/AudioLDM/Mustango cluster together), and rarity scoring reveals that Suno v5 exhibits strong mode collapse relative to human music's structural diversity — a finding independent of raw detection accuracy.

---

## 5. Suggested Next Steps

- Targeted fine-tuning or additional training data for **Echoes** and **Mubert-style loop-assembly generators**, the two persistent weak points.
- Investigate why the 130k out-of-domain "Unknown" bucket underperforms (13.4%) — likely a distinct generator family worth explicitly identifying and adding to training.
- Standardize held-out set composition across future evals so results are directly comparable run-to-run (several of the swings above are partly attributable to different sample splits rather than model changes).
