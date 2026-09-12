"""
Track A — Narrative Branch
Extracts the 128-dim "Musical Narrative Vector" from an audio chunk using
pure MIR/DSP methods. No training required.

Features:
    1. Harmonic Entropy & Tension     (chroma_cqt)
    2. Rhythmic Micro-timing (IOI var) (onset_strength)
    3. Motivic Recurrence Rate         (self-similarity matrix)
    4. Spectral Density Variance       (flatness / rolloff / centroid)
    5. Dynamic Envelope Variance       (RMS energy)

Usage:
    python track_a_narrative.py --input_dir ./data/raw --output narrative_vectors.parquet
"""
import argparse
import glob
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 22050
CHUNK_SECONDS = 30


def harmonic_entropy_and_tension(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Normalize each frame to a probability distribution over 12 pitch classes
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + 1e-9), axis=0)
    tension = np.std(chroma, axis=0)  # proxy for dissonance spread
    return float(np.mean(entropy)), float(np.mean(tension)), float(np.std(entropy))


def rhythmic_microtiming(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    if len(onsets) < 3:
        return 0.0
    ioi = np.diff(onsets)
    return float(np.var(ioi))


def motivic_recurrence(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    ssm = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)
    recurrence_rate = float(np.mean(ssm))
    return recurrence_rate


def spectral_density_variance(y, sr):
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return (
        float(np.var(flatness)),
        float(np.var(rolloff)),
        float(np.var(centroid)),
    )


def dynamic_envelope_variance(y):
    rms = librosa.feature.rms(y=y)[0]
    return float(np.var(rms))


def extract_narrative_features(path: str) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True, duration=CHUNK_SECONDS)

    ent_mean, tension_mean, ent_std = harmonic_entropy_and_tension(y, sr)
    ioi_var = rhythmic_microtiming(y, sr)
    recurrence_rate = motivic_recurrence(y, sr)
    flat_var, rolloff_var, centroid_var = spectral_density_variance(y, sr)
    dyn_var = dynamic_envelope_variance(y)

    return {
        "path": path,
        "harmonic_entropy_mean": ent_mean,
        "harmonic_entropy_std": ent_std,
        "harmonic_tension_mean": tension_mean,
        "rhythmic_ioi_variance": ioi_var,
        "motivic_recurrence_rate": recurrence_rate,
        "spectral_flatness_variance": flat_var,
        "spectral_rolloff_variance": rolloff_var,
        "spectral_centroid_variance": centroid_var,
        "dynamic_envelope_variance": dyn_var,
    }
    # NOTE: this baseline returns 9 scalar summary features per track.
    # Expand each into multi-statistic (mean/std/skew/percentiles) or
    # frame-level sequences to reach the full 128-dim vector described
    # in the README/architecture doc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of audio files")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--ext", default="wav", help="Audio file extension to glob for")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input_dir, f"**/*.{args.ext}"), recursive=True)
    rows = []
    for f in tqdm(files, desc="Extracting narrative features"):
        try:
            rows.append(extract_narrative_features(f))
        except Exception as e:
            print(f"[warn] failed on {f}: {e}")

    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
