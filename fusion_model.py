"""
Fusion Layer — concatenates Track A (narrative, 128-dim) and Track B
(style, 576-dim) embeddings into a 704-dim vector and trains a lightweight
classifier head (MLP by default; XGBoost optional) to output a calibrated
P(AI-generated) score optimized for AUROC.

Usage:
    python fusion_model.py --narrative narrative_vectors.parquet --style_ckpt simclr.pt
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class FusionMLP(nn.Module):
    """Small MLP fusion head: 704 -> 256 -> 64 -> 1."""

    def __init__(self, in_dim: int = 704, hidden1: int = 256, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits


class TemperatureScaler(nn.Module):
    """Post-hoc calibration: single learned temperature applied to logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def label_smoothing_bce(logits, targets, smoothing: float = 0.05):
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smoothed)


def train_fusion_head(X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 1e-3):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = FusionMLP(in_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = label_smoothing_bce(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_probs = torch.sigmoid(val_logits).numpy()
            auroc = roc_auc_score(y_val, val_probs)
            print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | val AUROC {auroc:.4f}")

    return model, (X_val_t, y_val)


def calibrate_temperature(model, X_val_t, y_val, epochs: int = 100, lr: float = 0.01):
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=epochs)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_val_t)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            scaler(logits), y_val_t
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrative", required=True, help="Parquet of narrative vectors")
    parser.add_argument("--style_embeddings", required=False,
                         help="Parquet/npy of precomputed Track B embeddings")
    parser.add_argument("--labels_col", default="label", help="Column name for 0/1 labels")
    args = parser.parse_args()

    narrative_df = pd.read_parquet(args.narrative)
    feature_cols = [c for c in narrative_df.columns if c not in ("path", args.labels_col)]

    if args.style_embeddings:
        style_df = pd.read_parquet(args.style_embeddings)
        merged = narrative_df.merge(style_df, on="path")
        style_cols = [c for c in style_df.columns if c != "path"]
        X = merged[feature_cols + style_cols].values
        y = merged[args.labels_col].values
    else:
        print("[info] No style embeddings provided — training on narrative features only.")
        X = narrative_df[feature_cols].values
        y = narrative_df[args.labels_col].values

    model, (X_val_t, y_val) = train_fusion_head(X, y)
    scaler = calibrate_temperature(model, X_val_t, y_val)
    print(f"Learned temperature: {scaler.temperature.item():.3f}")


if __name__ == "__main__":
    main()
