"""
Phase 1 — Unsupervised Representation Learning (SimCLR)
Pretrains the Track B (style) MobileNetV3-Small backbone on log-Mel
spectrograms using contrastive learning, so the CNN learns robust audio
texture representations before any label is used.

This is a skeleton — plug in your own augmentation pipeline (time/freq
masking, pitch shift, noise injection) and DataLoader over your dataset.

Usage:
    python train_simclr.py --input_dir ./data/raw
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from track_b_style import StyleEncoder


class ProjectionHead(nn.Module):
    """SimCLR projection head: 576 -> 256 -> 128."""

    def __init__(self, in_dim: int = 576, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    """Normalized Temperature-scaled Cross Entropy loss (SimCLR)."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss


def train(input_dir: str, epochs: int = 20, batch_size: int = 32, lr: float = 3e-4):
    """
    NOTE: This function expects a DataLoader yielding two augmented views
    of the same audio chunk per sample: (view1_logmel, view2_logmel).
    Implement `build_dataloader(input_dir, batch_size)` with your own
    augmentation strategy (time masking, freq masking, pitch/tempo jitter,
    additive noise) before running this training loop.
    """
    encoder = StyleEncoder(pretrained=True)
    projector = ProjectionHead(in_dim=encoder.out_dim)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=lr
    )

    # dataloader = build_dataloader(input_dir, batch_size)  # user-provided
    raise NotImplementedError(
        "Plug in build_dataloader(input_dir, batch_size) yielding paired "
        "augmented spectrogram views before running this training loop."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(args.input_dir, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
