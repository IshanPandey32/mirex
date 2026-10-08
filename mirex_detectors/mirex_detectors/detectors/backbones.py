"""Frozen self-supervised encoders shared by CLAM and FST.

Every encoder exposes   .pooled(wave) -> [B, n_layers, dim]   (time-mean of every hidden layer)
and is permanently frozen. `dummy=True` swaps in a tiny random conv encoder so the whole
pipeline can be smoke-tested without downloading 1 GB of weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _znorm(x):
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True) + 1e-7)


class HFEncoder(nn.Module):
    def __init__(self, hf_name: str, trust_remote_code=False):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(hf_name, trust_remote_code=trust_remote_code)
        self.m.eval().requires_grad_(False)
        cfg = self.m.config
        self.dim, self.n_layers = cfg.hidden_size, cfg.num_hidden_layers + 1

    @torch.no_grad()
    def pooled(self, wave):
        out = self.m(_znorm(wave.float()), output_hidden_states=True)
        return torch.stack([h.mean(1) for h in out.hidden_states], 1)


class DummyEncoder(nn.Module):
    def __init__(self, dim=64, n_layers=3):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.conv = nn.ModuleList([nn.Conv1d(1, dim, 400, stride=320)] +
                                  [nn.Conv1d(dim, dim, 3, padding=1) for _ in range(n_layers - 1)])
        self.requires_grad_(False)

    @torch.no_grad()
    def pooled(self, wave):
        x, hs = _znorm(wave.float()).unsqueeze(1), []
        for c in self.conv:
            x = torch.relu(c(x))
            hs.append(x.mean(-1))
        return torch.stack(hs, 1)


def load_mert(dummy=False):
    return DummyEncoder() if dummy else HFEncoder("m-a-p/MERT-v1-330M", trust_remote_code=True)


def load_w2v(dummy=False):
    return DummyEncoder(dim=48, n_layers=3) if dummy else HFEncoder("facebook/wav2vec2-xls-r-300m")
