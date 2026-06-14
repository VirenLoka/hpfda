"""Graph loading + label/loss helpers for transductive node classification.

Training is full-batch over a single heterogeneous graph, so instead of
mini-batch DataLoaders this module loads the saved graph, exposes the provider
train/test masks, and provides the imbalance-aware loss + class weights.
"""
from __future__ import annotations

import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import resolve_path


def load_metadata(path) -> dict:
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def load_graph(cfg: dict, device: torch.device) -> Tuple[object, dict]:
    """Load the saved HeteroData graph and its metadata, moved onto ``device``."""
    metadata = load_metadata(cfg["data"]["metadata_json"])
    # weights_only=False: the graph is our own trusted artifact (HeteroData).
    data = torch.load(resolve_path(cfg["data"]["graph_path"]), weights_only=False)
    data = data.to(device)
    return data, metadata


def compute_class_weights(data, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from the provider *train* mask."""
    y = data["provider"].y[data["provider"].train_mask]
    counts = torch.bincount(y, minlength=n_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(logp, target, weight=self.alpha, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_loss(cfg: dict, data, device: torch.device) -> nn.Module:
    """Construct the configured loss (focal or weighted cross-entropy)."""
    loss_cfg = cfg["loss"]
    n_classes = cfg["model"]["n_classes"]
    alpha = None
    if loss_cfg.get("class_weights") == "auto":
        alpha = compute_class_weights(data, n_classes).to(device)

    if loss_cfg["type"] == "focal":
        return FocalLoss(gamma=loss_cfg.get("focal_gamma", 2.0), alpha=alpha).to(device)
    return nn.CrossEntropyLoss(weight=alpha)
