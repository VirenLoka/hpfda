"""Dataset + DataLoader construction for the aggregated fraud data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import resolve_path


class FraudClaimsDataset(Dataset):
    """Represents engineered claim features (numeric + categorical) and labels.

    Numeric features are standardized using train statistics stored in the
    feature-metadata file, so train and test share the same scaling.
    """

    def __init__(self, csv_path, metadata: dict):
        self.numeric_cols = metadata["numeric"]
        self.categorical_cols = metadata["categorical"]
        self.label_col = metadata["label"]
        norm = metadata["normalization"]

        df = pd.read_csv(resolve_path(csv_path))

        num = df[self.numeric_cols].astype(np.float32).copy()
        for c in self.numeric_cols:
            mean, std = norm[c]["mean"], norm[c]["std"] or 1.0
            num[c] = (num[c] - mean) / std
        self.x_num = torch.tensor(num.values, dtype=torch.float32)

        self.x_cat = torch.tensor(
            df[self.categorical_cols].astype(np.int64).values, dtype=torch.long
        )
        self.y = torch.tensor(df[self.label_col].astype(np.int64).values, dtype=torch.long)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def load_metadata(path) -> dict:
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, dict]:
    """Build train/test dataloaders + the feature metadata from config."""
    metadata = load_metadata(cfg["data"]["metadata_json"])

    train_ds = FraudClaimsDataset(cfg["data"]["train_csv"], metadata)
    test_ds = FraudClaimsDataset(cfg["data"]["test_csv"], metadata)

    bs = cfg["training"]["batch_size"]
    workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=workers, drop_last=False
    )
    return train_loader, test_loader, metadata


def compute_class_weights(cfg: dict, metadata: dict) -> torch.Tensor:
    """Inverse-frequency class weights from the train CSV (for imbalance)."""
    df = pd.read_csv(resolve_path(cfg["data"]["train_csv"]))
    y = df[metadata["label"]].values
    counts = np.bincount(y, minlength=cfg["model"]["n_classes"]).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
