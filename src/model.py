"""Feature-tokenizer Transformer for tabular fraud classification.

This is an FT-Transformer-style architecture: every feature (numeric or
categorical) is turned into a ``d_model``-dimensional token, a learnable [CLS]
token is prepended, the sequence is passed through a standard Transformer
encoder, and the final [CLS] representation is fed to a classification head.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class NumericTokenizer(nn.Module):
    """Project each scalar numeric feature into its own d_model token."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # One learnable weight + bias vector per feature.
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_features) -> (B, n_features, d_model)
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CategoricalTokenizer(nn.Module):
    """Embed each categorical feature into its own d_model token."""

    def __init__(self, cardinalities: List[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cardinalities]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, n_cat) -> (B, n_cat, d_model)
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)


class FraudTransformer(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.num_tokenizer = NumericTokenizer(n_numeric, d_model) if n_numeric else None
        self.cat_tokenizer = (
            CategoricalTokenizer(cat_cardinalities, d_model) if cat_cardinalities else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = []
        if self.num_tokenizer is not None and x_num.shape[1] > 0:
            tokens.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None and x_cat.shape[1] > 0:
            tokens.append(self.cat_tokenizer(x_cat))
        seq = torch.cat(tokens, dim=1)  # (B, n_features, d_model)

        cls = self.cls_token.expand(batch, -1, -1)
        seq = torch.cat([cls, seq], dim=1)

        encoded = self.encoder(seq)
        cls_out = self.norm(encoded[:, 0])  # take [CLS]
        return self.head(cls_out)


def build_model(cfg: dict, metadata: dict) -> FraudTransformer:
    """Construct the model sized from config + feature metadata."""
    n_numeric = len(metadata["numeric"])
    cat_cardinalities = [
        metadata["categorical_meta"][c]["cardinality"] for c in metadata["categorical"]
    ]
    m = cfg["model"]
    return FraudTransformer(
        n_numeric=n_numeric,
        cat_cardinalities=cat_cardinalities,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        n_classes=m["n_classes"],
    )
