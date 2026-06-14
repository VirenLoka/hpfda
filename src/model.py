"""Heterogeneous GNN for provider fraud detection.

Per-node-type input encoders project raw features into a shared hidden space:

  * provider / physician : Linear over aggregate statistical features
  * beneficiary          : categorical embeddings + numeric features -> MLP
  * claim                : financial features + a learned "clinical state" vector
                           obtained by embedding the ICD diagnosis/procedure
                           codes and mean-pooling them (end-to-end, no Word2Vec)

The encoded node embeddings are then refined by several rounds of relation-wise
GraphSAGE message passing (PyG ``HeteroConv``). The final provider embeddings go
through an MLP head to produce fraud logits.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv


class ICDEncoder(nn.Module):
    """Embed ICD codes and mean-pool diagnosis & procedure codes per claim."""

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        # padding_idx=0 keeps padded slots at a zero vector.
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.out_dim = 2 * emb_dim  # diagnosis pool ++ procedure pool

    def _pool(self, idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(idx)                       # (N, L, E)
        mask = (idx != 0).unsqueeze(-1).float()         # (N, L, 1)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    def forward(self, diag_idx: torch.Tensor, proc_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._pool(diag_idx), self._pool(proc_idx)], dim=-1)


class ClaimEncoder(nn.Module):
    def __init__(self, n_num: int, icd: ICDEncoder, hidden: int):
        super().__init__()
        self.icd = icd
        self.mlp = nn.Sequential(
            nn.Linear(n_num + icd.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x_num, diag_idx, proc_idx) -> torch.Tensor:
        clinical = self.icd(diag_idx, proc_idx)
        return self.mlp(torch.cat([x_num, clinical], dim=-1))


class BeneficiaryEncoder(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: List[int], cat_dim: int, hidden: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, cat_dim) for card in cat_cardinalities]
        )
        in_dim = n_num + cat_dim * len(cat_cardinalities)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x_num, x_cat) -> torch.Tensor:
        cats = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.mlp(torch.cat([x_num] + cats, dim=-1))


class HeteroFraudGNN(nn.Module):
    def __init__(
        self,
        metadata: dict,
        edge_types: List[Tuple[str, str, str]],
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout
        dims = metadata["feature_dims"]
        icd_meta = metadata["icd"]
        cat_cards = [
            metadata["beneficiary_cat_meta"][c]["cardinality"]
            for c in metadata["beneficiary_categorical"]
        ]

        # --- per-type input encoders ---
        self.provider_enc = nn.Linear(dims["provider"], hidden)
        self.physician_enc = nn.Linear(dims["physician"], hidden)
        icd = ICDEncoder(icd_meta["code_vocab_size"], icd_meta["embedding_dim"])
        self.claim_enc = ClaimEncoder(dims["claim_num"], icd, hidden)
        self.bene_enc = BeneficiaryEncoder(
            dims["beneficiary_num"], cat_cards, metadata["cat_embedding_dim"], hidden
        )

        # --- heterogeneous message-passing stack ---
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), hidden) for et in edge_types}, aggr="sum"
            )
            self.convs.append(conv)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def encode(self, data) -> Dict[str, torch.Tensor]:
        return {
            "provider": self.provider_enc(data["provider"].x),
            "physician": self.physician_enc(data["physician"].x),
            "beneficiary": self.bene_enc(
                data["beneficiary"].x_num, data["beneficiary"].x_cat
            ),
            "claim": self.claim_enc(
                data["claim"].x_num, data["claim"].diag_idx, data["claim"].proc_idx
            ),
        }

    def forward(self, data) -> torch.Tensor:
        x_dict = self.encode(data)
        edge_index_dict = data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                      for k, v in x_dict.items()}
        return self.head(x_dict["provider"])


def build_model(cfg: dict, metadata: dict, edge_types) -> HeteroFraudGNN:
    m = cfg["model"]
    return HeteroFraudGNN(
        metadata=metadata,
        edge_types=[tuple(et) for et in edge_types],
        hidden=m["hidden_dim"],
        n_layers=m["n_layers"],
        dropout=m["dropout"],
        n_classes=m["n_classes"],
    )
