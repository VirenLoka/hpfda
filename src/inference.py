"""Run fraud predictions from a trained checkpoint on a series of feature inputs.

By default this loads the best checkpoint produced by training
(``checkpoint.dir/best.pt`` via the ``inference.checkpoint`` config key) and
scores an input CSV of *raw* feature values. Numeric features are standardized
and categorical features are encoded using the saved ``feature_metadata.json``,
exactly mirroring the training-time preprocessing.

Examples
--------
    # Use all defaults from configs/train_config.yaml
    python src/inference.py

    # Point at a specific checkpoint and input file
    python src/inference.py --checkpoint checkpoints/exp1/best.pt \
        --input samples/sample_claims.csv --output samples/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import load_metadata
from model import build_model
from utils import get_logger, load_config, resolve_path, set_seed


def preprocess(df: pd.DataFrame, metadata: dict):
    """Turn a raw-feature DataFrame into (x_num, x_cat) tensors.

    Numeric columns are standardized with the stored train mean/std; categorical
    columns are mapped through the saved vocab (unknown/missing -> 0).
    """
    num_cols = metadata["numeric"]
    cat_cols = metadata["categorical"]
    norm = metadata["normalization"]
    cat_meta = metadata["categorical_meta"]

    missing = [c for c in num_cols + cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    num = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).copy()
    for c in num_cols:
        mean, std = norm[c]["mean"], norm[c]["std"] or 1.0
        num[c] = (num[c].fillna(mean) - mean) / std
    x_num = torch.tensor(num.values, dtype=torch.float32)

    cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        vocab = cat_meta[c]["vocab"]
        codes = (
            df[c].astype("string").fillna("__nan__").map(lambda v: vocab.get(str(v), 0))
        )
        cat[:, j] = codes.fillna(0).astype(np.int64).values
    x_cat = torch.tensor(cat, dtype=torch.long)
    return x_num, x_cat


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Predict provider fraud from features.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override inference.checkpoint")
    parser.add_argument("--input", default=None, help="Override inference.input_csv")
    parser.add_argument("--output", default=None, help="Override inference.output_csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("inference", cfg["logging"].get("log_file"))

    infer_cfg = cfg.get("inference", {})
    ckpt_path = resolve_path(args.checkpoint or infer_cfg["checkpoint"])
    input_path = resolve_path(args.input or infer_cfg["input_csv"])
    output_path = resolve_path(args.output or infer_cfg["output_csv"])
    batch_size = infer_cfg.get("batch_size", 256)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train a model first (src/train.py)."
        )

    metadata = load_metadata(cfg["data"]["metadata_json"])
    device = resolve_device()

    logger.info("Loading checkpoint %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer the architecture the checkpoint was trained with, if present.
    model_cfg = ckpt.get("config", cfg)
    model = build_model(model_cfg, metadata).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(
        "Checkpoint trained for %s epoch(s); metrics=%s",
        ckpt.get("epoch", "?"), ckpt.get("metrics", {}),
    )

    df = pd.read_csv(input_path)
    logger.info("Scoring %d rows from %s", len(df), input_path)
    x_num, x_cat = preprocess(df, metadata)

    probs: List[float] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Predicting", unit="batch"):
            xn = x_num[i : i + batch_size].to(device)
            xc = x_cat[i : i + batch_size].to(device)
            logits = model(xn, xc)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())

    fraud_prob = np.asarray(probs)
    pred = (fraud_prob >= 0.5).astype(int)
    result = df.copy()
    result["FraudProbability"] = np.round(fraud_prob, 4)
    result["PredictedFraud"] = np.where(pred == 1, "Yes", "No")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)

    # Console preview
    preview_cols = ["FraudProbability", "PredictedFraud"]
    logger.info(
        "\n%s",
        result[preview_cols].to_string(max_rows=20),
    )
    logger.info(
        "Predicted fraud: %d / %d (%.1f%%)",
        int(pred.sum()), len(pred), 100.0 * pred.mean(),
    )


if __name__ == "__main__":
    main()
