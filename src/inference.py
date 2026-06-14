"""Score provider nodes for fraud from a trained HeteroGNN checkpoint.

Because fraud is a property of a provider's whole neighborhood (its claims,
patients and physicians), a provider cannot be scored from a flat feature row -
it is scored in the context of the graph. This script therefore loads the saved
graph + best checkpoint, runs one forward pass, and reports the fraud
probability for the providers listed in an input CSV (a "Provider" column).

Examples
--------
    python src/inference.py
    python src/inference.py --checkpoint checkpoints/gnn_exp1/best.pt \
        --input samples/sample_providers.csv --output samples/provider_predictions.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from dataset import load_metadata
from model import build_model
from utils import get_logger, load_config, resolve_path, set_seed


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Predict provider fraud with the GNN.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("inference", cfg["logging"].get("log_file"))

    infer_cfg = cfg.get("inference", {})
    ckpt_path = resolve_path(args.checkpoint or infer_cfg["checkpoint"])
    input_path = resolve_path(args.input or infer_cfg["input_csv"])
    output_path = resolve_path(args.output or infer_cfg["output_csv"])

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train a model first (src/train.py)."
        )

    device = resolve_device()
    metadata = load_metadata(cfg["data"]["metadata_json"])
    data = torch.load(resolve_path(cfg["data"]["graph_path"]), weights_only=False).to(device)
    edge_types = data.metadata()[1]

    logger.info("Loading checkpoint %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", cfg), metadata, edge_types).to(device)
    with torch.no_grad():
        model(data)                 # lazy-init SAGEConv params before loading
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Checkpoint epoch=%s metrics=%s", ckpt.get("epoch", "?"), ckpt.get("metrics", {}))

    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    prov_to_idx = {pid: i for i, pid in enumerate(metadata["provider_ids"])}

    df = pd.read_csv(input_path)
    if "Provider" not in df.columns:
        raise ValueError("Input CSV must contain a 'Provider' column.")

    rows = []
    unknown = 0
    for pid in df["Provider"].astype(str):
        idx = prov_to_idx.get(pid)
        if idx is None:
            unknown += 1
            rows.append({"Provider": pid, "FraudProbability": np.nan, "PredictedFraud": "UNKNOWN"})
        else:
            p = float(probs[idx])
            rows.append({"Provider": pid, "FraudProbability": round(p, 4),
                         "PredictedFraud": "Yes" if p >= 0.5 else "No"})
    result = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)
    if unknown:
        logger.warning("%d provider(s) were not in the graph and got UNKNOWN.", unknown)

    logger.info("\n%s", result.to_string(max_rows=30, index=False))
    flagged = (result["PredictedFraud"] == "Yes").sum()
    scored = (result["PredictedFraud"] != "UNKNOWN").sum()
    logger.info("Flagged fraud: %d / %d scored", int(flagged), int(scored))


if __name__ == "__main__":
    main()
