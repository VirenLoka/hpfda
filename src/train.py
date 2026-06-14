"""Train the heterogeneous fraud GNN (full-batch transductive node classification).

Forward-passes the whole graph each epoch, computes the loss on the provider
train mask, and periodically evaluates on the provider test mask. Includes
tqdm progress, a model/graph summary, per-epoch gradient-norm analysis, logging
and checkpointing - all paths configurable via configs/train_config.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import build_loss, load_graph
from model import build_model
from utils import get_logger, load_config, resolve_path, set_seed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def model_stats(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"total": total, "trainable": trainable, "size_mb": param_bytes / (1024 ** 2)}


def grad_global_norm(model: nn.Module, grad_clip: float | None) -> float:
    max_norm = grad_clip if grad_clip else float("inf")
    return float(nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


@torch.no_grad()
def evaluate(model, data, criterion) -> Dict[str, float]:
    model.eval()
    logits = model(data)
    mask = data["provider"].test_mask
    y = data["provider"].y[mask]
    out = logits[mask]
    loss = criterion(out, y).item()
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds = out.argmax(dim=1).cpu().numpy()
    return {"loss": loss, **_metrics(y.cpu().numpy(), preds, probs)}


class CheckpointManager:
    def __init__(self, cfg: dict, logger):
        ckpt = cfg["checkpoint"]
        self.dir = resolve_path(ckpt["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = ckpt["keep_last"]
        self.best_metric = ckpt["best_metric"]
        self.best_mode = ckpt["best_mode"]
        self.best_value = -np.inf if self.best_mode == "max" else np.inf
        self.saved: List[Path] = []
        self.logger = logger

    def _is_better(self, value: float) -> bool:
        return value > self.best_value if self.best_mode == "max" else value < self.best_value

    def save(self, epoch, model, optimizer, cfg, metrics):
        payload = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "metrics": metrics, "config": cfg,
        }
        path = self.dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, path)
        self.saved.append(path)
        while len(self.saved) > self.keep_last:
            old = self.saved.pop(0)
            if old.exists():
                old.unlink()
        value = metrics.get(self.best_metric)
        if value is not None and self._is_better(value):
            self.best_value = value
            torch.save(payload, self.dir / "best.pt")
            self.logger.info("New best %s=%.4f -> %s", self.best_metric, value,
                             self.dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the heterogeneous fraud GNN.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("train", cfg["logging"].get("log_file"))

    device = resolve_device(cfg["training"]["device"])
    logger.info("Using device: %s", device)

    data, metadata = load_graph(cfg, device)
    edge_types = data.metadata()[1]
    model = build_model(cfg, metadata, edge_types).to(device)

    # Lazy SAGEConv modules initialize their parameters on the first forward.
    with torch.no_grad():
        model(data)

    stats = model_stats(model)
    nc = metadata["node_counts"]
    logger.info("\n".join([
        "",
        "==================== Model / graph summary ==================",
        f"  Device          : {device}",
        f"  Architecture    : HeteroGNN (SAGE, hidden={cfg['model']['hidden_dim']}, "
        f"layers={cfg['model']['n_layers']})",
        f"  Parameters      : {stats['total']:,} total / {stats['trainable']:,} "
        f"trainable  (~{stats['size_mb']:.1f} MB)",
        f"  Nodes           : provider={nc['provider']:,} beneficiary={nc['beneficiary']:,} "
        f"physician={nc['physician']:,} claim={nc['claim']:,}",
        f"  ICD vocab       : {metadata['icd']['code_vocab_size']:,} codes "
        f"(emb_dim={metadata['icd']['embedding_dim']})",
        f"  Provider labels : train={metadata['n_train']:,} test={metadata['n_test']:,}",
        f"  Loss            : {cfg['loss']['type']} (class_weights={cfg['loss']['class_weights']})",
        "============================================================",
    ]))

    criterion = build_loss(cfg, data, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    writer = None
    if cfg["logging"].get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(resolve_path(cfg["logging"]["log_dir"])))
        except Exception as e:  # pragma: no cover
            logger.warning("TensorBoard unavailable (%s); continuing without it.", e)

    ckpt_mgr = CheckpointManager(cfg, logger)
    grad_clip = cfg["training"].get("grad_clip")
    log_every = cfg["logging"].get("log_every", 10)
    eval_every = cfg["training"].get("eval_every", 1)
    save_every = cfg["checkpoint"].get("save_every", 1)
    epochs = cfg["training"]["epochs"]

    train_mask = data["provider"].train_mask
    y_train = data["provider"].y[train_mask]

    pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_mask], y_train)
        loss.backward()
        gnorm = grad_global_norm(model, grad_clip)
        optimizer.step()

        train_loss = loss.item()
        metrics = {"loss": train_loss}
        postfix = {"loss": f"{train_loss:.4f}", "grad": f"{gnorm:.2f}"}

        if epoch % eval_every == 0:
            metrics = evaluate(model, data, criterion)
            postfix.update(f1=f"{metrics['f1']:.3f}", auc=f"{metrics['auc']:.3f}")
        pbar.set_postfix(postfix)

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/grad_norm", gnorm, epoch)
            if epoch % eval_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"test/{k}", v, epoch)

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                "epoch %d | train_loss %.4f | grad_norm %.3f | TEST loss %.4f "
                "acc %.4f precision %.4f recall %.4f f1 %.4f auc %.4f",
                epoch, train_loss, gnorm, metrics.get("loss", float("nan")),
                metrics.get("acc", 0), metrics.get("precision", 0),
                metrics.get("recall", 0), metrics.get("f1", 0), metrics.get("auc", 0),
            )

        if epoch % save_every == 0:
            ckpt_mgr.save(epoch, model, optimizer, cfg, metrics)
    pbar.close()

    if writer:
        writer.close()
    logger.info("Training complete. Best %s=%.4f", ckpt_mgr.best_metric, ckpt_mgr.best_value)


if __name__ == "__main__":
    main()
