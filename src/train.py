"""Train the FraudTransformer with logging, periodic eval and checkpointing.

Everything (paths, hyperparameters, logging and checkpoint locations) is driven
by configs/train_config.yaml. Run after src/aggregate_data.py has produced the
train/test CSVs and feature metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from dataset import build_dataloaders, compute_class_weights
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


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    return {"loss": total_loss / max(n, 1), **_metrics(labels_np, preds, probs)}


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


class CheckpointManager:
    """Rolling checkpoint saver that also tracks the best model by a metric."""

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

    def save(self, epoch: int, model, optimizer, cfg: dict, metrics: Dict[str, float]):
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }
        path = self.dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, path)
        self.saved.append(path)
        self.logger.info("Saved checkpoint %s", path)

        # rolling cleanup
        while len(self.saved) > self.keep_last:
            old = self.saved.pop(0)
            if old.exists():
                old.unlink()

        value = metrics.get(self.best_metric)
        if value is not None and self._is_better(value):
            self.best_value = value
            best_path = self.dir / "best.pt"
            torch.save(payload, best_path)
            self.logger.info(
                "New best %s=%.4f -> %s", self.best_metric, value, best_path
            )


def main():
    parser = argparse.ArgumentParser(description="Train the fraud transformer.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    log_file = cfg["logging"].get("log_file")
    logger = get_logger("train", log_file)
    logger.info("Loaded config from %s", args.config)

    device = resolve_device(cfg["training"]["device"])
    logger.info("Using device: %s", device)

    train_loader, test_loader, metadata = build_dataloaders(cfg)
    logger.info(
        "Train batches=%d test batches=%d (train rows=%d test rows=%d)",
        len(train_loader), len(test_loader), metadata["n_train"], metadata["n_test"],
    )

    model = build_model(cfg, metadata).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    # class weights for imbalance
    weight = None
    if cfg["training"].get("class_weights") == "auto":
        weight = compute_class_weights(cfg, metadata).to(device)
        logger.info("Class weights: %s", weight.tolist())
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # optional tensorboard
    writer = None
    if cfg["logging"].get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(resolve_path(cfg["logging"]["log_dir"])))
        except Exception as e:  # pragma: no cover
            logger.warning("TensorBoard unavailable (%s); continuing without it.", e)

    ckpt_mgr = CheckpointManager(cfg, logger)
    grad_clip = cfg["training"].get("grad_clip")
    log_every = cfg["logging"].get("log_every", 50)
    eval_every = cfg["training"].get("eval_every", 1)
    save_every = cfg["checkpoint"].get("save_every", 1)
    epochs = cfg["training"]["epochs"]

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (x_num, x_cat, y) in enumerate(train_loader, 1):
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += loss.item() * y.size(0)
            seen += y.size(0)
            global_step += 1
            if step % log_every == 0:
                avg = running / seen
                logger.info(
                    "epoch %d | step %d/%d | loss %.4f",
                    epoch, step, len(train_loader), avg,
                )
                if writer:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)

        train_loss = running / max(seen, 1)
        logger.info("epoch %d done | train_loss %.4f", epoch, train_loss)
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)

        metrics = {"loss": train_loss}
        if epoch % eval_every == 0:
            eval_metrics = evaluate(model, test_loader, device, criterion)
            metrics = eval_metrics
            logger.info(
                "epoch %d | TEST loss %.4f acc %.4f precision %.4f recall %.4f f1 %.4f auc %.4f",
                epoch, eval_metrics["loss"], eval_metrics["acc"],
                eval_metrics["precision"], eval_metrics["recall"],
                eval_metrics["f1"], eval_metrics["auc"],
            )
            if writer:
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"test/{k}", v, epoch)

        if epoch % save_every == 0:
            ckpt_mgr.save(epoch, model, optimizer, cfg, metrics)

    if writer:
        writer.close()
    logger.info("Training complete. Best %s=%.4f", ckpt_mgr.best_metric, ckpt_mgr.best_value)


if __name__ == "__main__":
    main()
