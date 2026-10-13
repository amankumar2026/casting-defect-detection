"""Train one model and keep the weights from its best validation epoch.

    python -m inspector.train small_cnn
    python -m inspector.train resnet18
    python -m inspector.train resnet18_relit             # wider brightness augmentation
    python -m inspector.train resnet18 --fraction 0.1    # only 10% of the training photos
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from inspector import config
from inspector.data import CastingDataset, load_split, subsample
from inspector.models import build_model, model_path, predict_proba


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validation_loss(model: nn.Module, images: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Returns (log loss, accuracy at threshold 0.5)."""
    probs = predict_proba(model, images).clip(1e-7, 1 - 1e-7)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(loss), float(((probs >= 0.5) == labels).mean())


def train(name: str, fraction: float = 1.0) -> dict:
    """Trains `name` on a fraction of the training photos. A smaller fraction
    gets more passes over its photos, so every run takes the same number of
    weight updates and is checked on validation the same number of times."""
    set_seed()
    images, labels = subsample(*load_split("train"), fraction)
    val_images, val_labels = load_split("val")
    loader = DataLoader(CastingDataset(images, labels, train=True,
                                       lighting=config.LIGHTING_RANGE[name]),
                        batch_size=config.BATCH_SIZE, shuffle=True)

    model = build_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE[name])
    loss_function = nn.BCEWithLogitsLoss()

    passes_per_check = round(1 / fraction)
    # The learning rate falls smoothly to zero over the whole run (cosine schedule).
    updates = config.EPOCHS[name] * passes_per_check * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    history, best_loss = [], float("inf")
    path = model_path(name, fraction)
    config.MODEL_DIR.mkdir(exist_ok=True)
    started = time.time()

    for check in range(1, config.EPOCHS[name] + 1):
        model.train()
        total, seen = 0.0, 0
        for _ in range(passes_per_check):
            for batch, target in loader:
                optimizer.zero_grad()
                loss = loss_function(model(batch), target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item() * len(target)
                seen += len(target)

        val_loss, val_accuracy = validation_loss(model, val_images, val_labels)
        history.append({"epoch": check, "train_loss": total / seen,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        kept = val_loss < best_loss
        if kept:
            best_loss = val_loss
            torch.save(model.state_dict(), path)
        print(f"{name} {check:>2}/{config.EPOCHS[name]}  train loss {total / seen:.4f}  "
              f"val loss {val_loss:.4f}  val acc {val_accuracy:.3f}  "
              f"{time.time() - started:.0f}s{'  saved' if kept else ''}", flush=True)

    result = {"model": name, "fraction": fraction, "train_photos": len(labels),
              "minutes": round((time.time() - started) / 60, 1), "history": history}
    path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=config.MODELS)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args.model, args.fraction)
