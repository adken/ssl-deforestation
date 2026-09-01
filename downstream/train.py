"""Train/evaluate supervised, frozen-SSL, and fine-tuned deforestation models.

Run from the repository root with, for example::

    python -m downstream.train --mode supervised --datapath /path/to/data
    python -m downstream.train --mode freeze --checkpoint checkpoints/best_checkpoint.pth --datapath /path/to/data
    python -m downstream.train --mode fine-tuning --checkpoint checkpoints/best_checkpoint.pth --datapath /path/to/data
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .models.tempCNN import TemporalCNN
from .utils.dataset import TimeSeriesDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(
    mode: str,
    num_classes: int,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> TemporalCNN:
    mode = mode.lower()
    model = TemporalCNN(num_classes=num_classes).to(device)

    if mode in {"freeze", "fine-tuning"}:
        if checkpoint_path is None:
            raise ValueError(f"--checkpoint is required for mode='{mode}'")
        model.load_pretrained_encoders(checkpoint_path, device=device)
        if mode == "freeze":
            model.freeze_encoders()
    elif mode != "supervised":
        raise ValueError(
            f"Invalid mode '{mode}'. Choose supervised, freeze, or fine-tuning."
        )

    return model


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "kappa": sklearn.metrics.cohen_kappa_score(y_true, y_pred),
        "f1_macro": sklearn.metrics.f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": sklearn.metrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": sklearn.metrics.recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_macro": sklearn.metrics.precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def train_epoch(model, optimizer, criterion, dataloader, device) -> float:
    model.train()
    total_loss = 0.0
    with tqdm(dataloader, desc="  train", leave=False) as bar:
        for s1, s2, y_true in bar:
            optimizer.zero_grad()
            logits = model(s1.to(device), s2.to(device))
            loss = criterion(logits, y_true.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)


def eval_epoch(model, criterion, dataloader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        with tqdm(dataloader, desc="  eval ", leave=False) as bar:
            for s1, s2, y_true in bar:
                logits = model(s1.to(device), s2.to(device))
                loss = criterion(logits, y_true.to(device))
                total_loss += loss.item()
                y_true_all.append(y_true.cpu())
                y_pred_all.append(logits.argmax(dim=1).cpu())
                bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return (
        avg_loss,
        torch.cat(y_true_all).numpy(),
        torch.cat(y_pred_all).numpy(),
    )


def _loader(dataset, args, *, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "drop_last": False,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def train(args) -> None:
    set_seed(args.seed)

    if not (0.0 < args.train_ratio < 1.0 and 0.0 < args.val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must both be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    dataset = TimeSeriesDataset(path=args.datapath)
    n = len(dataset)
    train_size = int(n * args.train_ratio)
    val_size = int(n * args.val_ratio)
    test_size = n - train_size - val_size
    if min(train_size, val_size, test_size) < 1:
        raise ValueError(
            f"Dataset/split combination produces an empty split: "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dl = _loader(train_ds, args, shuffle=True)
    val_dl = _loader(val_ds, args, shuffle=False)
    test_dl = _loader(test_ds, args, shuffle=False)

    device = torch.device(args.device)
    model = get_model(
        args.mode,
        num_classes=2,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    model.modelname += f"_lr={args.learning_rate}_wd={args.weight_decay}"

    optimizer = Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    logdir = os.path.join(args.logdir, model.modelname)
    os.makedirs(logdir, exist_ok=True)

    log = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_dl, device)
        val_loss, y_val, pred_val = eval_epoch(model, criterion, val_dl, device)
        val_scores = compute_metrics(y_val, pred_val)

        scores_str = ", ".join(f"{k}={v:.3f}" for k, v in val_scores.items())
        print(
            f"Epoch {epoch:>4d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {scores_str}"
        )

        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_scores,
            }
        )
        pd.DataFrame(log).set_index("epoch").to_csv(
            os.path.join(logdir, "trainlog.csv")
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No best model was selected; check the training configuration")

    model.load_state_dict(best_state)
    test_loss, y_test, pred_test = eval_epoch(model, criterion, test_dl, device)
    test_scores = compute_metrics(y_test, pred_test)
    test_scores["test_loss"] = test_loss

    scores_str = ", ".join(f"{k}={v:.3f}" for k, v in test_scores.items())
    print(f"Final test: {scores_str}")
    pd.DataFrame([test_scores]).to_csv(
        os.path.join(logdir, "testlog.csv"), index=False
    )
    torch.save(best_state, os.path.join(logdir, "best_model.pth"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream deforestation classification with TempCNN"
    )
    parser.add_argument(
        "--mode",
        default="supervised",
        choices=["supervised", "freeze", "fine-tuning"],
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to VICReg pretraining checkpoint"
    )
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
