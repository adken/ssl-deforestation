"""VICReg pretraining for paired Sentinel-1/Sentinel-2 time series."""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vicreg import VICRegNet
from utils.dataset import TimeSeriesDataset
from utils.loss import cov_loss, sim_loss, std_loss
from utils.utils import adjust_learning_rate, optim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, path)


def train(args) -> None:
    if args.batch_size < 2:
        raise ValueError("VICReg requires --batch_size >= 2 for variance/covariance terms")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard is required for training. Install requirements.txt first."
        ) from exc

    set_seed(args.seed)
    writer = SummaryWriter(log_dir=args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_chpt, exist_ok=True)

    dataset = TimeSeriesDataset(path=args.path)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "drop_last": True,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **loader_kwargs)
    if len(dataloader) == 0:
        raise ValueError(
            "No full VICReg batch can be formed. Reduce --batch_size or add more samples."
        )

    model = VICRegNet().to(args.device)
    optimizer = optim(model, args.weight_decay)
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)

        for view_a, view_b in loop:
            adjust_learning_rate(args, optimizer, dataloader, global_step)
            optimizer.zero_grad()
            view_a = view_a.to(args.device, dtype=torch.float32)
            view_b = view_b.to(args.device, dtype=torch.float32)

            repr_a, repr_b = model(view_a, view_b)
            loss_sim = sim_loss(repr_a, repr_b)
            loss_std = std_loss(repr_a, repr_b)
            loss_cov = cov_loss(repr_a, repr_b)
            loss = args.l * loss_sim + args.mu * loss_std + args.nu * loss_cov

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Loss/sim", loss_sim.item(), global_step)
            writer.add_scalar("Loss/std", loss_std.item(), global_step)
            writer.add_scalar("Loss/cov", loss_cov.item(), global_step)
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

            if global_step > 0 and global_step % args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        args.save_chpt, f"checkpoint_step{global_step}.pth"
                    ),
                )
            global_step += 1

        avg_loss = float(np.mean(epoch_losses))
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        print(f"Epoch {epoch:>4d}/{args.epochs} avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.save_chpt, "best_checkpoint.pth"),
            )

        with open(os.path.join(args.log_dir, "logs.txt"), "a") as log_file:
            log_file.write(f"Epoch {epoch}, avg_loss={avg_loss:.6f}\n")

    save_checkpoint(
        {
            "epoch": args.epochs,
            "step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(args.save_chpt, "final_checkpoint.pth"),
    )
    writer.flush()
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VICReg pretraining - S1/S2 time series")
    parser.add_argument("--path", required=True, help="Dataset directory")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l", type=float, default=25.0)
    parser.add_argument("--mu", type=float, default=25.0)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--save_chpt", default="checkpoints")
    parser.add_argument("--save_freq", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
