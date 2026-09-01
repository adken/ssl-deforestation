"""
Downstream classification dataset: paired S1/S2 time series with pixel labels.

Expected files in `path/`:
    s1.npy      — shape (N, T, 2)
    s2.npy      — shape (N, T, 10)
    labels.npy  — shape (N,) — integer class labels
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Labelled S1/S2 pixel time-series dataset for downstream classification.

    Parameters
    ----------
    path : str
        Directory containing ``s1.npy``, ``s2.npy``, and ``labels.npy``.
    transforms : callable, optional
        Optional transform applied to both s1 and s2 tensors after loading.
    """

    S2_SCALE = 10_000.0

    def __init__(self, path: str, transforms=None) -> None:
        self.path       = path
        self.transforms = transforms

        # FIX: mmap_mode='r+' in the original allowed accidental writes.
        # Use 'r' (read-only) throughout.
        self.s1     = np.load(os.path.join(path, "s1.npy"),     mmap_mode="r")
        self.s2     = np.load(os.path.join(path, "s2.npy"),     mmap_mode="r")
        self.labels = np.load(os.path.join(path, "labels.npy"))

        # FIX: original used `or` (logical OR) instead of asserting equal sizes.
        # `s1.shape[0] or s2.shape[0]` returns s1.shape[0] when it is non-zero,
        # silently ignoring a mismatch. The correct check is an assert.
        assert self.s1.shape[0] == self.s2.shape[0] == len(self.labels), (
            f"Mismatched sample counts: s1={self.s1.shape[0]}, "
            f"s2={self.s2.shape[0]}, labels={len(self.labels)}"
        )
        self.num_samples = self.s1.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # ── S1: per-sample z-score normalisation ────────────────────────────
        s1 = self.s1[idx].astype(np.float32)          # (T, 2)
        mean = s1.mean(axis=0, keepdims=True)
        std  = s1.std(axis=0,  keepdims=True) + 1e-6
        s1   = (s1 - mean) / std

        # ── S2: reflectance scaling ──────────────────────────────────────────
        s2 = self.s2[idx].astype(np.float32) / self.S2_SCALE  # (T, 10)

        # TempCNN expects (C, T)
        s1 = torch.from_numpy(s1).permute(1, 0)  # (2, T)
        s2 = torch.from_numpy(s2).permute(1, 0)  # (10, T)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transforms:
            s1 = self.transforms(s1)
            s2 = self.transforms(s2)

        return s1, s2, label
