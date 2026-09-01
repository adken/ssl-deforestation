"""
Pretraining dataset: loads paired S1 (SAR) and S2 (optical) pixel time series.

Expected files in `path/`:
    s1_stack.npy  — shape (N, T, 2)   — SAR (VV, VH)
    s2_stack.npy  — shape (N, T, 10)  — optical (10 bands, raw DN 0-10000)

Each __getitem__ returns a matched (s1, s2) pair for VICReg.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Paired S1/S2 pixel time-series dataset for SSL pretraining.

    Returns matched (s1, s2) tuples. S1 is z-score normalised per-sample;
    S2 is scaled to reflectance [0, 1] by dividing by 10 000.

    Parameters
    ----------
    path : str
        Directory containing ``s1_stack.npy`` and ``s2_stack.npy``.
    """

    S2_SCALE = 10_000.0

    def __init__(self, path: str) -> None:
        self.path = path
        # mmap_mode='r' is correct for read-only pretraining — avoids loading
        # the entire array into RAM at once.
        self.s1 = np.load(os.path.join(path, "s1_stack.npy"), mmap_mode="r")
        self.s2 = np.load(os.path.join(path, "s2_stack.npy"), mmap_mode="r")

        # FIX: original code set num_samples = s1.shape[0] + s2.shape[0] which
        # doubled the dataset size and caused the else-branch to index s1[-1]
        # (always the last sample) for any idx >= N. Both arrays must have the
        # same number of pixels; assert to catch data preparation errors early.
        assert self.s1.shape[0] == self.s2.shape[0], (
            f"S1 and S2 must have the same number of samples, "
            f"got {self.s1.shape[0]} vs {self.s2.shape[0]}"
        )
        self.num_samples = self.s1.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # ── S1: per-sample z-score normalisation ────────────────────────────
        s1 = self.s1[idx].astype(np.float32)          # (T, 2)
        mean = s1.mean(axis=0, keepdims=True)
        std  = s1.std(axis=0,  keepdims=True) + 1e-6  # avoid div-by-zero
        s1   = (s1 - mean) / std

        # ── S2: reflectance scaling ──────────────────────────────────────────
        s2 = self.s2[idx].astype(np.float32) / self.S2_SCALE  # (T, 10)

        # TempCNN expects (C, T) — channels first
        s1 = torch.from_numpy(s1).permute(1, 0)  # (2, T)
        s2 = torch.from_numpy(s2).permute(1, 0)  # (10, T)

        return s1, s2
