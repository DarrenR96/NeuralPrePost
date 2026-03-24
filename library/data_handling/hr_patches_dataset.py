import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HRPatchesDataset(Dataset):
    """
    Dataset for HR patches (Y channel only) produced by scripts/data_prep/extract_hr_patches.py.
    Each sample is one .npz file; only reference_hr_y is loaded. Returns a tensor normalized
    to [0, 1] with shape (1, H, W). Supports optional horizontal/vertical flips for training.
    """

    def __init__(self, data_path: str, tag: str):
        """
        Args:
            data_path: Root folder containing dataset_df.csv and numpy_data/.
            tag: 'train' or 'test'.
        """
        self.data_path = data_path
        self.tag = tag
        self.numpy_dir = os.path.join(data_path, "numpy_data")

        self.data_df = pd.read_csv(os.path.join(data_path, "dataset_df.csv"))
        self.data_df = self.data_df[self.data_df["tag"] == tag].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data_df)

    def _load_y(self, index: int) -> np.ndarray:
        npz_path = os.path.join(self.numpy_dir, self.data_df.iloc[index]["file"])
        data = np.load(npz_path)
        y = np.squeeze(data["reference_hr_y"].copy())
        return y

    def __getitem__(self, index: int) -> torch.Tensor:
        y = self._load_y(index)

        if self.tag == "train":
            if random.random() > 0.5:
                y = np.flipud(y)
            if random.random() > 0.5:
                y = np.fliplr(y)

        # Ensure (H, W) then add channel dim (1, H, W), normalize to [0, 1], float32
        if y.ndim != 2:
            y = np.squeeze(y)
        y = np.expand_dims(y, 0).astype(np.float32) / 1023.0
        return torch.from_numpy(y)


def fetch_hr_patches_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for the HR patches (Y-channel) dataset."""
    train_dataset = HRPatchesDataset(data_path, tag="train")
    test_dataset = HRPatchesDataset(data_path, tag="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
