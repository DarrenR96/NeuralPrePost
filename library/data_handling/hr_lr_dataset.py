from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HRLRDataset(Dataset):
    """
    Dataset for HR/LR patch pairs produced by scripts/data_prep/extract_lr_hr_patches.py.
    Each sample is one .npz file. Returns a dict of Y-channel tensors (reference_hr_y,
    compressed_hr_y, reference_lr_y) normalized to [0, 1]. Works with DataLoader:
    batching produces a dict of batched tensors with the same keys.
    """

    def __init__(
        self,
        data_path: str,
        tag: str,
        qps: Optional[List[int]] = None,
    ):
        """
        Args:
            data_path: Root folder containing dataset_df.csv and numpy_data/.
            tag: 'train' or 'test'.
            qps: If set, only include samples from these QP values; if None, use all.
        """
        self.data_path = data_path
        self.tag = tag
        self.numpy_dir = os.path.join(data_path, "numpy_data")

        self.data_df = pd.read_csv(os.path.join(data_path, "dataset_df.csv"))
        self.data_df = self.data_df[self.data_df["tag"] == tag]
        if qps is not None:
            self.data_df = self.data_df[self.data_df["qp"].isin(qps)]
        self.data_df = self.data_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data_df)

    def _load_sample(self, index: int) -> dict[str, np.ndarray]:
        npz_path = os.path.join(self.numpy_dir, self.data_df.iloc[index]["file"])
        data = np.load(npz_path)
        return {
            "reference_hr_y": data["reference_hr_y"],
            "compressed_hr_y": data["compressed_hr_y"],
            "reference_lr_y": data["reference_lr_y"],
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._load_sample(index)
        reference_hr_y = sample["reference_hr_y"]
        compressed_hr_y = sample["compressed_hr_y"]
        reference_lr_y = sample["reference_lr_y"]

        if self.tag == "train":
            if random.random() > 0.5:
                reference_hr_y = np.flipud(reference_hr_y)
                compressed_hr_y = np.flipud(compressed_hr_y)
                reference_lr_y = np.flipud(reference_lr_y)
            if random.random() > 0.5:
                reference_hr_y = np.fliplr(reference_hr_y)
                compressed_hr_y = np.fliplr(compressed_hr_y)
                reference_lr_y = np.fliplr(reference_lr_y)

        # Add channel dim, normalize, float32 (match video_dataset convention)
        def to_tensor(y: np.ndarray) -> torch.Tensor:
            y = np.expand_dims(y, 0).astype(np.float32) / 1023.0
            return torch.from_numpy(y)

        return {
            "reference_hr_y": to_tensor(reference_hr_y),
            "compressed_hr_y": to_tensor(compressed_hr_y),
            "reference_lr_y": to_tensor(reference_lr_y),
        }


def fetch_hr_lr_dataloaders(
    data_path: str,
    batch_size: int,
    qps: Optional[List[int]] = None,
    num_workers: int = 8,
):
    """Build train and test DataLoaders for the HR/LR patch dataset."""
    train_dataset = HRLRDataset(data_path, tag="train", qps=qps)
    test_dataset = HRLRDataset(data_path, tag="test", qps=qps)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, 
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
