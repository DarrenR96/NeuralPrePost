import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
from typing import List
import numpy as np 
import random
from tqdm import tqdm 

class VideoDataset(Dataset):
    def __init__(self, data_path: str, tag: str, qps: List[int], patch_size: int = 256):
        self.data_path = data_path
        self.qps = qps
        self.tag = tag
        self.patch_size = patch_size
        self.data_df = pd.read_csv(os.path.join(self.data_path, 'dataset_df.csv'))
        self.data_df = self.data_df[self.data_df['tag'] == self.tag]
        self.data_df = self.data_df[self.data_df['qp'].isin(self.qps)]
        self.data_df = self.data_df.reset_index(drop=True)
        self.data_collection = []
        for _row in tqdm(self.data_df.to_dict('records'), leave=False):
            _npy_data = np.load(os.path.join(self.data_path, 'numpy_data', _row['file']))
            self.data_collection.append({'reference_y': _npy_data['reference_y'], 'compressed_y': _npy_data['compressed_y']})

    def __len__(self):
        return len(self.data_collection)

    def __getitem__(self, index):
        npy_data = self.data_collection[index]
        reference_y = npy_data['reference_y']
        compressed_y = npy_data['compressed_y']

        if self.tag == 'train':
            h, w = reference_y.shape
            if h >= self.patch_size and w >= self.patch_size:
                i = random.randint(0, h - self.patch_size)
                j = random.randint(0, w - self.patch_size)
                reference_y = reference_y[i : i + self.patch_size, j : j + self.patch_size]
                compressed_y = compressed_y[i : i + self.patch_size, j : j + self.patch_size]
            if random.random() > 0.5:
                reference_y, compressed_y = np.flipud(reference_y), np.flipud(compressed_y)
            if random.random() > 0.5:
                reference_y, compressed_y = np.fliplr(reference_y), np.fliplr(compressed_y)
        else:
            reference_y = reference_y[0 : self.patch_size, 0 : self.patch_size]
            compressed_y = compressed_y[0 : self.patch_size, 0 : self.patch_size]

        reference_y, compressed_y = np.expand_dims(reference_y, 0), np.expand_dims(compressed_y, 0)
        reference_y, compressed_y = reference_y / 1023.0, compressed_y / 1023.0
        reference_y, compressed_y = reference_y.astype(np.float32), compressed_y.astype(np.float32)
        return compressed_y, reference_y


def fetch_video_dataloaders(data_path: str, batch_size, qps: List[int] = [0, 6, 11, 17, 23, 28, 40, 45, 51], patch_size: int = 256):
    train_dataset = VideoDataset(data_path, 'train', qps, patch_size=patch_size)
    test_dataset = VideoDataset(data_path, 'test', qps, patch_size=patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_dataloader, test_dataloader