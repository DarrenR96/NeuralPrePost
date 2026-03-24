import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os 
import random 

class QualityPredictionDataset(Dataset):
    def __init__(self, data_path: str, tag: str):
        super().__init__()
        self.data_path = data_path
        self.data_df = pd.read_csv(os.path.join(data_path, 'dataset_df.csv'))
        self.data_df = self.data_df[self.data_df['tag'] == tag]
        self.qp_list = np.array(sorted(self.data_df['qp'].unique().tolist()))
        video_frame_df = self.data_df[['video_name', 'frame']]
        self.video_frame_df = video_frame_df.drop_duplicates().reset_index(drop=True)
        self.weight_list =  np.linspace(0, 1, len(self.qp_list))

    def __len__(self) -> int:
        return len(self.video_frame_df)

    def __getitem__(self, index: int) -> torch.Tensor:

        chosen_video_frame = self.video_frame_df.iloc[index]
        chosen_weight = random.random()
        insert_loc = np.searchsorted(self.weight_list, chosen_weight, side='left')
        if insert_loc == 0:
            insert_loc += 1 
        left_loc, right_loc = max(0, insert_loc - 1), insert_loc
        denum = self.weight_list[right_loc] - self.weight_list[left_loc]
        left_scale, right_scale = (self.weight_list[right_loc] - chosen_weight)/denum, (chosen_weight - self.weight_list[left_loc])/denum
        left_qp, right_qp = self.qp_list[left_loc], self.qp_list[right_loc]

        left_frame_row = self.data_df[(self.data_df['video_name'] == chosen_video_frame['video_name']) & (self.data_df['frame'] == chosen_video_frame['frame']) & (self.data_df['qp'] == left_qp)].to_dict('records')[0]
        right_frame_row = self.data_df[(self.data_df['video_name'] == chosen_video_frame['video_name']) & (self.data_df['frame'] == chosen_video_frame['frame']) & (self.data_df['qp'] == right_qp)].to_dict('records')[0]

        left_frame = np.load(os.path.join(self.data_path, 'numpy_data', left_frame_row['file']))['compressed_y']
        right_frame = np.load(os.path.join(self.data_path, 'numpy_data', right_frame_row['file']))['compressed_y']
        
        target = chosen_weight

        left_scale = np.array([left_scale]).astype(np.float32)
        right_scale = np.array([right_scale]).astype(np.float32)
        target = np.array([target]).astype(np.float32)
        left_frame, right_frame = left_frame.astype(np.float32)/1023.0, right_frame.astype(np.float32)/1023.0
        left_frame, right_frame = np.expand_dims(left_frame, 0), np.expand_dims(right_frame, 0)

        return left_frame, right_frame, left_scale, right_scale, target


def fetch_qualtiy_prediction_dataloaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for the HR patches (Y-channel) dataset."""
    train_dataset = QualityPredictionDataset(data_path, tag="train")
    test_dataset = QualityPredictionDataset(data_path, tag="test")
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
