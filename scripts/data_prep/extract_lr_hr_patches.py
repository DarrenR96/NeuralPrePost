from metaflow import FlowSpec, step, Parameter
import glob
import os
import sys
import numpy as np
import random
import pandas as pd
import uuid
from skimage.util.shape import view_as_blocks 

# Ensure project root is on path (Metaflow may run from a different cwd)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from library.helper import yuv_frame_generator

class ExtractNumpyFlow(FlowSpec):

    reference_folder_hr = Parameter('reference_folder_hr', help='Path for folder containing uncompressed hr videos', required=True)
    compressed_folder_hr = Parameter('compressed_folder_hr', help='Path for folder containing compressed hr videos', required=True)
    reference_folder_lr = Parameter('reference_folder_lr', help='Path for folder containing uncompressed lr videos', required=True)
    hr_patch_size = Parameter('hr_patch_size', help='Patch size to be extracted from HR content', required=True, type=int)
    lr_patch_size = Parameter('lr_patch_size', help='Patch size to be extracted from LR content', required=True, type=int)
    test_split = Parameter('test_split', help='Fraction of videos to be used in test set', required=True)
    output_folder = Parameter('output_folder', help='Folder to write data to', required=True)

    @step 
    def start(self):
        extensions = ['*.mp4', '*.mkv']
        reference_video_paths = []
        for _extension in extensions:
            reference_video_paths.extend(glob.glob(os.path.join(self.reference_folder_hr, _extension)))
        total_videos = len(reference_video_paths)
        test_split = float(self.test_split)
        test_videos_num = int(test_split * total_videos)
        train_videos_num = total_videos - test_videos_num
        train_test_tags = ['train'] * train_videos_num + ['test'] * test_videos_num
        random.seed(0)
        random.shuffle(train_test_tags)
        train_test_values = []
        for _video, _tag in zip(reference_video_paths, train_test_tags):
            train_test_values.append({
                'video_name': os.path.basename(_video), 'tag': _tag
            })
        train_test_values_df = pd.DataFrame(train_test_values)

        compressed_video_paths = []
        for _extension in extensions:
            compressed_video_paths.extend(glob.glob(os.path.join(self.compressed_folder_hr, '**', _extension), recursive=True))
        compressed_videos = []
        for _video_path in compressed_video_paths:
            _video_path = os.path.relpath(_video_path, start=self.compressed_folder_hr)
            _qp, video_name = _video_path.split(os.sep)
            compressed_videos.append({
                'video_name': video_name, 'qp': _qp
            })

        compressed_videos_df = pd.DataFrame(compressed_videos)
        print(compressed_videos_df, train_test_values_df)
        video_df = pd.merge(train_test_values_df, compressed_videos_df, on='video_name')
        video_df = video_df[['video_name', 'qp', 'tag']]
        os.makedirs(self.output_folder, exist_ok=True)
        video_df.to_csv(os.path.join(self.output_folder, 'video_split.csv'), index=False)
        self.videos_to_extract = video_df.to_dict(orient='records')

        print(f"Total number of runs: {len(self.videos_to_extract)}")

        self.lr_video_paths = []
        for _extension in extensions:
            self.lr_video_paths.extend(glob.glob(os.path.join(self.reference_folder_lr, _extension)))
        self.hr_lr_ratio = self.hr_patch_size // self.lr_patch_size
        self.next(self.extract_patches, foreach='videos_to_extract')
    
    @step
    def extract_patches(self):
        reference_path = os.path.join(self.reference_folder_hr, self.input['video_name'])
        compressed_path = os.path.join(self.compressed_folder_hr, str(self.input['qp']), self.input['video_name'])
        video_identifier = self.input['video_name'].split('_')[0][1:]
        reference_lr_path = [x for x in self.lr_video_paths if video_identifier in x][0]
        output_folder = os.path.join(self.output_folder, 'numpy_data')
        os.makedirs(output_folder, exist_ok=True)
        reference_video = yuv_frame_generator(reference_path)
        compressed_video = yuv_frame_generator(compressed_path)
        reference_lr_video = yuv_frame_generator(reference_lr_path)
        self.written_data = []
        for frame_num, (_reference_frame, _compressed_frame, _reference_lr_frame) in enumerate(zip(reference_video, compressed_video, reference_lr_video)):
            _reference_frame_y = _reference_frame[0]
            _compressed_frame_y = _compressed_frame[0]
            _reference_frame_lr_y = _reference_lr_frame[0]
            _reference_uv = np.stack(_reference_frame[1:], -1)
            _compressed_uv = np.stack(_compressed_frame[1:], -1)
            _reference_lr_uv = np.stack(_reference_lr_frame[1:], -1)
            _reference_frame_y_blocks, _reference_uv_blocks = self.image_to_blocks(_reference_frame_y, self.hr_patch_size), self.image_to_blocks(_reference_uv, self.hr_patch_size)
            _compressed_frame_y_blocks, _compressed_uv_blocks = self.image_to_blocks(_compressed_frame_y, self.hr_patch_size), self.image_to_blocks(_compressed_uv, self.hr_patch_size)
            _reference_frame_lr_y_blocks, _reference_lr_uv_blocks = self.image_to_blocks(_reference_frame_lr_y, self.lr_patch_size), self.image_to_blocks(_reference_lr_uv, self.lr_patch_size)
            n_hr, n_lr = len(_reference_frame_y_blocks), len(_reference_frame_lr_y_blocks)
            if n_hr != n_lr:
                raise ValueError(
                    f"HR and LR block counts must match for same spatial regions (got {n_hr} HR blocks, {n_lr} LR blocks). "
                    "Ensure LR resolution is HR resolution divided by hr_patch_size/lr_patch_size."
                )
            for (_ref_hr_y, _ref_lr_y, _comp_hr_y, _ref_hr_uv, _ref_lr_uv, _comp_hr_uv) in zip(_reference_frame_y_blocks, _reference_frame_lr_y_blocks, _compressed_frame_y_blocks, _reference_uv_blocks, _reference_lr_uv_blocks, _compressed_uv_blocks):
                out_filename = str(uuid.uuid4()) + '.npz'
                out_filepath = os.path.join(output_folder, out_filename)
                np.savez(out_filepath, reference_hr_y=_ref_hr_y, reference_hr_uv=_ref_hr_uv, compressed_hr_y=_comp_hr_y, compressed_hr_uv=_comp_hr_uv, reference_lr_y=_ref_lr_y, reference_lr_uv=_ref_lr_uv)
                self.written_data.append({
                    'video_name': self.input['video_name'],
                    'qp': self.input['qp'],
                    'tag': self.input['tag'],
                    'file': out_filename,
                    'frame': frame_num
                })
        self.next(self.join_patches)
    
    @step 
    def join_patches(self, inputs):
        joined_written_data = []
        for input in inputs:
            joined_written_data.extend(
                input.written_data
            )
        dataset_df = pd.DataFrame(joined_written_data)
        dataset_df.to_csv(os.path.join(self.output_folder, 'dataset_df.csv'), index=False)
        self.next(self.end)

    @step
    def end(self):
        print("end")

    def image_to_blocks(self, img: np.ndarray, k: int) -> np.ndarray:
        if img.ndim == 2:        
            H, W = img.shape
            C = 1
            img = img[..., np.newaxis]  
        elif img.ndim == 3:       
            H, W, C = img.shape
        else:
            raise ValueError("Input image must be 2D (grayscale) or 3D (color).")

        crop_h = H % k
        crop_w = W % k
        img_cropped = img[:H-crop_h, :W-crop_w, :] if (crop_h or crop_w) else img
        Hc, Wc, _ = img_cropped.shape

        block_shape = (k, k, C)                 
        blocks = view_as_blocks(img_cropped, block_shape=block_shape)

        blocks = blocks.reshape(-1, k, k, C) 
        return blocks

if __name__ == '__main__':
    ExtractNumpyFlow()