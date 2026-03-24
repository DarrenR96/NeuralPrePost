from metaflow import FlowSpec, step, Parameter
import glob
import os
import sys
import numpy as np
import random
import pandas as pd
import uuid

# Ensure project root is on path (Metaflow may run from a different cwd)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from library.helper import yuv_frame_generator

class ExtractNumpyFlow(FlowSpec):

    reference_folder = Parameter('reference_folder', help='Path for folder containing uncompressed videos', required=True)
    compressed_folder = Parameter('compressed_folder', help='Path for folder containing compressed videos', required=True)
    test_split = Parameter('test_split', help='Fraction of videos to be used in test set', required=True)
    output_folder = Parameter('output_folder', help='Folder to write data to', required=True)

    @step 
    def start(self):
        extensions = ['*.mp4', '*.mkv']
        reference_video_paths = []
        for _extension in extensions:
            reference_video_paths.extend(glob.glob(os.path.join(self.reference_folder, _extension)))
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
            compressed_video_paths.extend(glob.glob(os.path.join(self.compressed_folder, '**', _extension), recursive=True))
        compressed_videos = []
        for _video_path in compressed_video_paths:
            _video_path = os.path.relpath(_video_path, start=self.compressed_folder)
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

        self.next(self.extract_patches, foreach='videos_to_extract')
    
    @step
    def extract_patches(self):
        reference_path = os.path.join(self.reference_folder, self.input['video_name'])
        compressed_path = os.path.join(self.compressed_folder, str(self.input['qp']), self.input['video_name'])
        output_folder = os.path.join(self.output_folder, 'numpy_data')
        os.makedirs(output_folder, exist_ok=True)
        reference_video = yuv_frame_generator(reference_path)
        compressed_video = yuv_frame_generator(compressed_path)
        self.written_data = []
        for frame_num, (_reference_frame, _compressed_frame) in enumerate(zip(reference_video, compressed_video)):
            _reference_frame_y = _reference_frame[0]
            _compressed_frame_y = _compressed_frame[0]
            _reference_uv = np.stack(_reference_frame[1:], -1)
            _compressed_uv = np.stack(_compressed_frame[1:], -1)
            out_filename = str(uuid.uuid4()) + '.npz'
            out_filepath = os.path.join(output_folder, out_filename)
            np.savez(out_filepath, reference_y=_reference_frame_y, reference_uv=_reference_uv, compressed_y=_compressed_frame_y, compressed_uv=_compressed_uv)
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

if __name__ == '__main__':
    ExtractNumpyFlow()