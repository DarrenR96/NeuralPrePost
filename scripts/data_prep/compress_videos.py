from metaflow import FlowSpec, step, Parameter
import glob 
import os 
import subprocess 

class CompressVideosFlow(FlowSpec):

    input_path = Parameter('input_path', help='Path to a folder containing videos to process', required=True)
    output_path = Parameter('output_path', help='Path to where the encoded videos will be saved', required=True)
    qps = Parameter('qps', help='QPs to set as target', required=True, separator=',')

    @step 
    def start(self):
        extensions = ['*.mp4', '*.mkv']
        input_video_paths = []
        for _extension in extensions:
            input_video_paths.extend(glob.glob(os.path.join(self.input_path, _extension)))
        qps_to_encode = [int(qp) for qp in self.qps]
        self.videos_to_compress = []
        for _video_path in input_video_paths:
            for _qp in qps_to_encode:
                self.videos_to_compress.append({'input': _video_path, 'qp': _qp})
        print(f"Found {len(input_video_paths)} to process in {self.input_path}")
        print(f"Target QPS: {len(self.qps)}, total encodes: {len(self.videos_to_compress)}")
        self.next(self.compress_video, foreach='videos_to_compress')
    
    @step
    def compress_video(self):
        input_videopath = self.input['input']
        input_videoname = os.path.basename(input_videopath)
        input_qp = self.input['qp']
        output_folder = os.path.join(self.output_path, str(input_qp))
        os.makedirs(output_folder, exist_ok=True)
        output_videopath = os.path.join(output_folder, input_videoname)
        self.encode_all_intra(input_videopath, output_videopath, input_qp)
        self.next(self.join_compression_step)
    
    @step 
    def join_compression_step(self, inputs):
        self.next(self.end)

    @step 
    def end(self):
        print(f"Finished CompressVideosFlow")

    def encode_all_intra(self, input_file, output_file, qp=22):
        cmd = [
            'ffmpeg', '-i', input_file,
            '-c:v', 'libx265', '-preset', 'veryfast',
            '-threads', '1',
            '-x265-params', f'qp={qp}:keyint=1:intra-refresh=1:frame-threads=1:pools=1:wpp-threads=1',
            '-an', output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)

if __name__ == '__main__':
    CompressVideosFlow()