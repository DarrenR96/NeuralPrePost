import av
import numpy as np

def plane_to_ndarray(plane, dtype):
    # 1D view over the raw buffer
    arr = np.frombuffer(plane, dtype=dtype)
    # Use line_size to handle padding; width is in samples
    bytes_per_sample = np.dtype(dtype).itemsize
    useful_line = plane.width * bytes_per_sample
    total_line = abs(plane.line_size)
    arr = arr.reshape(-1, total_line // bytes_per_sample)[:, :plane.width]
    return arr

def yuv_frame_generator(video_path):
    container = av.open(video_path)
    try:
        for packet in container.demux(video=0):
            for yuv_frame in packet.decode():
                fmt = yuv_frame.format.name
                if fmt == "yuv420p":
                    dtype = np.uint8
                elif fmt == "yuv420p10le":
                    dtype = np.uint16  # 10‑bit packed into 16‑bit words
                else:
                    continue  # or raise for unsupported formats

                y = plane_to_ndarray(yuv_frame.planes[0], dtype)
                u = plane_to_ndarray(yuv_frame.planes[1], dtype)
                v = plane_to_ndarray(yuv_frame.planes[2], dtype)

                yield y, u, v
    finally:
        container.close()
