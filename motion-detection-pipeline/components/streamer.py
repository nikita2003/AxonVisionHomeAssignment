from multiprocessing import shared_memory
from pathlib import Path
import time

import cv2
import numpy as np


class Streamer:
    def __init__(self, video_path: Path, output_queue, num_buffers=3):
        self.video_path = video_path
        self.output_queue = output_queue
        self.num_buffers = num_buffers

        self.shared_buffers = []
        self.buffer_names = []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: could not open video file: {self.video_path}")
            self.output_queue.put({'stop': True})
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps

        print(f"Streamer starting video stream at {fps} FPS")

        ret, first_frame = cap.read()
        if not ret:
            print("Streamer error: could not read first frame")
            self.output_queue.put({'stop': True})
            cap.release()
            return

        frame_shape = first_frame.shape
        frame_size = first_frame.size

        self._create_shared_buffers(frame_size)

        # Reset after getting first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_id = 0
        buffer_idx = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Streamer reached end of video")
                break

            shm = self.shared_buffers[buffer_idx]
            arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            arr[:] = frame[:]

            metadata = {
                'frame_id': frame_id,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                'buffer_index': buffer_idx,
                'buffer_name': self.buffer_names[buffer_idx],
                'shape': frame_shape,
                'stop': False
            }

            self.output_queue.put(metadata)

            buffer_idx = (buffer_idx + 1) % self.num_buffers
            frame_id += 1

            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed 
            if sleep_time > 0 :
                time.sleep(sleep_time)

        self.output_queue.put({'stop': True})

        cap.release()
        self._cleanup()
        print(f"Streamer finished streaming {frame_id} frames")

    def _create_shared_buffers(self, frame_size):
        for i in range(self.num_buffers):
            buffer_name  = f"frame_buffer_{i}_{id(self)}"
            shm = shared_memory.SharedMemory(create=True, size=frame_size, name=buffer_name)
            self.shared_buffers.append(shm)
            self.buffer_names.append(buffer_name)
            print(f"Streamer: Created buffer {i}: {buffer_name}")
            
    def _cleanup(self):
        for shm in self.shared_buffers:
            shm.close()
            shm.unlink()



def streamer_process(video_path: Path, output_queue):
    streamer = Streamer(video_path, output_queue)
    streamer.run()