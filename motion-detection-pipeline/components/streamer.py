from multiprocessing import shared_memory
from pathlib import Path
import time

import cv2
import numpy as np


class Streamer:

    def __init__(self, video_path: Path, output_queue, shutdown_event, enable_fps_logging=False, num_buffers=3):
        self.video_path = video_path
        self.output_queue = output_queue
        self.num_buffers = num_buffers
        self.shutdown_event = shutdown_event
        self.enable_fps_logging = enable_fps_logging
        self.shared_buffers = []
        self.buffer_names = []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: could not open video file: {self.video_path}")
            self.output_queue.put({'stop': True})
            self.shutdown_event.set()
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
        expected_time = time.time()

        if self.enable_fps_logging:
            frame_times = []
            last_log_time = time.time()

        while True:
            if self.shutdown_event.is_set():
                print("Streamer: Shutdown event detected")
                break

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Streamer reached end of video")
                self.shutdown_event.set()
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
            
            expected_time += frame_delay
            sleep_time = expected_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)

            if self.enable_fps_logging:
                actual_frame_time = time.time() - start_time
                frame_times.append(actual_frame_time)

                if time.time() - last_log_time >= 1.0:
                    recent_times = frame_times[-int(fps):]
                    avg_time = sum(recent_times) / len(recent_times)
                    current_fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"Streamer: Current FPS: {current_fps:.1f} (target: {fps:.1f})")
                    last_log_time = time.time()

        self.output_queue.put({'stop': True})
        
        if self.enable_fps_logging and frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            actual_fps = 1.0 / avg_frame_time
            print(f"Streamer: Average FPS: {actual_fps:.2f} (target: {fps:.2f})")

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


def streamer_process(
    video_path: Path, output_queue, shutdown_event, enable_fps_logging=False
):
    streamer = Streamer(video_path, output_queue, shutdown_event, enable_fps_logging)
    streamer.run()
