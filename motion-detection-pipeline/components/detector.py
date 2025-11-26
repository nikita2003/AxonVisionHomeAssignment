from multiprocessing import shared_memory

import cv2
import imutils
import numpy as np


class Detector:
    def __init__(self, input_queue, output_queue, num_buffers=3):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_buffers = num_buffers
        self.prev_frame = None
        self.output_buffers = []
        self.output_buffer_names = []

    def run(self):
        """Main detector process loop"""
        print("Detector Started")

        buffer_idx = 0
        frame_count = 0

        while True:
            metadata = self.input_queue.get()

            if metadata.get("stop", False):
                print("Detector: Received stop signal")
                break

            frame_id = metadata["frame_id"]
            timestamp = metadata["timestamp"]
            buffer_name = metadata["buffer_name"]
            shape = metadata["shape"]

            shm_input = shared_memory.SharedMemory(name=buffer_name)
            frame = np.ndarray(shape, dtype=np.uint8, buffer=shm_input.buf).copy()
            shm_input.close()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform motion detection
            if self.prev_frame is None:
                self.prev_frame = gray_frame
                detections = []
                motion_detected = False
            else:
                detections, motion_detected = self._detect_motion(gray_frame)
                self.prev_frame = gray_frame

            if not self.output_buffers:
                self._create_output_buffers(frame.nbytes)

            shm_output = self.output_buffers[buffer_idx]
            output_array = np.ndarray(shape, dtype=np.uint8, buffer=shm_output.buf)
            output_array[:] = frame[:]

            output_metadata = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "buffer_index": buffer_idx,
                "buffer_name": self.output_buffer_names[buffer_idx],
                "shape": shape,
                "detections": detections,
                "motion_detected": motion_detected,
                "stop": False,
            }
            self.output_queue.put(output_metadata)

            buffer_idx = (buffer_idx + 1) % self.num_buffers
            frame_count += 1

        self.output_queue.put({"stop": True})

        self._cleanup()
        print(f"Detector: Finished processing {frame_count+1} frames")

    def _detect_motion(self, gray_frame):
        diff = cv2.absdiff(gray_frame, self.prev_frame)

        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        min_area = 500
        significant_contours = []
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                significant_contours.append(c)

        motion_detected = len(significant_contours) > 0

        serializable_contours = []
        for contour in significant_contours:
            serializable_contours.append(contour.tolist())

        return serializable_contours, motion_detected

    def _create_output_buffers(self, frame_size):
        for i in range(self.num_buffers):
            buffer_name = f"detector_output_{i}_{id(self)}"
            shm = shared_memory.SharedMemory(
                create=True, size=frame_size, name=buffer_name
            )
            self.output_buffers.append(shm)
            self.output_buffer_names.append(buffer_name)
            print(f"Detector: Created output buffer {i}: {buffer_name}")

    def _cleanup(self):
        for shm in self.output_buffers:
            shm.close()
            shm.unlink()


def detector_process(input_queue, output_queue):
    detector = Detector(input_queue, output_queue)
    detector.run()
