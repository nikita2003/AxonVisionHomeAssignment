from datetime import datetime
from multiprocessing import shared_memory

import cv2
import numpy as np


class Visualizer:

    def __init__(self, input_queue, shutdown_event):
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event

    def run(self):
        print("Visualizer: Started")

        frame_count = 0

        while True:
            if self.shutdown_event.is_set():
                print("Visualizer: Shutdown event detected")
                break

            try:
                metadata = self.input_queue.get(timeout=0.5)
            except:
                continue

            if metadata.get('stop', False):
                print("Visualizer: Received stop signal")
                self.shutdown_event.set()
                break

            buffer_name = metadata["buffer_name"]
            shape = metadata["shape"]
            detections = metadata["detections"]
            motion_detected = metadata["motion_detected"]

            shm = shared_memory.SharedMemory(name=buffer_name)
            frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).copy()
            shm.close()

            frame_annotated = self._annotate_frame(frame, detections, motion_detected)

            cv2.imshow("Motion Detection Pipeline", frame_annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("z"):
                print("Visualizer: User pressed 'z' to quit")
                self.shutdown_event.set()
                break

            frame_count += 1

        cv2.destroyAllWindows()
        print(f"Visualizer: Finished displaying {frame_count} frames")

    def _annotate_frame(self, frame: np.array, detections, motion_detected):
        for contour_list in detections:
            contour_array = np.array(contour_list, dtype=np.int32)

            x,y,w,h = cv2.boundingRect(contour_array)

            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w<= 0 or h <= 0:
                continue

            roi = frame[y:y+h, x:x+w]

            blurred_roi = cv2.GaussianBlur(roi, (21,21), 0)

            frame[y:y+h, x:x+w] = blurred_roi

            cv2.drawContours(frame, [contour_array], -1, (0, 255, 0), 2)

        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            curr_time,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame


def visualizer_process(input_queue, shutdown_event):
    visualizer = Visualizer(input_queue, shutdown_event)
    visualizer.run()
