from datetime import datetime
from multiprocessing import shared_memory

import cv2
import numpy as np


class Visualizer:
    def __init__(self, input_queue):
        self.input_queue = input_queue

    def run(self):
        print("Visualizer: Started")

        frame_count = 0

        while True:
            metadata = self.input_queue.get()

            if metadata.get('stop', False):
                print("Visualizer: Received stop signal")
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
                break

            frame_count += 1

        cv2.destroyAllWindows()
        print(f"Visualizer: Finished displaying {frame_count+1} frames")

    def _annotate_frame(self, frame: np.array, detections, motion_detected):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, curr_time, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        for contour_list in detections:
            contour_array = np.array(contour_list, dtype=np.int32)
            cv2.drawContours(frame, [contour_array], -1, (0, 255, 0), 2)

        
        return frame
    
    
def visualizer_process(input_queue):
    visualizer = Visualizer(input_queue)
    visualizer.run()