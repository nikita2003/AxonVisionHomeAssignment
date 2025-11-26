import sys
from components.streamer import streamer_process
from components.detector import detector_process
from components.visualizer import visualizer_process

import multiprocessing as mp

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        print("Example: python main.py demo.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print(" Motion detection pipeline ")
    print(f"Video file: {video_path}")
    print("Press 'z' in the video window to quit early")
    print("="*15)

    streamer_to_detector = mp.Queue(maxsize=10)
    detector_to_visualizer = mp.Queue(maxsize=10)

    streamer = mp.Process(target=streamer_process, args=(video_path, streamer_to_detector))
    detector = mp.Process(target=detector_process, args=(streamer_to_detector, detector_to_visualizer))
    visualizer = mp.Process(target=visualizer_process, args=(detector_to_visualizer,))

    streamer.start()
    detector.start()
    visualizer.start()

    print("\nMain: All processes started")

    streamer.join()
    print("Main: Streamer finished")

    detector.join()
    print("Main: Detector finished")

    visualizer.join()
    print("Main: Visualizer finished")

    print("\nPipeline Complete ")

if __name__ == "__main__":
    main()
