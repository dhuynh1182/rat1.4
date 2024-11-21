import cv2
import requests
import threading
import queue  # Python 2.7 Queue or import queue for Python 3.x
import torch  # PyTorch for YOLOv5

import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath


class VideoCapture:
    def __init__(self, name):
        # Open the video capture stream
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()  # Thread-safe queue to hold frames
        # Start a background thread to read frames from the stream
        t = threading.Thread(target=self._reader)
        t.daemon = True  # Daemon thread will exit when the main program exits
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # If the queue is not empty, discard the previous frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            # Add the new frame to the queue
            self.q.put(frame)

    def read(self):
        # Get the next frame from the queue
        return self.q.get()

    def is_opened(self):
        # Check if the capture stream is opened
        return self.cap.isOpened()

    def release(self):
        # Release the capture stream
        self.cap.release()

def detect_objects(frame, model):
    # Perform inference using your custom model
    results = model(frame)  # Run inference using the custom model
    return results

def draw_results(frame, results):
    # Get bounding boxes and class labels from results
    for *box, conf, cls in results.xyxy[0]:
        label = f"{results.names[int(cls)]} {conf:.2f}"
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

if __name__ == '__main__':
    # Load pre-trained model (best.pt)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\David\\Desktop\\code\\best.pt')

    # Camera URL and stream
    URL = "http://192.168.0.192"
    cap = VideoCapture(URL + ":81/stream")  # Use the custom VideoCapture class

    # Adjust camera settings (e.g., frame size)
    requests.get(URL + "/control?var=framesize&val={}".format(8))

    while True:
        if cap.is_opened():
            # Get the next frame from the custom VideoCapture class
            frame = cap.read()

            # Perform object detection on the frame using your custom model
            results = detect_objects(frame, model)

            # Draw results on the frame (bounding boxes and labels)
            frame = draw_results(frame, results)

            # Display the frame with detections
            cv2.imshow("Output", frame)

        # Wait for a key press and exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()