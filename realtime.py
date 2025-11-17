# realtime.py
import os
# Suppress OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import onnxruntime as ort
import numpy as np
from tracker import PersonTracker

# Load CSRNet
session = ort.InferenceSession("csrnet.onnx", providers=['CUDAExecutionProvider'])

# Tracker
tracker = PersonTracker()

# Input source: RTSP, webcam, or file
SOURCE = 0  # Change to RTSP URL or video file path as needed
# SOURCE = "rtsp://your-camera:554/stream"  # RTSP example
# SOURCE = "video.mp4"  # Video file example

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print(f"Error: Cannot open video source '{SOURCE}'")
    exit(1)

print("Press 'q' to quit the stream...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Stream ended or error reading frame.")
        break

    # Resize for speed
    frame_resized = cv2.resize(frame, (800, 600))

    # Crowd counting
    blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (224, 224), swapRB=True, crop=False)
    inputs = {session.get_inputs()[0].name: blob}
    density_map = session.run(None, inputs)[0][0, 0]  # (H, W)
    crowd_count = int(np.sum(density_map))

    # Person tracking
    tracked_frame, entry_count = tracker.update(frame_resized.copy())

    # Overlay crowd count
    cv2.putText(tracked_frame, f"Crowd: {crowd_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    cv2.imshow("Real-time Crowd + Tracking", tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User closed the stream.")
        break

cap.release()
cv2.destroyAllWindows()
print("Stream closed.")