# tracker.py
import cv2
import numpy as np
from ultralytics import YOLO

class PersonTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.entry_counts = {}  # track_id → frame_first_seen
        self.entry_line_y = None  # Will be set as 70% of height

    def update(self, frame):
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().cpu().numpy()

        # Set entry line once
        if self.entry_line_y is None:
            self.entry_line_y = int(frame.shape[0] * 0.7)

        new_entries = 0
        if track_ids is not None:
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) // 2
                if tid not in self.entry_counts and cy > self.entry_line_y:
                    self.entry_counts[tid] = len(self.entry_counts) + 1
                    new_entries += 1

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw entry line
        cv2.line(frame, (0, self.entry_line_y), (frame.shape[1], self.entry_line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Entries: {len(self.entry_counts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        return frame, len(self.entry_counts)