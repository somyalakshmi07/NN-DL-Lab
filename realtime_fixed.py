#!/usr/bin/env python3
# realtime_fixed_robust.py  ← USE THIS FINAL VERSION
import os
# Workaround for OpenMP duplicate runtime issue seen with some builds
# (allows the app to continue; see warning about potential issues)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import time
import argparse

# ------------------- LOAD MODELS -------------------
print("Loading CSRNet ONNX model...")
crowd_session = ort.InferenceSession("csrnet.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print("Loading YOLOv8 tracker...")
detector = YOLO("yolov8n.pt")  # or yolov8s.pt / yolov8m.pt

# ------------------- CONFIG -------------------
# The ONNX model in this repo was exported with 224x224 input.
# Use 224x224 here so the runtime input matches the model.
# CLI args for tuning
parser = argparse.ArgumentParser(description="Real-time crowd counter + click-to-track (debuggable)")
parser.add_argument("--iou-threshold", type=float, default=0.4, help="IoU threshold for re-identification")
parser.add_argument("--hist-threshold", type=float, default=0.6, help="Histogram correlation threshold for re-ID")
parser.add_argument("--min-seen-frames", type=int, default=2, help="Min consecutive frames a detection must appear to be considered stable")
parser.add_argument("--debounce-frames", type=int, default=2, help="Debounce frames for live count (ignore IDs seen fewer frames)")
parser.add_argument("--stale-frames", type=int, default=300, help="Frames after which cached IDs are pruned")
parser.add_argument("--conf", type=float, default=0.4, help="YOLO detection confidence threshold")
parser.add_argument("--input-size", type=int, default=224, help="ONNX model input size (square)")
parser.add_argument("--source", default=0, help="Video source (0 for webcam, or path/RTSP)")
args = parser.parse_args()

INPUT_SIZE = (args.input_size, args.input_size)
CONF_THRESH = args.conf
ENTRY_LINE_RATIO = 0.5
COUNT_ONLY_HUMANS = True

# map args into local heuristic constants (used later)
IOU_THRESHOLD = args.iou_threshold
HIST_THRESHOLD = args.hist_threshold
MIN_SEEN_FRAMES = args.min_seen_frames
DEBOUNCE_FRAMES = args.debounce_frames
STALE_FRAMES = args.stale_frames

prev_count = 0
total_entries = 0
tracked_ids = set()
focused_id = None  # currently user-focused person ID
focused_tracker = None
focused_hist = None
id_to_box = {}  # map track_id -> last known bbox (x1,y1,x2,y2)
current_frame = None
# Per-ID temporal bookkeeping for debounce and cleanup
id_seen_counts = {}       # consecutive-frame appearance counts for each YOLO ID
id_last_seen_frame = {}   # last frame index where the ID was visible
frame_idx = 0

def _make_tracker():
    # create a CSRT tracker (fallback to legacy if needed)
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        return cv2.legacy.TrackerCSRT_create()

def _compute_hist(img):
    # compute HSV color histogram and normalize
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def _hist_match(h1, h2):
    # use correlation (higher is better, 1.0 is perfect)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def _iou(boxA, boxB):
    # box = (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    if boxAArea + boxBArea - inter == 0:
        return 0.0
    return inter / float(boxAArea + boxBArea - inter)


def mouse_click(event, x, y, flags, param):
    """Mouse callback: click on a person box to focus/unfocus tracking."""
    global focused_id, focused_tracker, focused_hist, id_to_box
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # find which id bbox contains the click
    clicked = None
    for tid, (x1, y1, x2, y2) in id_to_box.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            clicked = tid
            break

    if clicked is None:
        return

    # toggle focus
    if focused_id == clicked:
        # stop tracking
        focused_id = None
        focused_tracker = None
        focused_hist = None
        print(f"Stopped tracking ID {clicked}")
    else:
        focused_id = clicked
        # initialize tracker on current bbox using latest frame
        x1, y1, x2, y2 = id_to_box[clicked]
        w = x2 - x1
        h = y2 - y1
        focused_tracker = _make_tracker()
        try:
            if current_frame is not None:
                focused_tracker.init(current_frame, (x1, y1, w, h))
            else:
                # no current frame available yet
                focused_tracker = None
        except Exception:
            focused_tracker = None
        # compute and store appearance histogram from latest frame if available
        try:
            if current_frame is not None:
                patch = current_frame[y1:y2, x1:x2]
                focused_hist = _compute_hist(patch)
            else:
                focused_hist = None
        except Exception:
            focused_hist = None
        print(f"Now tracking ID {clicked}")


# ------------------- CAMERA SETUP (THE FIX!) -------------------
# Video source (from CLI args)
try:
    SOURCE = int(args.source)
except Exception:
    SOURCE = args.source
# SOURCE = "rtsp://your-ip:554/stream"   # Uncomment for RTSP
# SOURCE = "test_video.mp4"             # Or local video

cap = cv2.VideoCapture(SOURCE)

# === CRITICAL: Wait and retry camera ===
max_attempts = 50
attempt = 0
while not cap.isOpened() and attempt < max_attempts:
    print(f"Camera not ready... retry {attempt+1}/{max_attempts}")
    time.sleep(0.5)
    cap = cv2.VideoCapture(SOURCE)
    attempt += 1

if not cap.isOpened():
    print("\nERROR: Cannot open camera/video stream!")
    print("   → Check if camera is connected and not used by another app")
    print("   → For RTSP: test URL in VLC first")
    print("   → Try changing SOURCE = 1 or 2 (some laptops have multiple cams)")
    input("Press Enter to exit...")
    exit()

# Optional: Set resolution (helps many webcams)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Reduce lag

print("Camera opened successfully! Starting real-time counting...")

# ------------------- MAIN LOOP -------------------
cv2.namedWindow("Real-time Crowd Counter + Tracker", cv2.WINDOW_NORMAL)
# register mouse callback for click-to-track
cv2.setMouseCallback("Real-time Crowd Counter + Tracker", mouse_click)

def _show_placeholder(text, width=640, height=480, wait_ms=200):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = 30
    for i, line in enumerate(text.splitlines()):
        cv2.putText(img, line, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Real-time Crowd Counter + Tracker", img)
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q') or key == 27:
        return False
    # If window was closed by user, getWindowProperty will be <1
    if cv2.getWindowProperty("Real-time Crowd Counter + Tracker", cv2.WND_PROP_VISIBLE) < 1:
        return False
    return True

while True:
    ret, frame = cap.read()

    # If frame not available, show placeholder and keep trying until user closes
    if not ret:
        ok = _show_placeholder("No frame available - waiting...\nPress 'q' or close window to exit", 640, 480, wait_ms=300)
        if not ok:
            break
        continue

    orig_h, orig_w = frame.shape[:2]
    display_frame = frame.copy()
    # expose latest frame for mouse callback and re-ID routines
    current_frame = frame

    # If we have an active focused tracker, try to update it first
    tracker_seen = False
    if focused_tracker is not None and focused_id is not None:
        try:
            ok, tbbox = focused_tracker.update(frame)
            if ok:
                x, y, w, h = map(int, tbbox)
                id_to_box[focused_id] = (x, y, x + w, y + h)
                tracker_seen = True
        except Exception:
            tracker_seen = False

    # === YOLO Detection + Tracking ===
    try:
        results = detector.track(frame, persist=True, classes=[0], conf=CONF_THRESH, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else []
    except Exception as e:
        print("Detection error:", e)
        ok = _show_placeholder(f"Detection error:\n{str(e)}\nPress 'q' to exit", 640, 480, wait_ms=500)
        if not ok:
            break
        continue

    person_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # collect visible track IDs for live (current) count
    visible_ids = set()
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        person_mask[y1:y2, x1:x2] = 255
        visible_ids.add(int(tid))
        
        # Cache box for click detection and off-frame display
        id_to_box[int(tid)] = (x1, y1, x2, y2)

        # Update temporal bookkeeping: seen counts and last-seen frame
        tid_i = int(tid)
        if id_last_seen_frame.get(tid_i, -999) < frame_idx - 1:
            id_seen_counts[tid_i] = 1
        else:
            id_seen_counts[tid_i] = id_seen_counts.get(tid_i, 0) + 1
        id_last_seen_frame[tid_i] = frame_idx

        # Count entry only when ID has been stable for DEBOUNCE_FRAMES
        if id_seen_counts[tid_i] == DEBOUNCE_FRAMES and tid_i not in tracked_ids:
            tracked_ids.add(tid_i)
            total_entries += 1

        # Draw box with special color if this is the focused person
        if int(tid) == focused_id:
            # Focused person: thick red box with label "TRACKING: ID"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(display_frame, f"TRACKING: {int(tid)}", (x1, y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            # Other people: green box with ID
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID:{int(tid)}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # If our tracker reported a location, treat focused_id as visible
    if tracker_seen and focused_id is not None:
        visible_ids.add(focused_id)

    # Compute best IoU and hist scores for debug overlay (show even when not re-ID'ing)
    best_iou_display = 0.0
    best_iou_tid = None
    best_hist_display = -1.0
    best_hist_tid = None
    focused_last_box = id_to_box.get(focused_id) if focused_id in id_to_box else None
    for box, tid in zip(boxes, track_ids):
        try:
            x1, y1, x2, y2 = map(int, box)
            cand_box = (x1, y1, x2, y2)
            if focused_last_box is not None:
                sc = _iou(focused_last_box, cand_box)
                if sc > best_iou_display:
                    best_iou_display = sc
                    best_iou_tid = int(tid)
            if focused_hist is not None:
                # compute hist safely
                patch = current_frame[y1:y2, x1:x2]
                if patch.size > 0:
                    hist = _compute_hist(patch)
                    hs = _hist_match(focused_hist, hist)
                    if hs > best_hist_display:
                        best_hist_display = hs
                        best_hist_tid = int(tid)
        except Exception:
            continue

    # Re-identification: if focused person is not visible, try IoU first then appearance match
    if focused_id is not None and focused_id not in visible_ids and focused_hist is not None and len(boxes) > 0 and focused_id in id_to_box:
        focused_last_box = id_to_box.get(focused_id)
        best_iou = 0.0
        best_iou_tid = None
        best_iou_bbox = None
        # First pass: IoU-based matching (spatial heuristic)
        for box, tid in zip(boxes, track_ids):
            try:
                x1, y1, x2, y2 = map(int, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                # require the candidate ID to have been visible for a few consecutive frames
                if id_seen_counts.get(int(tid), 0) < MIN_SEEN_FRAMES:
                    continue
                cand_box = (x1, y1, x2, y2)
                score_iou = _iou(focused_last_box, cand_box)
                if score_iou > best_iou:
                    best_iou = score_iou
                    best_iou_tid = int(tid)
                    best_iou_bbox = cand_box
            except Exception:
                continue

        if best_iou_tid is not None and best_iou >= IOU_THRESHOLD:
            print(f"Re-identified focused person via IoU as new ID {best_iou_tid} (IoU={best_iou:.2f})")
            focused_id = best_iou_tid
            id_to_box[focused_id] = best_iou_bbox
            try:
                focused_tracker = _make_tracker()
                x1, y1, x2, y2 = best_iou_bbox
                w = x2 - x1
                h = y2 - y1
                if current_frame is not None:
                    focused_tracker.init(current_frame, (x1, y1, w, h))
                # refresh histogram
                patch = current_frame[y1:y2, x1:x2]
                focused_hist = _compute_hist(patch)
                visible_ids.add(focused_id)
            except Exception:
                focused_tracker = None
        else:
            # fallback: appearance-based matching but require the candidate to be stable
            best_score = -1.0
            best_tid = None
            best_bbox = None
            for box, tid in zip(boxes, track_ids):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    # require candidate to be seen across consecutive frames
                    if id_seen_counts.get(int(tid), 0) < MIN_SEEN_FRAMES:
                        continue
                    patch = current_frame[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    hist = _compute_hist(patch)
                    score = _hist_match(focused_hist, hist)
                    if score > best_score:
                        best_score = score
                        best_tid = int(tid)
                        best_bbox = (x1, y1, x2, y2)
                except Exception:
                    continue

            if best_score > 0.6 and best_tid is not None:
                print(f"Re-identified focused person as new ID {best_tid} (score={best_score:.2f})")
                focused_id = best_tid
                id_to_box[focused_id] = best_bbox
                # attempt to reinitialize tracker on the matched bbox
                try:
                    focused_tracker = _make_tracker()
                    x1, y1, x2, y2 = best_bbox
                    w = x2 - x1
                    h = y2 - y1
                    if current_frame is not None:
                        focused_tracker.init(current_frame, (x1, y1, w, h))
                    # refresh histogram
                    patch = current_frame[y1:y2, x1:x2]
                    focused_hist = _compute_hist(patch)
                    visible_ids.add(focused_id)
                except Exception:
                    focused_tracker = None
    
    # If focused person is off-frame but was tracked, show their last known position with dotted outline
    if focused_id is not None and focused_id not in visible_ids and focused_id in id_to_box:
        x1, y1, x2, y2 = id_to_box[focused_id]
        # Draw faint dotted box for off-frame focused person
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (100, 100, 255), 1)
        cv2.putText(display_frame, f"OFF-FRAME: {focused_id}", (x1, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    # live count = current visible people (debounced to reduce flicker)
    stable_ids = set([tid for tid in visible_ids if id_seen_counts.get(int(tid), 0) >= DEBOUNCE_FRAMES])
    # ensure focused_id is counted when tracker sees them
    if tracker_seen and focused_id is not None:
        stable_ids.add(focused_id)
    live_count = len(stable_ids)

    # Periodic cleanup: remove stale id caches to avoid unbounded growth
    to_remove = []
    for tid in list(id_to_box.keys()):
        if id_last_seen_frame.get(tid, -999) < frame_idx - STALE_FRAMES:
            to_remove.append(tid)
    for tid in to_remove:
        id_to_box.pop(tid, None)
        id_seen_counts.pop(tid, None)
        id_last_seen_frame.pop(tid, None)

    # === CSRNet Crowd Counting ===
    try:
        # Explicitly resize to INPUT_SIZE before creating the blob to avoid dimension mismatches.
        frame_for_model = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(frame_for_model, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)

        # Print model proto input shape and blob shape once for debugging
        if not globals().get("_realtime_shapes_printed", False):
            try:
                proto_shape = crowd_session.get_inputs()[0].shape
                print(f"ONNX model proto input shape: {proto_shape}")
                print(f"Blob shape (N,C,H,W): {blob.shape}")
            except Exception:
                pass
            globals()["_realtime_shapes_printed"] = True

        density_map = crowd_session.run(None, {crowd_session.get_inputs()[0].name: blob})[0][0,0]
    except Exception as e:
        print("Inference error:", e)
        ok = _show_placeholder(f"Inference error:\n{str(e)}\nPress 'q' or close window to exit", 640, 480, wait_ms=500)
        if not ok:
            break
        continue
    density_resized = cv2.resize(density_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    if COUNT_ONLY_HUMANS and person_mask.any():
        masked_density = density_resized * (person_mask > 0)
        crowd_count = int(masked_density.sum() + 0.5)
    else:
        crowd_count = int(density_resized.sum() + 0.5)

    # Smooth the count
    crowd_count = int(0.7 * prev_count + 0.3 * crowd_count)
    prev_count = crowd_count

    # === Draw Overlay ===
    line_y = int(orig_h * ENTRY_LINE_RATIO)
    cv2.line(display_frame, (0, line_y), (orig_w, line_y), (0,0,255), 3)
    # Show live (current) people count, cumulative entries, and density estimate
    cv2.putText(display_frame, f"Live: {live_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.putText(display_frame, f"Entries: {total_entries}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.putText(display_frame, f"Density: {crowd_count}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)

    # Debug overlay: show best IoU and histogram match candidate scores
    dbg_x = orig_w - 360
    dbg_y = 30
    if best_iou_tid is not None:
        cv2.putText(display_frame, f"Best IoU: {best_iou_display:.2f} (ID:{best_iou_tid})", (dbg_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
        dbg_y += 30
    else:
        cv2.putText(display_frame, f"Best IoU: N/A", (dbg_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
        dbg_y += 30
    if best_hist_tid is not None and best_hist_display >= 0:
        cv2.putText(display_frame, f"Best Hist: {best_hist_display:.2f} (ID:{best_hist_tid})", (dbg_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,200), 2)
    else:
        cv2.putText(display_frame, f"Best Hist: N/A", (dbg_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,200), 2)
    
    # Show focused tracking status
    if focused_id is not None:
        cv2.putText(display_frame, f"[Click to untrack ID {focused_id}]", (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Real-time Crowd Counter + Tracker", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or Esc
        break

    # advance frame index (for temporal heuristics)
    frame_idx += 1

# ------------------- CLEAN EXIT -------------------
cap.release()
cv2.destroyAllWindows()
print("Stopped gracefully. Total entries:", total_entries)