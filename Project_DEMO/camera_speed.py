"""
Live webcam vehicle classification + ROUGH speed estimation.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

#  Config 
MODEL_PATH = "transfer_best.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224
CAMERA_INDEX = 0

MIN_CONTOUR_AREA = 4000          # ignore blobs smaller than this (noise, small movement, phone icons etc.)
MAX_DISAPPEARED_FRAMES = 20      # forget a track if unseen for this many frames
MAX_MATCH_DISTANCE = 120         # px — max centroid jump allowed between frames to still count as "same" object
CLASSIFY_EVERY_N_FRAMES = 5      # how often to re-run the classifier per track (it's the slow step)
SPEED_WINDOW_SECONDS = 0.6       # smooth speed over this much history, not frame-to-frame jitter
CONF_THRESHOLD = 0.5
CROP_MARGIN_FRAC = 0.15          # pad crops by this fraction of box size — motion boxes often clip the vehicle's edges
MERGE_OVERLAP_IOU = 0.15         # merge two boxes into one if they overlap at least this much (fixes split/fragmented boxes)


def box_iou(a, b):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping_boxes(boxes):
    """Combines boxes that overlap significantly into one bounding box, so a single
    vehicle that background subtraction split into several fragments becomes one
    track instead of several duplicate ones."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if box_iou(merged[i], merged[j]) >= MERGE_OVERLAP_IOU:
                    x1, y1, w1, h1 = merged[i]
                    x2, y2, w2, h2 = merged[j]
                    nx1, ny1 = min(x1, x2), min(y1, y2)
                    nx2, ny2 = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
                    merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged

# Assumed real-world width (meters) per vehicle type, used ONLY to convert pixel
# movement into an approximate speed since there's no marked calibration distance.
# Matched by keyword against whatever your class_names.json labels are, case-insensitive.
# Edit these if you know your dataset's classes are narrower/wider than typical.
ASSUMED_WIDTH_M = {
    "bicycle": 0.6, "cycle": 0.6,
    "motorcycle": 0.8, "bike": 0.8, "scooter": 0.8,
    "auto": 1.4, "rickshaw": 1.4, "tempo": 1.4, "three": 1.4,
    "car": 1.8, "jeep": 1.8, "van": 1.9, "suv": 1.9,
    "tractor": 2.0,
    "truck": 2.5, "lorry": 2.5,
    "bus": 2.5, "minibus": 2.3,
    "ambulance": 2.0,
}
DEFAULT_WIDTH_M = 1.8  # fallback if the label doesn't match any keyword above


def guess_width_m(label):
    label_lower = label.lower()
    for keyword, width in ASSUMED_WIDTH_M.items():
        if keyword in label_lower:
            return width
    return DEFAULT_WIDTH_M


def load_class_names(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"ERROR: '{path}' not found. Generate it in Colab first (see export_class_names.py).")
    with open(p) as f:
        return json.load(f)


def preprocess_crop(frame_bgr, box):
    x, y, w, h = box
    mx, my = int(w * CROP_MARGIN_FRAC), int(h * CROP_MARGIN_FRAC)
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
    x_in = resized.astype(np.float32)  # raw [0,255] — model has its own internal rescaling
    return np.expand_dims(x_in, axis=0)


class Track:
    """One tracked moving object: its box, classification, and position history for speed."""
    def __init__(self, track_id, box, centroid):
        self.id = track_id
        self.box = box
        self.centroid = centroid
        self.history = [(time.time(), centroid)]   # (timestamp, (cx, cy))
        self.disappeared = 0
        self.label = "Classifying..."
        self.conf = 0.0
        self.speed_kmh = None
        self.frames_seen = 0

    def update_position(self, box, centroid):
        self.box = box
        self.centroid = centroid
        self.history.append((time.time(), centroid))
        # keep only recent history within the speed-averaging window (plus a little slack)
        cutoff = time.time() - (SPEED_WINDOW_SECONDS * 2)
        self.history = [h for h in self.history if h[0] >= cutoff]
        self.disappeared = 0
        self.frames_seen += 1

    def estimate_speed_kmh(self, meters_per_pixel):
        if len(self.history) < 2:
            return None
        t_new, c_new = self.history[-1]
        # find the oldest sample within the speed window
        t_old, c_old = self.history[0]
        for t, c in self.history:
            if t >= t_new - SPEED_WINDOW_SECONDS:
                t_old, c_old = t, c
                break
        dt = t_new - t_old
        if dt <= 0.05:  # not enough time elapsed yet for a stable estimate
            return None
        dist_px = np.hypot(c_new[0] - c_old[0], c_new[1] - c_old[1])
        dist_m = dist_px * meters_per_pixel
        speed_ms = dist_m / dt
        return speed_ms * 3.6  # m/s -> km/h


class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> Track

    def update(self, boxes):
        centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]

        if not self.tracks:
            for box, c in zip(boxes, centroids):
                self.tracks[self.next_id] = Track(self.next_id, box, c)
                self.next_id += 1
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        unmatched_boxes = list(range(len(boxes)))
        used_tracks = set()

        if boxes:
            dist_matrix = np.zeros((len(track_centroids), len(centroids)))
            for i, tc in enumerate(track_centroids):
                for j, c in enumerate(centroids):
                    dist_matrix[i, j] = np.hypot(tc[0] - c[0], tc[1] - c[1])

            # greedy nearest-match: repeatedly pick the closest remaining pair
            pairs = []
            dm = dist_matrix.copy()
            while dm.size and not np.all(np.isinf(dm)):
                i, j = np.unravel_index(np.argmin(dm), dm.shape)
                if dm[i, j] > MAX_MATCH_DISTANCE:
                    break
                pairs.append((i, j))
                dm[i, :] = np.inf
                dm[:, j] = np.inf

            for i, j in pairs:
                tid = track_ids[i]
                self.tracks[tid].update_position(boxes[j], centroids[j])
                used_tracks.add(tid)
                if j in unmatched_boxes:
                    unmatched_boxes.remove(j)

        # new tracks for unmatched boxes
        for j in unmatched_boxes:
            self.tracks[self.next_id] = Track(self.next_id, boxes[j], centroids[j])
            self.next_id += 1

        # age out tracks that weren't matched this frame
        for tid in track_ids:
            if tid not in used_tracks:
                self.tracks[tid].disappeared += 1

        self.tracks = {tid: t for tid, t in self.tracks.items()
                        if t.disappeared <= MAX_DISAPPEARED_FRAMES}
        return self.tracks


def main():
    class_names = load_class_names(CLASS_NAMES_PATH)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    print("\n*** Speed estimates below are ROUGH APPROXIMATIONS based on assumed vehicle")
    print("*** widths, not a real calibrated distance. Treat as relative, not exact.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera index {CAMERA_INDEX}.")

    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    tracker = CentroidTracker()
    frame_count = 0

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to grab frame from camera.")
            break
        frame_count += 1

        fg_mask = back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  # drop shadow gray (127)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        boxes = merge_overlapping_boxes(boxes)

        tracks = tracker.update(boxes)

        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue  # don't draw/classify objects not seen this frame

            if track.frames_seen % CLASSIFY_EVERY_N_FRAMES == 0 or track.conf == 0.0:
                x_in = preprocess_crop(frame, track.box)
                if x_in is not None:
                    probs = model.predict(x_in, verbose=0)[0]
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    if conf >= CONF_THRESHOLD:
                        track.label = class_names[top_idx]
                        track.conf = conf
                    else:
                        track.label = "Uncertain"
                        track.conf = conf

            meters_per_pixel = guess_width_m(track.label) / max(track.box[2], 1)
            speed = track.estimate_speed_kmh(meters_per_pixel)
            if speed is not None:
                track.speed_kmh = speed

            x, y, w, h = track.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            speed_text = f"~{track.speed_kmh:.0f} km/h" if track.speed_kmh is not None else "measuring..."
            label_text = f"#{tid} {track.label} ({track.conf*100:.0f}%) {speed_text}"
            cv2.putText(frame, label_text, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        cv2.putText(frame, "Rough/relative speed estimate only", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imshow("Vehicle Detection + Rough Speed (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()