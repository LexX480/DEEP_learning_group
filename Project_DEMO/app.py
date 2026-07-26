"""
Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, send_from_directory

import licencePlate

#  Config 
MODEL_PATH = "transfer_best.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.5
CLASSIFY_EVERY_N_FRAMES = 5

MIN_CONTOUR_AREA = 4000
MAX_DISAPPEARED_FRAMES = 20
MAX_MATCH_DISTANCE = 120
SPEED_WINDOW_SECONDS = 0.6
CROP_MARGIN_FRAC = 0.15
MERGE_OVERLAP_IOU = 0.15

CAPTURE_DIR = "captures"

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
DEFAULT_WIDTH_M = 1.8


def guess_width_m(label):
    label_lower = label.lower()
    for keyword, width in ASSUMED_WIDTH_M.items():
        if keyword in label_lower:
            return width
    return DEFAULT_WIDTH_M


#  Load class names + model once at startup 
def load_class_names(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"ERROR: '{path}' not found. Generate it in Colab first (see export_class_names.py).")
    with open(p) as f:
        return json.load(f)


CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)
print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
print("Loading model...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")
MODEL_LOCK = threading.Lock()  # tf models aren't guaranteed thread-safe for concurrent predict() calls

if MODEL.output_shape[-1] != len(CLASS_NAMES):
    sys.exit("ERROR: model output classes and class_names.json length don't match.")


#  Shared camera capture (single physical camera, read once, used by both pipelines) 
class SharedCamera:
    def __init__(self, index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit(f"ERROR: could not open camera index {index}.")
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


CAMERA = SharedCamera()


def predict(x_in):
    with MODEL_LOCK:
        return MODEL.predict(x_in, verbose=0)[0]


def classify_whole_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    x_in = np.expand_dims(resized.astype(np.float32), axis=0)  # raw [0,255]
    probs = predict(x_in)
    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])
    label = CLASS_NAMES[top_idx] if conf >= CONF_THRESHOLD else "Uncertain"
    return label, conf


# =============================================================================
# BOX 1 — Bike / vehicle classification (whole frame)
# =============================================================================
def gen_classify_stream():
    last_label, last_conf = "Warming up...", 0.0
    frame_count = 0
    while True:
        frame = CAMERA.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % CLASSIFY_EVERY_N_FRAMES == 0:
            last_label, last_conf = classify_whole_frame(frame)

        display = frame.copy()
        color = (0, 200, 0) if last_conf >= CONF_THRESHOLD else (0, 165, 255)
        text = f"{last_label} ({last_conf * 100:.1f}%)"
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ok, buf = cv2.imencode(".jpg", display)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# =============================================================================
# BOX 2 — Rough speed tracking (background subtraction + centroid tracker)
# =============================================================================
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


class Track:
    def __init__(self, track_id, box, centroid):
        self.id = track_id
        self.box = box
        self.centroid = centroid
        self.history = [(time.time(), centroid)]
        self.disappeared = 0
        self.label = "Classifying..."
        self.conf = 0.0
        self.speed_kmh = None
        self.frames_seen = 0

    def update_position(self, box, centroid):
        self.box = box
        self.centroid = centroid
        self.history.append((time.time(), centroid))
        cutoff = time.time() - (SPEED_WINDOW_SECONDS * 2)
        self.history = [h for h in self.history if h[0] >= cutoff]
        self.disappeared = 0
        self.frames_seen += 1

    def estimate_speed_kmh(self, meters_per_pixel):
        if len(self.history) < 2:
            return None
        t_new, c_new = self.history[-1]
        t_old, c_old = self.history[0]
        for t, c in self.history:
            if t >= t_new - SPEED_WINDOW_SECONDS:
                t_old, c_old = t, c
                break
        dt = t_new - t_old
        if dt <= 0.05:
            return None
        dist_px = np.hypot(c_new[0] - c_old[0], c_new[1] - c_old[1])
        dist_m = dist_px * meters_per_pixel
        return (dist_m / dt) * 3.6


class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

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

        for j in unmatched_boxes:
            self.tracks[self.next_id] = Track(self.next_id, boxes[j], centroids[j])
            self.next_id += 1

        for tid in track_ids:
            if tid not in used_tracks:
                self.tracks[tid].disappeared += 1

        self.tracks = {tid: t for tid, t in self.tracks.items()
                        if t.disappeared <= MAX_DISAPPEARED_FRAMES}
        return self.tracks


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
    return np.expand_dims(resized.astype(np.float32), axis=0)


def gen_speed_stream():
    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    tracker = CentroidTracker()

    while True:
        frame = CAMERA.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        fg_mask = back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        boxes = merge_overlapping_boxes(boxes)

        tracks = tracker.update(boxes)
        display = frame.copy()

        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue

            if track.frames_seen % CLASSIFY_EVERY_N_FRAMES == 0 or track.conf == 0.0:
                x_in = preprocess_crop(frame, track.box)
                if x_in is not None:
                    probs = predict(x_in)
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    track.label = CLASS_NAMES[top_idx] if conf >= CONF_THRESHOLD else "Uncertain"
                    track.conf = conf

            meters_per_pixel = guess_width_m(track.label) / max(track.box[2], 1)
            speed = track.estimate_speed_kmh(meters_per_pixel)
            if speed is not None:
                track.speed_kmh = speed

            x, y, w, h = track.box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
            speed_text = f"~{track.speed_kmh:.0f} km/h" if track.speed_kmh is not None else "measuring..."
            cv2.putText(display, f"#{tid} {track.label} ({track.conf*100:.0f}%) {speed_text}",
                        (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        cv2.putText(display, "Rough/relative speed estimate only", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        ok, buf = cv2.imencode(".jpg", display)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# =============================================================================
# Flask routes
# =============================================================================
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/classify")
def video_feed_classify():
    return Response(gen_classify_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/speed")
def video_feed_speed():
    return Response(gen_speed_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


# =============================================================================
# BOX 3 — License plate highlighting + capture
# =============================================================================
@app.route("/video_feed/plate")
def video_feed_plate():
    return Response(licencePlate.gen_plate_stream(CAMERA),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture_plate", methods=["POST"])
def capture_plate():
    result = licencePlate.capture_plate(CAMERA, save_dir=CAPTURE_DIR)
    if not result["success"]:
        return jsonify(result), 503

    result["full_image_url"] = f"/captures/{result['full_image']}"
    result["plate_image_url"] = (
        f"/captures/{result['plate_image']}" if result["plate_image"] else None
    )
    return jsonify(result)


@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(CAPTURE_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)