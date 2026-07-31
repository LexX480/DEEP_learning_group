"""
streamlit_app.py

Streamlit front-end for the Vehicle Detection / Speed Tracking project.
Reuses the same model, class list, tracking logic, and license-plate
module as app.py (Flask) — this is just a different UI on top of the
exact same pipeline.

Run:
    streamlit run streamlit_app.py
Then it opens automatically at:
    http://localhost:8501
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import licencePlate  # reused as-is from the Flask version

# =============================================================================
# Config (mirrors app.py)
# =============================================================================
MODEL_PATH = "transfer_best.keras"
CLASS_NAMES_PATH = "class_names.json"
TRAIN_LOG_PATH = "transfer_log.csv"
IMG_SIZE = 224
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


# =============================================================================
# Cached resources — loaded once per server process, shared across sessions
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    class_path = Path(CLASS_NAMES_PATH)
    if not class_path.is_file():
        st.error(f"'{CLASS_NAMES_PATH}' not found next to streamlit_app.py.")
        st.stop()
    with open(class_path) as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)
    if model.output_shape[-1] != len(class_names):
        st.error("Model output classes and class_names.json length don't match.")
        st.stop()
    return model, class_names


class SharedCamera:
    """Same idea as app.py's SharedCamera: one physical camera, read
    continuously in the background, so every tab/mode reads the latest
    frame without fighting over the device."""

    def __init__(self, index=0):
        import threading
        self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.frame = None
        self.running = self.cap.isOpened()
        if self.running:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
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

    def release(self):
        self.running = False
        self.cap.release()


@st.cache_resource(show_spinner="Opening camera...")
def get_camera(index):
    return SharedCamera(index)


model, CLASS_NAMES = load_model_and_classes()
MODEL_LOCK = None  # set on first use below (needs threading import)


def predict(x_in):
    import threading
    global MODEL_LOCK
    if MODEL_LOCK is None:
        MODEL_LOCK = threading.Lock()
    with MODEL_LOCK:
        return model.predict(x_in, verbose=0)[0]


def classify_whole_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    x_in = np.expand_dims(resized.astype(np.float32), axis=0)
    probs = predict(x_in)
    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])
    label = CLASS_NAMES[top_idx] if conf >= CONF_THRESHOLD else "Uncertain"
    return label, conf


# =============================================================================
# Speed-tracking helpers (mirrors app.py Box 2)
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


# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(page_title="Vehicle Detect", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Detection & Speed Tracking")
st.caption("Streamlit demo — vehicle classification, rough speed tracking, and license-plate highlighting, "
           "all running on the same TensorFlow model used in the Flask version.")

with st.sidebar:
    st.header("Settings")
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    st.caption("Change this if you have more than one camera and the wrong one opens.")
    st.divider()
    st.subheader("Model")
    st.write(f"**Classes ({len(CLASS_NAMES)}):**")
    st.write(", ".join(CLASS_NAMES))
    st.divider()
    st.info("Tick **Start camera** on a tab below to begin streaming. "
            "Only one tab should run at a time on shared hardware.")

tab_overview, tab_classify, tab_speed, tab_plate = st.tabs(
    ["📊 Overview", "🏷️ Classification", "⏱️ Speed Tracking", "🔍 License Plate"]
)

# =============================================================================
# Overview tab — training curves + quick model info
# =============================================================================
with tab_overview:
    st.subheader("Training history")
    log_path = Path(TRAIN_LOG_PATH)
    if log_path.is_file():
        df = pd.read_csv(log_path)
        df.index = df.index + 1
        df.index.name = "epoch"
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df[["accuracy", "val_accuracy"]])
            st.caption("Accuracy / validation accuracy per epoch")
        with col2:
            st.line_chart(df[["loss", "val_loss"]])
            st.caption("Loss / validation loss per epoch")
        best_epoch = int(df["val_accuracy"].idxmax())
        st.metric("Best validation accuracy",
                   f"{df['val_accuracy'].max()*100:.1f}%",
                   help=f"Achieved at epoch {best_epoch}")
    else:
        st.warning(f"'{TRAIN_LOG_PATH}' not found — skipping training curves.")

    st.subheader("How this demo works")
    st.markdown(
        "- **Classification** — the whole camera frame is fed to the model every "
        f"{CLASSIFY_EVERY_N_FRAMES} frames.\n"
        "- **Speed tracking** — background subtraction finds moving blobs, a simple "
        "centroid tracker follows them across frames, and speed is estimated from "
        "pixel displacement using an assumed real-world vehicle width.\n"
        "- **License plate** — a Haar cascade plus a classical edge/contour heuristic "
        "highlight likely plate regions; no plate-reading (OCR) is performed."
    )

# =============================================================================
# Classification tab (mirrors app.py Box 1)
# =============================================================================
with tab_classify:
    st.subheader("Vehicle classification")
    run_classify = st.checkbox("Start camera", key="run_classify")
    frame_slot = st.empty()
    label_slot = st.empty()

    if run_classify:
        camera = get_camera(camera_index)
        if not camera.running:
            st.error("Could not open the camera. Check the camera index in the sidebar.")
        else:
            last_label, last_conf = "Warming up...", 0.0
            frame_count = 0
            while st.session_state.get("run_classify"):
                frame = camera.get_frame()
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

                frame_slot.image(display, channels="BGR")
                label_slot.metric("Prediction", last_label, f"{last_conf*100:.1f}% confidence")
    else:
        st.info("Tick the box above to start streaming from the camera.")

# =============================================================================
# Speed tracking tab (mirrors app.py Box 2)
# =============================================================================
with tab_speed:
    st.subheader("Rough speed tracking")
    st.caption("Speed is a relative estimate only — it depends on an assumed vehicle "
               "width and is not calibrated against the real camera geometry.")
    run_speed = st.checkbox("Start camera", key="run_speed")
    frame_slot_speed = st.empty()

    if run_speed:
        camera = get_camera(camera_index)
        if not camera.running:
            st.error("Could not open the camera. Check the camera index in the sidebar.")
        else:
            back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
            tracker = CentroidTracker()

            while st.session_state.get("run_speed"):
                frame = camera.get_frame()
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

                frame_slot_speed.image(display, channels="BGR")
    else:
        st.info("Tick the box above to start streaming from the camera.")

# =============================================================================
# License plate tab (mirrors app.py Box 3 + reuses licencePlate.py as-is)
# =============================================================================
with tab_plate:
    st.subheader("License plate highlighting")
    run_plate = st.checkbox("Start camera", key="run_plate")
    frame_slot_plate = st.empty()
    capture_clicked = st.button("📸 Capture current frame", disabled=not run_plate)
    result_slot = st.empty()

    if run_plate:
        camera = get_camera(camera_index)
        if not camera.running:
            st.error("Could not open the camera. Check the camera index in the sidebar.")
        else:
            if capture_clicked:
                result = licencePlate.capture_plate(camera, save_dir=CAPTURE_DIR)
                with result_slot.container():
                    if result["success"]:
                        st.success(f"Captured — {result['plates_found']} plate(s) detected.")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(str(Path(CAPTURE_DIR) / result["full_image"]),
                                      caption="Full frame")
                        if result["plate_image"]:
                            with c2:
                                st.image(str(Path(CAPTURE_DIR) / result["plate_image"]),
                                          caption="Plate close-up")
                    else:
                        st.error(result.get("error", "Capture failed."))

            while st.session_state.get("run_plate"):
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                boxes = licencePlate.detect_plates(frame)
                display = licencePlate.draw_highlights(frame, boxes)

                status = f"{len(boxes)} plate(s) detected" if boxes else "Scanning..."
                cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255) if boxes else (200, 200, 200), 2)

                frame_slot_plate.image(display, channels="BGR")
    else:
        st.info("Tick the box above to start streaming from the camera.")
