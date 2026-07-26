"""
licencePlate.py

License plate detection / highlighting / capture module.
Plugs into the shared camera used by app.py (Box 1 = classification,
Box 2 = speed, Box 3 = this).

Detection uses two classical OpenCV techniques combined (no extra model
files or downloads needed, since both ship inside opencv-python):

  1. Haar cascade  -> cv2.data.haarcascades/haarcascade_russian_plate_number.xml
  2. Contour / edge heuristic -> bilateral filter + Canny edges + contour
     rectangularity + aspect-ratio filtering (the classic "ANPR without a
     model" approach)

Results from both are merged (overlap-suppressed) into one list of boxes
per frame. This won't be as accurate as a trained plate detector, but it
works out of the box for a student project and highlights plates in real
time.

Public API used by app.py:
    detect_plates(frame_bgr)              -> list[(x, y, w, h)]
    draw_highlights(frame_bgr, boxes)      -> annotated copy of the frame
    gen_plate_stream(camera)               -> MJPEG generator (Box 3 live feed)
    capture_plate(camera, save_dir)        -> dict with capture result
"""

import time
from pathlib import Path

import cv2
import numpy as np

#  Config 
MIN_PLATE_AREA = 1500
MAX_PLATE_AREA_FRAC = 0.15          # a plate shouldn't be > 15% of the frame
PLATE_ASPECT_RANGE = (1.0, 6.0)     # width / height of a typical plate
                                     # (motorcycle plates run squarer ~1.2-2,
                                     # car/truck plates run wider ~2-5)
MERGE_OVERLAP_IOU = 0.2
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS = 4

_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
try:
    _CASCADE = cv2.CascadeClassifier(str(_CASCADE_PATH))
    if _CASCADE.empty():
        print(f"WARNING: could not load plate cascade at {_CASCADE_PATH}; "
              "falling back to contour-only detection.")
        _CASCADE = None
except AttributeError:
    # OpenCV 5.0 moved CascadeClassifier into opencv_contrib. If only plain
    # opencv-python is installed (not opencv-contrib-python), the attribute
    # won't exist at all. Degrade gracefully to contour-only detection
    # instead of crashing the whole app on import.
    print("WARNING: cv2.CascadeClassifier is unavailable (install "
          "opencv-contrib-python for Haar cascade support on OpenCV 5+). "
          "Falling back to contour-only plate detection.")
    _CASCADE = None


#  Box helpers 
def _box_iou(a, b):
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


def _merge_boxes(boxes):
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _box_iou(merged[i], merged[j]) >= MERGE_OVERLAP_IOU:
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


#  Detection 
def _detect_haar(gray):
    if _CASCADE is None:
        return []
    rects = _CASCADE.detectMultiScale(
        gray, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(60, 20),
    )
    return [tuple(r) for r in rects]


def _detect_contours(gray, max_candidates=3):
    frame_area = gray.shape[0] * gray.shape[1]
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_PLATE_AREA or area > frame_area * MAX_PLATE_AREA_FRAC:
            continue

        # plates are near-rectangular -> approximate the contour and keep
        # shapes with roughly 4-8 vertices (allows a little noise on edges)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if not (4 <= len(approx) <= 8):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if not (PLATE_ASPECT_RANGE[0] <= aspect <= PLATE_ASPECT_RANGE[1]):
            continue

        # plates are close to a filled rectangle -> check how much of the
        # bounding box the contour actually fills (rejects sparse/odd shapes)
        rect_fill = area / float(w * h)
        if rect_fill < 0.55:
            continue

        candidates.append((area, (x, y, w, h)))

    # keep only the strongest few candidates (by area) to avoid cluttering
    # the frame with weak/spurious rectangles
    candidates.sort(key=lambda t: t[0], reverse=True)
    return [box for _, box in candidates[:max_candidates]]


def detect_plates(frame_bgr):
    """Return a merged list of (x, y, w, h) candidate license plate boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes = _detect_haar(gray) + _detect_contours(gray)
    merged = _merge_boxes(boxes)
    # cap results: a frame realistically has 1-2 visible plates, so don't
    # let residual noise clutter the highlight overlay
    merged.sort(key=lambda b: b[2] * b[3], reverse=True)
    return merged[:2]


#  Drawing 
def draw_highlights(frame_bgr, boxes, label="PLATE"):
    display = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        # bright highlight box + translucent fill so it visually "pops"
        overlay = display.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 3)

        tag = label
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(0, y - 8)
        cv2.rectangle(display, (x, ty - th - 6), (x + tw + 10, ty + 2), (0, 255, 255), -1)
        cv2.putText(display, tag, (x + 5, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    return display


#  Box 3 — live highlighted stream 
def gen_plate_stream(camera):
    """MJPEG generator: live camera feed with plate candidates highlighted."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        boxes = detect_plates(frame)
        display = draw_highlights(frame, boxes)

        status = f"{len(boxes)} plate(s) detected" if boxes else "Scanning..."
        cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255) if boxes else (200, 200, 200), 2)

        ok, buf = cv2.imencode(".jpg", display)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


#  Capture (still photo, called on-demand from a "Capture" button) 
def capture_plate(camera, save_dir="captures"):
    """
    Grab the current frame, detect + highlight plate(s), save the full
    annotated photo (and, if a plate was found, a cropped close-up) to
    disk. Returns a dict describing what was saved.
    """
    frame = camera.get_frame()
    if frame is None:
        return {"success": False, "error": "No camera frame available."}

    boxes = detect_plates(frame)
    annotated = draw_highlights(frame, boxes)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    full_name = f"capture_{stamp}.jpg"
    cv2.imwrite(str(out_dir / full_name), annotated)

    crop_name = None
    if boxes:
        # use the largest detected box as "the" plate for the close-up crop
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        mx, my = int(w * 0.15), int(h * 0.3)
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
        crop = frame[y0:y1, x0:x1]
        if crop.size > 0:
            crop_name = f"plate_{stamp}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)

    return {
        "success": True,
        "plates_found": len(boxes),
        "boxes": boxes,
        "full_image": full_name,
        "plate_image": crop_name,
    }