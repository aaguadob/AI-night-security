"""
dual_detect.py
--------------
Dual-channel (RGB + thermal) detection with fused bounding boxes and a
lingering-person alarm.

For each frame pair:
  1. A single forward pass runs on the [thermal, rgb] batch.
  2. Boxes from both channels are pooled and de-duplicated with NMS.
  3. A centroid tracker assigns stable IDs across frames.
  4. Any track that stays in the same grid zone for longer than `--linger`
     seconds (default: 10) triggers an audible alarm and a red alert overlay.

Usage:
    python dual_detect.py --rgb rgb.mp4 --thermal thermal.mp4 --weights best.pt
    python dual_detect.py --rgb 0 --thermal 1 --weights best.pt --linger 10
    python dual_detect.py --rgb rgb.mp4 --thermal thermal.mp4 --weights best.pt --conf 0.4 --linger 5
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from alarm import (
    draw_alarm_box, draw_normal_box,
    draw_channel_label, draw_hud,
    _play_alarm, ALARM_COOLDOWN,
)


# ── box fusion ────────────────────────────────────────────────────────────────

def _merge_boxes(res_t, res_r, nms_thresh: float = 0.45) -> list[tuple]:
    """
    Pool xyxy boxes from both channel results and de-duplicate with NMS.
    Returns list of (x1, y1, x2, y2, conf).
    """
    xywh_list: list[list[float]] = []
    confs:     list[float]       = []

    for result in (res_t, res_r):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            xywh_list.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            confs.append(float(box.conf[0]))

    if not xywh_list:
        return []

    indices = cv2.dnn.NMSBoxes(xywh_list, confs,
                                score_threshold=0.0, nms_threshold=nms_thresh)
    if len(indices) == 0:
        return []

    # OpenCV ≥ 4.7 returns a 1-D array; older versions return [[i], [j], ...]
    flat = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]

    merged = []
    for i in flat:
        x, y, w, h = xywh_list[i]
        merged.append((int(x), int(y), int(x + w), int(y + h), confs[i]))
    return merged


# ── centroid tracker ──────────────────────────────────────────────────────────

_ZONE_COLS = 8
_ZONE_ROWS = 6


def _get_zone(cx: int, cy: int, w: int, h: int) -> tuple[int, int]:
    col = min(int(cx / w * _ZONE_COLS), _ZONE_COLS - 1)
    row = min(int(cy / h * _ZONE_ROWS), _ZONE_ROWS - 1)
    return col, row


class CentroidTracker:
    """
    Lightweight tracker that assigns stable IDs by nearest-centroid matching.
    When a track stays in the same grid zone longer than `linger` seconds,
    it is flagged as an alarm.  Moves to a new zone reset the linger timer.
    """

    def __init__(self, linger: float = 10.0,
                 max_dist: float = 80.0, grace: float = 2.0):
        self.linger   = linger
        self.max_dist = max_dist
        self.grace    = grace
        self._next_id = 0
        self._state: dict[int, dict] = {}

    def update(self, detections: list[tuple], now: float,
               frame_w: int, frame_h: int) -> list[tuple]:
        """
        detections : [(x1, y1, x2, y2, conf), ...]
        Returns    : [(x1, y1, x2, y2, conf, track_id, elapsed, is_alarm), ...]
        Plays alarm sounds for qualifying tracks in-place.
        """
        self._match_and_update(detections, now, frame_w, frame_h)
        self._purge_stale(now)

        out = []
        for tid, s in self._state.items():
            elapsed  = now - s["entry_time"]
            is_alarm = elapsed >= self.linger
            if is_alarm and now - s["last_alarm"] >= ALARM_COOLDOWN:
                _play_alarm()
                s["last_alarm"] = now
            out.append((s["x1"], s["y1"], s["x2"], s["y2"],
                        s["conf"], tid, elapsed, is_alarm))
        return out

    @property
    def n_active(self) -> int:
        return len(self._state)

    # ── private ───────────────────────────────────────────────────────────────

    def _match_and_update(self, detections, now, frame_w, frame_h):
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2)
                     for x1, y1, x2, y2, _ in detections]

        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()

        for i, (cx, cy) in enumerate(centroids):
            best_id, best_d = None, self.max_dist
            for tid, s in self._state.items():
                if tid in matched_tracks:
                    continue
                d = np.hypot(cx - s["cx"], cy - s["cy"])
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is not None:
                matched_tracks.add(best_id)
                matched_dets.add(i)
                x1, y1, x2, y2, conf_val = detections[i]
                s = self._state[best_id]
                zone = _get_zone(cx, cy, frame_w, frame_h)
                if s["zone"] != zone:
                    s["zone"]       = zone
                    s["entry_time"] = now  # moved to new area — reset timer
                s.update(cx=cx, cy=cy, x1=x1, y1=y1, x2=x2, y2=y2,
                         conf=conf_val, last_seen=now)

        for i, (x1, y1, x2, y2, conf_val) in enumerate(detections):
            if i in matched_dets:
                continue
            cx, cy = centroids[i]
            zone   = _get_zone(cx, cy, frame_w, frame_h)
            self._state[self._next_id] = dict(
                cx=cx, cy=cy, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf_val,
                zone=zone, entry_time=now, last_seen=now, last_alarm=0.0,
            )
            self._next_id += 1

    def _purge_stale(self, now: float):
        stale = [tid for tid, s in self._state.items()
                 if now - s["last_seen"] > self.grace]
        for tid in stale:
            del self._state[tid]


# ── video loop ────────────────────────────────────────────────────────────────

def _open_source(src: str) -> cv2.VideoCapture:
    try:
        return cv2.VideoCapture(int(src))
    except ValueError:
        return cv2.VideoCapture(str(Path(src).resolve()))


def run(weights: str, rgb_src: str, thermal_src: str,
        imgsz: int, conf: float, device: str, linger: float):

    model   = YOLO(weights)
    tracker = CentroidTracker(linger=linger)

    cap_r = _open_source(rgb_src)
    cap_t = _open_source(thermal_src)

    if not cap_r.isOpened():
        raise RuntimeError(f"Cannot open RGB source: {rgb_src!r}")
    if not cap_t.isOpened():
        raise RuntimeError(f"Cannot open thermal source: {thermal_src!r}")

    print(f"RGB          : {rgb_src}")
    print(f"Thermal      : {thermal_src}")
    print(f"Weights      : {weights}")
    print(f"Device       : {device}")
    print(f"Linger alarm : {linger}s")
    print("Press  q  to quit.\n")

    fps_timer   = time.time()
    fps_display = 0.0
    frame_count = 0

    while True:
        ret_r, raw_r = cap_r.read()
        ret_t, raw_t = cap_t.read()
        if not ret_r or not ret_t:
            print("Stream ended.")
            break

        now     = time.time()
        frame_r = cv2.resize(raw_r, (imgsz, imgsz))
        frame_t = cv2.resize(raw_t, (imgsz, imgsz))

        # Single forward pass — batch of 2: [thermal, rgb]
        results = model(
            [frame_t, frame_r],
            conf    = conf,
            classes = [0],      # person only
            device  = device,
            verbose = False,
        )
        res_t, res_r = results[0], results[1]

        # Fuse boxes from both channels via NMS
        merged = _merge_boxes(res_t, res_r)

        # Track fused boxes; check for lingering persons
        frame_h, frame_w = frame_t.shape[:2]
        tracks = tracker.update(merged, now, frame_w, frame_h)

        # Draw detections on both frames
        n_alerts = 0
        for x1, y1, x2, y2, conf_val, tid, elapsed, is_alarm in tracks:
            if is_alarm:
                n_alerts += 1
                draw_alarm_box(frame_t, x1, y1, x2, y2, tid, elapsed)
                draw_alarm_box(frame_r, x1, y1, x2, y2, tid, elapsed)
            else:
                draw_normal_box(frame_t, x1, y1, x2, y2, tid, conf_val, elapsed)
                draw_normal_box(frame_r, x1, y1, x2, y2, tid, conf_val, elapsed)

        frame_count += 1
        if frame_count % 30 == 0:
            fps_display = 30 / (now - fps_timer)
            fps_timer   = now

        draw_channel_label(frame_t, "THERMAL")
        draw_channel_label(frame_r, "RGB")

        combined = np.hstack([frame_t, frame_r])
        draw_hud(combined, tracker.n_active, n_alerts, fps_display)

        cv2.imshow("AI Night Security — Dual Fusion Alarm", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_r.release()
    cap_t.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    default_device = (
        "mps"    if torch.backends.mps.is_available()  else
        "cuda:0" if torch.cuda.is_available()           else
        "cpu"
    )

    parser = argparse.ArgumentParser(
        description="Fused dual-channel (RGB + thermal) detection with lingering alarm")
    parser.add_argument("--rgb",     required=True,
                        help="RGB video path or camera index")
    parser.add_argument("--thermal", required=True,
                        help="Thermal video path or camera index")
    parser.add_argument("--weights", required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--device",  default=default_device)
    parser.add_argument("--linger",  type=float, default=10.0,
                        help="Seconds in same area before alarm fires (default: 10)")
    args = parser.parse_args()

    run(
        weights     = args.weights,
        rgb_src     = args.rgb,
        thermal_src = args.thermal,
        imgsz       = args.imgsz,
        conf        = args.conf,
        device      = args.device,
        linger      = args.linger,
    )


if __name__ == "__main__":
    main()
