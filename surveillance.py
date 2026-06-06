"""
surveillance.py
---------------
Video surveillance loop: reads thermal and RGB feeds, runs batch tracking,
and delegates alarm logic to AlarmTracker.

Detection uses model.track() so ByteTrack maintains stable IDs across frames.
Both frames are passed as a batch of 2 in each call (single forward pass).
Alarm logic and drawing are handled entirely by alarm.py.

Usage:
    python surveillance.py --thermal thermal.mp4 --rgb rgb.mp4
    python surveillance.py --thermal 1 --rgb 0            # two webcams
    python surveillance.py --thermal t.mp4 --rgb r.mp4 --linger 10 --conf 0.4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import load_model, DEFAULT_WEIGHTS
from alarm import AlarmTracker, draw_channel_label, draw_hud


def open_source(src: str) -> cv2.VideoCapture:
    try:
        return cv2.VideoCapture(int(src))
    except ValueError:
        return cv2.VideoCapture(str(Path(src).resolve()))


def run(weights: str, thermal_src: str, rgb_src: str,
        imgsz: int, conf: float, device: str, linger: float):

    model   = load_model(weights)
    tracker = AlarmTracker(linger=linger)

    cap_t = open_source(thermal_src)
    cap_r = open_source(rgb_src)

    if not cap_t.isOpened():
        raise RuntimeError(f"Cannot open thermal source: {thermal_src!r}")
    if not cap_r.isOpened():
        raise RuntimeError(f"Cannot open RGB source: {rgb_src!r}")

    print(f"Thermal      : {thermal_src}")
    print(f"RGB          : {rgb_src}")
    print(f"Weights      : {weights}")
    print(f"Device       : {device}")
    print(f"Linger alarm : {linger}s")
    print("Press  q  to quit.\n")

    fps_timer   = time.time()
    fps_display = 0.0
    frame_count = 0

    while True:
        ret_t, raw_t = cap_t.read()
        ret_r, raw_r = cap_r.read()
        if not ret_t or not ret_r:
            print("Stream ended.")
            break

        now     = time.time()
        frame_t = cv2.resize(raw_t, (imgsz, imgsz))
        frame_r = cv2.resize(raw_r, (imgsz, imgsz))

        # Single forward pass: batch of 2 — [thermal, rgb]
        results = model.track(
            [frame_t, frame_r],
            persist = True,
            conf    = conf,
            classes = [0],          # person only
            device  = device,
            tracker = "bytetrack.yaml",
            verbose = False,
        )

        # Alarm logic runs on thermal result (results[0]); draws on both frames
        n_alerts = tracker.update(results[0], frame_t, frame_r, now)

        frame_count += 1
        if frame_count % 30 == 0:
            fps_display = 30 / (now - fps_timer)
            fps_timer   = now

        draw_channel_label(frame_t, "THERMAL")
        draw_channel_label(frame_r, "RGB")

        combined = np.hstack([frame_t, frame_r])
        draw_hud(combined, len(tracker.active_ids), n_alerts, fps_display)

        cv2.imshow("AI Night Security — Lingering Alarm", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_t.release()
    cap_r.release()
    cv2.destroyAllWindows()


def main():
    default_device = "mps"    if torch.backends.mps.is_available()  else (
                     "cuda:0" if torch.cuda.is_available()           else "cpu")

    parser = argparse.ArgumentParser(
        description="Dual-feed surveillance with lingering-person alarm")
    parser.add_argument("--thermal", required=True,
                        help="Thermal video path or camera index")
    parser.add_argument("--rgb",     required=True,
                        help="RGB video path or camera index")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--device",  default=default_device)
    parser.add_argument("--linger",  type=float, default=15.0,
                        help="Seconds in same area before alarm (default: 15)")
    args = parser.parse_args()

    run(
        weights     = args.weights,
        thermal_src = args.thermal,
        rgb_src     = args.rgb,
        imgsz       = args.imgsz,
        conf        = args.conf,
        device      = args.device,
        linger      = args.linger,
    )


if __name__ == "__main__":
    main()
