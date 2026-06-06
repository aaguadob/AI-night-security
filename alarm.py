"""
alarm.py
--------
Lingering-person alarm logic, track state management, and drawing helpers.

AlarmTracker maintains per-track state (zone, entry time, last alarm time) and
applies alarm overlays to both the thermal and RGB frames when a person has been
in the same area for longer than `linger` seconds.

Import and use from surveillance.py:
    from alarm import AlarmTracker, draw_channel_label, draw_hud
"""

import subprocess
import time

import cv2
import numpy as np

# ── tuneable constants ────────────────────────────────────────────────────────

ZONE_COLS     = 8      # grid columns that define "areas"
ZONE_ROWS     = 6      # grid rows
GRACE_PERIOD  = 2.0    # seconds a track may vanish before its state is dropped
ALARM_COOLDOWN = 5.0   # min seconds between repeated alarm sounds per track

ALARM_SOUND_MAC = "/System/Library/Sounds/Sosumi.aiff"

COLOR_NORMAL    = (0, 200, 255)   # amber
COLOR_ALERT     = (0, 0, 255)     # red
COLOR_ALERT_DIM = (0, 80, 220)    # dim red for pulse


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_normal_box(frame, x1, y1, x2, y2, track_id, conf, elapsed):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NORMAL, 2)
    label = f"#{track_id} {conf:.2f} {elapsed:.0f}s"
    cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NORMAL, 1, cv2.LINE_AA)


def draw_alarm_box(frame, x1, y1, x2, y2, track_id, elapsed):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 160), -1)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    pulse     = int(time.time() * 2) % 2 == 0
    color     = COLOR_ALERT if pulse else COLOR_ALERT_DIM
    thickness = 4           if pulse else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"ALERT #{track_id}  {elapsed:.0f}s"
    cv2.putText(frame, label, (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_channel_label(frame, text):
    cv2.putText(frame, text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_hud(frame, n_persons, n_alerts, fps):
    text = f"Persons: {n_persons}  |  Alerts: {n_alerts}  |  {fps:.1f} FPS"
    cv2.putText(frame, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if n_alerts > 0 and int(time.time() * 2) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_ALERT, 6)


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_zone(cx: int, cy: int, w: int, h: int) -> tuple[int, int]:
    col = min(int(cx / w * ZONE_COLS), ZONE_COLS - 1)
    row = min(int(cy / h * ZONE_ROWS), ZONE_ROWS - 1)
    return col, row


def _play_alarm():
    try:
        subprocess.Popen(["afplay", ALARM_SOUND_MAC],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


# ── tracker ───────────────────────────────────────────────────────────────────

class AlarmTracker:
    """
    Stateful tracker that fires a lingering alarm when a person stays in the
    same grid area for longer than `linger` seconds.

    Call update() once per frame with the thermal YOLO result and both frames.
    It draws alarm/normal boxes on both frames in-place and returns n_alerts.
    """

    def __init__(self, linger: float = 15.0):
        self.linger      = linger
        self._state: dict[int, dict] = {}   # keyed by track_id
        self.active_ids: set[int]    = set()

    def update(self, thermal_result, frame_t: np.ndarray, frame_r: np.ndarray,
               now: float) -> int:
        """
        Process one frame's thermal detections.
        Draws boxes on frame_t and frame_r in-place.
        Returns the number of active alarm triggers.
        """
        self.active_ids = set()
        frame_h, frame_w = frame_t.shape[:2]

        if thermal_result.boxes.id is not None:
            for box, tid in zip(thermal_result.boxes,
                                thermal_result.boxes.id.int().tolist()):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf  = float(box.conf[0])
                cx    = (x1 + x2) // 2
                cy    = (y1 + y2) // 2
                zone  = _get_zone(cx, cy, frame_w, frame_h)

                self.active_ids.add(tid)
                self._update_state(tid, zone, conf, now)

                elapsed = now - self._state[tid]["entry_time"]

                if elapsed >= self.linger:
                    draw_alarm_box(frame_t, x1, y1, x2, y2, tid, elapsed)
                    draw_alarm_box(frame_r, x1, y1, x2, y2, tid, elapsed)
                    self._maybe_play_alarm(tid, now)
                else:
                    draw_normal_box(frame_t, x1, y1, x2, y2, tid, conf, elapsed)
                    draw_normal_box(frame_r, x1, y1, x2, y2, tid, conf, elapsed)

        self._purge_stale(now)

        return sum(
            1 for tid in self.active_ids
            if now - self._state[tid]["entry_time"] >= self.linger
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _update_state(self, tid: int, zone, conf: float, now: float):
        if tid not in self._state:
            self._state[tid] = {
                "zone":            zone,
                "entry_time":      now,
                "last_seen":       now,
                "last_alarm_time": 0.0,
                "conf":            conf,
            }
        else:
            s = self._state[tid]
            s["last_seen"] = now
            s["conf"]      = conf
            if s["zone"] != zone:
                s["zone"]       = zone
                s["entry_time"] = now  # moved to new area — reset timer

    def _maybe_play_alarm(self, tid: int, now: float):
        s = self._state[tid]
        if now - s["last_alarm_time"] >= ALARM_COOLDOWN:
            _play_alarm()
            s["last_alarm_time"] = now

    def _purge_stale(self, now: float):
        stale = [tid for tid, s in self._state.items()
                 if tid not in self.active_ids
                 and now - s["last_seen"] > GRACE_PERIOD]
        for tid in stale:
            del self._state[tid]
