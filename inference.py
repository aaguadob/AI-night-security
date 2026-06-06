"""
inference.py
------------
Model loading and single-shot image inference for thermal + RGB pairs.

Both images are passed as a batch of 2 in one forward pass.
For video with persistent tracking use surveillance.py instead.

Usage:
    python inference.py --thermal img_t.jpg --rgb img_r.jpg
    python inference.py --thermal img_t.jpg --rgb img_r.jpg --weights best.pt --save
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/detect/runs/thermal/llvip_finetune/weights/last.pt"


def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def infer(model: YOLO, thermal_img: np.ndarray, rgb_img: np.ndarray,
          conf: float = 0.35, device: str = "cpu", imgsz: int = 640):
    """
    Run a single forward pass on a [thermal, rgb] batch.
    Both images are resized to imgsz x imgsz internally.
    Returns (thermal_result, rgb_result).
    """
    frame_t = cv2.resize(thermal_img, (imgsz, imgsz))
    frame_r = cv2.resize(rgb_img,     (imgsz, imgsz))

    results = model(
        [frame_t, frame_r],
        conf    = conf,
        classes = [0],      # person only
        device  = device,
        verbose = False,
    )
    return results[0], results[1]


# ── CLI: visualise a single image pair ───────────────────────────────────────

def _draw_boxes(frame: np.ndarray, result, color=(0, 200, 255)) -> np.ndarray:
    out = frame.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    default_device = "mps"    if torch.backends.mps.is_available()  else (
                     "cuda:0" if torch.cuda.is_available()           else "cpu")

    parser = argparse.ArgumentParser(description="Single image-pair inference")
    parser.add_argument("--thermal", required=True, help="Thermal image path")
    parser.add_argument("--rgb",     required=True, help="RGB image path")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--device",  default=default_device)
    parser.add_argument("--save",    action="store_true",
                        help="Save side-by-side result to inference_out.png")
    args = parser.parse_args()

    img_t = cv2.imread(args.thermal)
    img_r = cv2.imread(args.rgb)
    if img_t is None:
        raise FileNotFoundError(args.thermal)
    if img_r is None:
        raise FileNotFoundError(args.rgb)

    model = load_model(args.weights)
    res_t, res_r = infer(model, img_t, img_r,
                         conf=args.conf, device=args.device, imgsz=args.imgsz)

    frame_t = cv2.resize(img_t, (args.imgsz, args.imgsz))
    frame_r = cv2.resize(img_r, (args.imgsz, args.imgsz))

    drawn_t = _draw_boxes(frame_t, res_t, color=(0, 200, 255))
    drawn_r = _draw_boxes(frame_r, res_r, color=(0, 255, 0))

    cv2.putText(drawn_t, f"THERMAL  {len(res_t.boxes)} det",
                (10, drawn_t.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(drawn_r, f"RGB  {len(res_r.boxes)} det",
                (10, drawn_r.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.hstack([drawn_t, drawn_r])

    if args.save:
        out_path = "inference_out.png"
        cv2.imwrite(out_path, combined)
        print(f"Saved → {out_path}")

    cv2.imshow("Inference", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
