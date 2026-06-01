"""
finetune_thermal.py
-------------------
Fine-tunes a YOLO model on the LLVIP thermal (infrared) split.

Run the dataloader first:
    python llvip_dataloader.py --llvip_root ./LLVIP --output_root ./dataset

Then fine-tune:
    python finetune_thermal.py
    python finetune_thermal.py --model yolov8n.pt --epochs 100 --device cuda

After training, test on a thermal image:
    python finetune_thermal.py --infer LLVIP/infrared/test/010001.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


# ── training ─────────────────────────────────────────────────────────────────

def train(
    model_weights: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    run_name: str,
):
    print(f"\n── Fine-tuning {model_weights} on thermal data ──")
    print(f"   data   : {data_yaml}")
    print(f"   epochs : {epochs}  |  imgsz : {imgsz}  |  batch : {batch}")
    print(f"   device : {device}\n")

    model = YOLO(model_weights)

    model.train(
        data        = data_yaml,
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        device      = device,
        classes     = [0],          # person only
        # augmentation — useful for thermal (greyscale-safe transforms)
        hsv_h       = 0.0,          # no hue shift (thermal is greyscale)
        hsv_s       = 0.0,          # no saturation shift
        hsv_v       = 0.3,          # brightness variation — simulates AGC differences
        fliplr      = 0.5,
        translate   = 0.1,
        scale       = 0.4,
        mosaic      = 1.0,
        # training dynamics
        lr0         = 1e-3,
        lrf         = 0.01,
        warmup_epochs = 3,
        patience    = 20,           # early stopping
        save_period = 10,
        # logging
        project     = project,
        name        = run_name,
        exist_ok    = True,
    )

    best_weights = Path(project) / run_name / "weights" / "best.pt"
    print(f"\n── Training complete. Best weights → {best_weights} ──")
    return str(best_weights)


# ── inference + plot ──────────────────────────────────────────────────────────

def to_tensor(img_bgr):
    t = torch.tensor(img_bgr, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)


def draw_boxes(img_bgr, results, color=(0, 200, 255)):
    img = img_bgr.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{conf:.2f}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def infer_and_plot(weights: str, image_path: str, imgsz: int = 640):
    print(f"\n── Inference with {weights} on {image_path} ──")
    model = YOLO(weights)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_resized = cv2.resize(img, (imgsz, imgsz))

    t0 = time.time()
    results = model(to_tensor(img_resized))
    elapsed = time.time() - t0

    n_det = len(results[0].boxes)
    print(f"   detections : {n_det}  |  inference time : {elapsed*1000:.1f} ms")

    img_drawn = draw_boxes(img_resized, results, color=(0, 200, 255))
    img_rgb   = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.title(f"Thermal — {n_det} detections  ({elapsed*1000:.0f} ms)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("thermal_inference.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   Plot saved → thermal_inference.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on LLVIP thermal split")

    # training args
    parser.add_argument("--model",    default="yolo26n.pt",
                        help="Base weights to fine-tune from")
    parser.add_argument("--data",     default="dataset/llvip_mixed.yaml",
                        help="Path to the YAML produced by llvip_dataloader.py")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--device",   default="0" if torch.cuda.is_available() else "cpu",
                        help="'0' for first GPU, 'cpu', or 'mps' for Apple Silicon")
    parser.add_argument("--project",  default="runs/thermal")
    parser.add_argument("--name",     default="llvip_finetune")

    # inference-only mode
    parser.add_argument("--infer",    default=None,
                        help="Skip training and run inference on this image path")
    parser.add_argument("--weights",  default=None,
                        help="Weights to use for --infer (defaults to best from training)")

    args = parser.parse_args()

    if args.infer:
        # inference-only: use provided weights or the default best path
        weights = args.weights or str(
            Path(args.project) / args.name / "weights" / "best.pt"
        )
        infer_and_plot(weights, args.infer, imgsz=args.imgsz)
    else:
        best = train(
            model_weights = args.model,
            data_yaml     = args.data,
            epochs        = args.epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            device        = args.device,
            project       = args.project,
            run_name      = args.name,
        )
        # quick sanity-check inference on the first test image
        test_imgs = sorted(Path("dataset/images/test").glob("*.jpg"))
        if test_imgs:
            infer_and_plot(best, str(test_imgs[0]), imgsz=args.imgsz)


if __name__ == "__main__":
    main()