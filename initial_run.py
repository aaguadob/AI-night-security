from ultralytics import YOLO
import cv2
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("runs/detect/runs/thermal/llvip_finetune/weights/best.pt")
t0 = time.time()

img0 = cv2.imread("LLVIP/visible/test/200122.jpg")
img1 = cv2.imread("LLVIP/infrared/test/200122.jpg")

process_size = (640, 640)
img0 = cv2.resize(img0, process_size)
img1 = cv2.resize(img1, process_size)

def to_tensor(img):
    t = torch.tensor(img, dtype=torch.float32)
    t = t.permute(2, 0, 1)
    t = t / 255.0
    t = t.unsqueeze(0)
    return t

results_rgb     = model(to_tensor(img0))
results_thermal = model(to_tensor(img1))

def draw_boxes(img_bgr, results, color=(0, 255, 0)):
    img = img_bgr.copy()
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

img0_drawn = draw_boxes(img0, results_rgb,     color=(0, 255, 0))   # green for RGB
img1_drawn = draw_boxes(img1, results_thermal, color=(0, 200, 255)) # amber for thermal

# BGR → RGB for matplotlib
img0_drawn = cv2.cvtColor(img0_drawn, cv2.COLOR_BGR2RGB)
img1_drawn = cv2.cvtColor(img1_drawn, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img0_drawn)
axes[0].set_title(f"RGB  —  {len(results_rgb[0].boxes)} detections", fontsize=13)
axes[0].axis("off")

axes[1].imshow(img1_drawn)
axes[1].set_title(f"Thermal  —  {len(results_thermal[0].boxes)} detections", fontsize=13)
axes[1].axis("off")

plt.suptitle("LLVIP 010001 — YOLO detections", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("detections.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Total time: {time.time() - t0:.3f}s")