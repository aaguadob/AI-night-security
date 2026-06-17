# AI Night Security

Person detection and lingering-person alarm system for night-time surveillance using fused RGB and thermal video feeds.

A YOLO model fine-tuned on the [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) dataset runs on both channels simultaneously. Detections are fused across modalities — matched pairs show the RGB box (sharper detail), unmatched thermal boxes are kept (catches what RGB misses in darkness), and unmatched RGB boxes are discarded (no thermal corroboration). Any person that stays in the same area for longer than a configurable threshold triggers an audible alarm.

---

## Project structure

```
.
├── data_downloader.py   # Download FLIR thermal dataset from Kaggle
├── Data_loader.py       # Convert LLVIP (Pascal VOC) → YOLO format
├── fine_tuning.py       # Fine-tune YOLO on thermal / RGB / mixed splits
├── initial_run.py       # Quick sanity-check: run the model on a single image pair
├── inference.py         # Single image-pair inference (thermal + RGB batch)
├── alarm.py             # Alarm logic, drawing helpers, AlarmTracker class
├── surveillance.py      # Video loop — ByteTrack on thermal, alarm on both feeds
├── dual_detect.py       # Video loop — IoU-fused dual-channel detection + alarm
├── yolo26n.pt           # Base model weights (starting point for fine-tuning)
├── dataset/             # YOLO-ready dataset produced by Data_loader.py
└── runs/                # Training outputs (best.pt lives here after fine-tuning)
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon:** PyTorch with MPS support is selected automatically at runtime.

### 2. Get the dataset

**Option A — LLVIP** (recommended, paired thermal + RGB):
Download from [LLVIP releases](https://bupt-ai-cz.github.io/LLVIP/) and place at `./LLVIP/`.

**Option B — FLIR** (thermal only):
```bash
python data_downloader.py   # requires kagglehub + Kaggle credentials
```

### 3. Prepare the dataset

```bash
# Mixed thermal + RGB (default — best for fine-tuning)
python Data_loader.py

# Thermal only
python Data_loader.py --modality infrared

# RGB only
python Data_loader.py --modality visible
```

Output goes to `dataset/` with a `llvip_<modality>.yaml` config file.

### 4. Fine-tune

```bash
python fine_tuning.py                                      # defaults: 50 epochs, mixed data
python fine_tuning.py --model yolo26n.pt --epochs 100 --device mps
```

Best weights are saved to `runs/thermal/llvip_finetune/weights/best.pt`.

### 5. Run dual-channel surveillance

```bash
# Two video files
python dual_detect.py --rgb rgb.mp4 --thermal thermal.mp4 --weights best.pt

# Two webcams (index 0 and 1)
python dual_detect.py --rgb 0 --thermal 1 --weights best.pt

# Custom alarm threshold and confidence
python dual_detect.py --rgb rgb.mp4 --thermal thermal.mp4 \
    --weights best.pt --linger 10 --conf 0.4
```

Press **q** to quit.

---

## Scripts

### `dual_detect.py` — fused dual-channel detection *(main script)*

Runs both feeds in a single forward pass per frame. Box fusion strategy:

| Case | Action |
|---|---|
| RGB box matches a thermal box (IoU ≥ threshold) | Keep the **RGB** box |
| Thermal box has no RGB match | Keep the **thermal** box |
| RGB box has no thermal match | **Discard** |

A centroid tracker assigns stable IDs. When a track stays in the same grid zone longer than `--linger` seconds, an alarm fires.

```
--rgb       RGB video path or camera index   (required)
--thermal   Thermal video path or camera index  (required)
--weights   Path to .pt weights file         (required)
--linger    Seconds before alarm fires       (default: 10)
--conf      Detection confidence threshold   (default: 0.35)
--imgsz     Inference resolution             (default: 640)
--device    cuda:0 / mps / cpu              (auto-detected)
```

### `surveillance.py` — thermal-primary tracking

Uses ByteTrack on the thermal channel with the RGB feed displayed alongside for context. Alarm logic is identical to `dual_detect.py`. Use this when the cameras are not well-aligned and IoU matching is unreliable.

```
--linger    default: 15s
```

### `inference.py` — single image-pair inference

```bash
python inference.py --thermal img_t.jpg --rgb img_r.jpg --weights best.pt --save
```

---

## Alarm behaviour

- A person that stays in the **same grid zone** (8 × 6 grid) for longer than `--linger` seconds triggers a red alert overlay and an audible sound (`afplay` on macOS).
- Moving to a new zone resets the linger timer for that track.
- The alarm repeats at most once every 5 seconds per track (configurable via `ALARM_COOLDOWN` in `alarm.py`).
- Tracks that disappear for more than 2 seconds are dropped (configurable via `GRACE_PERIOD`).

---

## Dataset: LLVIP

LLVIP provides 15,488 strictly-aligned infrared / visible image pairs with person annotations in Pascal VOC format. `Data_loader.py` converts them to YOLO format and supports three modes:

| Mode | Images |
|---|---|
| `infrared` | Thermal only |
| `visible` | RGB only |
| `mixed` | Both (prefixed `ir_` / `rgb_` to avoid name collisions) |

---

## Requirements

See `requirements.txt`. Key dependencies:

- `ultralytics` ≥ 8.2 — YOLO model and ByteTrack
- `torch` ≥ 2.2 — inference backend
- `opencv-python` ≥ 4.9 — video I/O and drawing
- `lapx` ≥ 0.5 — ByteTrack dependency
