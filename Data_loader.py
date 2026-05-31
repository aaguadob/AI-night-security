"""
llvip_dataloader.py
-------------------
Prepares the LLVIP dataset for YOLO fine-tuning on the thermal (infrared) split.

Expected LLVIP directory layout:
    LLVIP/
        visible/
            train/   *.jpg
            test/    *.jpg
        infrared/
            train/   *.jpg
            test/    *.jpg
        Annotations/   *.xml   (Pascal VOC format, shared by both modalities)

What this script does:
  1. Parses Pascal VOC XML annotations
  2. Converts boxes to YOLO format  (cx cy w h, normalised 0-1)
  3. Copies infrared images + writes per-image .txt label files into
     a clean dataset/  tree that Ultralytics can consume directly
  4. Writes llvip_thermal.yaml

Usage:
    python llvip_dataloader.py --llvip_root ./LLVIP --output_root ./dataset
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Return list of {label, x1, y1, x2, y2, img_w, img_h} from a VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip().lower()
        bndbox = obj.find("bndbox")
        x1 = float(bndbox.find("xmin").text)
        y1 = float(bndbox.find("ymin").text)
        x2 = float(bndbox.find("xmax").text)
        y2 = float(bndbox.find("ymax").text)
        objects.append(dict(label=label, x1=x1, y1=y1, x2=x2, y2=y2,
                            img_w=img_w, img_h=img_h))
    return objects


def voc_to_yolo(obj: dict) -> tuple[float, float, float, float]:
    """Convert VOC absolute coords to YOLO normalised (cx, cy, w, h)."""
    img_w, img_h = obj["img_w"], obj["img_h"]
    cx = (obj["x1"] + obj["x2"]) / 2 / img_w
    cy = (obj["y1"] + obj["y2"]) / 2 / img_h
    w  = (obj["x2"] - obj["x1"]) / img_w
    h  = (obj["y2"] - obj["y1"]) / img_h
    return cx, cy, w, h


# LLVIP only contains "person" — map to class 0
LABEL_MAP = {"person": 0}


def convert_split(
    infrared_img_dir: Path,
    annotation_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    split_name: str,
) -> tuple[int, int]:
    """
    Process one split (train / test).
    Returns (n_images_ok, n_images_skipped).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(infrared_img_dir.glob("*.jpg"))
    if not img_paths:
        img_paths = sorted(infrared_img_dir.glob("*.png"))

    ok, skipped = 0, 0
    for img_path in img_paths:
        stem = img_path.stem
        xml_path = annotation_dir / f"{stem}.xml"

        if not xml_path.exists():
            print(f"  [WARN] no annotation for {img_path.name}, skipping")
            skipped += 1
            continue

        objects = parse_voc_xml(xml_path)
        yolo_lines = []
        for obj in objects:
            cls_id = LABEL_MAP.get(obj["label"])
            if cls_id is None:
                continue  # skip non-person labels if any
            cx, cy, w, h = voc_to_yolo(obj)
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)

        # write label file (empty file = valid negative sample)
        lbl_path = out_lbl_dir / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines))

        ok += 1

    print(f"  [{split_name}] {ok} images converted, {skipped} skipped")
    return ok, skipped


def write_yaml(output_root: Path, class_names: list[str]) -> Path:
    yaml_path = output_root / "llvip_thermal.yaml"
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val:   images/test",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    yaml_path.write_text("\n".join(lines))
    print(f"\n  YAML written → {yaml_path}")
    return yaml_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare LLVIP thermal split for YOLO")
    parser.add_argument("--llvip_root",  type=Path, default=Path("LLVIP"),
                        help="Root of the raw LLVIP download")
    parser.add_argument("--output_root", type=Path, default=Path("dataset"),
                        help="Where to write the YOLO-ready dataset")
    args = parser.parse_args()

    llvip   = args.llvip_root
    out     = args.output_root

    annotation_dir = llvip / "Annotations"
    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotations directory not found at {annotation_dir}. "
            "Make sure --llvip_root points to the LLVIP root folder."
        )

    print("── Converting LLVIP thermal split to YOLO format ──")

    for split in ("train", "test"):
        infrared_dir = llvip / "infrared" / split
        if not infrared_dir.exists():
            print(f"  [INFO] split '{split}' not found, skipping")
            continue

        convert_split(
            infrared_img_dir = infrared_dir,
            annotation_dir   = annotation_dir,
            out_img_dir      = out / "images" / split,
            out_lbl_dir      = out / "labels" / split,
            split_name       = split,
        )

    write_yaml(out, class_names=["person"])
    print("\nDone. Dataset ready at:", out.resolve())


if __name__ == "__main__":
    main()