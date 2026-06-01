"""
llvip_dataloader.py
-------------------
Prepares the LLVIP dataset for YOLO fine-tuning.

Supports three modes via --modality:
  infrared  → thermal-only dataset
  visible   → RGB-only dataset
  mixed     → both modalities merged into one dataset (default)
              images are prefixed ir_ / rgb_ to avoid filename collisions

Expected LLVIP directory layout:
    LLVIP/
        visible/
            train/   *.jpg
            test/    *.jpg
        infrared/
            train/   *.jpg
            test/    *.jpg
        Annotations/   *.xml   (Pascal VOC format, shared by both modalities)

Output layout (same regardless of modality):
    dataset/
        images/
            train/   *.jpg
            test/    *.jpg
        labels/
            train/   *.txt
            test/    *.txt
        llvip_<modality>.yaml

Usage:
    python llvip_dataloader.py                                  # mixed (default)
    python llvip_dataloader.py --modality infrared
    python llvip_dataloader.py --modality visible
    python llvip_dataloader.py --modality mixed --output_root ./dataset_mixed
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
    img_dir: Path,
    annotation_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    split_name: str,
    prefix: str = "",           # e.g. "ir_" or "rgb_" to avoid name collisions in mixed mode
) -> tuple[int, int]:
    """
    Process one modality split (train / test).
    Returns (n_images_ok, n_images_skipped).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        img_paths = sorted(img_dir.glob("*.png"))

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
                continue
            cx, cy, w, h = voc_to_yolo(obj)
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_name = f"{prefix}{img_path.name}"
        shutil.copy2(img_path, out_img_dir / out_name)

        lbl_path = out_lbl_dir / f"{prefix}{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines))

        ok += 1

    print(f"  [{split_name}] {ok} images converted, {skipped} skipped")
    return ok, skipped


def write_yaml(output_root: Path, modality: str, class_names: list[str]) -> Path:
    yaml_path = output_root / f"llvip_{modality}.yaml"
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


# ── per-modality entry points ─────────────────────────────────────────────────

def process_single(llvip: Path, out: Path, modality: str):
    """Convert a single modality (infrared or visible) to YOLO format."""
    annotation_dir = llvip / "Annotations"

    print(f"── Converting LLVIP [{modality}] split to YOLO format ──")
    for split in ("train", "test"):
        img_dir = llvip / modality / split
        if not img_dir.exists():
            print(f"  [INFO] {modality}/{split} not found, skipping")
            continue
        convert_split(
            img_dir        = img_dir,
            annotation_dir = annotation_dir,
            out_img_dir    = out / "images" / split,
            out_lbl_dir    = out / "labels" / split,
            split_name     = f"{modality}/{split}",
        )


def process_mixed(llvip: Path, out: Path):
    """
    Merge infrared + visible into one dataset.
    Images are prefixed ir_ / rgb_ to prevent filename collisions
    (both modalities share the same stems e.g. 010001.jpg).
    Labels are identical for both since annotations are modality-agnostic.
    """
    annotation_dir = llvip / "Annotations"

    print("── Converting LLVIP [mixed: infrared + visible] to YOLO format ──")
    for split in ("train", "test"):
        for modality, prefix in (("infrared", "ir_"), ("visible", "rgb_")):
            img_dir = llvip / modality / split
            if not img_dir.exists():
                print(f"  [INFO] {modality}/{split} not found, skipping")
                continue
            convert_split(
                img_dir        = img_dir,
                annotation_dir = annotation_dir,
                out_img_dir    = out / "images" / split,
                out_lbl_dir    = out / "labels" / split,
                split_name     = f"{modality}/{split}",
                prefix         = prefix,
            )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare LLVIP dataset for YOLO")
    parser.add_argument("--llvip_root",  type=Path, default=Path("LLVIP"),
                        help="Root of the raw LLVIP download")
    parser.add_argument("--output_root", type=Path, default=Path("dataset"),
                        help="Where to write the YOLO-ready dataset")
    parser.add_argument("--modality",
                        choices=["infrared", "visible", "mixed"],
                        default="mixed",
                        help="Which modality to prepare (default: mixed)")
    args = parser.parse_args()

    llvip = args.llvip_root
    out   = args.output_root

    annotation_dir = llvip / "Annotations"
    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotation_dir}. "
            "Make sure --llvip_root points to the LLVIP root folder."
        )

    if args.modality == "mixed":
        process_mixed(llvip, out)
    else:
        process_single(llvip, out, args.modality)

    write_yaml(out, modality=args.modality, class_names=["person"])
    print("\nDone. Dataset ready at:", out.resolve())


if __name__ == "__main__":
    main()