"""
crop_exhibition_sources.py

Reads selection_v1.txt, runs every referenced source image through the
prepare_dataset.py head-crop pipeline (YOLOv8, scale=2.5, resolution=1024),
and saves results to exhibition_source/normalized/{category}/{stem}.png.

Images where YOLOv8 finds no head are centre-cropped as a fallback
(with a warning) so the generation can still proceed.

Usage:
    python spikes/flux_lora_training/crop_exhibition_sources.py
    python spikes/flux_lora_training/crop_exhibition_sources.py --overwrite
    python spikes/flux_lora_training/crop_exhibition_sources.py --scale 2.0
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SPIKE_DIR     = Path(__file__).parent
REPO_ROOT     = SPIKE_DIR.parent.parent
SELECTION     = SPIKE_DIR / "exhibition_source" / "selection_v1.txt"
DRIVE_FACES   = Path.home() / "Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/My Drive/Gh0st in the Loop/faces_exhibition"
OUT_BASE      = SPIKE_DIR / "exhibition_source" / "normalized"
REJECTED_DIR  = SPIKE_DIR / "exhibition_source" / "rejected"

DETECTION_LONG_EDGE = 1024


# ── Helpers (mirrored from tools/prepare_dataset.py) ─────────────────────────

def stem(name: str) -> str:
    s = Path(name).stem
    return re.sub(r"[\s()]+", "_", s).strip("_")


def downsample_for_detection(image: Image.Image):
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= DETECTION_LONG_EDGE:
        return image, 1.0, 1.0
    scale = DETECTION_LONG_EDGE / long_edge
    small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return small, w / small.width, h / small.height


def get_head_keypoints(path: Path, pose_model, scale_x: float, scale_y: float):
    results = pose_model(str(path), verbose=False)
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None
    kpts = results[0].keypoints.xy[0].tolist()
    head_kpts = [
        (int(x * scale_x), int(y * scale_y))
        for x, y in kpts[:5]
        if x > 0 and y > 0
    ]
    return head_kpts if head_kpts else None


def compute_crop_box(head_kpts, scale: float):
    xs = [p[0] for p in head_kpts]
    ys = [p[1] for p in head_kpts]
    cx = (min(xs) + max(xs)) // 2
    cy = (min(ys) + max(ys)) // 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    half = int(span * scale / 2)
    return cx - half, cy - half, cx + half, cy + half


def sample_background_colour(image: Image.Image, crop_left, crop_top,
                              crop_right, crop_bottom, hits):
    w, h = image.size
    arr = np.array(image)
    samples = []
    if "top" not in hits:
        ty = max(0, crop_top)
        samples.append(arr[ty, max(0, min(w - 1, crop_left))].tolist())
        samples.append(arr[ty, max(0, min(w - 1, crop_right))].tolist())
    else:
        by = min(h - 1, crop_bottom)
        samples.append(arr[by, max(0, min(w - 1, crop_left))].tolist())
        samples.append(arr[by, max(0, min(w - 1, crop_right))].tolist())
    return tuple(int(x) for x in np.mean(samples, axis=0)[:3])


def pad_to_square(image: Image.Image, crop_left, crop_top,
                  crop_right, crop_bottom, bg_colour):
    w, h = image.size
    clamped = image.crop((
        max(0, crop_left), max(0, crop_top),
        min(w, crop_right), min(h, crop_bottom),
    ))
    target_size = max(crop_right - crop_left, crop_bottom - crop_top)
    square = Image.new("RGB", (target_size, target_size), bg_colour)
    paste_x = max(0, -crop_left)
    paste_y = max(0, -crop_top)
    square.paste(clamped, (paste_x, paste_y))
    return square


def centre_crop_fallback(image: Image.Image, resolution: int) -> Image.Image:
    """Centre-square crop for images where YOLOv8 finds no head."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (resolution, resolution), Image.LANCZOS
    )


def parse_selection(path: Path) -> list:
    """Returns list of (category, filename) in selection order."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("/", 1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))
    return entries


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crop exhibition source images with YOLOv8.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess images that already exist")
    parser.add_argument("--scale", type=float, default=2.5,
                        help="Crop size as multiple of keypoint span (default: 2.5)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Output resolution (default: 1024)")
    args = parser.parse_args()

    if not DRIVE_FACES.exists():
        print(f"ERROR: Drive faces folder not found:\n  {DRIVE_FACES}")
        sys.exit(1)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    entries = parse_selection(SELECTION)
    print(f"Selection  : {len(entries)} images")
    print(f"Source     : {DRIVE_FACES}")
    print(f"Output     : {OUT_BASE}")
    print(f"Rejected   : {REJECTED_DIR}")
    print(f"Scale      : {args.scale}")
    print(f"Resolution : {args.resolution}x{args.resolution}")
    print()

    print("Loading YOLOv8 pose model...")
    pose_model = YOLO("yolov8n-pose.pt")
    print("Ready.\n")

    counts = {"ok": 0, "skipped": 0, "rejected_fallback": 0, "missing": 0, "error": 0}
    rejected_list = []

    for i, (cat, fname) in enumerate(entries, 1):
        src = DRIVE_FACES / cat / fname
        out_stem = stem(fname)
        out_path = OUT_BASE / cat / f"{out_stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i:2d}/{len(entries)}] {cat}/{fname} → {out_stem}.png ... ", end="", flush=True)

        if not args.overwrite and out_path.exists():
            print("skipped")
            counts["skipped"] += 1
            continue

        if not src.exists():
            print("MISSING")
            counts["missing"] += 1
            continue

        try:
            image = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
            small, scale_x, scale_y = downsample_for_detection(image)

            tmp_path = Path("/tmp") / src.name
            small.save(tmp_path)
            head_kpts = get_head_keypoints(tmp_path, pose_model, scale_x, scale_y)
            tmp_path.unlink(missing_ok=True)

            if not head_kpts:
                # Fallback: centre crop
                result = centre_crop_fallback(image, args.resolution)
                result.save(out_path)
                rej_copy = REJECTED_DIR / cat / fname
                rej_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, rej_copy)
                print("FALLBACK (centre crop — no head detected)")
                counts["rejected_fallback"] += 1
                rejected_list.append(f"{cat}/{fname}")
                continue

            crop_left, crop_top, crop_right, crop_bottom = compute_crop_box(
                head_kpts, args.scale
            )

            w, h = image.size
            hits = []
            if crop_left < 0:   hits.append("left")
            if crop_right > w:  hits.append("right")
            if crop_top < 0:    hits.append("top")
            if crop_bottom > h: hits.append("bottom")

            if hits:
                bg = sample_background_colour(
                    image, crop_left, crop_top, crop_right, crop_bottom, hits
                )
                result = pad_to_square(
                    image, crop_left, crop_top, crop_right, crop_bottom, bg
                )
            else:
                result = image.crop((crop_left, crop_top, crop_right, crop_bottom))

            result = result.resize((args.resolution, args.resolution), Image.LANCZOS)
            result.save(out_path)
            print("ok")
            counts["ok"] += 1

        except Exception as e:
            print(f"ERROR: {e}")
            counts["error"] += 1

    print()
    print("── Summary ──────────────────────────────────")
    print(f"  Processed     : {counts['ok']}")
    print(f"  Skipped       : {counts['skipped']}")
    print(f"  Centre-cropped: {counts['rejected_fallback']}  (no head detected, review these)")
    print(f"  Missing src   : {counts['missing']}")
    print(f"  Errors        : {counts['error']}")
    total_done = counts["ok"] + counts["rejected_fallback"] + counts["skipped"]
    print(f"  Total output  : {total_done} / {len(entries)}")

    if rejected_list:
        print()
        print("Images where YOLOv8 found no head (centre-crop fallback used):")
        for r in rejected_list:
            print(f"  {r}")
        print()
        print("Review these — if the face is heavily obscured, the FLUX output may")
        print("be inconsistent. Swap for a clearer image from the same style set.")

    print()
    print(f"Output: {OUT_BASE}")


if __name__ == "__main__":
    main()
