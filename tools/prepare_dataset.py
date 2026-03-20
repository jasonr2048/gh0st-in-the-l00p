"""
prepare_dataset.py

Prepares a dataset of images for StyleGAN2-ADA training by:
  1. Detecting head position using YOLOv8 pose estimation (keypoints)
  2. Cropping a square region centred on the head
  3. Padding with sampled background colour if crop exceeds image bounds
  4. Resizing to target resolution
  5. Mirroring input subfolder structure in output
  6. Moving unprocessable images to dataset/rejected/

Originals are never modified.

Usage:
    python tools/prepare_dataset.py \
        --input_dir  /path/to/dataset/raw \
        --output_dir /path/to/dataset/prepared \
        --rejected_dir /path/to/dataset/rejected \
        --resolution 512 \
        --scale 3.0 \
        --overwrite

Requirements:
    pip install ultralytics Pillow numpy
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO

# Allow large images without error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DETECTION_LONG_EDGE  = 1024  # downsample to this for detection only


# ── Helpers ───────────────────────────────────────────────────────────────────

def downsample_for_detection(image: Image.Image):
    """Return downsampled copy and scale factors (scale_x, scale_y)."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= DETECTION_LONG_EDGE:
        return image, 1.0, 1.0
    scale = DETECTION_LONG_EDGE / long_edge
    small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return small, w / small.width, h / small.height


def get_head_keypoints(path: Path, pose_model, scale_x: float, scale_y: float):
    """
    Run pose estimation and return head keypoints (nose, eyes, ears)
    scaled back to original image coordinates.
    Returns list of (x, y) or None if nothing detected.
    """
    results = pose_model(str(path), verbose=False)
    if results[0].keypoints is None:
        return None

    kpts = results[0].keypoints.xy[0].tolist()
    head_kpts = [
        (int(x * scale_x), int(y * scale_y))
        for x, y in kpts[:5]
        if x > 0 and y > 0
    ]
    return head_kpts if head_kpts else None


def compute_crop_box(head_kpts, scale: float):
    """Compute square crop box from head keypoints and scale factor."""
    xs = [p[0] for p in head_kpts]
    ys = [p[1] for p in head_kpts]
    cx = (min(xs) + max(xs)) // 2
    cy = (min(ys) + max(ys)) // 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    half = int(span * scale / 2)
    return cx - half, cy - half, cx + half, cy + half


def sample_background_colour(image: Image.Image, crop_left, crop_top,
                              crop_right, crop_bottom, hits):
    """
    Sample background colour from top corners of crop box.
    Falls back to bottom corners if top edge was hit.
    """
    w, h = image.size
    arr = np.array(image)
    samples = []

    if "top" not in hits:
        ty = max(0, crop_top)
        samples.append(arr[ty, max(0, min(w-1, crop_left))].tolist())
        samples.append(arr[ty, max(0, min(w-1, crop_right))].tolist())
    else:
        by = min(h-1, crop_bottom)
        samples.append(arr[by, max(0, min(w-1, crop_left))].tolist())
        samples.append(arr[by, max(0, min(w-1, crop_right))].tolist())

    return tuple(int(x) for x in np.mean(samples, axis=0)[:3])


def pad_to_square(image: Image.Image, crop_left, crop_top,
                  crop_right, crop_bottom, bg_colour):
    """
    Crop clamped region and pad with bg_colour to produce a square,
    preserving correct spatial position of content.
    """
    w, h = image.size
    clamped = image.crop((
        max(0, crop_left), max(0, crop_top),
        min(w, crop_right), min(h, crop_bottom)
    ))
    target_size = max(crop_right - crop_left, crop_bottom - crop_top)
    square = Image.new("RGB", (target_size, target_size), bg_colour)
    paste_x = max(0, -crop_left)
    paste_y = max(0, -crop_top)
    square.paste(clamped, (paste_x, paste_y))
    return square


def process_image(path: Path, input_dir: Path, output_dir: Path,
                  rejected_dir: Path, pose_model, resolution: int,
                  scale: float, overwrite: bool) -> str:
    """
    Process a single image. Returns 'ok', 'skipped', 'rejected', or 'error'.
    """
    # Mirror subfolder structure
    rel_path = path.relative_to(input_dir)
    out_path = output_dir / rel_path.parent / (path.stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        return "skipped"

    try:
        image = Image.open(path).convert("RGB")
        small, scale_x, scale_y = downsample_for_detection(image)

        # Save small version temporarily for YOLO (needs a file path)
        tmp_path = Path("/tmp") / path.name
        small.save(tmp_path)

        head_kpts = get_head_keypoints(tmp_path, pose_model, scale_x, scale_y)
        tmp_path.unlink(missing_ok=True)

        if not head_kpts:
            rejected_out = rejected_dir / rel_path
            rejected_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, rejected_out)
            return "rejected"

        crop_left, crop_top, crop_right, crop_bottom = compute_crop_box(
            head_kpts, scale
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

        result = result.resize((resolution, resolution), Image.LANCZOS)
        result.save(out_path)
        return "ok"

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return "error"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare head/face dataset for StyleGAN training."
    )
    parser.add_argument("--input_dir",    required=True,
                        help="Root folder of raw images (read-only)")
    parser.add_argument("--output_dir",   required=True,
                        help="Root folder for processed images")
    parser.add_argument("--rejected_dir", default=None,
                        help="Folder for unprocessable images (default: output_dir/../rejected)")
    parser.add_argument("--resolution",   type=int,   default=512,
                        help="Output resolution in pixels (default: 512)")
    parser.add_argument("--scale",        type=float, default=3.0,
                        help="Crop size as multiple of keypoint span (default: 3.0)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Reprocess images that already exist in output_dir")
    parser.add_argument("--sample",       type=int,   default=None,
                        help="Process only a random sample of N images")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rejected_dir = Path(args.rejected_dir) if args.rejected_dir \
                   else input_dir.parent / "rejected"

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        print("No supported images found.")
        sys.exit(0)

    if args.sample:
        import random
        images = random.sample(images, min(args.sample, len(images)))

    print(f"Images found    : {len(images)}")
    print(f"Resolution      : {args.resolution}x{args.resolution}")
    print(f"Scale           : {args.scale}")
    print(f"Overwrite       : {args.overwrite}")
    print(f"Output          : {output_dir}")
    print(f"Rejected        : {rejected_dir}")
    print()

    print("Loading YOLOv8 pose model...")
    pose_model = YOLO("yolov8n-pose.pt")
    print("Ready.\n")

    counts = {"ok": 0, "skipped": 0, "rejected": 0, "error": 0}

    for i, path in enumerate(images, 1):
        rel = path.relative_to(input_dir)
        print(f"[{i}/{len(images)}] {rel} ... ", end="", flush=True)
        result = process_image(
            path, input_dir, output_dir, rejected_dir,
            pose_model, args.resolution, args.scale, args.overwrite
        )
        counts[result] += 1
        print(result)

    print()
    print("── Summary ──────────────────────────────────")
    print(f"  Processed : {counts['ok']}")
    print(f"  Skipped   : {counts['skipped']}  (already exist, use --overwrite to reprocess)")
    print(f"  Rejected  : {counts['rejected']}  → {rejected_dir}")
    print(f"  Errors    : {counts['error']}")
    print(f"  Total     : {len(images)}")


if __name__ == "__main__":
    main()
