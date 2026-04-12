"""
prepare_dataset_v2.py

Identical to prepare_dataset.py with one improvement: an estimated mouth
position is added to the keypoint set before computing the crop box.

The COCO pose model (yolov8n-pose.pt) has no mouth keypoints. With only
nose + eyes + ears the crop centre sits between eye and nose level, cutting
off the lower face. The mouth is estimated as:

    mouth_y = nose_y + (nose_y − eye_midpoint_y) × mouth_scale

i.e. roughly as far below the nose as the eyes are above it.  The default
mouth_scale of 0.8 places the estimated point near the upper lip; increase
toward 1.5 to include more chin. Set --mouth_scale 0 to disable (identical
to v1 behaviour).

Usage:
    python tools/prepare_dataset_v2.py \
        --input_dir  /path/to/dataset/raw/ready \
        --output_dir /path/to/dataset/prepared \
        --rejected_dir /path/to/dataset/rejected \
        --resolution 512 \
        --scale 2.5 \
        --mouth_scale 0.8 \
        --overwrite

Requirements:
    pip install ultralytics Pillow numpy
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DETECTION_LONG_EDGE  = 1024


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


def get_head_keypoints(path: Path, pose_model, scale_x: float, scale_y: float,
                       mouth_scale: float = 0.8):
    """
    Run pose estimation and return head keypoints (nose, eyes, ears, estimated mouth)
    scaled back to original image coordinates.

    mouth_scale controls how far below the nose the estimated mouth point is placed,
    as a multiple of the eye-to-nose vertical distance.  Set to 0 to disable.

    Returns list of (x, y) or None if nothing detected.
    """
    results = pose_model(str(path), verbose=False)
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    kpts = results[0].keypoints.xy[0].tolist()

    def get(idx):
        if idx < len(kpts):
            x, y = kpts[idx]
            if x > 0 and y > 0:
                return (int(x * scale_x), int(y * scale_y))
        return None

    nose      = get(0)
    left_eye  = get(1)
    right_eye = get(2)
    left_ear  = get(3)
    right_ear = get(4)

    head_kpts = [p for p in [nose, left_eye, right_eye, left_ear, right_ear]
                 if p is not None]

    # Estimate mouth position from nose + eyes
    if mouth_scale > 0 and nose is not None:
        eye_pts = [p for p in [left_eye, right_eye] if p is not None]
        if eye_pts:
            eye_mid_y = sum(p[1] for p in eye_pts) / len(eye_pts)
            mouth_y   = int(nose[1] + (nose[1] - eye_mid_y) * mouth_scale)
            mouth_x   = nose[0]  # mouth sits roughly under the nose
            head_kpts.append((mouth_x, mouth_y))

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
                  scale: float, mouth_scale: float, overwrite: bool) -> str:
    """Process a single image. Returns 'ok', 'skipped', 'rejected', or 'error'."""
    rel_path = path.relative_to(input_dir)
    out_path = output_dir / rel_path.parent / (path.stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        return "skipped"

    try:
        image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        small, scale_x, scale_y = downsample_for_detection(image)

        tmp_path = Path("/tmp") / path.name
        small.save(tmp_path)

        head_kpts = get_head_keypoints(tmp_path, pose_model, scale_x, scale_y,
                                       mouth_scale=mouth_scale)
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
        description="Prepare head/face dataset (v2: mouth-aware cropping)."
    )
    parser.add_argument("--input_dir",    required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--rejected_dir", default=None)
    parser.add_argument("--resolution",   type=int,   default=512)
    parser.add_argument("--scale",        type=float, default=2.5,
                        help="Crop size as multiple of keypoint span (default: 2.5)")
    parser.add_argument("--mouth_scale",  type=float, default=0.8,
                        help="Estimated mouth offset as multiple of eye-to-nose "
                             "vertical distance (default: 0.8, set 0 to disable)")
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument("--sample",       type=int, default=None)
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    output_dir   = Path(args.output_dir)
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
    print(f"Mouth scale     : {args.mouth_scale}"
          + (" (disabled)" if args.mouth_scale == 0 else ""))
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
            pose_model, args.resolution, args.scale, args.mouth_scale,
            args.overwrite
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
