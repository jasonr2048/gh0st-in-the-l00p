"""
prepare_dataset_v3.py

Like v1, but with two-tier mouth detection for better crop centring:

  1. YOLOv8 face model (yolov8n-face.pt, derronqi — 5 face keypoints):
       left_eye(0)  right_eye(1)  nose(2)  left_mouth(3)  right_mouth(4)
     Mouth corners used directly when detected.
     Model is downloaded automatically on first run.

  2. Fallback — COCO pose model (yolov8n-pose.pt) + geometric estimation:
     Used per-image whenever the face model fails to detect.
     Mouth estimated as:
       r        = euclidean_distance(nose, centroid of visible eyes+ears)
       mouth_xy = (nose_x, nose_y + r * mouth_scale)
     Using the full 2D distance makes the estimate rotation-robust: on a
     side-facing face the eyes shift horizontally, keeping r stable even as
     the vertical component shrinks.

Usage:
    python tools/prepare_dataset_v3.py \
        --input_dir  /path/to/dataset/raw/ready \
        --output_dir /path/to/dataset/prepared_v3 \
        --resolution 512 \
        --scale 2.5 \
        --mouth_scale 0.8 \
        --overwrite

    # Skip face model entirely (pose + geometry only):
    python tools/prepare_dataset_v3.py ... --no_face_model

Requirements:
    pip install ultralytics Pillow numpy
"""

import argparse
import math
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DETECTION_LONG_EDGE  = 1024

# Face model: keypoint indices (derronqi yolov8-face convention)
FACE_LEFT_EYE   = 0
FACE_RIGHT_EYE  = 1
FACE_NOSE       = 2
FACE_LEFT_MOUTH = 3
FACE_RIGHT_MOUTH= 4

# Pose model: keypoint indices (COCO)
POSE_NOSE      = 0
POSE_LEFT_EYE  = 1
POSE_RIGHT_EYE = 2
POSE_LEFT_EAR  = 3
POSE_RIGHT_EAR = 4

FACE_MODEL_URL  = (
    "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
)
FACE_MODEL_NAME = "yolov8n-face.pt"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_face_model() -> YOLO | None:
    """Download yolov8n-face.pt if needed, load it, validate 5-keypoint output.
    Returns the model or None if anything fails."""
    model_path = Path(FACE_MODEL_NAME)
    if not model_path.exists():
        print(f"Downloading face model from {FACE_MODEL_URL} ...")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, model_path)
            print("Download complete.")
        except Exception as e:
            print(f"  Face model download failed: {e}")
            return None

    try:
        model = YOLO(str(model_path))
        # Smoke-test: check the model metadata claims 5 keypoints
        if hasattr(model, 'kpt_shape') and model.kpt_shape[0] < 5:
            print(f"  Face model has only {model.kpt_shape[0]} keypoints — skipping.")
            return None
        print(f"Face model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"  Face model load failed: {e}")
        return None


# ── Keypoint helpers ──────────────────────────────────────────────────────────

def downsample_for_detection(image: Image.Image):
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= DETECTION_LONG_EDGE:
        return image, 1.0, 1.0
    scale = DETECTION_LONG_EDGE / long_edge
    small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return small, w / small.width, h / small.height


def _parse_kpts(results, scale_x, scale_y, min_idx):
    """Extract keypoints from a YOLO result, scaled to original image coords.
    Returns list of (x, y) or None per index, up to min_idx+1 points needed."""
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None
    kpts = results[0].keypoints.xy[0].tolist()

    def get(idx):
        if idx < len(kpts):
            x, y = kpts[idx]
            if x > 0 and y > 0:
                return (int(x * scale_x), int(y * scale_y))
        return None

    return get


def get_keypoints_face_model(tmp_path, face_model, scale_x, scale_y):
    """Run face model; return flat list of detected keypoints or None.
    Includes mouth corners when detected — that's the whole point."""
    results = face_model(str(tmp_path), verbose=False)
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    kpts = results[0].keypoints.xy[0].tolist()

    def get(idx):
        if idx < len(kpts):
            x, y = kpts[idx]
            if x > 0 and y > 0:
                return (int(x * scale_x), int(y * scale_y))
        return None

    points = [get(i) for i in range(5)]
    detected = [p for p in points if p is not None]
    return detected if detected else None


def get_keypoints_pose_model(tmp_path, pose_model, scale_x, scale_y,
                              mouth_scale: float):
    """Run pose model; return flat list including estimated mouth point."""
    results = pose_model(str(tmp_path), verbose=False)
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    kpts = results[0].keypoints.xy[0].tolist()

    def get(idx):
        if idx < len(kpts):
            x, y = kpts[idx]
            if x > 0 and y > 0:
                return (int(x * scale_x), int(y * scale_y))
        return None

    nose      = get(POSE_NOSE)
    left_eye  = get(POSE_LEFT_EYE)
    right_eye = get(POSE_RIGHT_EYE)
    left_ear  = get(POSE_LEFT_EAR)
    right_ear = get(POSE_RIGHT_EAR)

    head_kpts = [p for p in [nose, left_eye, right_eye, left_ear, right_ear]
                 if p is not None]
    if not head_kpts:
        return None

    # Estimate mouth: nose + euclidean(nose, eye/ear centroid) × mouth_scale
    if mouth_scale > 0 and nose is not None:
        upper = [p for p in [left_eye, right_eye, left_ear, right_ear]
                 if p is not None]
        if upper:
            cx = sum(p[0] for p in upper) / len(upper)
            cy = sum(p[1] for p in upper) / len(upper)
            r  = math.hypot(nose[0] - cx, nose[1] - cy)
            head_kpts.append((nose[0], int(nose[1] + r * mouth_scale)))

    return head_kpts


def get_head_keypoints(tmp_path, face_model, pose_model,
                       scale_x, scale_y, mouth_scale):
    """Try face model first; fall back to pose + geometry per image."""
    source = "face"
    if face_model is not None:
        kpts = get_keypoints_face_model(tmp_path, face_model, scale_x, scale_y)
        if kpts:
            return kpts, source

    source = "pose"
    kpts = get_keypoints_pose_model(tmp_path, pose_model, scale_x, scale_y,
                                    mouth_scale)
    return kpts, source


# ── Crop helpers (unchanged from v1) ─────────────────────────────────────────

def compute_crop_box(head_kpts, scale: float):
    xs = [p[0] for p in head_kpts]
    ys = [p[1] for p in head_kpts]
    cx = (min(xs) + max(xs)) // 2
    cy = (min(ys) + max(ys)) // 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    half = int(span * scale / 2)
    return cx - half, cy - half, cx + half, cy + half


def sample_background_colour(image, crop_left, crop_top, crop_right, crop_bottom, hits):
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


def pad_to_square(image, crop_left, crop_top, crop_right, crop_bottom, bg_colour):
    w, h = image.size
    clamped = image.crop((
        max(0, crop_left), max(0, crop_top),
        min(w, crop_right), min(h, crop_bottom)
    ))
    target_size = max(crop_right - crop_left, crop_bottom - crop_top)
    square = Image.new("RGB", (target_size, target_size), bg_colour)
    square.paste(clamped, (max(0, -crop_left), max(0, -crop_top)))
    return square


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(path, input_dir, output_dir, rejected_dir,
                  face_model, pose_model,
                  resolution, scale, mouth_scale, overwrite) -> str:
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

        head_kpts, source = get_head_keypoints(
            tmp_path, face_model, pose_model, scale_x, scale_y, mouth_scale
        )
        tmp_path.unlink(missing_ok=True)

        if not head_kpts:
            rejected_out = rejected_dir / rel_path
            rejected_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, rejected_out)
            return f"rejected"

        crop_left, crop_top, crop_right, crop_bottom = compute_crop_box(
            head_kpts, scale
        )

        w, h = image.size
        hits = []
        if crop_left  < 0: hits.append("left")
        if crop_right > w: hits.append("right")
        if crop_top   < 0: hits.append("top")
        if crop_bottom> h: hits.append("bottom")

        if hits:
            bg = sample_background_colour(
                image, crop_left, crop_top, crop_right, crop_bottom, hits
            )
            result = pad_to_square(
                image, crop_left, crop_top, crop_right, crop_bottom, bg
            )
        else:
            result = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        result.resize((resolution, resolution), Image.LANCZOS).save(out_path)
        return source   # 'face' or 'pose' — useful for reviewing fallback rate

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return "error"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare head/face dataset (v3: face model + pose fallback)."
    )
    parser.add_argument("--input_dir",     required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--rejected_dir",  default=None)
    parser.add_argument("--resolution",    type=int,   default=512)
    parser.add_argument("--scale",         type=float, default=2.5)
    parser.add_argument("--mouth_scale",   type=float, default=0.8,
                        help="Mouth offset for pose fallback, as multiple of "
                             "nose-to-eye/ear-centroid distance (default: 0.8)")
    parser.add_argument("--no_face_model", action="store_true",
                        help="Skip face model entirely; use pose + geometry only")
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--sample",        type=int, default=None)
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

    images = [p for p in input_dir.rglob("*")
              if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        print("No supported images found.")
        sys.exit(0)

    if args.sample:
        import random
        images = random.sample(images, min(args.sample, len(images)))

    face_model = None if args.no_face_model else load_face_model()
    if face_model is None and not args.no_face_model:
        print("  → Falling back to pose model + geometry for all images.\n")

    print(f"\nLoading pose model (fallback)...")
    pose_model = YOLO("yolov8n-pose.pt")

    print(f"\nImages found    : {len(images)}")
    print(f"Resolution      : {args.resolution}x{args.resolution}")
    print(f"Scale           : {args.scale}")
    print(f"Mouth scale     : {args.mouth_scale}")
    print(f"Face model      : {'yes' if face_model else 'no (pose+geometry only)'}")
    print(f"Overwrite       : {args.overwrite}")
    print(f"Output          : {output_dir}\n")

    counts: dict[str, int] = {"face": 0, "pose": 0, "skipped": 0,
                               "rejected": 0, "error": 0}

    for i, path in enumerate(images, 1):
        rel = path.relative_to(input_dir)
        print(f"[{i}/{len(images)}] {rel} ... ", end="", flush=True)
        result = process_image(
            path, input_dir, output_dir, rejected_dir,
            face_model, pose_model,
            args.resolution, args.scale, args.mouth_scale, args.overwrite
        )
        counts[result] = counts.get(result, 0) + 1
        print(result)

    print()
    print("── Summary ──────────────────────────────────")
    print(f"  Face model    : {counts['face']}")
    print(f"  Pose fallback : {counts['pose']}")
    print(f"  Skipped       : {counts['skipped']}")
    print(f"  Rejected      : {counts['rejected']}  → {rejected_dir}")
    print(f"  Errors        : {counts['error']}")
    print(f"  Total         : {len(images)}")


if __name__ == "__main__":
    main()
