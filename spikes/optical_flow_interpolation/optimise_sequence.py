"""
optimise_sequence.py

Orders images within and across style sets to minimise face position jumps,
using YOLOv8 pose keypoints (nose + eyes centroid).

Usage:
    python tools/optimise_sequence.py \
        --prepared_dir /path/to/dataset/prepared \
        --sets rhinestones,clown,blue_face,latex_skin,horror_drip \
        --output data/sequence.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
HEAD_KPT_INDICES = [0, 1, 2]  # nose, left eye, right eye
IMAGE_SIZE = 512


# ── Keypoint extraction ───────────────────────────────────────────────────────

def extract_keypoints(image_paths: list[Path], model: YOLO) -> dict[str, tuple | None]:
    """Run YOLOv8 pose on each image; return {str(path): (cx, cy)} or None."""
    result = {}
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {path.name} ... ", end="", flush=True)
        detections = model(str(path), verbose=False)
        if detections[0].keypoints is None or len(detections[0].keypoints.xy) == 0:
            result[str(path)] = None
            print("no detection")
            continue

        kpts = detections[0].keypoints.xy[0].tolist()  # [[x, y], ...]
        visible = [
            (kpts[i][0], kpts[i][1])
            for i in HEAD_KPT_INDICES
            if i < len(kpts) and kpts[i][0] > 0 and kpts[i][1] > 0
        ]
        if not visible:
            result[str(path)] = None
            print("no visible head kpts")
            continue

        cx = sum(x for x, _ in visible) / len(visible) / IMAGE_SIZE
        cy = sum(y for _, y in visible) / len(visible) / IMAGE_SIZE
        result[str(path)] = (cx, cy)
        print(f"({cx:.3f}, {cy:.3f})")

    return result


# ── Ordering ──────────────────────────────────────────────────────────────────

def dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def greedy_order(paths: list[str], kpts: dict) -> list[str]:
    """Nearest-neighbour ordering, starting from image closest to set centroid."""
    if len(paths) <= 1:
        return list(paths)

    valid = [p for p in paths if kpts.get(p) is not None]
    if not valid:
        return list(paths)

    cx = sum(kpts[p][0] for p in valid) / len(valid)
    cy = sum(kpts[p][1] for p in valid) / len(valid)
    centroid = (cx, cy)

    start = min(valid, key=lambda p: dist(kpts[p], centroid))

    remaining = list(paths)
    ordered = [start]
    remaining.remove(start)

    while remaining:
        last = ordered[-1]
        if kpts.get(last) is None:
            ordered.append(remaining.pop(0))
            continue
        candidates = [p for p in remaining if kpts.get(p) is not None]
        if not candidates:
            ordered.extend(remaining)
            break
        nearest = min(candidates, key=lambda p: dist(kpts[last], kpts[p]))
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered


def sequence_distance(sequence: list[str], kpts: dict) -> float:
    total = 0.0
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        if kpts.get(a) is not None and kpts.get(b) is not None:
            total += dist(kpts[a], kpts[b])
    return total


def first_detected(paths: list[str], kpts: dict) -> str | None:
    return next((p for p in paths if kpts.get(p) is not None), None)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_sequence(ax, sequence: list[str], kpts: dict, img_to_set: dict,
                  colour_map: dict, title: str):
    for i, p in enumerate(sequence):
        if kpts.get(p) is None:
            continue
        cx, cy = kpts[p]
        colour = colour_map.get(img_to_set.get(p, ""), "grey")
        ax.scatter(cx, 1 - cy, color=colour, s=40, zorder=3)
        if i > 0 and kpts.get(sequence[i - 1]) is not None:
            prev_cx, prev_cy = kpts[sequence[i - 1]]
            ax.plot([prev_cx, cx], [1 - prev_cy, 1 - cy], "k-", alpha=0.2, lw=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("cx (normalised)")
    ax.set_ylabel("cy (normalised, flipped)")
    ax.set_title(title)
    ax.set_aspect("equal")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimise image sequence for optical flow interpolation."
    )
    parser.add_argument("--prepared_dir", required=True,
                        help="Path to dataset/prepared/")
    parser.add_argument("--sets", required=True,
                        help="Comma-separated list of set names in desired order")
    parser.add_argument("--output", default="data/sequence.json",
                        help="Output JSON path (default: data/sequence.json)")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    set_names = [s.strip() for s in args.sets.split(",")]
    output_path = Path(args.output)

    if not prepared_dir.exists():
        print(f"prepared_dir not found: {prepared_dir}")
        sys.exit(1)

    # ── Collect images per set ────────────────────────────────────────────────
    set_images: dict[str, list[Path]] = {}
    for name in set_names:
        folder = prepared_dir / name
        if not folder.exists():
            print(f"WARNING: set folder not found: {folder}")
            set_images[name] = []
            continue
        imgs = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        set_images[name] = imgs
        print(f"  {name}: {len(imgs)} images")

    all_images = [p for name in set_names for p in set_images[name]]
    print(f"\nTotal images: {len(all_images)}")

    # ── Extract keypoints ─────────────────────────────────────────────────────
    print("\nLoading YOLOv8 pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("\nExtracting keypoints...")
    kpts = extract_keypoints(all_images, model)

    undetected = [p for p, v in kpts.items() if v is None]
    detected = len(all_images) - len(undetected)
    print(f"\nKeypoints detected: {detected}/{len(all_images)}")

    # ── Optimise within each set ──────────────────────────────────────────────
    original_sequence = [str(p) for p in all_images]
    original_dist = sequence_distance(original_sequence, kpts)

    ordered_sets: dict[str, list[str]] = {
        name: greedy_order([str(p) for p in set_images[name]], kpts)
        for name in set_names
    }

    # ── Optimise across sets (try reversing each successive set) ──────────────
    final_ordered: dict[str, list[str]] = {set_names[0]: ordered_sets[set_names[0]]}
    prev_name = set_names[0]

    for curr_name in set_names[1:]:
        prev_last = final_ordered[prev_name][-1] if final_ordered[prev_name] else None
        fwd = ordered_sets[curr_name]
        rev = list(reversed(ordered_sets[curr_name]))

        if prev_last is None or kpts.get(prev_last) is None or not fwd:
            choice = fwd
        else:
            fwd_first = first_detected(fwd, kpts)
            rev_first = first_detected(rev, kpts)
            fwd_dist = dist(kpts[prev_last], kpts[fwd_first]) if fwd_first else float("inf")
            rev_dist = dist(kpts[prev_last], kpts[rev_first]) if rev_first else float("inf")
            choice = fwd if fwd_dist <= rev_dist else rev
            direction = "forward" if fwd_dist <= rev_dist else "reversed"
            print(f"  {prev_name} → {curr_name}: {direction} (fwd={fwd_dist:.4f}, rev={rev_dist:.4f})")

        final_ordered[curr_name] = choice
        prev_name = curr_name

    optimised_sequence = [p for name in set_names for p in final_ordered[name]]
    optimised_dist = sequence_distance(optimised_sequence, kpts)

    print(f"\nTotal keypoint distance:")
    print(f"  Before : {original_dist:.4f}")
    print(f"  After  : {optimised_dist:.4f}")
    if original_dist > 0:
        print(f"  Improvement: {(1 - optimised_dist / original_dist) * 100:.1f}%")

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "sets": [
            {
                "name": name,
                "images": final_ordered[name],
                "keypoints": [kpts.get(p) for p in final_ordered[name]],
            }
            for name in set_names
        ],
        "total_images": len(optimised_sequence),
        "undetected": undetected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSequence saved to {output_path}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    img_to_set = {
        str(p): name
        for name in set_names
        for p in set_images[name]
    }
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(set_names), 1)))
    colour_map = {name: colours[i] for i, name in enumerate(set_names)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_sequence(ax1, original_sequence, kpts, img_to_set, colour_map, "Original order")
    plot_sequence(ax2, optimised_sequence, kpts, img_to_set, colour_map, "Optimised order")

    legend = [Patch(facecolor=colour_map[n], label=n) for n in set_names]
    fig.legend(handles=legend, loc="lower center", ncol=len(set_names),
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    vis_path = output_path.with_suffix(".png")
    plt.savefig(vis_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Visualisation saved to {vis_path}")


if __name__ == "__main__":
    main()
