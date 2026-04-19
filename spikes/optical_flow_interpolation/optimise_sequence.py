"""
optimise_sequence.py

Orders images within and across style sets to minimise face position jumps,
using YOLOv8 pose keypoints.

Distance metric: Euclidean in (nose_x, nose_y, left_eye_x − nose_x, right_eye_x − nose_x)
space. The eye offsets capture face orientation (rotation left/right), not just position.
Falls back to eye midpoint when nose is undetected.

Usage:
    python spikes/optical_flow_interpolation/optimise_sequence.py \
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
IMAGE_SIZE = 512

# COCO keypoint indices
NOSE      = 0
LEFT_EYE  = 1
RIGHT_EYE = 2


# ── Keypoint extraction ───────────────────────────────────────────────────────

def extract_raw_keypoints(image_paths: list[Path], model: YOLO) -> dict[str, dict | None]:
    """Run YOLOv8 pose; return {str(path): {'nose': (x,y), 'left_eye': (x,y), 'right_eye': (x,y)}}.
    Any undetected point is None. Returns None for the whole image if nothing detected."""
    result = {}
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {path.name} ... ", end="", flush=True)
        detections = model(str(path), verbose=False)
        if detections[0].keypoints is None or len(detections[0].keypoints.xy) == 0:
            result[str(path)] = None
            print("no detection")
            continue

        kpts = detections[0].keypoints.xy[0].tolist()

        def get_kpt(idx):
            if idx < len(kpts):
                x, y = kpts[idx]
                if x > 0 and y > 0:
                    return (x / IMAGE_SIZE, y / IMAGE_SIZE)
            return None

        raw = {
            'nose':      get_kpt(NOSE),
            'left_eye':  get_kpt(LEFT_EYE),
            'right_eye': get_kpt(RIGHT_EYE),
        }

        if not any(v is not None for v in raw.values()):
            result[str(path)] = None
            print("no visible head kpts")
            continue

        result[str(path)] = raw

        nose  = raw['nose']
        left  = raw['left_eye']
        right = raw['right_eye']
        print(
            f"nose={'({:.3f},{:.3f})'.format(*nose) if nose else 'x'}  "
            f"L={'({:.3f},{:.3f})'.format(*left) if left else 'x'}  "
            f"R={'({:.3f},{:.3f})'.format(*right) if right else 'x'}"
        )

    return result


def feature_vector(raw: dict | None) -> tuple | None:
    """4D feature: (nose_x, nose_y, left_eye_x − nose_x, right_eye_x − nose_x).
    The eye offsets encode face left/right rotation relative to the nose vertical."""
    if raw is None:
        return None

    nose  = raw.get('nose')
    left  = raw.get('left_eye')
    right = raw.get('right_eye')

    if nose:
        nx, ny = nose
        l_off = (left[0]  - nx) if left  else 0.0
        r_off = (right[0] - nx) if right else 0.0
        return (nx, ny, l_off, r_off)

    # Nose missing — fall back to eye midpoint, zero offsets
    if left and right:
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2, 0.0, 0.0)
    if left:
        return (left[0], left[1], 0.0, 0.0)
    if right:
        return (right[0], right[1], 0.0, 0.0)
    return None


def build_feature_map(raw_kpts: dict) -> dict[str, tuple | None]:
    return {path: feature_vector(raw) for path, raw in raw_kpts.items()}


# ── Distance & ordering ───────────────────────────────────────────────────────

def dist(a: tuple, b: tuple) -> float:
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5


def greedy_order(paths: list[str], fvecs: dict) -> list[str]:
    """Nearest-neighbour ordering, starting from image closest to set centroid.
    Used as the initial solution for two_opt."""
    if len(paths) <= 1:
        return list(paths)

    valid = [p for p in paths if fvecs.get(p) is not None]
    if not valid:
        return list(paths)

    n_dims = len(fvecs[valid[0]])
    centroid = tuple(
        sum(fvecs[p][d] for p in valid) / len(valid)
        for d in range(n_dims)
    )
    start = min(valid, key=lambda p: dist(fvecs[p], centroid))

    remaining = list(paths)
    ordered = [start]
    remaining.remove(start)

    while remaining:
        last = ordered[-1]
        if fvecs.get(last) is None:
            ordered.append(remaining.pop(0))
            continue
        candidates = [p for p in remaining if fvecs.get(p) is not None]
        if not candidates:
            ordered.extend(remaining)
            break
        nearest = min(candidates, key=lambda p: dist(fvecs[last], fvecs[p]))
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered


def two_opt(paths: list[str], fvecs: dict) -> list[str]:
    """Improve a path ordering using 2-opt.

    Repeatedly tries reversing every sub-segment of the sequence; keeps the
    reversal whenever it reduces total path length. Converges to a local
    optimum where no single reversal helps further.

    Seeds from greedy_order. O(N²) per pass; fast for the small N typical
    of each style set.
    """
    seq = greedy_order(paths, fvecs)
    n = len(seq)
    if n <= 2:
        return seq

    def edge(p, q) -> float:
        fp, fq = fvecs.get(p), fvecs.get(q)
        return dist(fp, fq) if fp is not None and fq is not None else 0.0

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Edges removed: (i → i+1)  and, if not at end, (j → j+1)
                # Edges added:   (i → j)    and, if not at end, (i+1 → j+1)
                old = edge(seq[i], seq[i + 1])
                new = edge(seq[i], seq[j])
                if j + 1 < n:
                    old += edge(seq[j], seq[j + 1])
                    new += edge(seq[i + 1], seq[j + 1])
                if new < old - 1e-10:
                    seq[i + 1:j + 1] = seq[i + 1:j + 1][::-1]
                    improved = True

    return seq


def sequence_distance(sequence: list[str], fvecs: dict) -> float:
    total = 0.0
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        if fvecs.get(a) is not None and fvecs.get(b) is not None:
            total += dist(fvecs[a], fvecs[b])
    return total


def first_detected(paths: list[str], fvecs: dict) -> str | None:
    return next((p for p in paths if fvecs.get(p) is not None), None)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_sequence(ax, sequence, fvecs, img_to_set, colour_map, title):
    """Plot spatial position (nose_x, nose_y) with lines showing sequence order."""
    for i, p in enumerate(sequence):
        fv = fvecs.get(p)
        if fv is None:
            continue
        x, y = fv[0], 1 - fv[1]
        colour = colour_map.get(img_to_set.get(p, ""), "grey")
        ax.scatter(x, y, color=colour, s=40, zorder=3)
        if i > 0:
            prev = fvecs.get(sequence[i - 1])
            if prev is not None:
                ax.plot([prev[0], x], [1 - prev[1], y], 'k-', alpha=0.2, lw=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nose_x (normalised)")
    ax.set_ylabel("nose_y (normalised, flipped)")
    ax.set_title(title)
    ax.set_aspect('equal')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimise image sequence for optical flow interpolation."
    )
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--sets",         required=True,
                        help="Comma-separated set names in desired order")
    parser.add_argument("--output", default="data/sequence.json")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    set_names    = [s.strip() for s in args.sets.split(",")]
    output_path  = Path(args.output)

    if not prepared_dir.exists():
        print(f"prepared_dir not found: {prepared_dir}")
        sys.exit(1)

    # ── Collect images ────────────────────────────────────────────────────────
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

    # ── Extract keypoints & build feature vectors ─────────────────────────────
    print("\nLoading YOLOv8 pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("\nExtracting keypoints...")
    raw_kpts = extract_raw_keypoints(all_images, model)
    fvecs    = build_feature_map(raw_kpts)

    undetected = [p for p, v in raw_kpts.items() if v is None]
    print(f"\nKeypoints detected: {len(all_images) - len(undetected)}/{len(all_images)}")
    for p in undetected:
        print(f"  undetected: {Path(p).name}")

    # ── Optimise within each set ──────────────────────────────────────────────
    original_sequence = [str(p) for p in all_images]
    original_dist     = sequence_distance(original_sequence, fvecs)

    ordered_sets = {
        name: two_opt([str(p) for p in set_images[name]], fvecs)
        for name in set_names
    }

    # ── Optimise across sets ──────────────────────────────────────────────────
    final_ordered: dict[str, list[str]] = {set_names[0]: ordered_sets[set_names[0]]}
    prev_name = set_names[0]

    for curr_name in set_names[1:]:
        prev_last = final_ordered[prev_name][-1] if final_ordered[prev_name] else None
        fwd = ordered_sets[curr_name]
        rev = list(reversed(ordered_sets[curr_name]))

        if prev_last is None or fvecs.get(prev_last) is None or not fwd:
            choice = fwd
        else:
            fwd_first = first_detected(fwd, fvecs)
            rev_first = first_detected(rev, fvecs)
            fwd_d = dist(fvecs[prev_last], fvecs[fwd_first]) if fwd_first else float("inf")
            rev_d = dist(fvecs[prev_last], fvecs[rev_first]) if rev_first else float("inf")
            choice    = fwd if fwd_d <= rev_d else rev
            direction = "forward" if fwd_d <= rev_d else "reversed"
            print(f"  {prev_name} → {curr_name}: {direction} (fwd={fwd_d:.4f}, rev={rev_d:.4f})")

        final_ordered[curr_name] = choice
        prev_name = curr_name

    optimised_sequence = [p for name in set_names for p in final_ordered[name]]
    optimised_dist     = sequence_distance(optimised_sequence, fvecs)

    print(f"\nTotal sequence distance:")
    print(f"  Before : {original_dist:.4f}")
    print(f"  After  : {optimised_dist:.4f}")
    if original_dist > 0:
        print(f"  Improvement: {(1 - optimised_dist / original_dist) * 100:.1f}%")

    # ── Write output ──────────────────────────────────────────────────────────
    img_to_set = {str(p): name for name in set_names for p in set_images[name]}

    output = {
        "sets": [
            {
                "name": name,
                "images": final_ordered[name],
                "keypoints": [
                    list(fvecs[p]) if fvecs.get(p) is not None else None
                    for p in final_ordered[name]
                ],
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
    colours    = plt.cm.tab10(np.linspace(0, 1, max(len(set_names), 1)))
    colour_map = {name: colours[i] for i, name in enumerate(set_names)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_sequence(ax1, original_sequence, fvecs, img_to_set, colour_map, "Original order")
    plot_sequence(ax2, optimised_sequence, fvecs, img_to_set, colour_map, "Optimised order")

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
