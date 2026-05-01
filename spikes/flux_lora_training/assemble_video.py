#!/usr/bin/env python3
"""
Assemble animated clips into the final sequence video.

Takes the Runway clips from sequence/clips/, concatenates them in
style-transition order, and optionally runs through app.py to add the
scan + text overlay.

Sequence order:
  pure rhinestones → [transition A] → pure bead_cage
  → [transition B] → pure crybabyglitch
  → [transition C] → pure red_blackliner

Output:
  sequence/assembled/raw_sequence.mp4         — concatenated clips only
  sequence/assembled/screen_A.mp4             — with scan + text overlay
  sequence/assembled/screen_B.mp4             — same (mirrored for screen B)

Usage:
  # Assemble only (no overlay):
  python spikes/flux_lora_training/assemble_video.py --no-overlay

  # Full pipeline (requires venv active):
  python spikes/flux_lora_training/assemble_video.py

  # Custom text payload for the overlay:
  python spikes/flux_lora_training/assemble_video.py --payload data/my_text.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPIKE_DIR    = Path(__file__).resolve().parent
REPO_ROOT    = SPIKE_DIR.parent.parent
SEQUENCE_DIR = SPIKE_DIR / "output" / "gh0st_flux_lora_v2" / "sequence"
CLIPS_DIR    = SEQUENCE_DIR / "clips"
OUT_DIR      = SEQUENCE_DIR / "assembled"

ALL_ALPHAS = [round(v / 100, 2) for v in range(5, 100, 5)]


def clip_path(name: str) -> Path:
    return CLIPS_DIR / name


def pure_clip(style: str) -> Path:
    return CLIPS_DIR / f"pure_{style}.mp4"


def transition_clips(set_a: str, set_b: str) -> list[Path]:
    """Return clips in ascending alpha order (style_b → style_a)."""
    pair_dir = CLIPS_DIR / f"{set_a}_x_{set_b}"
    clips = []
    for alpha in ALL_ALPHAS:
        pct_a = int(round(alpha * 100))
        pct_b = 100 - pct_a
        stem = f"{pct_a:03d}{set_a}__{pct_b:03d}{set_b}"
        p = pair_dir / f"{stem}.mp4"
        if p.exists():
            clips.append(p)
        else:
            print(f"  WARNING: missing clip {p.relative_to(CLIPS_DIR)}")
    return clips


def build_clip_list() -> list[Path]:
    """Full ordered sequence of clips."""
    return [
        pure_clip("rhinestones"),
        *transition_clips("bead_cage", "rhinestones"),
        pure_clip("bead_cage"),
        *transition_clips("crybabyglitch", "bead_cage"),
        pure_clip("crybabyglitch"),
        *transition_clips("red_blackliner", "crybabyglitch"),
        pure_clip("red_blackliner"),
    ]


def concatenate(clips: list[Path], out_path: Path) -> None:
    """Concatenate MP4 clips using ffmpeg concat demuxer."""
    missing = [c for c in clips if not c.exists()]
    if missing:
        print(f"ERROR: {len(missing)} clips missing:")
        for m in missing[:10]:
            print(f"  {m}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")
        concat_file = Path(f.name)

    print(f"Concatenating {len(clips)} clips → {out_path.name} …")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print("ffmpeg error:")
        print(result.stderr[-2000:])
        sys.exit(1)

    duration = _probe_duration(out_path)
    print(f"  → {out_path.name}  ({duration:.1f}s)")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def apply_overlay(raw_video: Path, payload: Path | None) -> tuple[Path, Path]:
    """Run raw_video through app.py and return (screen_A, screen_B)."""
    sys.path.insert(0, str(REPO_ROOT))
    from config import load_default_config
    from exhibition.video_runtime import VideoExhibitionRuntime

    config = load_default_config()
    config.exhibition.video_path_a = raw_video
    config.exhibition.video_path_b = raw_video
    config.exhibition.overwrite = True
    config.exhibition.export_output_dir = OUT_DIR

    if payload is not None:
        config.exhibition.text_payload = payload

    # Match video duration exactly
    duration = _probe_duration(raw_video)
    if duration > 0:
        config.exhibition.proof_duration_seconds = duration

    runtime = VideoExhibitionRuntime(config)
    screen_a, screen_b = runtime.export_videos()
    return screen_a, screen_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble sequence video")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip the app.py overlay step")
    parser.add_argument("--payload", type=Path, default=None,
                        help="Custom text payload JSON for overlay")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show clip list without assembling")
    args = parser.parse_args()

    clips = build_clip_list()

    if args.dry_run:
        print(f"Clip sequence ({len(clips)} clips):")
        for c in clips:
            status = "✓" if c.exists() else "✗ MISSING"
            print(f"  {status}  {c.name}")
        total_s = sum(_probe_duration(c) for c in clips if c.exists())
        print(f"\nEstimated duration: {total_s:.0f}s ({total_s / 60:.1f} min)")
        return

    ready = [c for c in clips if c.exists()]
    missing = [c for c in clips if not c.exists()]
    print(f"Clips ready  : {len(ready)}/{len(clips)}")
    if missing:
        print(f"Missing      : {len(missing)} — run animate_sequence.py first")
        if len(missing) > 5:
            print("  (use --dry-run to see full list)")
        else:
            for m in missing:
                print(f"  {m.name}")
        if len(missing) == len(clips):
            sys.exit(1)
        print("Proceeding with available clips…\n")
        clips = ready  # assemble partial

    raw_path = OUT_DIR / "raw_sequence.mp4"
    concatenate(clips, raw_path)

    if args.no_overlay:
        print(f"\nRaw video: {raw_path}")
        return

    print("\nApplying scan + text overlay via app.py…")
    screen_a, screen_b = apply_overlay(raw_path, args.payload)
    print(f"\nscreen_A : {screen_a}")
    print(f"screen_B : {screen_b}")
    print("\nDONE — ready for A to review.")


if __name__ == "__main__":
    main()
