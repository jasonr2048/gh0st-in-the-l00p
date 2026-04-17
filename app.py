"""CLI entry point for Gh0st in the L00p exhibition system.

Modes:
    video       — VideoExhibitionRuntime: interpolation video + overlay (default)
    exhibition  — Gh0stExhibitionRuntime: still images + overlay (A's original)

Examples:
    python app.py                                         # preview video mode
    python app.py --export                                # export video mode
    python app.py --video data/my_morph.mp4 --export
    python app.py --duration 180 --export --overwrite
    python app.py --mode exhibition --export
    python app.py --export --output exports/test_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gh0st in the L00p — exhibition runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["video", "exhibition"],
        default="video",
        help="Runtime mode (default: video)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/interpolation_flow.mp4"),
        metavar="PATH",
        help="Background video file (video mode only, default: data/interpolation_flow.mp4)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Render to MP4 files instead of opening preview windows",
    )
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Override total duration in seconds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Use a fixed output directory (no timestamp suffix); overwrites previous export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="Explicit output directory (overrides default naming)",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Print FPS and jitter stats to stdout during playback",
    )

    args = parser.parse_args()

    from config import load_default_config
    config = load_default_config()

    if args.perf:
        config.exhibition.perf_debug = True
    if args.overwrite:
        config.exhibition.overwrite = True
    if args.duration is not None:
        config.exhibition.proof_duration_seconds = args.duration
    if args.output is not None:
        config.exhibition.export_output_dir = args.output

    if args.mode == "video":
        if not args.video.exists():
            print(f"Error: video file not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        config.exhibition.video_path_a = args.video
        config.exhibition.video_path_b = args.video

        from exhibition.video_runtime import VideoExhibitionRuntime
        runtime = VideoExhibitionRuntime(config)

        if args.export:
            screen_a, screen_b = runtime.export_videos()
            print(f"Screen A: {screen_a}")
            print(f"Screen B: {screen_b}")
        else:
            runtime.run()

    else:  # exhibition
        from exhibition.runtime import Gh0stExhibitionRuntime
        runtime = Gh0stExhibitionRuntime(config)

        if args.export:
            screen_a, screen_b = runtime.export_videos()
            print(f"Screen A: {screen_a}")
            print(f"Screen B: {screen_b}")
        else:
            runtime.run()


if __name__ == "__main__":
    main()
