from __future__ import annotations

import argparse
from pathlib import Path

from config import load_default_config
from exhibition.runtime import Gh0stExhibitionRuntime
from sim.loop import Gh0stSimulation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gh0st in the L00p runtime")
    parser.add_argument(
        "--mode",
        choices=("live", "exhibition"),
        default="live",
        help="Select the runtime mode. Defaults to the existing live/archive simulation.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override the exhibition proof-of-concept duration in seconds.",
    )
    parser.add_argument(
        "--perf-debug",
        action="store_true",
        help="Print exhibition FPS and frame cadence diagnostics to the terminal.",
    )
    parser.add_argument(
        "--lightweight-render",
        action="store_true",
        help="Use a lower-cost exhibition render path for smoother proof-of-concept playback.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Open exhibition mode as a fullscreen presentation surface.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Render deterministic exhibition videos for both screens instead of realtime playback.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the exhibition export output directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_default_config()

    if args.mode == "exhibition":
        if args.duration_seconds is not None:
            config.exhibition.proof_duration_seconds = args.duration_seconds
        if args.perf_debug:
            config.exhibition.perf_debug = True
        if args.lightweight_render:
            config.exhibition.lightweight_render = True
            config.exhibition.target_fps = 24
            config.exhibition.logic_hz = 12.0
            config.exhibition.frame_delay_ms = int(1000 / config.exhibition.target_fps)
        if args.fullscreen:
            config.exhibition.fullscreen = True
        if args.output_dir is not None:
            config.exhibition.export_output_dir = Path(args.output_dir)
        runtime = Gh0stExhibitionRuntime(config)
        if args.export:
            try:
                screen_a_path, screen_b_path = runtime.export_videos()
                print(f"Exported {screen_a_path}")
                print(f"Exported {screen_b_path}")
            except RuntimeError as exc:
                print(f"[exhibition] export error: {exc}")
                raise SystemExit(1) from exc
        else:
            runtime.run()
        return

    simulation = Gh0stSimulation(config)
    simulation.run()


if __name__ == "__main__":
    main()
