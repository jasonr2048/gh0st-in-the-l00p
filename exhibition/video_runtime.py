"""VideoExhibitionRuntime — uses video files as the background layer.

The scan effect, text overlay, tinting, and corruption progression all come
from A's exhibition system unchanged.  The only difference is that background
frames come from cv2.VideoCapture instead of still image files.

Sidecar JSON:
    If a file named <video_stem>.json exists next to the video, it is loaded
    automatically.  Fields used:
        duration_seconds  — overrides config.exhibition.proof_duration_seconds
        fps               — overrides config.exhibition.export_fps
        experiment_id     — used to name the output directory
        source_sets       — informational, passed through

Output directory:
    exports/exhibition/<experiment_id>/            (when overwrite=True)
    exports/exhibition/<experiment_id>_<ts>/       (default — unique per run)
    Overridden entirely if config.exhibition.export_output_dir is set explicitly
    via --output in the CLI.

Usage (export):
    from config import load_default_config
    from exhibition.video_runtime import VideoExhibitionRuntime
    from pathlib import Path

    config = load_default_config()
    config.exhibition.video_path_a = Path("data/interpolation_flow.mp4")
    config.exhibition.video_path_b = Path("data/interpolation_flow.mp4")

    runtime = VideoExhibitionRuntime(config)
    screen_a, screen_b = runtime.export_videos()

Usage (live preview):
    runtime.run()
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from random import Random
from time import monotonic, sleep

import cv2
import numpy as np

from config import ROOT, SimulationConfig
from exhibition.models import VideoFrameState
from exhibition.payload_loader import load_text_payload
from exhibition.text_state_machine import CorruptionTextStateMachine
from exhibition.typewriter import ExhibitionTypewriter
from render.exhibition_renderer import ExhibitionRenderer


# Maps phase progress → (state_name, category_a, category_b).
# state_name keys match STATE_PRIMARY_CATEGORY; category_b is the first entry
# from PAIRING_MATRIX for each category_a.
_STATE_SCHEDULE: tuple[tuple[float, str, str, str], ...] = (
    (0.0,       "state_1", "stable",    "extreme"),
    (1.0 / 6,   "state_2", "ambiguous", "synthetic"),
    (2.0 / 6,   "state_3", "glitch",    "stable"),
    (3.0 / 6,   "state_4", "extreme",   "stable"),
    (4.0 / 6,   "state_5", "synthetic", "ambiguous"),
    (5.0 / 6,   "state_6", "collapse",  "glitch"),
)


def _state_for_progress(progress: float) -> tuple[str, str, str]:
    """Return (state_name, category_a, category_b) for a 0–1 progress value."""
    result = _STATE_SCHEDULE[0][1:]
    for threshold, state_name, cat_a, cat_b in _STATE_SCHEDULE:
        if progress >= threshold:
            result = (state_name, cat_a, cat_b)
    return result  # type: ignore[return-value]


def _load_sidecar(video_path: Path) -> dict:
    """Load <video_stem>.json from the same directory as the video, if present."""
    sidecar = video_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


class VideoExhibitionRuntime:
    """Exhibition runtime backed by video files rather than still images.

    The overlay (scan line, text, tinting, corruption effects) progresses
    independently of the video playback rate.  The video loops automatically
    if it is shorter than the target duration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        ecfg = config.exhibition
        if ecfg.video_path_a is None or ecfg.video_path_b is None:
            raise ValueError(
                "VideoExhibitionRuntime requires config.exhibition.video_path_a "
                "and video_path_b to be set."
            )
        self.sidecar = _load_sidecar(ecfg.video_path_a)
        # Sidecar values override config if present
        self.duration_s: float = self.sidecar.get("duration_seconds", ecfg.proof_duration_seconds)
        self.export_fps: float = float(self.sidecar.get("fps", ecfg.export_fps or ecfg.target_fps))
        self.experiment_id: str = self.sidecar.get(
            "experiment_id", ecfg.video_path_a.stem
        )
        self.rng = Random(config.random_seed)
        self.payload = load_text_payload(ecfg.text_payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open preview windows and play in real time. Press Q or ESC to quit."""
        ecfg = self.config.exhibition
        self.rng = Random(self.config.random_seed)
        typewriter_a, typewriter_b = self._build_typewriters()
        renderer = self._build_renderer(open_windows=True)

        cap_a = _open_capture(ecfg.video_path_a)
        cap_b = _open_capture(ecfg.video_path_b)

        start_s = monotonic()
        next_frame_at = start_s
        frame_period_s = max(0.001, 1.0 / max(1, ecfg.target_fps))
        logic_period_s = max(0.001, 1.0 / max(1.0, ecfg.logic_hz))
        next_logic_at = start_s
        current_a: tuple[str, str, str] = ("", "normal", "")
        current_b: tuple[str, str, str] = ("", "normal", "")
        last_cue_index = -1
        frame_samples: list[float] = []
        report_started_at = start_s
        last_frame_completed_at = start_s

        try:
            keep_running = True
            while keep_running:
                now_s = monotonic()
                if now_s < next_frame_at:
                    sleep(min(0.002, next_frame_at - now_s))
                    continue

                elapsed_s = min(now_s - start_s, self.duration_s)
                progress = elapsed_s / max(1.0, self.duration_s)
                state_name, cat_a, cat_b = _state_for_progress(progress)
                corruption = progress
                cue_index = int(elapsed_s / ecfg.image_hold_seconds)

                if cue_index != last_cue_index:
                    typewriter_a.reset_for_new_face(f"vcue_{cue_index}_a")
                    typewriter_b.reset_for_new_face(f"vcue_{cue_index}_b")
                    last_cue_index = cue_index

                allow_text = (elapsed_s % ecfg.image_hold_seconds) >= ecfg.image_text_delay_s

                if now_s >= next_logic_at:
                    current_a, current_b = self._advance_streams(
                        elapsed_s, state_name, cat_a, cat_b,
                        typewriter_a, typewriter_b, allow_text=allow_text,
                    )
                    next_logic_at += logic_period_s

                bg_a = _read_next_frame(cap_a)
                bg_b = _read_next_frame(cap_b)

                scan_progress = (elapsed_s % ecfg.scan_cycle_seconds) / ecfg.scan_cycle_seconds
                state_a, state_b = self._build_video_states(
                    corruption, scan_progress, current_a, current_b,
                )
                renderer.render_video(bg_a, bg_b, state_a, state_b)

                next_frame_at += frame_period_s
                wait_ms = int(max(1.0, (next_frame_at - monotonic()) * 1000.0))
                keep_running = renderer.process_events(wait_ms)

                frame_now = monotonic()
                frame_samples.append((frame_now - last_frame_completed_at) * 1000.0)
                last_frame_completed_at = frame_now
                if ecfg.perf_debug and frame_now - report_started_at >= 1.0 and frame_samples:
                    avg_ms = sum(frame_samples) / len(frame_samples)
                    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                    jitter = statistics.pstdev(frame_samples) if len(frame_samples) > 1 else 0.0
                    print(
                        f"[video_exhibition] fps={fps:05.2f} avg_ms={avg_ms:05.2f} "
                        f"jitter={jitter:04.2f}",
                        flush=True,
                    )
                    frame_samples.clear()
                    report_started_at = frame_now

                if elapsed_s >= self.duration_s:
                    break
        finally:
            cap_a.release()
            cap_b.release()
            renderer.close()

    def export_videos(self) -> tuple[Path, Path]:
        """Render to MP4 files and return (screen_a_path, screen_b_path)."""
        ecfg = self.config.exhibition
        self.rng = Random(self.config.random_seed)
        typewriter_a, typewriter_b = self._build_typewriters()
        renderer = self._build_renderer(open_windows=False)

        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        screen_a_path = output_dir / "screen_A.mp4"
        screen_b_path = output_dir / "screen_B.mp4"

        fps = self.export_fps
        size_a = (self.config.window_a.canvas_width, self.config.window_a.canvas_height)
        size_b = (self.config.window_b.canvas_width, self.config.window_b.canvas_height)

        # avc1 (H.264) works on macOS; mp4v is the cross-platform fallback.
        writer_a, writer_b = None, None
        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer_a = cv2.VideoWriter(str(screen_a_path), fourcc, fps, size_a)
            writer_b = cv2.VideoWriter(str(screen_b_path), fourcc, fps, size_b)
            if writer_a.isOpened() and writer_b.isOpened():
                print(f"Using codec: {codec}")
                break
            writer_a.release()
            writer_b.release()
            writer_a, writer_b = None, None

        if writer_a is None or writer_b is None:
            raise RuntimeError(
                f"Failed to open video writers for {screen_a_path} / {screen_b_path}. "
                "Tried codecs: avc1, mp4v."
            )

        cap_a = _open_capture(ecfg.video_path_a)
        cap_b = _open_capture(ecfg.video_path_b)

        total_frames = max(1, int(round(self.duration_s * fps))) + 1
        current_a: tuple[str, str, str] = ("", "normal", "")
        current_b: tuple[str, str, str] = ("", "normal", "")
        last_cue_index = -1

        try:
            for frame_index in range(total_frames):
                elapsed_s = min(self.duration_s, frame_index / fps)
                progress = elapsed_s / max(1.0, self.duration_s)
                state_name, cat_a, cat_b = _state_for_progress(progress)
                corruption = progress
                cue_index = int(elapsed_s / ecfg.image_hold_seconds)

                if cue_index != last_cue_index:
                    typewriter_a.reset_for_new_face(f"vcue_{cue_index}_a")
                    typewriter_b.reset_for_new_face(f"vcue_{cue_index}_b")
                    last_cue_index = cue_index

                allow_text = (elapsed_s % ecfg.image_hold_seconds) >= ecfg.image_text_delay_s

                current_a, current_b = self._advance_streams(
                    elapsed_s, state_name, cat_a, cat_b,
                    typewriter_a, typewriter_b, allow_text=allow_text,
                )

                bg_a = _read_next_frame(cap_a)
                bg_b = _read_next_frame(cap_b)

                scan_progress = (elapsed_s % ecfg.scan_cycle_seconds) / ecfg.scan_cycle_seconds
                state_a, state_b = self._build_video_states(
                    corruption, scan_progress, current_a, current_b,
                )
                rendered_a, rendered_b = renderer.compose_video_frames(bg_a, bg_b, state_a, state_b)
                writer_a.write(rendered_a)
                writer_b.write(rendered_b)
        finally:
            writer_a.release()
            writer_b.release()
            cap_a.release()
            cap_b.release()
            renderer.close()

        return screen_a_path, screen_b_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_output_dir(self) -> Path:
        """Return the export output directory.

        - If config.exhibition.export_output_dir was set explicitly (e.g. via
          --output in app.py), use it as-is.
        - Otherwise: exports/exhibition/<experiment_id>/  (overwrite=True)
                  or exports/exhibition/<experiment_id>_<ts>/  (default)
        """
        ecfg = self.config.exhibition
        base = ecfg.export_output_dir
        # If the base is the dataclass default, derive from experiment_id.
        if base == ROOT / "exports" / "exhibition":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = self.experiment_id if ecfg.overwrite else f"{self.experiment_id}_{ts}"
            return base / folder
        return base

    def _build_typewriters(self) -> tuple[ExhibitionTypewriter, ExhibitionTypewriter]:
        ecfg = self.config.exhibition
        return (
            ExhibitionTypewriter(
                CorruptionTextStateMachine(self.payload),
                screen_role="node_a",
                char_interval_s=ecfg.typewriter_char_interval_s,
                normal_pause_s=ecfg.normal_pause_s,
                log_dump_pause_s=ecfg.log_dump_pause_s,
                normal_idle_s=ecfg.normal_idle_s,
                log_dump_idle_s=ecfg.log_dump_idle_s,
                log_dump_prefreeze_s=ecfg.log_dump_prefreeze_s,
                log_dump_cooldown_s=ecfg.log_dump_cooldown_s,
                burst_max_lines_per_face=ecfg.burst_max_lines_per_face,
                burst_pause_s=ecfg.burst_pause_s,
            ),
            ExhibitionTypewriter(
                CorruptionTextStateMachine(self.payload),
                screen_role="node_b",
                char_interval_s=ecfg.typewriter_char_interval_s,
                normal_pause_s=ecfg.normal_pause_s,
                log_dump_pause_s=ecfg.log_dump_pause_s,
                normal_idle_s=ecfg.normal_idle_s,
                log_dump_idle_s=ecfg.log_dump_idle_s,
                log_dump_prefreeze_s=ecfg.log_dump_prefreeze_s,
                log_dump_cooldown_s=ecfg.log_dump_cooldown_s,
                burst_max_lines_per_face=ecfg.burst_max_lines_per_face,
                burst_pause_s=ecfg.burst_pause_s,
            ),
        )

    def _build_renderer(self, *, open_windows: bool) -> ExhibitionRenderer:
        ecfg = self.config.exhibition
        return ExhibitionRenderer(
            self.config.window_a,
            self.config.window_b,
            overlay_margin_px=ecfg.overlay_margin_px,
            lightweight_render=ecfg.lightweight_render,
            fullscreen=ecfg.fullscreen,
            clean_presentation=ecfg.clean_presentation,
            open_windows=open_windows,
        )

    def _advance_streams(
        self,
        elapsed_s: float,
        state_name: str,
        cat_a: str,
        cat_b: str,
        typewriter_a: ExhibitionTypewriter,
        typewriter_b: ExhibitionTypewriter,
        *,
        allow_text: bool,
    ) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        corruption = min(max(elapsed_s / max(1.0, self.duration_s), 0.0), 1.0)
        b_text = typewriter_b.current_full_text()
        current_a = typewriter_a.update(
            elapsed_s, state_name, cat_a, corruption, self.rng,
            forbidden_texts=((b_text,) if b_text else ()),
            allow_progress=allow_text,
        )
        a_text = typewriter_a.current_full_text()
        current_b = typewriter_b.update(
            elapsed_s, state_name, cat_b, corruption, self.rng,
            forbidden_texts=((a_text,) if a_text else ()),
            allow_progress=allow_text,
        )
        return current_a, current_b

    def _build_video_states(
        self,
        corruption: float,
        scan_progress: float,
        current_a: tuple[str, str, str],
        current_b: tuple[str, str, str],
    ) -> tuple[VideoFrameState, VideoFrameState]:
        text_a, kind_a, label_a = current_a
        text_b, kind_b, label_b = current_b
        return (
            VideoFrameState(
                corruption_score=corruption,
                revealed_text=text_a,
                line_kind=kind_a,
                state_label=label_a,
                scan_progress=scan_progress,
            ),
            VideoFrameState(
                corruption_score=corruption,
                revealed_text=text_b,
                line_kind=kind_b,
                state_label=label_b,
                scan_progress=scan_progress,
            ),
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _read_next_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Read the next frame, looping back to the start on exhaustion."""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    if not ret or frame is None:
        # Return a black frame — should only happen if the file is unreadable
        return np.zeros((1920, 1080, 3), dtype=np.uint8)
    return frame
