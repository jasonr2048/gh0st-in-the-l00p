from __future__ import annotations

import statistics
from pathlib import Path
from random import Random
from time import monotonic
from time import sleep

import cv2

from config import SimulationConfig
from exhibition.models import ExhibitionFrameState
from exhibition.payload_loader import load_text_payload
from exhibition.text_state_machine import CorruptionTextStateMachine
from exhibition.timeline import build_proof_timeline
from exhibition.typewriter import ExhibitionTypewriter
from render.exhibition_renderer import ExhibitionRenderer


class Gh0stExhibitionRuntime:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = Random(config.random_seed)
        self.timeline = build_proof_timeline(
            config.exhibition.faces_root,
            self.rng,
            duration_s=config.exhibition.proof_duration_seconds,
            hold_seconds=config.exhibition.image_hold_seconds,
            soft_landing_seconds=config.exhibition.loop_soft_landing_seconds,
        )
        self.payload = load_text_payload(config.exhibition.text_payload)

    def run(self) -> None:
        self.rng = Random(self.config.random_seed)
        screen_a_typewriter, screen_b_typewriter = self._build_typewriters()
        renderer = self._build_renderer(open_windows=True)
        print(
            f"[exhibition] starting playback duration={self.timeline.duration_s:.1f}s fps={self.config.exhibition.target_fps}",
            flush=True,
        )
        start_s = monotonic()
        next_frame_at = start_s
        frame_period_s = max(0.001, 1.0 / max(1, self.config.exhibition.target_fps))
        logic_period_s = max(0.001, 1.0 / max(1.0, self.config.exhibition.logic_hz))
        next_logic_at = start_s
        current_a = ("", "normal", "")
        current_b = ("", "normal", "")
        last_cue_time = -1.0
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

                elapsed_s = min(now_s - start_s, self.timeline.duration_s)
                previous_cue, cue = self.timeline.frame_at(elapsed_s)
                next_cue_time = self.timeline.next_time_after(elapsed_s)
                current_a, current_b, last_cue_time = self._reset_for_cue_if_needed(
                    cue.time_s,
                    cue.screen_b_path,
                    cue.screen_a_path,
                    screen_a_typewriter,
                    screen_b_typewriter,
                    current_a,
                    current_b,
                    last_cue_time,
                )
                inspection_elapsed = max(0.0, elapsed_s - cue.time_s)
                allow_text = inspection_elapsed >= self.config.exhibition.image_text_delay_s

                if now_s >= next_logic_at:
                    current_a, current_b = self._advance_streams(
                        elapsed_s,
                        cue.state_name,
                        cue.screen_b_category,
                        cue.screen_a_category,
                        min(next_cue_time, self.timeline.duration_s) - elapsed_s,
                        self.timeline.duration_s - elapsed_s,
                        screen_a_typewriter,
                        screen_b_typewriter,
                        allow_text=allow_text,
                    )
                    next_logic_at += logic_period_s

                frame_state_a, frame_state_b = self._build_frame_states(
                    elapsed_s,
                    inspection_elapsed,
                    previous_cue.screen_a_path,
                    previous_cue.screen_b_path,
                    cue.screen_a_path,
                    cue.screen_b_path,
                    cue.corruption_score,
                    current_a,
                    current_b,
                )
                renderer.render(frame_state_a, frame_state_b)

                next_frame_at += frame_period_s
                wait_ms = int(max(1.0, (next_frame_at - monotonic()) * 1000.0))
                keep_running = renderer.process_events(wait_ms)

                frame_now = monotonic()
                frame_samples.append((frame_now - last_frame_completed_at) * 1000.0)
                last_frame_completed_at = frame_now
                if self.config.exhibition.perf_debug and frame_now - report_started_at >= 1.0 and frame_samples:
                    avg_ms = sum(frame_samples) / len(frame_samples)
                    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                    jitter = statistics.pstdev(frame_samples) if len(frame_samples) > 1 else 0.0
                    cadence = "stable" if jitter < 2.5 else "unstable"
                    print(
                        f"[exhibition] fps={fps:05.2f} avg_ms={avg_ms:05.2f} jitter={jitter:04.2f} cadence={cadence}",
                        flush=True,
                    )
                    frame_samples.clear()
                    report_started_at = frame_now
                if elapsed_s >= self.timeline.duration_s:
                    break
        finally:
            renderer.close()

    def export_videos(self) -> tuple[Path, Path]:
        self.rng = Random(self.config.random_seed)
        screen_a_typewriter, screen_b_typewriter = self._build_typewriters()
        renderer = self._build_renderer(open_windows=False)
        output_dir = self.config.exhibition.export_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        screen_a_path = output_dir / "screen_A.avi"
        screen_b_path = output_dir / "screen_B.avi"
        fps = float(self.config.exhibition.export_fps or self.config.exhibition.target_fps)
        total_frames = max(1, int(round(self.timeline.duration_s * fps))) + 1
        codec_name = "MJPG"
        print(
            f"[exhibition] starting export duration={self.timeline.duration_s:.1f}s fps={fps:.1f} frames={total_frames}",
            flush=True,
        )
        print(f"[exhibition] codec/container={codec_name}/AVI", flush=True)
        print(f"[exhibition] saving {screen_a_path}", flush=True)
        print(f"[exhibition] saving {screen_b_path}", flush=True)
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer_a = cv2.VideoWriter(
            str(screen_a_path),
            fourcc,
            fps,
            (self.config.window_a.canvas_width, self.config.window_a.canvas_height),
        )
        writer_b = cv2.VideoWriter(
            str(screen_b_path),
            fourcc,
            fps,
            (self.config.window_b.canvas_width, self.config.window_b.canvas_height),
        )
        if not writer_a.isOpened() or not writer_b.isOpened():
            writer_a.release()
            writer_b.release()
            print("[exhibition] export error: failed to open video writers", flush=True)
            raise RuntimeError("Failed to open exhibition export video writers")

        current_a = ("", "normal", "")
        current_b = ("", "normal", "")
        last_cue_time = -1.0
        try:
            for frame_index in range(total_frames):
                elapsed_s = min(self.timeline.duration_s, frame_index / fps)
                previous_cue, cue = self.timeline.frame_at(elapsed_s)
                next_cue_time = self.timeline.next_time_after(elapsed_s)
                current_a, current_b, last_cue_time = self._reset_for_cue_if_needed(
                    cue.time_s,
                    cue.screen_b_path,
                    cue.screen_a_path,
                    screen_a_typewriter,
                    screen_b_typewriter,
                    current_a,
                    current_b,
                    last_cue_time,
                )
                inspection_elapsed = max(0.0, elapsed_s - cue.time_s)
                allow_text = inspection_elapsed >= self.config.exhibition.image_text_delay_s
                current_a, current_b = self._advance_streams(
                    elapsed_s,
                    cue.state_name,
                    cue.screen_b_category,
                    cue.screen_a_category,
                    min(next_cue_time, self.timeline.duration_s) - elapsed_s,
                    self.timeline.duration_s - elapsed_s,
                    screen_a_typewriter,
                    screen_b_typewriter,
                    allow_text=allow_text,
                )
                frame_state_a, frame_state_b = self._build_frame_states(
                    elapsed_s,
                    inspection_elapsed,
                    previous_cue.screen_a_path,
                    previous_cue.screen_b_path,
                    cue.screen_a_path,
                    cue.screen_b_path,
                    cue.corruption_score,
                    current_a,
                    current_b,
                )
                frame_a, frame_b = renderer.compose_frames(frame_state_a, frame_state_b)
                writer_a.write(frame_a)
                writer_b.write(frame_b)
        finally:
            writer_a.release()
            writer_b.release()
            renderer.close()
        self._validate_export(screen_a_path, screen_b_path)
        print("[exhibition] export complete", flush=True)
        return screen_a_path, screen_b_path

    def _build_typewriters(self) -> tuple[ExhibitionTypewriter, ExhibitionTypewriter]:
        screen_a_state_machine = CorruptionTextStateMachine(self.payload)
        screen_b_state_machine = CorruptionTextStateMachine(self.payload)
        return (
            ExhibitionTypewriter(
                screen_a_state_machine,
                screen_role="node_a",
                char_interval_s=self.config.exhibition.typewriter_char_interval_s,
                normal_pause_s=self.config.exhibition.normal_pause_s,
                log_dump_pause_s=self.config.exhibition.log_dump_pause_s,
                normal_idle_s=self.config.exhibition.normal_idle_s,
                log_dump_idle_s=self.config.exhibition.log_dump_idle_s,
                log_dump_prefreeze_s=self.config.exhibition.log_dump_prefreeze_s,
                log_dump_cooldown_s=self.config.exhibition.log_dump_cooldown_s,
                burst_max_lines_per_face=self.config.exhibition.burst_max_lines_per_face,
                burst_pause_s=self.config.exhibition.burst_pause_s,
            ),
            ExhibitionTypewriter(
                screen_b_state_machine,
                screen_role="node_b",
                char_interval_s=self.config.exhibition.typewriter_char_interval_s,
                normal_pause_s=self.config.exhibition.normal_pause_s,
                log_dump_pause_s=self.config.exhibition.log_dump_pause_s,
                normal_idle_s=self.config.exhibition.normal_idle_s,
                log_dump_idle_s=self.config.exhibition.log_dump_idle_s,
                log_dump_prefreeze_s=self.config.exhibition.log_dump_prefreeze_s,
                log_dump_cooldown_s=self.config.exhibition.log_dump_cooldown_s,
                burst_max_lines_per_face=self.config.exhibition.burst_max_lines_per_face,
                burst_pause_s=self.config.exhibition.burst_pause_s,
            ),
        )

    def _build_renderer(self, *, open_windows: bool) -> ExhibitionRenderer:
        return ExhibitionRenderer(
            self.config.window_a,
            self.config.window_b,
            overlay_margin_px=self.config.exhibition.overlay_margin_px,
            lightweight_render=self.config.exhibition.lightweight_render,
            fullscreen=self.config.exhibition.fullscreen,
            clean_presentation=self.config.exhibition.clean_presentation,
            open_windows=open_windows,
        )

    def _advance_streams(
        self,
        elapsed_s: float,
        state_name: str,
        screen_a_face_category: str,
        screen_b_face_category: str,
        remaining_face_time_s: float,
        remaining_total_time_s: float,
        screen_a_typewriter: ExhibitionTypewriter,
        screen_b_typewriter: ExhibitionTypewriter,
        *,
        allow_text: bool,
    ) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        remaining_budget_s = max(0.0, min(remaining_face_time_s, remaining_total_time_s))
        soft_landing = remaining_total_time_s <= self.config.exhibition.loop_soft_landing_seconds
        screen_b_current = screen_b_typewriter.current_full_text()
        current_a = screen_a_typewriter.update(
            elapsed_s,
            state_name,
            screen_a_face_category,
            min(max(elapsed_s / self.timeline.duration_s, 0.0), 1.0),
            self.rng,
            forbidden_texts=((screen_b_current,) if screen_b_current else ()),
            allow_progress=allow_text,
            allow_new_lines=allow_text and remaining_budget_s > 0.5,
            allow_log_dumps=not soft_landing,
            remaining_time_s=remaining_budget_s,
        )
        screen_a_current = screen_a_typewriter.current_full_text()
        current_b = screen_b_typewriter.update(
            elapsed_s,
            state_name,
            screen_b_face_category,
            min(max(elapsed_s / self.timeline.duration_s, 0.0), 1.0),
            self.rng,
            forbidden_texts=((screen_a_current,) if screen_a_current else ()),
            allow_progress=allow_text,
            allow_new_lines=allow_text and remaining_budget_s > 0.5,
            allow_log_dumps=not soft_landing,
            remaining_time_s=remaining_budget_s,
        )
        return current_a, current_b

    def _build_frame_states(
        self,
        elapsed_s: float,
        inspection_elapsed: float,
        previous_a_path: Path,
        previous_b_path: Path,
        current_a_path: Path,
        current_b_path: Path,
        cue_corruption_score: float,
        current_a: tuple[str, str, str],
        current_b: tuple[str, str, str],
    ) -> tuple[ExhibitionFrameState, ExhibitionFrameState]:
        corruption_score = min(max(elapsed_s / self.timeline.duration_s, cue_corruption_score), 1.0)
        scan_progress = min(
            1.0,
            (inspection_elapsed % self.config.exhibition.scan_cycle_seconds)
            / self.config.exhibition.scan_cycle_seconds,
        )
        transition_alpha = min(1.0, inspection_elapsed / self.config.exhibition.transition_duration_s)
        text_a, line_kind_a, state_label_a = current_a
        text_b, line_kind_b, state_label_b = current_b
        return (
            ExhibitionFrameState(
                image_path=current_a_path,
                previous_image_path=previous_a_path if previous_a_path != current_a_path else None,
                corruption_score=corruption_score,
                revealed_text=text_a,
                line_kind=line_kind_a,
                state_label=state_label_a,
                scan_progress=scan_progress,
                transition_alpha=transition_alpha,
            ),
            ExhibitionFrameState(
                image_path=current_b_path,
                previous_image_path=previous_b_path if previous_b_path != current_b_path else None,
                corruption_score=corruption_score,
                revealed_text=text_b,
                line_kind=line_kind_b,
                state_label=state_label_b,
                scan_progress=scan_progress,
                transition_alpha=transition_alpha,
            ),
        )

    @staticmethod
    def _reset_for_cue_if_needed(
        cue_time: float,
        screen_a_face_token: Path,
        screen_b_face_token: Path,
        screen_a_typewriter: ExhibitionTypewriter,
        screen_b_typewriter: ExhibitionTypewriter,
        current_a: tuple[str, str, str],
        current_b: tuple[str, str, str],
        last_cue_time: float,
    ) -> tuple[tuple[str, str, str], tuple[str, str, str], float]:
        if cue_time == last_cue_time:
            return current_a, current_b, last_cue_time
        screen_a_typewriter.reset_for_new_face(str(screen_a_face_token))
        screen_b_typewriter.reset_for_new_face(str(screen_b_face_token))
        return ("", "normal", ""), ("", "normal", ""), cue_time

    @staticmethod
    def _validate_export(screen_a_path: Path, screen_b_path: Path) -> None:
        missing = []
        for path in (screen_a_path, screen_b_path):
            if not path.exists():
                missing.append(path)
                continue
            if path.stat().st_size < 1024:
                missing.append(path)
                continue
            capture = cv2.VideoCapture(str(path))
            ok, _ = capture.read()
            capture.release()
            if not ok:
                missing.append(path)
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise RuntimeError(f"Exhibition export failed to write video files: {joined}")
