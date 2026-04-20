from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class ScreenConfig:
    window_name: str
    canvas_width: int = 1080
    canvas_height: int = 1920
    window_width: int = 405
    window_height: int = 720
    window_x: int = 0
    window_y: int = 0


@dataclass(slots=True)
class ExhibitionConfig:
    faces_root: Path = ROOT / "faces_exhibition"
    text_payload: Path = ROOT / "exhibition" / "text_payload.json"
    export_output_dir: Path = ROOT / "exports" / "exhibition"
    proof_duration_seconds: float = 72.0
    frame_delay_ms: int = 33
    target_fps: int = 30
    export_fps: int = 30
    logic_hz: float = 15.0
    min_text_display_s: float = 2.0
    max_face_duration_s: float = 15.0
    typewriter_char_interval_s: float = 0.058
    normal_pause_s: float = 3.2
    log_dump_pause_s: float = 2.4
    overlay_margin_px: int = 64
    image_hold_seconds: float = 15.0
    image_text_delay_s: float = 1.05
    scan_cycle_seconds: float = 2.8
    normal_idle_s: float = 3.0
    log_dump_idle_s: float = 2.5
    log_dump_prefreeze_s: float = 0.9
    log_dump_cooldown_s: float = 7.5
    transition_duration_s: float = 1.25
    loop_soft_landing_seconds: float = 24.0
    burst_max_lines_per_face: int = 5
    burst_pause_s: float = 0.9
    perf_debug: bool = False
    lightweight_render: bool = False
    fullscreen: bool = False
    clean_presentation: bool = True


@dataclass(slots=True)
class SimulationConfig:
    faces_root: Path = ROOT / "faces_archive"
    dialogue_source: Path = ROOT / "dialogue_source" / "dialogue_source_master.txt"
    turn_interval_ms: int = 2400
    random_seed: int = 42
    exhibition: ExhibitionConfig = field(default_factory=ExhibitionConfig)
    window_a: ScreenConfig = field(
        default_factory=lambda: ScreenConfig(window_name="SCREEN A", window_x=0, window_y=0)
    )
    window_b: ScreenConfig = field(
        default_factory=lambda: ScreenConfig(window_name="SCREEN B", window_x=440, window_y=0)
    )


def load_default_config() -> SimulationConfig:
    return SimulationConfig()
