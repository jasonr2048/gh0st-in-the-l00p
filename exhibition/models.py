from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True, frozen=True)
class PayloadLine:
    text: str
    kind: str = "normal"
    speaker: str = "system"


@dataclass(slots=True, frozen=True)
class VisualCue:
    time_s: float
    screen_a_path: Path
    screen_b_path: Path
    screen_a_category: str
    screen_b_category: str
    corruption_score: float
    state_name: str


@dataclass(slots=True, frozen=True)
class ExhibitionFrameState:
    image_path: Path
    previous_image_path: Path | None
    corruption_score: float
    revealed_text: str
    line_kind: str
    state_label: str
    scan_progress: float = 0.0
    transition_alpha: float = 1.0
    image_frame: object = field(default=None, hash=False, compare=False)


@dataclass(slots=True, frozen=True)
class VideoFrameState:
    corruption_score: float
    revealed_text: str
    line_kind: str
    state_label: str
    scan_progress: float = 0.0
