from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

from exhibition.models import VisualCue


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
STATE_PRIMARY_CATEGORY = {
    "state_1": "stable",
    "state_2": "ambiguous",
    "state_3": "glitch",
    "state_4": "extreme",
    "state_5": "synthetic",
    "state_6": "collapse",
}
PAIRING_MATRIX = {
    "extreme": ("stable", "ambiguous"),
    "glitch": ("stable", "synthetic"),
    "synthetic": ("ambiguous", "glitch"),
    "stable": ("extreme", "glitch"),
    "ambiguous": ("synthetic", "extreme"),
    "collapse": ("glitch", "extreme", "synthetic"),
}


@dataclass(slots=True, frozen=True)
class ExhibitionTimeline:
    cues: tuple[VisualCue, ...]
    duration_s: float

    def frame_at(self, elapsed_s: float) -> tuple[VisualCue, VisualCue]:
        previous = self.cues[0]
        current = self.cues[0]
        for cue in self.cues:
            if elapsed_s >= cue.time_s:
                previous = current
                current = cue
            else:
                break
        return previous, current


def build_proof_timeline(
    faces_root: Path,
    rng: Random,
    *,
    duration_s: float,
    hold_seconds: float,
) -> ExhibitionTimeline:
    curated_pool = _load_curated_pool(faces_root)
    recent_a: list[Path] = []
    recent_b: list[Path] = []
    cues: list[VisualCue] = []
    cue_count = max(2, int(duration_s / hold_seconds) + 1)

    for index in range(cue_count):
        phase_progress = min(1.0, index / max(1, cue_count - 1))
        state_name = _state_name_for_progress(phase_progress)
        category_a = _choose_screen_a_category(state_name, phase_progress, rng)
        category_b = _choose_screen_b_category(category_a, state_name, phase_progress, rng)
        cues.append(
            VisualCue(
                time_s=min(duration_s, index * hold_seconds),
                screen_a_path=_select_curated_image(
                    curated_pool,
                    category_a,
                    recent_a,
                    rng,
                    corruption_score=phase_progress,
                    avoid_paths=tuple(recent_b[-4:]),
                ),
                screen_b_path=_select_curated_image(
                    curated_pool,
                    category_b,
                    recent_b,
                    rng,
                    corruption_score=phase_progress,
                    avoid_paths=tuple({*recent_a[-4:], cues[-1].screen_a_path if cues else None} - {None}),
                ),
                screen_a_category=category_a,
                screen_b_category=category_b,
                corruption_score=phase_progress,
                state_name=state_name,
            )
        )
    return ExhibitionTimeline(cues=tuple(cues), duration_s=duration_s)


def _load_curated_pool(faces_root: Path) -> dict[str, list[Path]]:
    pool: dict[str, list[Path]] = {name: [] for name in ("stable", "ambiguous", "synthetic", "glitch", "extreme")}
    for category in pool:
        category_root = faces_root / category
        if not category_root.exists():
            continue
        for path in category_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                pool[category].append(path)
    if not any(pool.values()):
        raise ValueError(f"No curated exhibition images found under {faces_root}")
    return pool


def _state_name_for_progress(progress: float) -> str:
    scaled = min(5, int(progress * 6))
    return f"state_{scaled + 1}"


def _choose_screen_a_category(state_name: str, corruption_score: float, rng: Random) -> str:
    primary = STATE_PRIMARY_CATEGORY.get(state_name, "stable")
    if primary != "collapse":
        return primary
    collapse_choices = [("glitch", 0.4), ("extreme", 0.33), ("synthetic", 0.27)]
    roll = rng.random()
    cursor = 0.0
    for category, weight in collapse_choices:
        cursor += weight
        if roll <= cursor:
            return category
    return "glitch" if corruption_score > 0.85 else "extreme"


def _choose_screen_b_category(category_a: str, state_name: str, corruption_score: float, rng: Random) -> str:
    preferred = list(PAIRING_MATRIX.get(category_a, ("stable", "ambiguous")))
    if state_name == "state_6":
        preferred = [category for category in ("glitch", "extreme", "synthetic") if category != category_a]
        if not preferred:
            preferred = [category for category in ("stable", "ambiguous") if category != category_a]
    if rng.random() < 0.82:
        return preferred[int(rng.random() * len(preferred))]

    fallback = [category for category in ("stable", "ambiguous", "synthetic", "glitch", "extreme") if category != category_a]
    if corruption_score >= 0.75:
        fallback = [category for category in fallback if category in ("glitch", "extreme", "synthetic")] or fallback
    return fallback[int(rng.random() * len(fallback))]


def _select_curated_image(
    curated_pool: dict[str, list[Path]],
    category: str,
    recent_paths: list[Path],
    rng: Random,
    *,
    corruption_score: float,
    avoid_paths: tuple[Path | None, ...] = (),
) -> Path:
    paths = curated_pool.get(category, [])
    if not paths:
        paths = [path for values in curated_pool.values() for path in values]

    recent_window = 10
    recent_block = set(recent_paths[-recent_window:])
    avoid_block = {path for path in avoid_paths if path is not None}
    weighted_candidates: list[tuple[float, Path]] = []

    for path in paths:
        score = rng.random()
        if path in recent_block:
            score -= 10.0
        if path in avoid_block:
            score -= 14.0
        if category == "stable" and corruption_score < 0.35:
            score += 1.8
        if category == "ambiguous" and 0.15 <= corruption_score <= 0.55:
            score += 1.2
        if category == "glitch" and corruption_score >= 0.45:
            score += 1.8
        if category == "extreme" and corruption_score >= 0.58:
            score += 1.8
        if category == "synthetic" and corruption_score >= 0.65:
            score += 1.6
        weighted_candidates.append((score, path))

    weighted_candidates.sort(key=lambda item: item[0], reverse=True)
    chosen_path = weighted_candidates[0][1]
    recent_paths.append(chosen_path)
    if len(recent_paths) > recent_window:
        del recent_paths[:-recent_window]
    return chosen_path
