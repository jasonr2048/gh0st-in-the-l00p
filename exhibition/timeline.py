from __future__ import annotations

import math
import re
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

    def next_time_after(self, elapsed_s: float) -> float:
        for cue in self.cues:
            if cue.time_s > elapsed_s:
                return cue.time_s
        return self.duration_s


def build_proof_timeline(
    faces_root: Path,
    rng: Random,
    *,
    duration_s: float,
    hold_seconds: float,
    soft_landing_seconds: float = 24.0,
) -> ExhibitionTimeline:
    curated_pool = _load_curated_pool(faces_root)
    recent_a: list[Path] = []
    recent_b: list[Path] = []
    recent_id_a: list[str] = []
    recent_id_b: list[str] = []
    identity_usage_a: dict[str, int] = {}
    identity_usage_b: dict[str, int] = {}
    path_usage_a: dict[Path, int] = {}
    path_usage_b: dict[Path, int] = {}
    cues: list[VisualCue] = []
    cue_count = max(2, int(round(duration_s / hold_seconds)))
    interval_s = duration_s / cue_count

    for index in range(cue_count):
        time_s = min(duration_s, index * interval_s)
        phase_progress = _loop_phase_progress(time_s, duration_s, soft_landing_seconds)
        state_name = _state_name_for_progress(phase_progress)
        category_a = _choose_screen_a_category(state_name, phase_progress, rng)
        category_b = _choose_screen_b_category(category_a, state_name, phase_progress, rng)
        cues.append(
            VisualCue(
                time_s=time_s,
                screen_a_path=_select_curated_image(
                    curated_pool,
                    category_a,
                    recent_a,
                    recent_id_a,
                    identity_usage_a,
                    path_usage_a,
                    rng,
                    corruption_score=phase_progress,
                    avoid_paths=tuple(recent_b[-4:]),
                    avoid_identities=tuple(recent_id_b[-14:]),
                ),
                screen_b_path=_select_curated_image(
                    curated_pool,
                    category_b,
                    recent_b,
                    recent_id_b,
                    identity_usage_b,
                    path_usage_b,
                    rng,
                    corruption_score=phase_progress,
                    avoid_paths=tuple({*recent_a[-4:], cues[-1].screen_a_path if cues else None} - {None}),
                    avoid_identities=tuple({*recent_id_a[-14:]}),
                    recent_window=24,
                    identity_window=22,
                    identity_penalty=8.5,
                    path_penalty=12.0,
                ),
                screen_a_category=category_a,
                screen_b_category=category_b,
                corruption_score=phase_progress,
                state_name=state_name,
            )
        )
    return ExhibitionTimeline(cues=tuple(cues), duration_s=duration_s)


def _loop_phase_progress(time_s: float, duration_s: float, soft_landing_seconds: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    landing = max(0.0, min(soft_landing_seconds, duration_s * 0.4))
    if landing <= 0.0 or time_s <= duration_s - landing:
        base = min(1.0, time_s / max(duration_s - landing, 0.001))
        return min(1.0, base**0.84)

    landing_progress = (time_s - (duration_s - landing)) / max(landing, 0.001)
    eased = 0.5 * (1.0 + math.cos(math.pi * min(max(landing_progress, 0.0), 1.0)))
    floor = 0.08
    return floor + ((1.0 - floor) * eased)


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
    if primary == "stable":
        if corruption_score >= 0.08 and rng.random() < 0.38:
            return "ambiguous"
        if corruption_score >= 0.18 and rng.random() < 0.22:
            return "glitch"
        return "stable"
    if primary == "ambiguous":
        if corruption_score >= 0.34 and rng.random() < 0.28:
            return "glitch"
        return "ambiguous"
    if primary == "glitch":
        if corruption_score >= 0.52 and rng.random() < 0.22:
            return "extreme"
        return "glitch"
    if primary == "extreme":
        if corruption_score >= 0.68 and rng.random() < 0.18:
            return "synthetic"
        return "extreme"
    if primary == "synthetic":
        if corruption_score >= 0.78 and rng.random() < 0.14:
            return "glitch"
        return "synthetic"
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
    recent_identities: list[str],
    identity_usage: dict[str, int],
    path_usage: dict[Path, int],
    rng: Random,
    *,
    corruption_score: float,
    avoid_paths: tuple[Path | None, ...] = (),
    avoid_identities: tuple[str, ...] = (),
    recent_window: int = 16,
    identity_window: int = 14,
    identity_penalty: float = 5.6,
    path_penalty: float = 8.0,
) -> Path:
    paths = curated_pool.get(category, [])
    if not paths:
        paths = [path for values in curated_pool.values() for path in values]

    recent_block = set(recent_paths[-recent_window:])
    avoid_block = {path for path in avoid_paths if path is not None}
    identity_block = set(recent_identities[-identity_window:]) | set(avoid_identities)
    weighted_candidates: list[tuple[float, Path]] = []

    unseen_candidates = [
        path for path in paths
        if identity_usage.get(_identity_key(path), 0) == 0 and _identity_key(path) not in identity_block
    ]
    unused_variant_candidates = [
        path for path in paths
        if path_usage.get(path, 0) == 0 and _identity_key(path) not in identity_block
    ]
    primary_candidates = [path for path in paths if _identity_key(path) not in identity_block]
    if unseen_candidates:
        candidate_paths = unseen_candidates
    elif unused_variant_candidates:
        candidate_paths = unused_variant_candidates
    elif primary_candidates:
        candidate_paths = primary_candidates
    else:
        candidate_paths = paths

    for path in candidate_paths:
        identity = _identity_key(path)
        score = rng.random()
        if path in recent_block:
            score -= 14.0
        if path in avoid_block:
            score -= 18.0
        if identity in identity_block:
            score -= 24.0
        score -= identity_usage.get(identity, 0) * identity_penalty
        score -= path_usage.get(path, 0) * path_penalty
        if category == "stable":
            if corruption_score < 0.18:
                score -= 1.8
            elif corruption_score < 0.35:
                score -= 0.8
        if category == "ambiguous" and 0.10 <= corruption_score <= 0.48:
            score += 1.4
        if category == "glitch" and corruption_score >= 0.28:
            score += 1.9
        if category == "extreme" and corruption_score >= 0.44:
            score += 1.6
        if category == "synthetic" and corruption_score >= 0.58:
            score += 1.3
        weighted_candidates.append((score, path))

    weighted_candidates.sort(key=lambda item: item[0], reverse=True)
    chosen_path = _choose_weighted_candidate(weighted_candidates, rng)
    recent_paths.append(chosen_path)
    chosen_identity = _identity_key(chosen_path)
    recent_identities.append(chosen_identity)
    identity_usage[chosen_identity] = identity_usage.get(chosen_identity, 0) + 1
    path_usage[chosen_path] = path_usage.get(chosen_path, 0) + 1
    if len(recent_paths) > recent_window:
        del recent_paths[:-recent_window]
    if len(recent_identities) > identity_window:
        del recent_identities[:-identity_window]
    return chosen_path


def _choose_weighted_candidate(weighted_candidates: list[tuple[float, Path]], rng: Random) -> Path:
    best = weighted_candidates[0][0]
    band = [item for item in weighted_candidates[:24] if item[0] >= best - 7.0]
    floor = min(score for score, _ in band)
    weights = [(score - floor) + 0.55 for score, _ in band]
    total = sum(weights)
    roll = rng.random() * total
    cursor = 0.0
    for (_, path), weight in zip(band, weights):
        cursor += weight
        if roll <= cursor:
            return path
    return band[-1][1]


def _identity_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"\(.*?\)", "", stem)
    stem = re.sub(r"\d+", "", stem)
    stem = re.sub(r"[_\-\s]+", "", stem)
    stem = stem.replace("copy", "")
    stem = stem.replace("final", "")
    return stem or path.stem.lower()
