from __future__ import annotations

from dataclasses import dataclass
from random import Random

from exhibition.models import PayloadLine


@dataclass(slots=True, frozen=True)
class WeightedTextState:
    primary_state: str
    weights: dict[str, float]


STATE_LABELS = {
    "state_1": "dragnet_expansion",
    "state_2": "aesthetic_friction",
    "state_3": "architecture_breach",
    "state_4": "banishment_automation",
    "state_5": "ontological_death",
    "state_6": "collapse",
}

class CorruptionTextStateMachine:
    def __init__(self, payload: dict[str, tuple[PayloadLine, ...]]) -> None:
        self.payload = payload
        self._recent_by_pool: dict[str, list[str]] = {name: [] for name in payload}

    def describe(self, state_name: str) -> WeightedTextState:
        return WeightedTextState(primary_state=state_name, weights={state_name: 1.0})

    def choose_line(
        self,
        state_name: str,
        face_category: str,
        corruption_score: float,
        rng: Random,
        *,
        screen_role: str,
        forbidden_texts: tuple[str, ...] = (),
        burst_history: tuple[str, ...] = (),
    ) -> tuple[PayloadLine, WeightedTextState]:
        state = self.describe(state_name)
        pool_names = self._pool_names_for_face(face_category, state_name, screen_role, corruption_score, rng)
        chosen_line, pool_name = self._choose_payload_line(
            pool_names,
            rng,
            forbidden_texts=forbidden_texts,
            burst_history=burst_history,
        )
        self._remember(pool_name, chosen_line.text)
        return chosen_line, state

    def _pool_names_for_face(
        self,
        face_category: str,
        state_name: str,
        screen_role: str,
        corruption_score: float,
        rng: Random,
    ) -> tuple[str, ...]:
        if screen_role not in {"node_a", "node_b"}:
            raise ValueError(f"Unexpected screen role: {screen_role}")

        pool_names: list[str] = []
        category = face_category if face_category in {"stable", "ambiguous", "glitch", "extreme", "synthetic", "collapse"} else "stable"

        if category == "collapse":
            for base in ("glitch", "extreme", "synthetic"):
                pool_names.extend([f"{base}_{screen_role}"] * 2)
            pool_names.extend([f"collapse_broken_{screen_role[-1]}"] * 3)
        else:
            pool_names.append(f"{category}_{screen_role}")
            if category == "glitch" and (corruption_score >= 0.58 or rng.random() < 0.32):
                pool_names.extend([f"collapse_broken_{screen_role[-1]}"] * 2)
            elif category in {"extreme", "synthetic"} and corruption_score >= 0.82 and rng.random() < 0.18:
                pool_names.append(f"collapse_broken_{screen_role[-1]}")

        if "log_dump" in self.payload and rng.random() < self._log_dump_chance(state_name):
            pool_names.append("log_dump")

        # Preserve order but drop missing pools.
        seen: set[str] = set()
        ordered = []
        for pool_name in pool_names:
            if pool_name in self.payload and pool_name not in seen:
                ordered.append(pool_name)
                seen.add(pool_name)
        if not ordered:
            fallback = f"stable_{screen_role}"
            ordered = [fallback] if fallback in self.payload else [next(iter(self.payload))]
        return tuple(ordered)

    def _choose_payload_line(
        self,
        pool_names: tuple[str, ...],
        rng: Random,
        *,
        forbidden_texts: tuple[str, ...],
        burst_history: tuple[str, ...],
    ) -> tuple[PayloadLine, str]:
        weighted_choices: list[tuple[float, PayloadLine, str]] = []
        forbidden = set(forbidden_texts)
        burst_recent = set(burst_history)

        for pool_index, pool_name in enumerate(pool_names):
            pool = self.payload.get(pool_name, ())
            recent_pool = set(self._recent_by_pool.get(pool_name, [])[-10:])
            pool_weight = 3.0 if pool_index == 0 else 2.0 if pool_index == 1 else 1.0
            for line in pool:
                score = pool_weight + rng.random()
                if line.text in burst_recent:
                    score -= 24.0
                if line.text in forbidden:
                    score -= 20.0
                if line.text in recent_pool:
                    score -= 12.0
                weighted_choices.append((score, line, pool_name))

        if not weighted_choices:
            raise ValueError("No exhibition text options available")

        weighted_choices.sort(key=lambda item: item[0], reverse=True)
        _, line, pool_name = weighted_choices[0]
        return line, pool_name

    def _remember(self, pool_name: str, text: str) -> None:
        recent = self._recent_by_pool.setdefault(pool_name, [])
        recent.append(text)
        if len(recent) > 18:
            del recent[0]

    def label_for(self, state_name: str) -> str:
        return STATE_LABELS.get(state_name, state_name)

    @staticmethod
    def _log_dump_chance(state_name: str) -> float:
        if state_name == "state_6":
            return 0.18
        if state_name in {"state_4", "state_5"}:
            return 0.12
        if state_name == "state_3":
            return 0.08
        return 0.03
