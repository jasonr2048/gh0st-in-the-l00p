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
        self._recent_global: list[str] = []
        self._recent_signatures: list[str] = []

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
            if category == "stable" and corruption_score >= 0.08:
                pool_names.append(f"ambiguous_{screen_role}")
            elif category == "ambiguous":
                pool_names.append(f"glitch_{screen_role}")
            elif category == "glitch":
                pool_names.append(f"ambiguous_{screen_role}")
            elif category == "extreme":
                pool_names.append(f"synthetic_{screen_role}")
            elif category == "synthetic":
                pool_names.append(f"glitch_{screen_role}")
            if category == "glitch" and (corruption_score >= 0.58 or rng.random() < 0.32):
                pool_names.extend([f"collapse_broken_{screen_role[-1]}"] * 2)
            elif category in {"extreme", "synthetic"} and corruption_score >= 0.82 and rng.random() < 0.18:
                pool_names.append(f"collapse_broken_{screen_role[-1]}")

        if "log_dump" in self.payload and rng.random() < self._log_dump_chance(state_name, screen_role):
            pool_names.insert(0, "log_dump")

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
        burst_signatures = {self._functional_signature(text) for text in burst_recent}
        recent_signatures = set(self._recent_signatures[-18:])

        for pool_index, pool_name in enumerate(pool_names):
            pool = self.payload.get(pool_name, ())
            recent_pool = set(self._recent_by_pool.get(pool_name, [])[-14:])
            recent_global = set(self._recent_global[-24:])
            pool_weight = 3.0 if pool_index == 0 else 2.0 if pool_index == 1 else 1.0
            for line in pool:
                signature = self._functional_signature(line.text)
                score = pool_weight + self._base_line_weight(line) + rng.random()
                if pool_name == "log_dump":
                    score += 2.4
                if line.text in burst_recent:
                    score -= 24.0
                if signature in burst_signatures:
                    score -= 9.0
                if line.text in forbidden:
                    score -= 20.0
                if line.text in recent_pool:
                    score -= 12.0
                if line.text in recent_global:
                    score -= 16.0
                if signature in recent_signatures:
                    score -= 6.5
                weighted_choices.append((score, line, pool_name))

        if not weighted_choices:
            raise ValueError("No exhibition text options available")

        weighted_choices.sort(key=lambda item: item[0], reverse=True)
        _, line, pool_name = self._choose_from_top_band(weighted_choices, rng)
        return line, pool_name

    def _remember(self, pool_name: str, text: str) -> None:
        recent = self._recent_by_pool.setdefault(pool_name, [])
        recent.append(text)
        if len(recent) > 18:
            del recent[0]
        self._recent_global.append(text)
        if len(self._recent_global) > 36:
            del self._recent_global[0]
        self._recent_signatures.append(self._functional_signature(text))
        if len(self._recent_signatures) > 42:
            del self._recent_signatures[0]

    def label_for(self, state_name: str) -> str:
        return STATE_LABELS.get(state_name, state_name)

    @staticmethod
    def _log_dump_chance(state_name: str, screen_role: str) -> float:
        if state_name == "state_6":
            base = 0.28
        elif state_name in {"state_4", "state_5"}:
            base = 0.20
        elif state_name == "state_3":
            base = 0.15
        else:
            base = 0.055
        if screen_role == "node_a":
            return base * 1.15
        if screen_role == "node_b":
            return base * 0.78
        return base

    @staticmethod
    def _base_line_weight(line: PayloadLine) -> float:
        lowered = line.text.lower()
        score = 0.0
        if any(token in lowered for token in ("req_rcv", "200 ok", "retry?", "resolving...", "null.", "[node_b] > _")):
            score -= 1.8
        if any(token in lowered for token in ("project_shadow", "molecular photofitting", "face refuses categorisation", "danger identified before materialisation", "nightshade", "fawkes", "bad gateway", "psy-op", "synthetic probability", "loss function")):
            score -= 0.8
        if any(token in lowered for token in ("categorise. fix. exile.", "paritance achieved", "autophagous loop", "the copy is ontologically sufficient")):
            score -= 1.1
        if any(token in lowered for token in ("biometric markers", "berlin_suedkreuz_node_04", "prum_ii", "bounding box oscillation", "target is uncommon", "digital dead zone", "consensus reality database")):
            score += 0.4
        return score

    @staticmethod
    def _choose_from_top_band(
        weighted_choices: list[tuple[float, PayloadLine, str]],
        rng: Random,
    ) -> tuple[float, PayloadLine, str]:
        top_score = weighted_choices[0][0]
        band = [choice for choice in weighted_choices[:10] if choice[0] >= top_score - 3.5]
        floor = min(choice[0] for choice in band)
        weights = [(choice[0] - floor) + 0.65 for choice in band]
        total = sum(weights)
        roll = rng.random() * total
        cursor = 0.0
        for choice, weight in zip(band, weights):
            cursor += weight
            if roll <= cursor:
                return choice
        return band[-1]

    @staticmethod
    def _functional_signature(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("prum", "cross-referencing", "cross-ref", "schengen")):
            return "cross_ref"
        if any(token in lowered for token in ("liveness", "biometric", "ges/match", "facial")):
            return "biometric_match"
        if any(token in lowered for token in ("loitering", "heatmap", "behavioural", "behavioral", "watch zone")):
            return "behavioural_scan"
        if any(token in lowered for token in ("bounding box", "pixel", "ratio", "geometry", "photofitting")):
            return "image_geometry"
        if any(token in lowered for token in ("project_shadow", "dead zone", "banishment", "erase", "consensus reality")):
            return "banishment"
        if any(token in lowered for token in ("nightshade", "fawkes", "poison", "toxic", "weights corrupted")):
            return "poisoned_model"
        if any(token in lowered for token in ("synthetic", "deepfake", "psy-op", "platform", "engagement")):
            return "synthetic_platform"
        if any(token in lowered for token in ("null", "eof", "fpr", "_", "unreadable")):
            return "collapse_noise"
        return lowered.split(">", 1)[-1].strip().split(" ", 1)[0] or "generic"
