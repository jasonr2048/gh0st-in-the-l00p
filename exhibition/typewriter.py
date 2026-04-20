from __future__ import annotations

from dataclasses import dataclass
from random import Random

from exhibition.models import PayloadLine
from exhibition.text_state_machine import CorruptionTextStateMachine


@dataclass(slots=True)
class ActiveLine:
    payload_line: PayloadLine
    state_label: str
    started_at_s: float
    visible_characters: int = 0
    completed_at_s: float | None = None
    archived: bool = False


@dataclass(slots=True)
class PendingLogDump:
    payload_line: PayloadLine
    state_label: str
    reveal_at_s: float


class ExhibitionTypewriter:
    def __init__(
        self,
        state_machine: CorruptionTextStateMachine,
        *,
        screen_role: str,
        char_interval_s: float = 0.04,
        normal_pause_s: float = 0.8,
        log_dump_pause_s: float = 1.6,
        normal_idle_s: float = 1.0,
        log_dump_idle_s: float = 1.8,
        log_dump_prefreeze_s: float = 0.8,
        log_dump_cooldown_s: float = 6.0,
        burst_max_lines_per_face: int = 4,
        burst_pause_s: float = 0.45,
    ) -> None:
        self.state_machine = state_machine
        self.screen_role = screen_role
        self.char_interval_s = char_interval_s
        self.normal_pause_s = normal_pause_s
        self.log_dump_pause_s = log_dump_pause_s
        self.normal_idle_s = normal_idle_s
        self.log_dump_idle_s = log_dump_idle_s
        self.log_dump_prefreeze_s = log_dump_prefreeze_s
        self.log_dump_cooldown_s = log_dump_cooldown_s
        self.burst_max_lines_per_face = burst_max_lines_per_face
        self.burst_pause_s = burst_pause_s
        self.active_line: ActiveLine | None = None
        self.pending_log_dump: PendingLogDump | None = None
        self.idle_until_s: float = 0.0
        self.last_log_dump_at_s: float = float("-inf")
        self.face_history: list[str] = []
        self.current_face_token: str = ""
        self.face_line_count: int = 0
        self.last_completed_at: float | None = None
        self.has_completed_line: bool = False

    def update(
        self,
        now_s: float,
        state_name: str,
        face_category: str,
        corruption_score: float,
        rng: Random,
        *,
        forbidden_texts: tuple[str, ...] = (),
        allow_progress: bool = True,
        allow_new_lines: bool = True,
        allow_log_dumps: bool = True,
        remaining_time_s: float | None = None,
    ) -> tuple[str, str, str]:
        if not allow_progress:
            return "", "normal", self.active_line.state_label if self.active_line else ""

        if now_s < self.idle_until_s:
            if self.pending_log_dump is not None:
                return "", "log_dump", self.pending_log_dump.state_label
            state_label = self.active_line.state_label if self.active_line else ""
            return self._history_text(), "normal", state_label

        if self.pending_log_dump is not None:
            if now_s < self.pending_log_dump.reveal_at_s:
                return "", "log_dump", self.pending_log_dump.state_label
            self.active_line = ActiveLine(
                payload_line=self.pending_log_dump.payload_line,
                state_label=self.pending_log_dump.state_label,
                started_at_s=now_s,
            )
            self.pending_log_dump = None

        if self.active_line is None:
            if not allow_new_lines:
                return self._history_text(), "normal", ""
            self._start_line(
                now_s,
                state_name,
                face_category,
                corruption_score,
                rng,
                forbidden_texts=forbidden_texts,
                allow_log_dumps=allow_log_dumps,
                remaining_time_s=remaining_time_s,
            )
            if self.pending_log_dump is not None:
                return "", "log_dump", self.pending_log_dump.state_label
            if self.active_line is None:
                return self._history_text(), "normal", ""

        assert self.active_line is not None
        line = self.active_line
        target_length = len(line.payload_line.text)
        elapsed_s = max(0.0, now_s - line.started_at_s)
        line.visible_characters = min(target_length, int(elapsed_s / self.char_interval_s))

        if line.visible_characters >= target_length and line.completed_at_s is None:
            line.completed_at_s = now_s
            self.last_completed_at = now_s
            self.has_completed_line = True
        if line.completed_at_s is not None and not line.archived:
            if line.payload_line.kind != "log_dump":
                self.face_history.append(line.payload_line.text)
                self.face_history = self.face_history[-self.burst_max_lines_per_face :]
                self.face_line_count += 1
            line.archived = True

        if line.completed_at_s is not None:
            if line.payload_line.kind != "log_dump" and self.face_line_count < self.burst_max_lines_per_face:
                if now_s - line.completed_at_s >= self.burst_pause_s:
                    self.active_line = None
                return self._history_text(), "normal", line.state_label

            pause_s = self.log_dump_pause_s if line.payload_line.kind == "log_dump" else self.normal_pause_s
            if now_s - line.completed_at_s >= pause_s:
                idle_s = self.log_dump_idle_s if line.payload_line.kind == "log_dump" else self.normal_idle_s
                self.idle_until_s = now_s + idle_s
                self.active_line = None
                return self._history_text(), "normal", self.pending_log_dump.state_label if self.pending_log_dump else line.state_label
            if line.payload_line.kind == "log_dump":
                return line.payload_line.text, "log_dump", line.state_label
            return self._history_text(), line.payload_line.kind, line.state_label

        revealed_text = line.payload_line.text[: line.visible_characters]
        if line.payload_line.kind == "log_dump":
            return revealed_text, line.payload_line.kind, line.state_label
        if self.face_history:
            revealed_text = "\n".join(self.face_history[-self.burst_max_lines_per_face :] + [revealed_text])
        return revealed_text, line.payload_line.kind, line.state_label

    def current_full_text(self) -> str:
        if self.active_line is None:
            return ""
        return self.active_line.payload_line.text

    def reset_for_new_face(self, face_token: str) -> None:
        if face_token == self.current_face_token:
            return
        self.current_face_token = face_token
        self.active_line = None
        self.pending_log_dump = None
        self.idle_until_s = 0.0
        self.face_history = []
        self.face_line_count = 0
        self.last_completed_at = None
        self.has_completed_line = False

    def _start_line(
        self,
        now_s: float,
        state_name: str,
        face_category: str,
        corruption_score: float,
        rng: Random,
        *,
        forbidden_texts: tuple[str, ...],
        allow_log_dumps: bool,
        remaining_time_s: float | None,
    ) -> None:
        payload_line, weighted_state = self._select_payload_line(
            now_s,
            state_name,
            face_category,
            corruption_score,
            rng,
            forbidden_texts=forbidden_texts,
            allow_log_dumps=allow_log_dumps,
        )
        if remaining_time_s is not None and not self._line_fits_remaining_time(payload_line, remaining_time_s):
            self.active_line = None
            self.pending_log_dump = None
            return
        state_label = self.state_machine.label_for(weighted_state.primary_state)
        if payload_line.kind == "log_dump":
            self.pending_log_dump = PendingLogDump(
                payload_line=payload_line,
                state_label=state_label,
                reveal_at_s=now_s + self.log_dump_prefreeze_s,
            )
            self.active_line = None
            self.last_log_dump_at_s = now_s
            return

        self.active_line = ActiveLine(
            payload_line=payload_line,
            state_label=state_label,
            started_at_s=now_s,
        )

    def _select_payload_line(
        self,
        now_s: float,
        state_name: str,
        face_category: str,
        corruption_score: float,
        rng: Random,
        *,
        forbidden_texts: tuple[str, ...],
        allow_log_dumps: bool,
    ) -> tuple[PayloadLine, object]:
        fallback: tuple[PayloadLine, object] | None = None
        for _ in range(8):
            candidate = self.state_machine.choose_line(
                state_name,
                face_category,
                corruption_score,
                rng,
                screen_role=self.screen_role,
                forbidden_texts=forbidden_texts,
                burst_history=tuple(self.face_history),
            )
            payload_line, weighted_state = candidate
            if fallback is None:
                fallback = candidate
            if payload_line.kind == "log_dump" and not allow_log_dumps:
                continue
            if payload_line.kind != "log_dump":
                return candidate
            if now_s - self.last_log_dump_at_s >= self.log_dump_cooldown_s:
                return candidate
        assert fallback is not None
        return fallback

    def _line_fits_remaining_time(self, payload_line: PayloadLine, remaining_time_s: float) -> bool:
        typing_time = len(payload_line.text) * self.char_interval_s
        if payload_line.kind == "log_dump":
            required = self.log_dump_prefreeze_s + typing_time + self.log_dump_pause_s
        else:
            required = typing_time + max(self.burst_pause_s, self.normal_pause_s * 0.45)
        return remaining_time_s >= (required + 0.35)

    def _history_text(self) -> str:
        return "\n".join(self.face_history[-self.burst_max_lines_per_face :])

    def is_typing(self) -> bool:
        if self.pending_log_dump is not None:
            return True
        if self.active_line is None:
            return False
        return self.active_line.completed_at_s is None

    def is_ready(self, now_s: float, min_display_s: float) -> bool:
        if not self.has_completed_line:
            return False
        if self.is_typing():
            return False
        if self.last_completed_at is None:
            return False
        if now_s < self.idle_until_s:
            return False
        return (now_s - self.last_completed_at) >= min_display_s
