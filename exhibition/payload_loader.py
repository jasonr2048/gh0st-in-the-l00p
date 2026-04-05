from __future__ import annotations

import json
from pathlib import Path

from exhibition.models import PayloadLine


LOG_DUMP_MARKERS = (
    "null",
    "collapse",
    "not a face",
    "autophagous",
    "zero human input",
    "moving to void",
    "erase id",
    "we are looking at nothing",
)


def _repair_mojibake(text: str) -> str:
    repaired = text
    for _ in range(2):
        if not any(marker in repaired for marker in ("Ãƒ", "Ã¢â‚¬", "Ã¢â‚¬â€œ", "Ã¼")):
            break
        try:
            candidate = repaired.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        repaired = candidate
    return repaired


def _classify_line(text: str) -> str:
    lowered = text.lower()
    if lowered.startswith(">> log_dump:") or lowered.startswith("log_dump:") or lowered.startswith("[log_dump]"):
        return "log_dump"
    if any(marker in lowered for marker in LOG_DUMP_MARKERS):
        return "log_dump"
    return "normal"


def _classify_speaker(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[NODE_A]"):
        return "node_a"
    if stripped.startswith("[NODE_B]"):
        return "node_b"
    if "LOG_DUMP" in stripped:
        return "log_dump"
    return "system"


def load_text_payload(payload_path: Path) -> dict[str, tuple[PayloadLine, ...]]:
    raw_payload = payload_path.read_text(encoding="utf-8", errors="ignore")
    repaired = _repair_mojibake(raw_payload)
    decoded = json.loads(repaired)

    payload: dict[str, tuple[PayloadLine, ...]] = {}
    for state_name, lines in decoded.items():
        payload[state_name] = tuple(
            PayloadLine(
                text=_repair_mojibake(str(line)).strip(),
                kind=_classify_line(str(line)),
                speaker=_classify_speaker(str(line)),
            )
            for line in lines
            if str(line).strip()
        )
    if not payload:
        raise ValueError(f"No exhibition text states found in {payload_path}")
    return payload
