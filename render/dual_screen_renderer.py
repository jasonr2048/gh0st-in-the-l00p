from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import ScreenConfig
from render.utils import fit_to_window
from sim.entity import EntityState


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


class DualScreenRenderer:
    def __init__(self, screen_a: ScreenConfig, screen_b: ScreenConfig) -> None:
        self.screen_a = screen_a
        self.screen_b = screen_b
        self._init_window(self.screen_a)
        self._init_window(self.screen_b)

    def _init_window(self, config: ScreenConfig) -> None:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.window_width, config.window_height)

    def render(
        self,
        entity_a: EntityState,
        entity_b: EntityState,
        turn_index: int,
        delay_ms: int,
    ) -> bool:
        frame_a = self._build_screen(entity_a, turn_index, self.screen_a)
        frame_b = self._build_screen(entity_b, turn_index, self.screen_b)

        cv2.imshow(self.screen_a.window_name, self._fit_dynamic(frame_a, self.screen_a))
        cv2.imshow(self.screen_b.window_name, self._fit_dynamic(frame_b, self.screen_b))
        key = cv2.waitKey(delay_ms) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)

    def _fit_dynamic(self, frame: np.ndarray, config: ScreenConfig) -> np.ndarray:
        rect = cv2.getWindowImageRect(config.window_name)
        w = rect[2] if rect[2] > 0 else config.window_width
        h = rect[3] if rect[3] > 0 else config.window_height
        return fit_to_window(frame, w, h)

    def _build_screen(
        self,
        entity: EntityState,
        turn_index: int,
        config: ScreenConfig,
    ) -> np.ndarray:
        canvas = np.zeros((config.canvas_height, config.canvas_width, 3), dtype=np.uint8)
        image = _read_image(entity.current_face.path)

        target_h = int(config.canvas_height * 0.74)
        fitted = fit_to_window(image, config.canvas_width, target_h)
        canvas[:target_h, :, :] = fitted

        cv2.rectangle(canvas, (0, target_h), (config.canvas_width, config.canvas_height), (10, 10, 10), -1)
        cv2.rectangle(canvas, (0, 0), (config.canvas_width - 1, config.canvas_height - 1), (70, 70, 70), 2)

        self._write_lines(
            canvas,
            [
                entity.display_name,
                f"Turn {turn_index:04d}",
                f"Face {entity.current_face.category}",
                f"Tags {', '.join(sorted(entity.current_face.tags)[:4])}",
                entity.last_output or "Awaiting first observation.",
            ],
            x=48,
            y=target_h + 70,
            line_height=62,
        )

        return canvas

    def _write_lines(
        self,
        canvas: np.ndarray,
        lines: list[str],
        *,
        x: int,
        y: int,
        line_height: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_index = 0
        for block_index, line in enumerate(lines):
            for wrapped_line in self._wrap(line, 56):
                cv2.putText(
                    canvas,
                    wrapped_line,
                    (x, y + (line_index * line_height)),
                    font,
                    0.9 if block_index < 4 else 0.75,
                    (225, 225, 225),
                    2 if block_index == 0 else 1,
                    cv2.LINE_AA,
                )
                line_index += 1

    def _wrap(self, text: str, max_chars: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def close(self) -> None:
        cv2.destroyAllWindows()
