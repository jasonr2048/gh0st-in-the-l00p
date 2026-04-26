from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import ScreenConfig
from exhibition.models import ExhibitionFrameState
from render.utils import fit_to_window


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


class ExhibitionRenderer:
    NORMAL_TEXT_COLOR = (92, 255, 128)
    NORMAL_GLOW_COLOR = (28, 110, 42)
    LOG_DUMP_TEXT_COLOR = (170, 255, 180)
    LOG_DUMP_GLOW_COLOR = (76, 145, 88)
    PANEL_COLOR = (0, 0, 0)
    PANEL_BORDER_COLOR = (30, 140, 46)
    PANEL_DIVIDER_COLOR = (18, 80, 28)
    SCANLINE_COLOR = (8, 22, 8)

    def __init__(
        self,
        screen_a: ScreenConfig,
        screen_b: ScreenConfig,
        *,
        overlay_margin_px: int,
        lightweight_render: bool = False,
        fullscreen: bool = False,
        clean_presentation: bool = True,
        open_windows: bool = True,
    ) -> None:
        self.screen_a = screen_a
        self.screen_b = screen_b
        self.overlay_margin_px = overlay_margin_px
        self.lightweight_render = lightweight_render
        self.fullscreen = fullscreen
        self.clean_presentation = clean_presentation
        self.open_windows = open_windows
        self.normal_font = self._load_font(28)
        self.log_dump_font = self._load_font(34)
        self.header_font = self._load_font(18)
        self._fitted_image_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self._text_sprite_cache: OrderedDict[tuple[str, int, tuple[int, int, int], tuple[int, int, int]], np.ndarray] = OrderedDict()
        self._panel_geometry_cache: dict[tuple[int, int, str], dict[str, object]] = {}
        self._window_rect_cache: dict[str, tuple[int, int]] = {}
        self._window_rect_refresh: dict[str, int] = {}
        if self.open_windows:
            self._init_window(screen_a)
            self._init_window(screen_b)

    def _init_window(self, config: ScreenConfig) -> None:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(config.window_name, config.window_width, config.window_height)

    def render(
        self,
        screen_a_state: ExhibitionFrameState,
        screen_b_state: ExhibitionFrameState,
    ) -> None:
        frame_a, frame_b = self.compose_frames(screen_a_state, screen_b_state)

        if not self.open_windows:
            return
        cv2.imshow(self.screen_a.window_name, self._fit_dynamic(frame_a, self.screen_a))
        cv2.imshow(self.screen_b.window_name, self._fit_dynamic(frame_b, self.screen_b))

    def compose_frames(
        self,
        screen_a_state: ExhibitionFrameState,
        screen_b_state: ExhibitionFrameState,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._build_screen(screen_a_state, self.screen_a),
            self._build_screen(screen_b_state, self.screen_b),
        )

    def compose_video_frames(
        self,
        background_a: np.ndarray,
        background_b: np.ndarray,
        state_a: object,
        state_b: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compose overlay onto raw video background frames. Used for export."""
        return (
            self._build_screen_from_frame(background_a, state_a, self.screen_a),
            self._build_screen_from_frame(background_b, state_b, self.screen_b),
        )

    def render_video(
        self,
        background_a: np.ndarray,
        background_b: np.ndarray,
        state_a: object,
        state_b: object,
    ) -> None:
        """Render overlay onto raw video background frames to preview windows."""
        frame_a = self._build_screen_from_frame(background_a, state_a, self.screen_a)
        frame_b = self._build_screen_from_frame(background_b, state_b, self.screen_b)
        cv2.imshow(self.screen_a.window_name, self._fit_dynamic(frame_a, self.screen_a))
        cv2.imshow(self.screen_b.window_name, self._fit_dynamic(frame_b, self.screen_b))

    def _build_screen_from_frame(
        self,
        background: np.ndarray,
        state: object,
        config: ScreenConfig,
    ) -> np.ndarray:
        """Build a full canvas from a raw video frame and a VideoFrameState."""
        canvas = fit_to_window(background, config.canvas_width, config.canvas_height).copy()
        canvas = self._harden_frame(canvas, state.corruption_score)
        canvas = self._tint_canvas(canvas, state.corruption_score, state.line_kind)
        self._draw_active_scan(canvas, state.scan_progress)
        proxy = ExhibitionFrameState(
            image_path=Path("."),
            previous_image_path=None,
            corruption_score=state.corruption_score,
            revealed_text=state.revealed_text,
            line_kind=state.line_kind,
            state_label=state.state_label,
            scan_progress=state.scan_progress,
            transition_alpha=1.0,
        )
        self._draw_overlay_panel(canvas, proxy)
        if not self.clean_presentation:
            self._draw_border(canvas, state.corruption_score, state.line_kind)
        return canvas

    def process_events(self, wait_ms: int) -> bool:
        if not self.open_windows:
            return True
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)

    def _fit_dynamic(self, frame: np.ndarray, config: ScreenConfig) -> np.ndarray:
        refresh = self._window_rect_refresh.get(config.window_name, 0)
        cached = self._window_rect_cache.get(config.window_name)
        if cached is None or refresh <= 0:
            rect = cv2.getWindowImageRect(config.window_name)
            w = rect[2] if rect[2] > 0 else config.window_width
            h = rect[3] if rect[3] > 0 else config.window_height
            self._window_rect_cache[config.window_name] = (w, h)
            self._window_rect_refresh[config.window_name] = 12
        else:
            w, h = cached
            self._window_rect_refresh[config.window_name] = refresh - 1
        return fit_to_window(frame, w, h)

    def _build_screen(self, frame_state: ExhibitionFrameState, config: ScreenConfig) -> np.ndarray:
        canvas = self._build_transition_canvas(frame_state, config)
        canvas = self._harden_frame(canvas, frame_state.corruption_score)
        canvas = self._tint_canvas(canvas, frame_state.corruption_score, frame_state.line_kind)
        self._draw_active_scan(canvas, frame_state.scan_progress)
        self._draw_overlay_panel(canvas, frame_state)
        if not self.clean_presentation:
            self._draw_border(canvas, frame_state.corruption_score, frame_state.line_kind)
        return canvas

    def _tint_canvas(self, canvas: np.ndarray, corruption_score: float, line_kind: str) -> np.ndarray:
        overlay = np.zeros_like(canvas)
        overlay[:, :] = (
            int(20 + (90 * corruption_score)),
            int(6 + (30 * corruption_score)),
            int(18 + (110 * corruption_score)),
        )
        if line_kind == "log_dump":
            overlay[:, :] = (
                int(35 + (120 * corruption_score)),
                int(18 + (65 * corruption_score)),
                int(30 + (165 * corruption_score)),
            )
        return cv2.addWeighted(canvas, 1.0, overlay, 0.12 + (0.12 * corruption_score), 0.0)

    def _draw_overlay_panel(self, canvas: np.ndarray, frame_state: ExhibitionFrameState) -> None:
        h, w = canvas.shape[:2]
        geometry = self._panel_geometry(h, w, frame_state.line_kind)
        margin = int(geometry["margin"])
        panel_w = int(geometry["panel_w"])
        panel_y = int(geometry["panel_y"])
        top_left = geometry["top_left"]
        bottom_right = geometry["bottom_right"]
        base_y = int(geometry["base_y"])
        line_step = int(geometry["line_step"])

        cv2.rectangle(canvas, top_left, bottom_right, self.PANEL_COLOR, -1)
        cv2.rectangle(canvas, top_left, bottom_right, self.PANEL_BORDER_COLOR, 2)
        cv2.line(
            canvas,
            (margin, panel_y + 34),
            (margin + panel_w, panel_y + 34),
            self.PANEL_DIVIDER_COLOR,
            1,
        )
        if not self.lightweight_render:
            self._draw_scanlines(canvas, top_left, bottom_right)

        accent_width = int(panel_w * frame_state.corruption_score)
        if accent_width > 0:
            accent_color = (60, 155, 72) if frame_state.line_kind == "log_dump" else (42, 126, 54)
            cv2.rectangle(
                canvas,
                (margin + 2, panel_y + 4),
                (margin + 2 + accent_width, panel_y + 12),
                accent_color,
                -1,
            )

        header_text = self._build_header(frame_state)
        self._draw_terminal_text(
            canvas,
            header_text,
            (margin + 18, panel_y + 10),
            font_key="header",
            text_color=(66, 180, 82),
            glow_color=(14, 55, 20),
        )

        lines = self._wrap(frame_state.revealed_text, 28 if frame_state.line_kind == "log_dump" else 42)
        if frame_state.line_kind == "log_dump":
            text_color = self.LOG_DUMP_TEXT_COLOR
            glow_color = self.LOG_DUMP_GLOW_COLOR
            font_key = "log_dump"
        else:
            text_color = self.NORMAL_TEXT_COLOR
            glow_color = self.NORMAL_GLOW_COLOR
            font_key = "normal"

        visible_lines = lines[:3] if frame_state.line_kind == "log_dump" else lines[:5]
        for index, line in enumerate(visible_lines):
            position = (margin + 28, base_y + (index * line_step) + (10 if frame_state.line_kind == "log_dump" else 0))
            self._draw_terminal_text(
                canvas,
                f">> {line}" if frame_state.line_kind == "log_dump" and index == 0 else f"> {line}" if index == 0 else f"  {line}",
                position,
                font_key=font_key,
                text_color=text_color,
                glow_color=glow_color,
            )

    def _draw_border(self, canvas: np.ndarray, corruption_score: float, line_kind: str) -> None:
        color = (75, 75, 75)
        if line_kind == "log_dump":
            color = (
                int(100 + (100 * corruption_score)),
                int(60 + (30 * corruption_score)),
                int(120 + (90 * corruption_score)),
            )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), color, 2)

    def _wrap(self, text: str, max_chars: int) -> list[str]:
        lines: list[str] = []
        for block in text.splitlines() or [""]:
            words = block.split()
            if not words:
                continue
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

    def _build_transition_canvas(self, frame_state: ExhibitionFrameState, config: ScreenConfig) -> np.ndarray:
        current = self._get_fitted_image(frame_state.image_path, config)
        if frame_state.previous_image_path is None or frame_state.transition_alpha >= 1.0:
            return current.copy()

        previous = self._get_fitted_image(frame_state.previous_image_path, config)
        alpha = max(0.0, min(1.0, frame_state.transition_alpha))
        if self.lightweight_render:
            return cv2.addWeighted(previous, 1.0 - alpha, current, alpha, 0.0)

        instability = 1.0 + (frame_state.corruption_score * 1.35)
        shifted_previous = self._transform_transition_frame(
            previous,
            1.0 - alpha,
            drift=int(-18 * instability),
            corruption_score=frame_state.corruption_score,
            direction=-1.0,
        )
        shifted_current = self._transform_transition_frame(
            current,
            alpha,
            drift=int(18 * instability),
            corruption_score=frame_state.corruption_score,
            direction=1.0,
        )
        return cv2.addWeighted(shifted_previous, 1.0 - alpha, shifted_current, alpha, 0.0)

    def _transform_transition_frame(
        self,
        frame: np.ndarray,
        alpha: float,
        *,
        drift: int,
        corruption_score: float,
        direction: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        contamination = (1.0 - alpha) * (0.35 + corruption_score)
        scale = 1.0 + ((1.0 - alpha) * (0.08 + (0.04 * corruption_score)))
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
        matrix[0, 1] += 0.02 * contamination * direction
        matrix[1, 0] -= 0.012 * contamination * direction
        matrix[0, 2] += drift * (1.0 - alpha)
        matrix[1, 2] += drift * 0.55 * (1.0 - alpha)
        transformed = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if corruption_score >= 0.45:
            transformed = self._ripple_distortion(transformed, contamination, direction)
        return transformed

    def _ripple_distortion(self, frame: np.ndarray, contamination: float, direction: float) -> np.ndarray:
        h, w = frame.shape[:2]
        rows = np.arange(h, dtype=np.float32)
        amplitude = min(9.0, 2.0 + (7.0 * contamination))
        offsets = np.sin((rows / max(1.0, h)) * 18.0) * amplitude * direction
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(rows[:, None], (1, w))
        map_x += offsets[:, None]
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _harden_frame(self, canvas: np.ndarray, corruption_score: float) -> np.ndarray:
        contrast = 1.08 + (0.20 * corruption_score)
        hardened = cv2.convertScaleAbs(canvas, alpha=contrast, beta=-10)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(hardened, -1, kernel)
        return cv2.addWeighted(hardened, 0.72, sharpened, 0.28, 0.0)

    def _draw_terminal_text(
        self,
        canvas: np.ndarray,
        text: str,
        position: tuple[int, int],
        *,
        font_key: str,
        text_color: tuple[int, int, int],
        glow_color: tuple[int, int, int],
    ) -> None:
        x, y = position
        sprite = self._text_sprite(text, font_key, text_color, glow_color)
        self._blend_sprite(canvas, sprite, x, y)

    def _draw_scanlines(
        self,
        canvas: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
    ) -> None:
        start_x, start_y = top_left
        end_x, end_y = bottom_right
        for y in range(start_y + 14, end_y - 6, 6):
            cv2.line(canvas, (start_x + 8, y), (end_x - 8, y), self.SCANLINE_COLOR, 1)

    def _draw_active_scan(self, canvas: np.ndarray, scan_progress: float) -> None:
        if scan_progress <= 0.0:
            return
        h, w = canvas.shape[:2]
        scan_y = int((h * 0.05) + ((h * 0.78) * scan_progress))
        if self.lightweight_render:
            cv2.line(canvas, (0, scan_y), (w, scan_y), (96, 255, 128), 1)
            return
        band_y0 = max(0, scan_y - 8)
        band_y1 = min(h, scan_y + 8)
        if band_y1 > band_y0:
            local = canvas[band_y0:band_y1, :, :].astype(np.int16)
            local[:, :, 1] += 18
            local[:, :, 0] -= 6
            local[:, :, 2] -= 6
            canvas[band_y0:band_y1, :, :] = np.clip(local, 0, 255).astype(np.uint8)
            green_band = canvas[band_y0:band_y1, :, 1].astype(np.int16)
            green_band += 12
            canvas[band_y0:band_y1, :, 1] = np.clip(green_band, 0, 255).astype(np.uint8)
        cv2.line(canvas, (0, scan_y), (w, scan_y), (96, 255, 128), 1)
        cv2.line(canvas, (0, max(0, scan_y - 1)), (w, max(0, scan_y - 1)), (34, 120, 48), 1)
        cv2.line(canvas, (0, min(h - 1, scan_y + 1)), (w, min(h - 1, scan_y + 1)), (34, 120, 48), 1)

    def _panel_geometry(self, h: int, w: int, line_kind: str) -> dict[str, object]:
        key = (h, w, line_kind)
        if key in self._panel_geometry_cache:
            return self._panel_geometry_cache[key]

        margin = self.overlay_margin_px
        panel_w = w - (margin * 2)
        panel_h = int(h * (0.29 if line_kind == "log_dump" else 0.245))
        panel_y = h - margin - panel_h
        geometry = {
            "margin": margin,
            "panel_w": panel_w,
            "panel_h": panel_h,
            "panel_y": panel_y,
            "top_left": (margin, panel_y),
            "bottom_right": (margin + panel_w, panel_y + panel_h),
            "base_y": panel_y + 54,
            "line_step": 60 if line_kind == "log_dump" else 34,
        }
        self._panel_geometry_cache[key] = geometry
        return geometry

    def _font(self, font_key: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_key == "header":
            return self.header_font
        if font_key == "log_dump":
            return self.log_dump_font
        return self.normal_font

    def _text_sprite(
        self,
        text: str,
        font_key: str,
        text_color: tuple[int, int, int],
        glow_color: tuple[int, int, int],
    ) -> np.ndarray:
        cache_key = (text, font_key, text_color, glow_color)
        cached = self._text_sprite_cache.get(cache_key)
        if cached is not None:
            self._text_sprite_cache.move_to_end(cache_key)
            return cached

        font = self._font(font_key)
        bbox = font.getbbox(text)
        width = max(1, bbox[2] - bbox[0] + 4)
        height = max(1, bbox[3] - bbox[1] + 6)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        offset = (2 - bbox[0], 2 - bbox[1])
        if self.lightweight_render:
            draw.text((offset[0] + 1, offset[1] + 1), text, font=font, fill=(*glow_color, 220))
        else:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                draw.text((offset[0] + dx, offset[1] + dy), text, font=font, fill=(*glow_color, 255))
        draw.text(offset, text, font=font, fill=(*text_color, 255))
        sprite = np.array(image)
        self._text_sprite_cache[cache_key] = sprite
        if len(self._text_sprite_cache) > 256:
            self._text_sprite_cache.popitem(last=False)
        return sprite

    def _blend_sprite(self, canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        h, w = canvas.shape[:2]
        sh, sw = sprite.shape[:2]
        if x >= w or y >= h:
            return
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + sw)
        y1 = min(h, y + sh)
        if x0 >= x1 or y0 >= y1:
            return

        sprite_x0 = x0 - x
        sprite_y0 = y0 - y
        sprite_crop = sprite[sprite_y0:sprite_y0 + (y1 - y0), sprite_x0:sprite_x0 + (x1 - x0)]
        alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
        if not np.any(alpha):
            return
        rgb_crop = sprite_crop[:, :, :3][:, :, ::-1].astype(np.float32)
        canvas_crop = canvas[y0:y1, x0:x1].astype(np.float32)
        blended = (rgb_crop * alpha) + (canvas_crop * (1.0 - alpha))
        canvas[y0:y1, x0:x1] = blended.astype(np.uint8)

    def _get_fitted_image(self, image_path: Path, config: ScreenConfig) -> np.ndarray:
        key = (str(image_path), config.canvas_width, config.canvas_height)
        cached = self._fitted_image_cache.get(key)
        if cached is not None:
            self._fitted_image_cache.move_to_end(key)
            return cached
        image = _read_image(image_path)
        fitted = fit_to_window(image, config.canvas_width, config.canvas_height)
        self._fitted_image_cache[key] = fitted
        if len(self._fitted_image_cache) > 48:
            self._fitted_image_cache.popitem(last=False)
        return fitted

    def _build_header(self, frame_state: ExhibitionFrameState) -> str:
        corruption = int(frame_state.corruption_score * 100)
        if frame_state.line_kind == "log_dump":
            return f"SYS.LOG_DUMP :: CORRUPTION={corruption:03d}% :: EVENT=CRITICAL_RUPTURE"
        return f"INTEL.NODE :: CORRUPTION={corruption:03d}% :: STATUS=ACTIVE"

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = (
            Path("C:/Windows/Fonts/consola.ttf"),
            Path("C:/Windows/Fonts/lucon.ttf"),
            Path("C:/Windows/Fonts/cour.ttf"),
        )
        for font_path in font_candidates:
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def close(self) -> None:
        if self.open_windows:
            cv2.destroyAllWindows()
