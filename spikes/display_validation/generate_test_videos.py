"""
Generate placeholder face videos for display validation.

Reads dimensions, paths and fps from config.toml — set those to match your
target monitors before generating. Run from repo root:

    uv run spikes/display_validation/generate_test_videos.py
"""

from pathlib import Path

import tomli
import cv2
import numpy as np

CONFIG_PATH = Path(__file__).parent / "config.toml"

# Face occupies this fraction of video width (centred, black background).
# At ~30% of a 1080-wide portrait display: roughly life-size on a 42" screen.
FACE_WIDTH_RATIO = 0.30


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def draw_frame(
    w: int,
    h: int,
    color: tuple[int, int, int],
    label: str,
    video_path: str,
    fps: int,
    frame_num: int,
    total_frames: int,
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    t = frame_num / total_frames

    # Faint grid
    grid_color = (18, 18, 18)
    step = max(1, w // 14)
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), grid_color, 1)

    # Border — validates that content fills the screen edge-to-edge
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
    tick = max(20, w // 18)
    for px, py in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        sx = 1 if px == 0 else -1
        sy = 1 if py == 0 else -1
        cv2.line(frame, (px, py), (px + sx * tick, py), color, 2)
        cv2.line(frame, (px, py), (px, py + sy * tick), color, 2)

    # Wireframe face — centred, sized as a fixed ratio of video width
    face_w = int(w * FACE_WIDTH_RATIO)
    face_h = int(face_w * 340 / 260)
    dim = tuple(max(0, c - 90) for c in color)

    cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, color, 1)
    cv2.line(frame, (cx, cy - face_h), (cx, cy + face_h), dim, 1)
    cv2.line(frame, (cx - face_w, cy - int(face_h * 0.24)),
             (cx + face_w, cy - int(face_h * 0.24)), dim, 1)

    eye_r = max(4, int(face_w * 0.14))
    eye_offset = int(face_w * 0.40)
    eye_y = cy - int(face_h * 0.28)
    for eye_x in (cx - eye_offset, cx + eye_offset):
        cv2.circle(frame, (eye_x, eye_y), eye_r, color, 1)
        cv2.circle(frame, (eye_x, eye_y), max(1, eye_r // 8), color, -1)

    nose_base = cy + int(face_h * 0.16)
    cv2.line(frame, (cx, cy - int(face_h * 0.12)), (cx, nose_base), color, 1)
    cv2.line(frame, (cx - int(face_w * 0.08), nose_base),
             (cx + int(face_w * 0.08), nose_base), color, 1)
    cv2.ellipse(frame, (cx, cy + int(face_h * 0.54)),
                (int(face_w * 0.26), int(face_h * 0.08)), 0, 0, 180, color, 1)

    # Scan line
    scan_y = int(t * h) % h
    bright = tuple(min(255, c + 60) for c in color)
    cv2.line(frame, (0, scan_y), (w, scan_y), bright, 1)

    # Labels — show actual config values rather than fake data
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, w / 700)
    th = max(1, int(fs * 1.5))
    margin = max(10, w // 40)

    cv2.putText(frame, label,
                (margin, margin + int(fs * 36)),
                font, fs * 1.6, color, th, cv2.LINE_AA)
    cv2.putText(frame, Path(video_path).name,
                (margin, margin + int(fs * 80)),
                font, fs * 0.85, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{w} x {h}  {fps} fps",
                (margin, margin + int(fs * 115)),
                font, fs * 0.85, color, 1, cv2.LINE_AA)

    cv2.putText(frame, "TOP",
                (cx - int(fs * 25), margin + int(fs * 155)),
                font, fs * 0.9, color, th, cv2.LINE_AA)
    cv2.putText(frame, "BOTTOM",
                (cx - int(fs * 50), h - margin),
                font, fs * 0.9, color, th, cv2.LINE_AA)
    cv2.putText(frame, f"FRAME {frame_num:04d} / {total_frames}",
                (margin, h - margin),
                font, fs * 0.75, color, 1, cv2.LINE_AA)

    return frame


def generate_video(screen_cfg: dict, color: tuple[int, int, int], label: str) -> None:
    w = screen_cfg["video_width"]
    h = screen_cfg["video_height"]
    fps = screen_cfg["fps"]
    path = Path(screen_cfg["video_path"])
    total_frames = fps * 10

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for i in range(total_frames):
        writer.write(draw_frame(w, h, color, label, screen_cfg["video_path"], fps, i, total_frames))

    writer.release()
    print(f"Written: {path}  ({w}x{h} @ {fps}fps)")


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    generate_video(cfg["screen_a"], color=(0, 200, 80),  label="SCREEN A")
    generate_video(cfg["screen_b"], color=(200, 160, 0), label="SCREEN B")
    print("Done.")
