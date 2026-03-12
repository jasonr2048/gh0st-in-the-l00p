"""
Display validation player.

Opens two windows and plays the configured videos, letterboxed with a black
background to fit the window. Resize windows freely — content adapts.

Run from repo root:
    uv run spikes/display_validation/player.py

Press Q (with an OpenCV window focused) or Ctrl+C in the terminal to quit.
Videos loop automatically.
"""

import signal
import sys
from pathlib import Path

import tomli
import cv2
import numpy as np

CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def fit_to_window(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale frame to fit within target dimensions, centred on a black canvas."""
    fh, fw = frame.shape[:2]
    scale = min(target_w / fw, target_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    cfg = load_config(CONFIG_PATH)

    screens = []
    for key, label in (("screen_a", "SCREEN A"), ("screen_b", "SCREEN B")):
        screen_cfg = cfg[key]
        cap = cv2.VideoCapture(screen_cfg["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {screen_cfg['video_path']}")
        w = screen_cfg.get("window_width", 405)
        h = screen_cfg.get("window_height", 720)
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(label, w, h)
        fps = screen_cfg.get("fps", 30)
        screens.append([label, cap, fps, w, h])

    delay_ms = int(1000 / screens[0][2])

    while True:
        for screen in screens:
            label, cap, _, w, h = screen
            rect = cv2.getWindowImageRect(label)
            if rect[2] > 0 and rect[3] > 0:
                screen[3], screen[4] = rect[2], rect[3]
                w, h = screen[3], screen[4]

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                cv2.imshow(label, fit_to_window(frame, w, h))

        if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    for _, cap, *_ in screens:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
