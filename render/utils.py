from __future__ import annotations

import cv2
import numpy as np


def fit_to_window(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(target_w / fw, target_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas
