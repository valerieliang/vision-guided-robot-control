"""Camera helpers for webcam capture."""

from __future__ import annotations

import cv2
import numpy as np


def open_camera(camera_index: int, frame_width: int, frame_height: int) -> cv2.VideoCapture:
    """Open a webcam and configure its requested frame size."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return capture


def read_frame(capture: cv2.VideoCapture) -> np.ndarray | None:
    """Read a frame from an open camera, returning None for empty reads."""
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame
