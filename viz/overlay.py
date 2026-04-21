"""OpenCV drawing helpers for hand landmark visualization."""

from __future__ import annotations

import cv2
import numpy as np

from cv.hand_tracker import HandLandmarks


def draw_hands(
    frame_bgr: np.ndarray,
    hands: list[HandLandmarks],
    connections: list[tuple[int, int]],
) -> np.ndarray:
    """Draw hand landmarks and handedness labels on a frame."""
    for hand in hands:
        points = _landmark_pixels(frame_bgr, hand)
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame_bgr, points[start], points[end], (80, 180, 255), 2)

        for point in points:
            cv2.circle(frame_bgr, point, 4, (30, 220, 30), -1)

        if hand.landmarks:
            height, width = frame_bgr.shape[:2]
            wrist = hand.landmarks[0]
            label_x = max(0, min(width - 1, int(wrist.x * width)))
            label_y = max(20, min(height - 1, int(wrist.y * height) - 12))
            label = f"{hand.handedness} {hand.score:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )

    return frame_bgr


def _landmark_pixels(frame_bgr: np.ndarray, hand: HandLandmarks) -> list[tuple[int, int]]:
    """Convert normalized landmarks to image pixel coordinates."""
    height, width = frame_bgr.shape[:2]
    return [
        (
            max(0, min(width - 1, int(landmark.x * width))),
            max(0, min(height - 1, int(landmark.y * height))),
        )
        for landmark in hand.landmarks
    ]
