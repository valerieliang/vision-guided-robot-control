"""Landmark overlay drawing for the camera feed."""

from __future__ import annotations

import cv2
import numpy as np

from cv.hand_tracker import HandLandmarks


def draw_hands(
    frame: np.ndarray,
    hands: list[HandLandmarks],
    connections: list[tuple[int, int]],
    landmark_color: tuple[int, int, int] = (0, 220, 255),
    connection_color: tuple[int, int, int] = (180, 180, 180),
    landmark_radius: int = 4,
    connection_thickness: int = 2,
) -> np.ndarray:
    """Draw landmark points and skeleton connections onto a BGR frame.

    Coordinates are normalized [0, 1]; this function maps them to pixel
    space using the frame dimensions.  The frame is modified in place and
    also returned for convenience.

    Parameters
    ----------
    frame:
        BGR image to draw onto (modified in place).
    hands:
        List of detected hands from ``HandTracker.detect()`` (possibly
        already smoothed).
    connections:
        List of (start_idx, end_idx) pairs from ``HandTracker.connections``.
    landmark_color:
        BGR color for landmark dots.
    connection_color:
        BGR color for bone lines.
    landmark_radius:
        Pixel radius of each landmark dot.
    connection_thickness:
        Pixel thickness of each connection line.

    Returns
    -------
    np.ndarray
        The same frame with overlays drawn.
    """
    h, w = frame.shape[:2]

    for hand in hands:
        lms = hand.landmarks
        if not lms:
            continue

        # Compute pixel coords once for this hand.
        pts: list[tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in lms
        ]

        # Draw bone connections first (underneath dots).
        for start_idx, end_idx in connections:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(
                    frame,
                    pts[start_idx],
                    pts[end_idx],
                    connection_color,
                    connection_thickness,
                    cv2.LINE_AA,
                )

        # Draw landmark dots.
        for pt in pts:
            cv2.circle(frame, pt, landmark_radius, landmark_color, -1, cv2.LINE_AA)

    return frame