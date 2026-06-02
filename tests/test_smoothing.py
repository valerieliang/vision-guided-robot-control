from __future__ import annotations

import numpy as np

from cv.hand_tracker import HandLandmarks, Landmark
from cv.smoothing import AngleSmoother, HandLandmarkSmoother


def _hand(x_offset: float, score: float = 0.9) -> HandLandmarks:
    return HandLandmarks(
        handedness="Right",
        score=score,
        landmarks=[
            Landmark(x=0.1 + x_offset + i * 0.01, y=0.2 + i * 0.01, z=0.0)
            for i in range(21)
        ],
        raw_landmarks=None,
    )


def test_landmark_smoother_blends_landmarks() -> None:
    smoother = HandLandmarkSmoother(alpha=0.5, hold_frames=1)
    smoother.update([_hand(0.0)])

    result = smoother.update([_hand(0.2)])

    assert result[0].landmarks[0].x == 0.2


def test_landmark_smoother_holds_brief_dropout() -> None:
    smoother = HandLandmarkSmoother(alpha=0.5, hold_frames=1)
    first = smoother.update([_hand(0.0)])

    held = smoother.update([])
    expired = smoother.update([])

    assert held == first
    assert expired == []


def test_angle_smoother_deadbands_small_changes() -> None:
    smoother = AngleSmoother(alpha=0.5, deadband_rad=0.1)
    first = np.zeros(3, dtype=np.float32)
    second = np.array([0.05, 0.2, -0.2], dtype=np.float32)

    smoother.update(first)
    result = smoother.update(second)

    np.testing.assert_allclose(result, np.array([0.0, 0.1, -0.1], dtype=np.float32))
