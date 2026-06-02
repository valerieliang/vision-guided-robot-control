from __future__ import annotations

import numpy as np

from cv.hand_tracker import HandLandmarks, Landmark
from kinematics.retargeting import retarget, retarget_all


def _synthetic_open_hand() -> HandLandmarks:
    coords = [
        (0.0, 0.0, 0.0),
        (-0.08, -0.02, 0.0),
        (-0.13, -0.06, 0.0),
        (-0.17, -0.10, 0.0),
        (-0.20, -0.14, 0.0),
        (-0.05, -0.12, 0.0),
        (-0.05, -0.22, 0.0),
        (-0.05, -0.32, 0.0),
        (-0.05, -0.42, 0.0),
        (0.0, -0.13, 0.0),
        (0.0, -0.24, 0.0),
        (0.0, -0.35, 0.0),
        (0.0, -0.46, 0.0),
        (0.05, -0.12, 0.0),
        (0.05, -0.22, 0.0),
        (0.05, -0.32, 0.0),
        (0.05, -0.42, 0.0),
        (0.10, -0.10, 0.0),
        (0.10, -0.19, 0.0),
        (0.10, -0.28, 0.0),
        (0.10, -0.37, 0.0),
    ]
    return HandLandmarks(
        handedness="Right",
        score=0.95,
        landmarks=[Landmark(*point) for point in coords],
        raw_landmarks=None,
    )


def test_retarget_returns_flat_19_angle_array() -> None:
    angles = retarget(_synthetic_open_hand())
    flat = angles.as_flat_array()

    assert flat.shape == (19,)
    assert flat.dtype == np.float32
    assert np.all(np.isfinite(flat))


def test_retarget_all_retains_one_result_per_hand() -> None:
    hand = _synthetic_open_hand()

    results = retarget_all([hand, hand])

    assert len(results) == 2
