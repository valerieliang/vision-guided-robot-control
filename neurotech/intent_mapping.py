"""Map decoded neurotech labels to robot-hand pose vectors."""

from __future__ import annotations

import numpy as np

from neurotech.decoders import NeuroIntent


class NeuroHandPoseMapper:
    """Convert discrete decoded labels into 19 retargeted hand angles."""

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self._confidence_threshold = confidence_threshold
        self._last_pose = _open_pose()

    def to_retargeted(self, intent: NeuroIntent) -> np.ndarray:
        """Return a 19-angle retargeting vector for a decoded intent."""
        if intent.confidence < self._confidence_threshold:
            return self._last_pose.copy()

        label = intent.normalized_label()
        if label in {"rest", "open"}:
            pose = _open_pose()
        elif label in {"close", "feet"}:
            pose = _close_pose()
        elif label == "pinch":
            pose = _pinch_pose()
        elif label == "point":
            pose = _point_pose()
        elif label == "left_hand":
            pose = _left_hand_pose()
        elif label == "right_hand":
            pose = _right_hand_pose()
        elif label == "tongue":
            pose = _wave_pose()
        else:
            pose = _open_pose()

        self._last_pose = pose.astype(np.float32)
        return self._last_pose.copy()


def _open_pose() -> np.ndarray:
    return np.zeros(19, dtype=np.float32)


def _close_pose() -> np.ndarray:
    pose = np.zeros(19, dtype=np.float32)
    for offset in (0, 4, 8, 12):
        pose[offset : offset + 3] = [1.15, 1.05, 0.8]
    pose[16:19] = [0.9, 0.75, 0.25]
    return pose


def _pinch_pose() -> np.ndarray:
    pose = np.zeros(19, dtype=np.float32)
    pose[0:3] = [0.9, 0.85, 0.45]
    pose[4:7] = [0.25, 0.15, 0.05]
    pose[8:15] = 0.15
    pose[16:19] = [1.1, 0.9, 0.3]
    return pose


def _point_pose() -> np.ndarray:
    pose = _close_pose()
    pose[0:3] = [0.0, 0.0, 0.0]
    return pose


def _left_hand_pose() -> np.ndarray:
    pose = _open_pose()
    pose[12:15] = [0.95, 0.9, 0.55]
    return pose


def _right_hand_pose() -> np.ndarray:
    pose = _open_pose()
    pose[0:3] = [0.95, 0.9, 0.55]
    return pose


def _wave_pose() -> np.ndarray:
    pose = _open_pose()
    pose[3] = -0.2
    pose[7] = -0.08
    pose[11] = 0.08
    pose[15] = 0.2
    return pose
