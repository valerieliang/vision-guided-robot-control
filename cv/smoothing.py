"""Temporal smoothing utilities for MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np

from cv.hand_tracker import HandLandmarks, Landmark


def passthrough(hands: list[HandLandmarks]) -> list[HandLandmarks]:
    """Return hand landmarks unchanged."""
    return hands


class HandLandmarkSmoother:
    """Stateful exponential landmark smoother with short dropout hold.

    MediaPipe landmarks can jitter by a few pixels even when the hand is still.
    Retargeting turns that small landmark noise into larger joint-angle changes,
    so this filter smooths the normalized landmarks before kinematic conversion.
    """

    def __init__(
        self,
        alpha: float = 0.45,
        min_score: float = 0.35,
        hold_frames: int = 3,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0.0, 1.0].")
        if hold_frames < 0:
            raise ValueError("hold_frames must be non-negative.")
        self._alpha = alpha
        self._min_score = min_score
        self._hold_frames = hold_frames
        self._last: list[HandLandmarks] = []
        self._misses = 0

    def update(self, hands: list[HandLandmarks]) -> list[HandLandmarks]:
        """Return smoothed hands, holding briefly through tracking dropouts."""
        valid = [
            hand
            for hand in hands
            if hand.score >= self._min_score and len(hand.landmarks) >= 21
        ]
        if not valid:
            self._misses += 1
            if self._last and self._misses <= self._hold_frames:
                return self._last
            self._last = []
            return []

        self._misses = 0
        if not self._last:
            self._last = valid
            return valid

        smoothed = [
            self._smooth_hand(current, self._match_previous(current, index))
            for index, current in enumerate(valid)
        ]
        self._last = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear filter history."""
        self._last = []
        self._misses = 0

    def _match_previous(
        self,
        current: HandLandmarks,
        fallback_index: int,
    ) -> HandLandmarks | None:
        for previous in self._last:
            if previous.handedness == current.handedness:
                return previous
        if fallback_index < len(self._last):
            return self._last[fallback_index]
        return None

    def _smooth_hand(
        self,
        current: HandLandmarks,
        previous: HandLandmarks | None,
    ) -> HandLandmarks:
        if previous is None or len(previous.landmarks) != len(current.landmarks):
            return current
        landmarks = [
            Landmark(
                x=_ema(prev.x, point.x, self._alpha),
                y=_ema(prev.y, point.y, self._alpha),
                z=_ema(prev.z, point.z, self._alpha),
            )
            for prev, point in zip(previous.landmarks, current.landmarks)
        ]
        return replace(current, landmarks=landmarks)


class AngleSmoother:
    """Smooth and deadband retargeted joint-angle vectors."""

    def __init__(
        self,
        alpha: float = 0.35,
        deadband_rad: float = 0.015,
        max_step_rad: float | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0.0, 1.0].")
        if deadband_rad < 0.0:
            raise ValueError("deadband_rad must be non-negative.")
        self._alpha = alpha
        self._deadband_rad = deadband_rad
        self._max_step_rad = max_step_rad
        self._last: np.ndarray | None = None

    def update(self, values: np.ndarray) -> np.ndarray:
        """Return a smoothed numpy-like vector without importing numpy here."""
        if self._last is None:
            self._last = values.copy()
            return values

        delta = values - self._last
        if self._deadband_rad:
            delta = delta.copy()
            delta[np.abs(delta) < self._deadband_rad] = 0.0
        step = delta * self._alpha
        if self._max_step_rad is not None:
            step = step.clip(-self._max_step_rad, self._max_step_rad)
        self._last = self._last + step
        return self._last.copy()

    def reset(self) -> None:
        """Clear filter history."""
        self._last = None


class OneEuroScalar:
    """Small scalar One Euro filter for future per-angle tuning."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._last_value: float | None = None
        self._last_derivative = 0.0

    def update(self, value: float, dt: float) -> float:
        """Filter one scalar sample."""
        if self._last_value is None or dt <= 0.0:
            self._last_value = value
            return value
        derivative = (value - self._last_value) / dt
        derivative_alpha = _cutoff_alpha(self.derivative_cutoff, dt)
        derivative_hat = _ema(self._last_derivative, derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        value_alpha = _cutoff_alpha(cutoff, dt)
        value_hat = _ema(self._last_value, value, value_alpha)
        self._last_value = value_hat
        self._last_derivative = derivative_hat
        return value_hat


def _ema(previous: float, current: float, alpha: float) -> float:
    return previous + alpha * (current - previous)


def _cutoff_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
    return 1.0 - exp(-dt / tau)
