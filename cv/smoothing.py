"""Placeholder smoothing utilities for future perception refinements."""

from __future__ import annotations

from cv.hand_tracker import HandLandmarks


def passthrough(hands: list[HandLandmarks]) -> list[HandLandmarks]:
    """Return hand landmarks unchanged."""
    return hands
