"""End-to-end loop latency tracker."""

from __future__ import annotations

from collections import deque


class LatencyMonitor:
    """Rolling average of control loop elapsed times.

    Parameters
    ----------
    window:
        Number of recent frames to average over.
    """

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._samples: deque[float] = deque(maxlen=window)

    def record(self, elapsed: float) -> None:
        """Record one loop iteration's elapsed time in seconds."""
        self._samples.append(elapsed)

    @property
    def mean_ms(self) -> float:
        """Rolling mean latency in milliseconds."""
        if not self._samples:
            return 0.0
        return (sum(self._samples) / len(self._samples)) * 1000.0

    @property
    def max_ms(self) -> float:
        """Rolling max latency in milliseconds."""
        if not self._samples:
            return 0.0
        return max(self._samples) * 1000.0

    def __str__(self) -> str:
        return f"latency  mean={self.mean_ms:.1f}ms  max={self.max_ms:.1f}ms"