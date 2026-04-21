"""JSONL recording for detected hand landmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import TracebackType
from typing import TextIO

from cv.hand_tracker import HandLandmarks


class LandmarkRecorder:
    """Append timestamped hand landmark detections to a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self._file: TextIO | None = None

    def open(self) -> None:
        """Open the output file, creating parent directories as needed."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("a", encoding="utf-8")

    def record(self, hands: list[HandLandmarks]) -> None:
        """Write one frame of hand detections to the JSONL file."""
        if self._file is None:
            raise RuntimeError("Recorder is not open.")

        payload = {
            "timestamp": time.time(),
            "hands": [
                {
                    "handedness": hand.handedness,
                    "score": hand.score,
                    "landmarks": [
                        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                        for landmark in hand.landmarks
                    ],
                }
                for hand in hands
            ],
        }
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the output file if it is open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> LandmarkRecorder:
        """Open and return this recorder for context manager use."""
        self.open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the recorder when leaving a context manager."""
        self.close()
