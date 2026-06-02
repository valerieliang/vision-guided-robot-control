"""Decoded EEG/EMG intent replay sources.

This module intentionally starts from decoded labels rather than raw EEG
classification. Public datasets can be added as offline training/evaluation
inputs, while teleop consumes the same small intent schema either way.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Literal


NeuroLabel = Literal[
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
    "rest",
    "open",
    "close",
    "pinch",
    "point",
]


LABEL_ALIASES = {
    "left": "left_hand",
    "left_hand": "left_hand",
    "right": "right_hand",
    "right_hand": "right_hand",
    "feet": "feet",
    "foot": "feet",
    "tongue": "tongue",
    "rest": "rest",
    "stop": "rest",
    "open": "open",
    "close": "close",
    "grab": "close",
    "pinch": "pinch",
    "point": "point",
}


@dataclass(frozen=True)
class NeuroIntent:
    """A decoded neurotech command for high-level robot-hand control."""

    label: str
    confidence: float = 0.75
    source: str = "neurotech_replay"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_label(self) -> str:
        """Return the canonical label used by hand-pose mapping."""
        return LABEL_ALIASES.get(self.label.strip().lower(), "rest")


class NeuroReplaySource:
    """Cycle through decoded neurotech predictions from JSONL or CSV."""

    def __init__(
        self,
        path: str | Path,
        interval_s: float = 0.35,
        realtime: bool = True,
    ) -> None:
        self._events = _load_events(Path(path))
        self._interval_s = interval_s
        self._realtime = realtime
        self._index = 0
        self._last_emit = 0.0

    def next_intent(self) -> NeuroIntent:
        """Return the next decoded event, optionally paced in realtime."""
        if self._realtime:
            elapsed = time.time() - self._last_emit
            if self._last_emit and elapsed < self._interval_s:
                time.sleep(self._interval_s - elapsed)
        self._last_emit = time.time()

        event = self._events[self._index % len(self._events)]
        self._index += 1
        return NeuroIntent(
            label=event.label,
            confidence=event.confidence,
            source=event.source,
            metadata=event.metadata,
        )


class SyntheticNeuroSource:
    """Deterministic no-hardware stream for demos and pipeline tests."""

    def __init__(
        self,
        labels: Iterable[str] | None = None,
        interval_s: float = 0.45,
        realtime: bool = True,
    ) -> None:
        self._labels = list(labels or ["rest", "open", "close", "pinch", "point"])
        self._interval_s = interval_s
        self._realtime = realtime
        self._index = 0
        self._last_emit = 0.0

    def next_intent(self) -> NeuroIntent:
        """Return a deterministic synthetic decoded intent."""
        if self._realtime:
            elapsed = time.time() - self._last_emit
            if self._last_emit and elapsed < self._interval_s:
                time.sleep(self._interval_s - elapsed)
        self._last_emit = time.time()
        label = self._labels[self._index % len(self._labels)]
        self._index += 1
        return NeuroIntent(
            label=label,
            confidence=0.82,
            source="synthetic_neurotech",
            metadata={"note": "synthetic decoded label, not raw EEG"},
        )


def _load_events(path: Path) -> list[NeuroIntent]:
    if not path.exists():
        raise FileNotFoundError(f"Missing neurotech replay file: {path}")
    if path.suffix.lower() == ".jsonl":
        events = _load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        events = _load_csv(path)
    else:
        raise ValueError("Neurotech replay file must be .jsonl or .csv.")
    if not events:
        raise ValueError(f"Neurotech replay file has no events: {path}")
    return events


def _load_jsonl(path: Path) -> list[NeuroIntent]:
    events: list[NeuroIntent] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object.")
            events.append(_row_to_intent(row))
    return events


def _load_csv(path: Path) -> list[NeuroIntent]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [_row_to_intent(row) for row in csv.DictReader(file)]


def _row_to_intent(row: dict[str, Any]) -> NeuroIntent:
    label = str(row.get("label") or row.get("intent") or "rest")
    confidence = float(row.get("confidence") or 0.75)
    confidence = max(0.0, min(1.0, confidence))
    return NeuroIntent(
        label=label,
        confidence=confidence,
        source=str(row.get("source") or "neurotech_replay"),
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"label", "intent", "confidence", "source"}
        },
    )
