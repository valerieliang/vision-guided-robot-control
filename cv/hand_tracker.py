"""Hand landmark detection using MediaPipe Hands."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import time
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np


@dataclass(frozen=True)
class Landmark:
    """A normalized hand landmark from MediaPipe."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandLandmarks:
    """Detected landmarks and label for one hand."""

    handedness: str
    score: float
    landmarks: list[Landmark]
    raw_landmarks: Any


class HandTracker:
    """Small wrapper around MediaPipe Hands for frame-by-frame detection."""

    def __init__(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_asset_path: str | None = None,
        model_asset_url: str | None = None,
    ) -> None:
        legacy_hands = _load_legacy_hands_solution()
        self._backend = "legacy" if legacy_hands is not None else "tasks"
        self._connections: list[tuple[int, int]]
        self._timestamp_ms = int(time.monotonic() * 1000)

        if legacy_hands is not None:
            self._hands = legacy_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._connections = _normalize_connections(legacy_hands.HAND_CONNECTIONS)
            return

        task_modules = _load_tasks_modules()
        model_path = _ensure_model_asset(model_asset_path, model_asset_url)
        options = task_modules.hand_landmarker.HandLandmarkerOptions(
            base_options=task_modules.base_options.BaseOptions(
                model_asset_path=str(model_path),
                delegate=task_modules.base_options.BaseOptions.Delegate.CPU,
            ),
            running_mode=task_modules.running_mode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = task_modules.hand_landmarker.HandLandmarker.create_from_options(
            options
        )
        self._image_module = task_modules.image
        self._connections = _normalize_connections(
            task_modules.hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
        )

    @property
    def connections(self) -> list[tuple[int, int]]:
        """Return the landmark connection graph used for drawing."""
        return self._connections

    def detect(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        """Detect hands in a BGR frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        if self._backend == "tasks":
            return self._detect_with_tasks(frame_rgb)

        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        detected: list[HandLandmarks] = []
        handedness_results = results.multi_handedness or []

        for index, raw_landmarks in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if index < len(handedness_results):
                classification = handedness_results[index].classification[0]
                label = classification.label
                score = float(classification.score)

            detected.append(
                HandLandmarks(
                    handedness=label,
                    score=score,
                    landmarks=[
                        Landmark(x=float(point.x), y=float(point.y), z=float(point.z))
                        for point in raw_landmarks.landmark
                    ],
                    raw_landmarks=raw_landmarks,
                )
            )

        return detected

    def _detect_with_tasks(self, frame_rgb: np.ndarray) -> list[HandLandmarks]:
        """Detect hands with the MediaPipe Tasks API."""
        image = self._image_module.Image(
            image_format=self._image_module.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        self._timestamp_ms += 1
        results = self._hands.detect_for_video(image, self._timestamp_ms)
        if not results.hand_landmarks:
            return []

        detected: list[HandLandmarks] = []
        handedness_results = results.handedness or []
        for index, hand_landmarks in enumerate(results.hand_landmarks):
            label = "Unknown"
            score = 0.0
            if index < len(handedness_results) and handedness_results[index]:
                category = handedness_results[index][0]
                label = category.category_name or "Unknown"
                score = float(category.score)

            detected.append(
                HandLandmarks(
                    handedness=label,
                    score=score,
                    landmarks=[
                        Landmark(x=float(point.x), y=float(point.y), z=float(point.z))
                        for point in hand_landmarks
                    ],
                    raw_landmarks=None,
                )
            )

        return detected

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self) -> HandTracker:
        """Return this tracker for context manager use."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release resources when leaving a context manager."""
        self.close()


def _load_hands_solution() -> Any:
    """Load MediaPipe Hands across supported package layouts."""
    hands = _load_legacy_hands_solution()
    if hands is not None:
        return hands

    version, location = _mediapipe_version_and_location()
    raise ImportError(
        "Could not load legacy MediaPipe Hands from this mediapipe installation "
        f"(version={version}, location={location})."
    )


def _load_legacy_hands_solution() -> Any | None:
    """Load the legacy MediaPipe Hands solution if this install provides it."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    solutions = getattr(mp, "solutions", None)
    hands = getattr(solutions, "hands", None)
    if hands is not None:
        return hands

    for module_name in (
        "mediapipe.solutions.hands",
        "mediapipe.python.solutions.hands",
    ):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            continue

    version = getattr(mp, "__version__", "unknown")
    location = getattr(mp, "__file__", "unknown")
    if hasattr(mp, "tasks"):
        return None
    raise ImportError(
        "MediaPipe is installed, but neither legacy solutions nor tasks are "
        f"available (version={version}, location={location})."
    )


@dataclass(frozen=True)
class _TasksModules:
    """Container for MediaPipe Tasks modules used by the tracker."""

    hand_landmarker: Any
    base_options: Any
    running_mode: Any
    image: Any


def _load_tasks_modules() -> _TasksModules:
    """Load MediaPipe Tasks modules for newer MediaPipe packages."""
    try:
        return _TasksModules(
            hand_landmarker=import_module(
                "mediapipe.tasks.python.vision.hand_landmarker"
            ),
            base_options=import_module("mediapipe.tasks.python.core.base_options"),
            running_mode=import_module(
                "mediapipe.tasks.python.vision.core.vision_task_running_mode"
            ).VisionTaskRunningMode,
            image=import_module("mediapipe.tasks.python.vision.core.image"),
        )
    except ModuleNotFoundError as exc:
        version, location = _mediapipe_version_and_location()
        raise ImportError(
            "Could not load MediaPipe Tasks Hand Landmarker modules "
            f"(version={version}, location={location})."
        ) from exc


def _ensure_model_asset(
    model_asset_path: str | None,
    model_asset_url: str | None,
) -> Path:
    """Return a local hand-landmarker model path, downloading it if needed."""
    if not model_asset_path:
        raise FileNotFoundError(
            "This MediaPipe version requires a hand-landmarker .task model. "
            "Set `model_asset_path` in config.yaml."
        )

    path = Path(model_asset_path)
    if path.exists():
        return path

    if not model_asset_url:
        raise FileNotFoundError(
            f"Missing model asset: {path}. Set `model_asset_url` in config.yaml "
            "or place the model file at `model_asset_path`."
        )

    print(f"Downloading MediaPipe hand model to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(model_asset_url, path)
    except OSError as exc:
        raise RuntimeError(
            f"Could not download MediaPipe hand model from {model_asset_url}. "
            f"Download it manually and save it to {path}."
        ) from exc

    return path


def _normalize_connections(connections: Any) -> list[tuple[int, int]]:
    """Convert MediaPipe connection objects to plain index tuples."""
    normalized: list[tuple[int, int]] = []
    for connection in connections:
        if hasattr(connection, "start") and hasattr(connection, "end"):
            normalized.append((int(connection.start), int(connection.end)))
        else:
            start, end = connection
            normalized.append((int(start), int(end)))
    return normalized


def _mediapipe_version_and_location() -> tuple[str, str]:
    """Return MediaPipe version and import location for error messages."""
    try:
        import mediapipe as mp
    except ImportError:
        return "not installed", "not installed"

    return (
        getattr(mp, "__version__", "unknown"),
        getattr(mp, "__file__", "unknown"),
    )
