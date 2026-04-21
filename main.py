"""Milestone 1 entry point: webcam hand landmarks and optional recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from cv.camera import open_camera, read_frame
from cv.hand_tracker import HandTracker
from cv.smoothing import passthrough
from data.recorder import LandmarkRecorder
from viz.overlay import draw_hands


CONFIG_PATH = Path("config.yaml")
WINDOW_NAME = "Vision-Guided Robot Control - Hand Tracking"


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the perception MVP."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    model_asset_path: str = "models/hand_landmarker.task"
    model_asset_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    draw_landmarks: bool = True
    record_landmarks: bool = False
    output_path: str = "data/sessions/landmarks.jsonl"


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load application config from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    return AppConfig(**raw_config)


def run(config: AppConfig) -> None:
    """Run the webcam hand-landmark visualization loop."""
    capture = None
    recorder = LandmarkRecorder(config.output_path) if config.record_landmarks else None

    try:
        if recorder is not None:
            recorder.open()

        with HandTracker(
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_asset_path=config.model_asset_path,
            model_asset_url=config.model_asset_url,
        ) as tracker:
            capture = open_camera(
                config.camera_index,
                config.frame_width,
                config.frame_height,
            )

            while True:
                frame = read_frame(capture)
                if frame is None:
                    print("Warning: empty camera frame; skipping.")
                    continue

                hands = passthrough(tracker.detect(frame))

                if recorder is not None:
                    recorder.record(hands)

                if config.draw_landmarks:
                    frame = draw_hands(frame, hands, tracker.connections)

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if recorder is not None:
            recorder.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Load config and start the Milestone 1 app."""
    try:
        config = load_config()
        run(config)
    except (FileNotFoundError, RuntimeError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
