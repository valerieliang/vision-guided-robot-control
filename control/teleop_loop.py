"""Real-time teleoperation loop.

Wires the full pipeline at a fixed control rate:

  Camera → HandTracker → retarget() → AllegroController → AllegroSimEnv

The loop runs at ``target_hz`` (default 30 Hz).  If processing takes longer
than one period, it skips the sleep and logs a timing warning rather than
falling behind silently.

Usage (replace main.py for Milestone 2):

    from control.teleop_loop import TeleopLoop, TeleopConfig
    cfg = TeleopConfig()
    with TeleopLoop(cfg) as loop:
        loop.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from control.latency_monitor import LatencyMonitor
from cv.camera import open_camera, read_frame
from cv.hand_tracker import HandTracker
from cv.smoothing import passthrough
from kinematics.retargeting import retarget_all
from simulation.robot_controller import AllegroController
from simulation.sim_env import AllegroSimEnv
from viz.overlay import draw_hands


@dataclass
class TeleopConfig:
    """All runtime settings for the teleoperation loop."""

    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Hand tracker
    max_num_hands: int = 1          # only need one hand for teleoperation
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    model_asset_path: str = "models/hand_landmarker.task"
    model_asset_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

    # Simulation
    urdf_path: str = "simulation/assets/allegro/allegro_hand_right.urdf"
    gui: bool = True
    gravity: float = 0.0           # disable gravity — hand is mounted

    # Control
    target_hz: float = 30.0
    limits_path: str = "kinematics/joint_limits.yaml"

    # Display
    show_camera: bool = True        # show the CV overlay window
    camera_window: str = "Hand Tracking — Teleop"

    # Latency logging
    log_latency: bool = True
    latency_window: int = 60        # frames to average over


class TeleopLoop:
    """Full teleoperation pipeline.

    Parameters
    ----------
    config:
        A ``TeleopConfig`` instance.  All fields have sensible defaults.
    """

    def __init__(self, config: TeleopConfig | None = None) -> None:
        self._cfg = config or TeleopConfig()
        self._capture = None
        self._tracker: HandTracker | None = None
        self._sim: AllegroSimEnv | None = None
        self._controller: AllegroController | None = None
        self._monitor: LatencyMonitor | None = None

    def _setup(self) -> None:
        cfg = self._cfg

        self._tracker = HandTracker(
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            model_asset_path=cfg.model_asset_path,
            model_asset_url=cfg.model_asset_url,
        )

        self._capture = open_camera(
            cfg.camera_index, cfg.frame_width, cfg.frame_height
        )

        self._sim = AllegroSimEnv(
            urdf_path=cfg.urdf_path,
            gui=cfg.gui,
            gravity=cfg.gravity,
        )

        self._controller = AllegroController(
            limits_path=cfg.limits_path,
            control_rate_hz=cfg.target_hz,
        )

        if cfg.log_latency:
            self._monitor = LatencyMonitor(window=cfg.latency_window)

    def _teardown(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
        if self._capture is not None:
            self._capture.release()
        if self._sim is not None:
            self._sim.close()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Block until the user presses 'q' or closes the window."""
        cfg = self._cfg
        period = 1.0 / cfg.target_hz

        while True:
            t0 = time.perf_counter()

            # ── 1. Capture ────────────────────────────────────────────
            frame = read_frame(self._capture)
            if frame is None:
                print("Warning: empty frame, skipping.")
                continue

            # ── 2. Detect ─────────────────────────────────────────────
            hands = passthrough(self._tracker.detect(frame))

            # ── 3. Retarget ───────────────────────────────────────────
            if hands:
                # Use only the first detected hand for control.
                joint_angles_list = retarget_all(hands[:1])
                flat = joint_angles_list[0].as_flat_array()
                allegro_cmd = self._controller.retargeted_to_allegro(flat)
            else:
                # No hand detected — hold previous position (controller
                # keeps _prev; just send that).
                allegro_cmd = self._controller.retargeted_to_allegro(
                    np.zeros(19, dtype=np.float32)
                )

            # ── 4. Simulate ───────────────────────────────────────────
            self._sim.set_joint_angles(allegro_cmd)
            # Step physics several times per control cycle to keep sim stable
            # at 240 Hz internal rate while the loop runs at 30 Hz.
            for _ in range(8):
                self._sim.step()

            # ── 5. Display ────────────────────────────────────────────
            if cfg.show_camera:
                frame = draw_hands(frame, hands, self._tracker.connections)
                _draw_fps_overlay(frame, 1.0 / max(time.perf_counter() - t0, 1e-6))
                cv2.imshow(cfg.camera_window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── 6. Timing ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t0

            if self._monitor is not None:
                self._monitor.record(elapsed)

            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif elapsed > period * 1.5:
                print(
                    f"[teleop] loop overrun: {elapsed * 1000:.1f} ms "
                    f"(budget {period * 1000:.1f} ms)"
                )

    def __enter__(self) -> TeleopLoop:
        self._setup()
        return self

    def __exit__(self, *_: object) -> None:
        self._teardown()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_fps_overlay(frame: np.ndarray, fps: float) -> None:
    """Burn FPS into the top-left corner of the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 80),
        2,
        cv2.LINE_AA,
    )
