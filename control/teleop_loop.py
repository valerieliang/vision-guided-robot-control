"""Real-time teleoperation loop.

Wires the full pipeline at a fixed control rate:

  Camera → HandTracker → retarget() → AllegroController → AllegroSimEnv

Changes from v1:
  - Camera window pinned to top-left; PyBullet GUI spawns to the right.
  - Handedness label ("YOUR LEFT/RIGHT HAND") burned into camera feed.
  - Target Hz lowered to a CPU-safe default (15 Hz); configurable.
  - PyBullet disconnect guard: loop exits cleanly if the GUI is closed.
  - Overrun threshold raised to 3× budget to reduce log noise at startup.
  - Physics sub-steps reduced from 8 to 4 to cut per-loop sim cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as pb

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
    max_num_hands: int = 1
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
    gravity: float = 0.0

    # Control — 15 Hz is safe on CPU-only; raise if you have a fast machine
    target_hz: float = 15.0
    limits_path: str = "kinematics/joint_limits.yaml"

    # Display
    show_camera: bool = True
    camera_window: str = "Hand Tracking"

    # Camera window screen position (pixels from top-left of monitor)
    camera_window_x: int = 0
    camera_window_y: int = 30
    # Display size for the camera window (independent of capture resolution)
    display_width: int = 640
    display_height: int = 360

    # Latency logging
    log_latency: bool = True
    latency_window: int = 30


class TeleopLoop:
    """Full teleoperation pipeline."""

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

        # Pin camera window to the left side of the screen.
        if cfg.show_camera:
            cv2.namedWindow(cfg.camera_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cfg.camera_window, cfg.display_width, cfg.display_height)
            cv2.moveWindow(cfg.camera_window, cfg.camera_window_x, cfg.camera_window_y)

    def _teardown(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
        if self._capture is not None:
            self._capture.release()
        if self._sim is not None:
            self._sim.close()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Block until the user presses 'q', closes the PyBullet window, or Ctrl+C."""
        cfg = self._cfg
        period = 1.0 / cfg.target_hz

        while True:
            t0 = time.perf_counter()

            # Exit cleanly if the PyBullet GUI was closed by the user.
            if cfg.gui and not pb.isConnected():
                print("[teleop] PyBullet window closed — stopping.")
                break

            # ── 1. Capture ────────────────────────────────────────────
            frame = read_frame(self._capture)
            if frame is None:
                print("Warning: empty frame, skipping.")
                continue

            # ── 2. Detect ─────────────────────────────────────────────
            hands = passthrough(self._tracker.detect(frame))

            # ── 3. Retarget ───────────────────────────────────────────
            if hands:
                joint_angles_list = retarget_all(hands[:1])
                flat = joint_angles_list[0].as_flat_array()
                allegro_cmd = self._controller.retargeted_to_allegro(flat)
                active_hand = hands[0]
            else:
                allegro_cmd = self._controller.retargeted_to_allegro(
                    np.zeros(19, dtype=np.float32)
                )
                active_hand = None

            # ── 4. Simulate ───────────────────────────────────────────
            try:
                self._sim.set_joint_angles(allegro_cmd)
                for _ in range(4):   # 4 sub-steps at 15 Hz ≈ 60 Hz internal
                    self._sim.step()
            except Exception:
                print("[teleop] Physics server disconnected — stopping.")
                break

            # ── 5. Display ────────────────────────────────────────────
            if cfg.show_camera:
                display = cv2.resize(
                    frame, (cfg.display_width, cfg.display_height)
                )
                # Scale connections to match the smaller display frame.
                display = draw_hands(display, hands, self._tracker.connections)
                fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                _draw_hud(display, fps, active_hand)
                cv2.imshow(cfg.camera_window, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── 6. Timing ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t0

            if self._monitor is not None:
                self._monitor.record(elapsed)
                if len(self._monitor._samples) == cfg.latency_window:
                    print(f"[teleop] {self._monitor}")

            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif elapsed > period * 3.0:
                print(
                    f"[teleop] severe overrun: {elapsed * 1000:.1f} ms "
                    f"(budget {period * 1000:.1f} ms)"
                )

    def __enter__(self) -> TeleopLoop:
        self._setup()
        return self

    def __exit__(self, *_: object) -> None:
        self._teardown()


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    fps: float,
    active_hand: object | None,
) -> None:
    """Burn FPS, hand label, and control status onto the display frame."""
    h, w = frame.shape[:2]

    # FPS — top left
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA,
    )

    if active_hand is not None:
        handedness = getattr(active_hand, "handedness", "Unknown")
        # MediaPipe labels from camera POV (mirrored), so we flip for the user.
        if handedness == "Left":
            display_label = "YOUR RIGHT HAND"
            colour = (0, 220, 255)
        elif handedness == "Right":
            display_label = "YOUR LEFT HAND"
            colour = (255, 180, 0)
        else:
            display_label = "HAND DETECTED"
            colour = (200, 200, 200)

        (tw, _), _ = cv2.getTextSize(
            display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.putText(
            frame, display_label,
            (w - tw - 12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA,
        )

        cv2.putText(
            frame, "SIM: ACTIVE",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "NO HAND DETECTED",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA,
        )

    # Quit hint — bottom right
    hint = "Q to quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame, hint,
        (w - hw - 12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )