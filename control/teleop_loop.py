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
from typing import Any

import cv2
import numpy as np

from control.latency_monitor import LatencyMonitor
from cv.camera import open_camera, read_frame
from cv.hand_tracker import HandTracker
from cv.smoothing import AngleSmoother, HandLandmarkSmoother
from kinematics.retargeting import retarget_all
from neurotech.decoders import NeuroIntent, NeuroReplaySource, SyntheticNeuroSource
from neurotech.intent_mapping import NeuroHandPoseMapper
from simulation.robot_controller import AllegroController
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
    target_hz: float = 60.0
    limits_path: str = "kinematics/joint_limits.yaml"
    input_source: str = "hand"

    # Smoothing / dropout handling
    landmark_smoothing_alpha: float = 0.45
    hand_min_score: float = 0.35
    hand_hold_frames: int = 3
    angle_smoothing_alpha: float = 0.35
    angle_deadband_rad: float = 0.015
    angle_max_step_rad: float = 0.22

    # Neurotech decoded-data replay
    neuro_replay_path: str | None = None
    neuro_replay_interval_s: float = 0.35
    neuro_confidence_threshold: float = 0.55

    # Display
    show_camera: bool = True
    camera_window: str = "Hand Tracking"
    handedness_display: str = "mediapipe"

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
        self._sim: Any | None = None
        self._controller: AllegroController | None = None
        self._monitor: LatencyMonitor | None = None
        self._hand_smoother: HandLandmarkSmoother | None = None
        self._angle_smoother: AngleSmoother | None = None
        self._neuro_source: NeuroReplaySource | SyntheticNeuroSource | None = None
        self._neuro_mapper: NeuroHandPoseMapper | None = None
        self._last_neuro_intent: NeuroIntent | None = None
        self._pb: Any | None = None

    def _setup(self) -> None:
        cfg = self._cfg
        self._validate_input_source()

        if self._uses_camera:
            self._tracker = HandTracker(
                max_num_hands=cfg.max_num_hands,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                model_asset_path=cfg.model_asset_path,
                model_asset_url=cfg.model_asset_url,
            )
            self._hand_smoother = HandLandmarkSmoother(
                alpha=cfg.landmark_smoothing_alpha,
                min_score=cfg.hand_min_score,
                hold_frames=cfg.hand_hold_frames,
            )
            self._capture = open_camera(
                cfg.camera_index, cfg.frame_width, cfg.frame_height
            )

        self._angle_smoother = AngleSmoother(
            alpha=cfg.angle_smoothing_alpha,
            deadband_rad=cfg.angle_deadband_rad,
            max_step_rad=cfg.angle_max_step_rad,
        )

        if self._uses_neurotech:
            self._neuro_mapper = NeuroHandPoseMapper(cfg.neuro_confidence_threshold)
            if cfg.input_source == "synthetic_neurotech":
                self._neuro_source = SyntheticNeuroSource(
                    interval_s=cfg.neuro_replay_interval_s
                )
            elif cfg.neuro_replay_path:
                self._neuro_source = NeuroReplaySource(
                    cfg.neuro_replay_path,
                    interval_s=cfg.neuro_replay_interval_s,
                )
            else:
                self._neuro_source = SyntheticNeuroSource(
                    interval_s=cfg.neuro_replay_interval_s
                )

        self._sim = self._create_sim_backend()

        self._controller = AllegroController(
            limits_path=cfg.limits_path,
            control_rate_hz=cfg.target_hz,
        )

        if cfg.log_latency:
            self._monitor = LatencyMonitor(window=cfg.latency_window)

        # Pin camera window to the left side of the screen.
        if cfg.show_camera and self._uses_camera:
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
            if self._pb is not None and cfg.gui and not self._pb.isConnected():
                print("[teleop] PyBullet window closed — stopping.")
                break

            frame, hands, active_hand = self._read_smoothed_hands()
            flat, source_label = self._next_retargeted_command(hands)
            flat = self._angle_smoother.update(flat)
            allegro_cmd = self._controller.retargeted_to_allegro(flat)

            # ── 4. Simulate ───────────────────────────────────────────
            try:
                self._sim.set_joint_angles(allegro_cmd)
                for _ in range(4):   # 4 sub-steps at 15 Hz ≈ 60 Hz internal
                    self._sim.step()
            except Exception:
                print("[teleop] Physics server disconnected — stopping.")
                break

            # ── 5. Display ────────────────────────────────────────────
            if cfg.show_camera and self._uses_camera and frame is not None:
                display = cv2.resize(
                    frame, (cfg.display_width, cfg.display_height)
                )
                # Scale connections to match the smaller display frame.
                display = draw_hands(display, hands, self._tracker.connections)
                fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
                _draw_hud(
                    display,
                    fps,
                    active_hand,
                    source_label,
                    self._last_neuro_intent,
                    cfg.handedness_display,
                )
                cv2.imshow(cfg.camera_window, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif not self._uses_camera:
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

    @property
    def _uses_camera(self) -> bool:
        return self._cfg.input_source in {"hand", "hybrid"}

    @property
    def _uses_neurotech(self) -> bool:
        return self._cfg.input_source in {
            "neurotech_replay",
            "synthetic_neurotech",
            "hybrid",
        }

    def _validate_input_source(self) -> None:
        valid = {"hand", "neurotech_replay", "synthetic_neurotech", "hybrid"}
        if self._cfg.input_source not in valid:
            raise ValueError(
                f"Unsupported input_source {self._cfg.input_source!r}; "
                f"expected one of {sorted(valid)}."
            )
        valid_handedness = {"mediapipe", "mirrored"}
        if self._cfg.handedness_display not in valid_handedness:
            raise ValueError(
                f"Unsupported handedness_display {self._cfg.handedness_display!r}; "
                f"expected one of {sorted(valid_handedness)}."
            )

    def _create_sim_backend(self) -> Any:
        cfg = self._cfg
        try:
            import pybullet as pb

            from simulation.sim_env import AllegroSimEnv
        except ImportError as exc:
            raise RuntimeError(
                "The previous-commit visualization uses PyBullet and the "
                "Allegro URDF. Install pybullet to run it."
            ) from exc
        self._pb = pb
        return AllegroSimEnv(
            urdf_path=cfg.urdf_path,
            gui=cfg.gui,
            gravity=cfg.gravity,
        )

    def _read_smoothed_hands(
        self,
    ) -> tuple[np.ndarray | None, list[object], object | None]:
        if not self._uses_camera:
            return None, [], None
        frame = read_frame(self._capture)
        if frame is None:
            print("Warning: empty frame, skipping.")
            return None, [], None
        detected = self._tracker.detect(frame)
        hands = self._hand_smoother.update(detected)
        active_hand = hands[0] if hands else None
        return frame, hands, active_hand

    def _next_retargeted_command(self, hands: list[object]) -> tuple[np.ndarray, str]:
        if hands:
            joint_angles_list = retarget_all(hands[:1])
            return joint_angles_list[0].as_flat_array(), "hand"
        if self._uses_neurotech:
            intent = self._neuro_source.next_intent()
            self._last_neuro_intent = intent
            return self._neuro_mapper.to_retargeted(intent), intent.normalized_label()
        return np.zeros(19, dtype=np.float32), "open"


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    fps: float,
    active_hand: object | None,
    source_label: str,
    neuro_intent: NeuroIntent | None,
    handedness_display: str,
) -> None:
    """Burn FPS, hand label, and control status onto the display frame."""
    h, w = frame.shape[:2]

    # FPS — top left
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"CTRL: {source_label}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    if neuro_intent is not None:
        cv2.putText(
            frame,
            f"NEURO: {neuro_intent.normalized_label()} {neuro_intent.confidence:.2f}",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 190, 255),
            2,
            cv2.LINE_AA,
        )

    if active_hand is not None:
        handedness = getattr(active_hand, "handedness", "Unknown")
        user_handedness = _display_handedness(handedness, handedness_display)
        if user_handedness == "Left":
            display_label = "LEFT HAND"
            colour = (0, 220, 255)
        elif user_handedness == "Right":
            display_label = "RIGHT HAND"
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


def _display_handedness(handedness: str, mode: str) -> str:
    """Return the HUD handedness label.

    MediaPipe's handedness can be interpreted differently depending on whether
    the camera feed is mirrored. The default keeps MediaPipe's anatomical label.
    """
    if mode != "mirrored":
        return handedness
    if handedness == "Left":
        return "Right"
    if handedness == "Right":
        return "Left"
    return handedness