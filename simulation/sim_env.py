"""PyBullet simulation environment for the Allegro Hand right.

Loads the URDF, exposes a clean set_joint_angles() interface, and runs
the physics step.  Mesh paths in the URDF are relative to the URDF file's
own directory, so we set the PyBullet search path accordingly.

Window layout
-------------
PyBullet's GUI window is positioned to the RIGHT of the camera feed.
Assuming the camera window is 640 px wide at x=0, the sim window starts
at x=660.  Adjust SIM_WINDOW_X if your screen layout differs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

_DEFAULT_URDF = Path("simulation/assets/allegro/allegro_hand_right.urdf")
_NUM_JOINTS = 16

# Screen position for the PyBullet GUI (pixels from top-left).
# Place it to the right of the 640-wide camera window.
SIM_WINDOW_X = 660
SIM_WINDOW_Y = 30

# Allegro thumb joint 12 (CMC) has URDF limits lower=0.263, upper=1.396.
# Sending 0 parks the thumb fully retracted behind the palm — unintuitive.
# We initialise it at the midpoint so it starts in a natural opposition pose.
_THUMB_CMC_NEUTRAL = 0.829   # (0.263 + 1.396) / 2

# Full neutral pose: fingers open, thumb naturally opposed.
# Order matches _collect_revolute_joints() — i.e. Allegro joint 0–15.
_NEUTRAL_POSE = np.array([
    # pinky  (joints 0–3):  spread=0, flexion=0
     0.0,  0.0,  0.0,  0.0,
    # ring   (joints 4–7)
     0.0,  0.0,  0.0,  0.0,
    # middle (joints 8–11)
     0.0,  0.0,  0.0,  0.0,
    # index  (joints 12–15) — NOTE: Allegro finger 3 is index
     0.0,  0.0,  0.0,  0.0,
], dtype=np.float64)
# Thumb is joints 12–15 in the URDF joint numbering used by robot_controller.
# sim_env receives the already-mapped 16-element array from AllegroController,
# so we don't need to special-case thumb here — AllegroController handles it.


class AllegroSimEnv:
    """Minimal PyBullet wrapper for the Allegro Hand right.

    Parameters
    ----------
    urdf_path:
        Path to ``allegro_hand_right.urdf``.
    gui:
        If True, open the PyBullet GUI window.
    timestep:
        Physics timestep in seconds (default 1/240 s).
    gravity:
        Gravitational acceleration (m/s²).  0 = disabled (hand floats).
    """

    def __init__(
        self,
        urdf_path: str | Path = _DEFAULT_URDF,
        gui: bool = True,
        timestep: float = 1.0 / 240.0,
        gravity: float = 0.0,
    ) -> None:
        self._urdf_path = Path(urdf_path).resolve()
        self._timestep = timestep
        self._gui = gui

        mode = pb.GUI if gui else pb.DIRECT
        self._client = pb.connect(mode)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -gravity)
        pb.setTimeStep(timestep)

        if gui:
            self._configure_gui()

        self._hand_id = self._load_hand()
        self._joint_indices = self._collect_revolute_joints()
        self._init_controllers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Drive all 16 revolute joints to the given angles (radians).

        Parameters
        ----------
        angles:
            Length-16 array produced by ``AllegroController.retargeted_to_allegro()``.
            Allegro joint order: 0–3 pinky, 4–7 ring, 8–11 middle, 12–15 thumb.
        """
        if len(angles) != _NUM_JOINTS:
            raise ValueError(
                f"Expected {_NUM_JOINTS} joint angles, got {len(angles)}."
            )
        for allegro_idx, joint_idx in enumerate(self._joint_indices):
            pb.setJointMotorControl2(
                self._hand_id,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=float(angles[allegro_idx]),
                force=0.8,
                maxVelocity=2.0,
            )

    def step(self) -> None:
        """Advance the simulation by one physics timestep."""
        pb.stepSimulation()

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint positions (radians) for all 16 joints."""
        states = pb.getJointStates(self._hand_id, self._joint_indices)
        return np.array([s[0] for s in states], dtype=np.float32)

    def reset(self) -> None:
        """Reset all joints to the neutral open-hand pose."""
        for i, joint_idx in enumerate(self._joint_indices):
            pb.resetJointState(self._hand_id, joint_idx, _NEUTRAL_POSE[i])

    def close(self) -> None:
        """Disconnect from PyBullet."""
        if pb.isConnected(self._client):
            pb.disconnect(self._client)

    def __enter__(self) -> AllegroSimEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hand(self) -> int:
        """Load the Allegro URDF and return its body ID."""
        pb.setAdditionalSearchPath(str(self._urdf_path.parent))
        hand_id = pb.loadURDF(
            str(self._urdf_path),
            basePosition=[0.0, 0.0, 0.2],
            baseOrientation=pb.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=pb.URDF_USE_SELF_COLLISION,
        )
        return hand_id

    def _collect_revolute_joints(self) -> list[int]:
        """Return PyBullet joint indices for all revolute joints in order."""
        revolute: list[int] = []
        for i in range(pb.getNumJoints(self._hand_id)):
            info = pb.getJointInfo(self._hand_id, i)
            if info[2] == pb.JOINT_REVOLUTE:
                revolute.append(i)
        if len(revolute) != _NUM_JOINTS:
            raise RuntimeError(
                f"Expected {_NUM_JOINTS} revolute joints, found {len(revolute)}."
            )
        return revolute

    def _init_controllers(self) -> None:
        """Initialise position controllers and set the neutral pose."""
        for i, joint_idx in enumerate(self._joint_indices):
            pb.resetJointState(self._hand_id, joint_idx, _NEUTRAL_POSE[i])
            pb.setJointMotorControl2(
                self._hand_id,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=_NEUTRAL_POSE[i],
                force=0.8,
            )

    def _configure_gui(self) -> None:
        """Configure camera viewpoint and move the window to the right side."""
        pb.resetDebugVisualizerCamera(
            cameraDistance=0.4,
            cameraYaw=35,
            cameraPitch=-20,
            cameraTargetPosition=[0.0, 0.0, 0.2],
        )
        # Disable default mouse picking panel to reduce visual clutter.
        pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

        # Position the window to the right of the camera feed.
        # PyBullet doesn't have a Python API for window position, but we can
        # set it via the OpenGL window title argument on some platforms.
        # On Windows the reliable method is to use the win32 API post-spawn.
        try:
            import ctypes
            import threading
            import time as _time

            def _move_window() -> None:
                _time.sleep(1.5)   # wait for the GL window to appear
                hwnd = ctypes.windll.user32.FindWindowW(None, "Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build")
                if not hwnd:
                    # Try alternate title seen on some PyBullet versions
                    hwnd = ctypes.windll.user32.FindWindowW(None, "OpenGL 3+")
                if hwnd:
                    # SWP_NOSIZE = 0x0001, SWP_NOZORDER = 0x0004
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, None,
                        SIM_WINDOW_X, SIM_WINDOW_Y,
                        0, 0,
                        0x0001 | 0x0004,
                    )

            threading.Thread(target=_move_window, daemon=True).start()
        except Exception:
            pass   # Non-Windows or ctypes unavailable — window stays wherever PyBullet puts it
