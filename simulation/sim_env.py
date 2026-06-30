"""PyBullet simulation environment for the Allegro Hand right.

URDF joint structure (verified from joint info output):
  joint_0.0  .. joint_3.0  : finger 0 [spread, mcp, pip, dip]  lo=-0.47..1.61
  joint_4.0  .. joint_7.0  : finger 1 [spread, mcp, pip, dip]
  joint_8.0  .. joint_11.0 : finger 2 [spread, mcp, pip, dip]
  joint_12.0 .. joint_15.0 : thumb    [CMC, MCP_lat, MCP_flex, IP_flex]
  joint_12.0 lower limit = 0.263 (thumb CMC must be >= 0.263, never 0)

Motor control:
  Disable default velocity brake (VELOCITY_CONTROL, force=0) before arming
  position control. Use explicit positionGain/velocityGain, not just force.
  Tuned values: kp=10, kd=5.0, force=10.0
  (kd=5.0 keeps system overdamped even for large link inertia ~0.025 kg*m^2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

_DEFAULT_URDF = Path("simulation/assets/allegro/allegro_hand_right.urdf")
_NUM_JOINTS = 16

SIM_WINDOW_X = 660
SIM_WINDOW_Y = 30

_KP = 10.0
_KD = 5.0
_MAX_FORCE = 10.0

# Thumb CMC (joint 12) lower limit is 0.263 -- it can never reach 0.
_THUMB_CMC_NEUTRAL = 0.263

# Neutral pose: fingers open, thumb naturally opposed.
# Spread joints (0, 4, 8) at 0; flexion joints at 0; thumb CMC at 0.263.
_NEUTRAL_POSE = np.array([
    0.0, 0.0, 0.0, 0.0,   # finger 0: spread=0, mcp=0, pip=0, dip=0
    0.0, 0.0, 0.0, 0.0,   # finger 1
    0.0, 0.0, 0.0, 0.0,   # finger 2
    _THUMB_CMC_NEUTRAL, 0.0, 0.0, 0.0,  # thumb: CMC=0.263, lat=0, mcp=0, ip=0
], dtype=np.float64)


class AllegroSimEnv:
    """Minimal PyBullet wrapper for the Allegro Hand right."""

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

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Drive all 16 revolute joints to the given angles (radians).

        Joint order matches URDF enumeration:
          0-3  = finger 0 [spread, mcp, pip, dip]
          4-7  = finger 1
          8-11 = finger 2
          12-15 = thumb [CMC, MCP_lat, MCP_flex, IP_flex]
        """
        if len(angles) != _NUM_JOINTS:
            raise ValueError(f"Expected {_NUM_JOINTS} joint angles, got {len(angles)}.")
        for allegro_idx, joint_idx in enumerate(self._joint_indices):
            pb.setJointMotorControl2(
                self._hand_id, joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=float(angles[allegro_idx]),
                positionGain=_KP,
                velocityGain=_KD,
                force=_MAX_FORCE,
            )

    def step(self) -> None:
        pb.stepSimulation()

    def get_joint_angles(self) -> np.ndarray:
        states = pb.getJointStates(self._hand_id, self._joint_indices)
        return np.array([s[0] for s in states], dtype=np.float32)

    def reset(self) -> None:
        for i, joint_idx in enumerate(self._joint_indices):
            pb.resetJointState(self._hand_id, joint_idx, _NEUTRAL_POSE[i])
        self._arm_position_controllers(_NEUTRAL_POSE)

    def close(self) -> None:
        if pb.isConnected(self._client):
            pb.disconnect(self._client)

    def __enter__(self) -> AllegroSimEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load_hand(self) -> int:
        pb.setAdditionalSearchPath(str(self._urdf_path.parent))
        return pb.loadURDF(
            str(self._urdf_path),
            basePosition=[0.0, 0.0, 0.2],
            baseOrientation=pb.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            # self-collision disabled: causes inter-finger impulses that corrupt joint control
        )

    def _collect_revolute_joints(self) -> list[int]:
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
        """Snap to neutral, disable velocity brakes, arm PD controllers."""
        for i, joint_idx in enumerate(self._joint_indices):
            pb.resetJointState(self._hand_id, joint_idx, _NEUTRAL_POSE[i])
        for joint_idx in self._joint_indices:
            pb.setJointMotorControl2(
                self._hand_id, joint_idx,
                pb.VELOCITY_CONTROL, force=0.0,
            )
        self._arm_position_controllers(_NEUTRAL_POSE)

    def _arm_position_controllers(self, targets: np.ndarray) -> None:
        for i, joint_idx in enumerate(self._joint_indices):
            pb.setJointMotorControl2(
                self._hand_id, joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=float(targets[i]),
                positionGain=_KP,
                velocityGain=_KD,
                force=_MAX_FORCE,
            )

    def _configure_gui(self) -> None:
        pb.resetDebugVisualizerCamera(
            cameraDistance=0.4, cameraYaw=35, cameraPitch=-20,
            cameraTargetPosition=[0.0, 0.0, 0.2],
        )
        pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        try:
            import ctypes, threading, time as _time
            def _move_window() -> None:
                _time.sleep(1.5)
                hwnd = ctypes.windll.user32.FindWindowW(
                    None,
                    "Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build",
                )
                if not hwnd:
                    hwnd = ctypes.windll.user32.FindWindowW(None, "OpenGL 3+")
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, None, SIM_WINDOW_X, SIM_WINDOW_Y, 0, 0, 0x0001 | 0x0004,
                    )
            threading.Thread(target=_move_window, daemon=True).start()
        except Exception:
            pass