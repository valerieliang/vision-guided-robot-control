"""PyBullet simulation environment for the Allegro Hand right.

Loads the URDF, exposes a clean set_joint_angles() interface, and runs
the physics step.  Mesh paths in the URDF are relative to the URDF file's
own directory, so we set the PyBullet search path accordingly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

# Path to the URDF relative to the project root.
_DEFAULT_URDF = Path("simulation/assets/allegro/allegro_hand_right.urdf")

# Allegro revolute joint indices in the loaded model (0-indexed as PyBullet
# reports them).  Fixed joints (the four *_tip joints) are excluded — PyBullet
# skips them in its joint index sequence, so the 16 revolute joints map to
# indices 0–15 directly.
_NUM_JOINTS = 16


class AllegroSimEnv:
    """Minimal PyBullet wrapper for the Allegro Hand right.

    Parameters
    ----------
    urdf_path:
        Path to ``allegro_hand_right.urdf``.  Defaults to the project-standard
        location ``simulation/assets/allegro/allegro_hand_right.urdf``.
    gui:
        If True, open the PyBullet GUI window.  Pass False for headless use.
    timestep:
        Physics timestep in seconds.  Default matches a 240 Hz sim (PyBullet
        default); the teleop loop steps this manually at ~30 Hz.
    gravity:
        Gravitational acceleration (m/s²).  Set to 0 to disable gravity so
        the unsupported hand doesn't fall.
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

        # Connect to PyBullet.
        mode = pb.GUI if gui else pb.DIRECT
        self._client = pb.connect(mode)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -gravity)
        pb.setTimeStep(timestep)

        if gui:
            self._configure_camera()

        self._hand_id = self._load_hand()
        self._joint_indices = self._collect_revolute_joints()

        # Enable position control on all revolute joints with low forces so
        # the hand moves smoothly without snapping.
        for idx in self._joint_indices:
            pb.setJointMotorControl2(
                self._hand_id,
                idx,
                pb.POSITION_CONTROL,
                targetPosition=0.0,
                force=0.5,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Drive all 16 revolute joints to the given angles (radians).

        Parameters
        ----------
        angles:
            A length-16 array in Allegro joint order:
            joints 0–3   → pinky   (index 0 = MCP spread, 1–3 = flexion)
            joints 4–7   → ring
            joints 8–11  → middle
            joints 12–15 → thumb
            This matches the order produced by
            ``robot_controller.AllegroController.retargeted_to_allegro()``.
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
                force=0.5,
                maxVelocity=2.0,
            )

    def step(self) -> None:
        """Advance the simulation by one physics timestep."""
        pb.stepSimulation()

    def get_joint_angles(self) -> np.ndarray:
        """Return the current joint positions (radians) for all 16 joints."""
        states = pb.getJointStates(self._hand_id, self._joint_indices)
        return np.array([s[0] for s in states], dtype=np.float32)

    def reset(self) -> None:
        """Reset all joints to the neutral open-hand pose."""
        for idx in self._joint_indices:
            pb.resetJointState(self._hand_id, idx, 0.0)

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
        # PyBullet resolves relative mesh paths from the URDF's own directory.
        pb.setAdditionalSearchPath(str(self._urdf_path.parent))

        hand_id = pb.loadURDF(
            str(self._urdf_path),
            basePosition=[0.0, 0.0, 0.2],   # lifted so it's visible
            baseOrientation=pb.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=pb.URDF_USE_SELF_COLLISION,
        )
        return hand_id

    def _collect_revolute_joints(self) -> list[int]:
        """Return PyBullet joint indices for all revolute joints in order."""
        revolute: list[int] = []
        num_joints = pb.getNumJoints(self._hand_id)
        for i in range(num_joints):
            info = pb.getJointInfo(self._hand_id, i)
            joint_type = info[2]
            if joint_type == pb.JOINT_REVOLUTE:
                revolute.append(i)

        if len(revolute) != _NUM_JOINTS:
            raise RuntimeError(
                f"Expected {_NUM_JOINTS} revolute joints in Allegro URDF, "
                f"found {len(revolute)}.  Check the URDF path and mesh replacement."
            )
        return revolute

    def _configure_camera(self) -> None:
        """Set a sensible default viewpoint for the GUI."""
        pb.resetDebugVisualizerCamera(
            cameraDistance=0.45,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.0, 0.0, 0.2],
        )
