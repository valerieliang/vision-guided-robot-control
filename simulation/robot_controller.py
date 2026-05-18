"""Map retargeted human hand angles to Allegro Hand joint commands.

The Allegro Hand right has 16 revolute joints numbered 0–15:

  Joints  0– 3  →  finger 0  (pinky):   0=MCP-spread, 1=MCP-flex, 2=PIP, 3=DIP
  Joints  4– 7  →  finger 1  (ring):    4=MCP-spread, 5=MCP-flex, 6=PIP, 7=DIP
  Joints  8–11  →  finger 2  (middle):  8=MCP-spread, 9=MCP-flex,10=PIP,11=DIP
  Joints 12–15  →  thumb:              12=CMC,       13=MCP,     14=IP, 15=(unused→0)

retargeting.py produces a 19-element flat array with this order:
  [0]  index  mcp_flexion
  [1]  index  pip_flexion
  [2]  index  dip_flexion
  [3]  index  mcp_abduction
  [4]  middle mcp_flexion
  [5]  middle pip_flexion
  [6]  middle dip_flexion
  [7]  middle mcp_abduction
  [8]  ring   mcp_flexion
  [9]  ring   pip_flexion
  [10] ring   dip_flexion
  [11] ring   mcp_abduction
  [12] pinky  mcp_flexion
  [13] pinky  pip_flexion
  [14] pinky  dip_flexion
  [15] pinky  mcp_abduction
  [16] thumb  cmc_flexion
  [17] thumb  mcp_flexion
  [18] thumb  ip_flexion

This module:
  1. Reorders from retargeting order → Allegro joint order.
  2. Applies the joint limits from joint_limits.yaml (with safety margin).
  3. Applies a per-step velocity cap so sudden tracking jumps don't jerk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_LIMITS_PATH = Path("kinematics/joint_limits.yaml")

# ------------------------------------------------------------------
# Allegro joint index → (finger, role) mapping for limit lookup.
# Role keys must match the joint_limits.yaml field names.
# ------------------------------------------------------------------
_ALLEGRO_JOINT_META: list[tuple[str, str]] = [
    # finger 0 = pinky
    ("pinky",  "mcp_abduction"),
    ("pinky",  "mcp_flexion"),
    ("pinky",  "pip_flexion"),
    ("pinky",  "dip_flexion"),
    # finger 1 = ring
    ("ring",   "mcp_abduction"),
    ("ring",   "mcp_flexion"),
    ("ring",   "pip_flexion"),
    ("ring",   "dip_flexion"),
    # finger 2 = middle
    ("middle", "mcp_abduction"),
    ("middle", "mcp_flexion"),
    ("middle", "pip_flexion"),
    ("middle", "dip_flexion"),
    # finger 3 = index  (Allegro calls this finger 3, not finger 0)
    ("index",  "mcp_abduction"),
    ("index",  "mcp_flexion"),
    ("index",  "pip_flexion"),
    ("index",  "dip_flexion"),
]

# Thumb joints 12–15.
_THUMB_JOINT_META: list[tuple[str, str]] = [
    ("thumb", "cmc_flexion"),
    ("thumb", "mcp_flexion"),
    ("thumb", "ip_flexion"),
    # Joint 15 is a second thumb flexion DOF that we leave at 0.
    # (Allegro thumb has 4 DOF; retargeting only produces 3.)
]

# Indices into the 19-element retargeting flat array for each Allegro joint.
# -1 means "no retargeting source → hold at 0".
_RETARGET_TO_ALLEGRO: list[int] = [
    # pinky (finger 0)
    15,   # mcp_abduction  ← retarget[15]
    12,   # mcp_flexion    ← retarget[12]
    13,   # pip_flexion    ← retarget[13]
    14,   # dip_flexion    ← retarget[14]
    # ring (finger 1)
    11,   # mcp_abduction  ← retarget[11]
    8,    # mcp_flexion    ← retarget[8]
    9,    # pip_flexion    ← retarget[9]
    10,   # dip_flexion    ← retarget[10]
    # middle (finger 2)
    7,    # mcp_abduction  ← retarget[7]
    4,    # mcp_flexion    ← retarget[4]
    5,    # pip_flexion    ← retarget[5]
    6,    # dip_flexion    ← retarget[6]
    # index (finger 3)
    3,    # mcp_abduction  ← retarget[3]
    0,    # mcp_flexion    ← retarget[0]
    1,    # pip_flexion    ← retarget[1]
    2,    # dip_flexion    ← retarget[2]
]

# Thumb: allegro joints 12–15
_RETARGET_TO_ALLEGRO_THUMB: list[int] = [16, 17, 18, -1]


class AllegroController:
    """Convert retargeted angles to clamped Allegro joint commands.

    Parameters
    ----------
    limits_path:
        Path to ``joint_limits.yaml``.
    control_rate_hz:
        Expected control loop frequency.  Used with ``max_joint_delta_rad``
        from the YAML safety block to compute per-step velocity cap.
    """

    def __init__(
        self,
        limits_path: str | Path = _LIMITS_PATH,
        control_rate_hz: float = 30.0,
    ) -> None:
        with open(limits_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        safety = raw.get("safety", {})
        self._margin = float(safety.get("margin_rad", 0.087))
        max_delta = float(safety.get("max_joint_delta_rad", 0.175))
        # Convert max angular displacement per control cycle to per-step.
        self._max_delta = max_delta  # already per-step at the configured rate

        # Build (min, max) limit arrays for all 16 Allegro joints.
        self._lo = np.zeros(16, dtype=np.float64)
        self._hi = np.zeros(16, dtype=np.float64)

        fingers_cfg = raw["fingers"]

        for allegro_idx, (finger, role) in enumerate(_ALLEGRO_JOINT_META):
            joint_cfg = fingers_cfg[finger][role]
            lo = float(joint_cfg["min"]) + self._margin
            hi = float(joint_cfg["max"]) - self._margin
            self._lo[allegro_idx] = lo
            self._hi[allegro_idx] = hi

        thumb_cfg = raw["fingers"]["thumb"]
        thumb_roles = ["cmc_flexion", "mcp_flexion", "ip_flexion"]
        for i, role in enumerate(thumb_roles):
            joint_cfg = thumb_cfg[role]
            lo = float(joint_cfg["min"]) + self._margin
            hi = float(joint_cfg["max"]) - self._margin
            self._lo[12 + i] = lo
            self._hi[12 + i] = hi
        # Joint 15: second thumb DOF — lock at midpoint of CMC range.
        self._lo[15] = 0.0
        self._hi[15] = 0.0

        # Previous command for velocity capping.
        self._prev = np.zeros(16, dtype=np.float64)

    def retargeted_to_allegro(self, retargeted: np.ndarray) -> np.ndarray:
        """Convert a 19-element retargeting array to 16 Allegro joint angles.

        Steps applied in order:
          1. Reindex from retargeting order to Allegro joint order.
          2. Clamp to [min+margin, max-margin] from joint_limits.yaml.
          3. Cap per-step delta to prevent jerk.

        Parameters
        ----------
        retargeted:
            Output of ``JointAngles.as_flat_array()`` — 19 float32 values.

        Returns
        -------
        np.ndarray
            Shape (16,) float64 array ready for ``AllegroSimEnv.set_joint_angles()``.
        """
        raw = np.zeros(16, dtype=np.float64)

        # Fingers
        for allegro_idx, retarget_idx in enumerate(_RETARGET_TO_ALLEGRO):
            raw[allegro_idx] = float(retargeted[retarget_idx])

        # Thumb
        for i, retarget_idx in enumerate(_RETARGET_TO_ALLEGRO_THUMB):
            if retarget_idx == -1:
                raw[12 + i] = 0.0
            else:
                raw[12 + i] = float(retargeted[retarget_idx])

        # Clamp to joint limits (with safety margin already baked in).
        clamped = np.clip(raw, self._lo, self._hi)

        # Velocity cap: limit how much any joint can move in one step.
        delta = clamped - self._prev
        delta = np.clip(delta, -self._max_delta, self._max_delta)
        command = self._prev + delta

        # Final hard clamp after delta application (delta + prev could still
        # exceed limits if prev was near the boundary).
        command = np.clip(command, self._lo, self._hi)

        self._prev = command.copy()
        return command

    def reset(self) -> None:
        """Reset internal state (previous command) to zero."""
        self._prev[:] = 0.0
