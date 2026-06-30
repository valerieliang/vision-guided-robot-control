"""Map retargeted human hand angles to Allegro Hand joint commands.

Allegro Hand right -- 16 revolute joints, verified from URDF info:
-------------------------------------------------------------------
Joints 0-3:   finger 0 (index)  [spread, mcp, pip, dip]
              spread lo=-0.47  hi=0.47
              mcp    lo=-0.196 hi=1.610
              pip    lo=-0.174 hi=1.709
              dip    lo=-0.227 hi=1.618

Joints 4-7:   finger 1 (middle) [spread, mcp, pip, dip]
Joints 8-11:  finger 2 (ring)   [spread, mcp, pip, dip]

Joints 12-15: thumb
              CMC (12): lo=0.263  hi=1.396  -- must always be >= 0.263
              lat (13): lo=-0.105 hi=1.163
              mcp (14): lo=-0.189 hi=1.644
              ip  (15): lo=-0.162 hi=1.719

NOTE: This URDF has 3 non-thumb fingers (index/middle/ring).
Pinky retarget values (indices 12-15) are unused.

retargeting.py flat array (19 elements)
----------------------------------------
[0]  index  mcp_flexion      [4]  middle mcp_flexion
[1]  index  pip_flexion      [5]  middle pip_flexion
[2]  index  dip_flexion      [6]  middle dip_flexion
[3]  index  mcp_abduction    [7]  middle mcp_abduction
[8]  ring   mcp_flexion      [12] pinky  mcp_flexion  (UNUSED)
[9]  ring   pip_flexion      [13] pinky  pip_flexion  (UNUSED)
[10] ring   dip_flexion      [14] pinky  dip_flexion  (UNUSED)
[11] ring   mcp_abduction    [15] pinky  mcp_abduction (UNUSED)
[16] thumb  cmc_flexion
[17] thumb  mcp_flexion
[18] thumb  ip_flexion
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

_LIMITS_PATH = Path("kinematics/joint_limits.yaml")

# ------------------------------------------------------------------
# Allegro joint -> retarget flat-array index, for joints 0-11.
# Each finger block: [spread, mcp_flex, pip_flex, dip_flex].
# ------------------------------------------------------------------
_FINGER_MAP: list[int] = [
    # finger 0 = index (joints 0-3)
    3,   # joint 0: spread  <- retarget[3]  index mcp_abduction
    0,   # joint 1: mcp     <- retarget[0]  index mcp_flexion
    1,   # joint 2: pip     <- retarget[1]  index pip_flexion
    2,   # joint 3: dip     <- retarget[2]  index dip_flexion
    # finger 1 = middle (joints 4-7)
    7,   # joint 4: spread  <- retarget[7]  middle mcp_abduction
    4,   # joint 5: mcp     <- retarget[4]  middle mcp_flexion
    5,   # joint 6: pip     <- retarget[5]  middle pip_flexion
    6,   # joint 7: dip     <- retarget[6]  middle dip_flexion
    # finger 2 = ring (joints 8-11)
    11,  # joint 8:  spread <- retarget[11] ring mcp_abduction
    8,   # joint 9:  mcp    <- retarget[8]  ring mcp_flexion
    9,   # joint 10: pip    <- retarget[9]  ring pip_flexion
    10,  # joint 11: dip    <- retarget[10] ring dip_flexion
]

# Limit metadata for joints 0-11 only (thumb handled separately).
_FINGER_JOINT_META: list[tuple[str, str]] = [
    ("index",  "mcp_abduction"), ("index",  "mcp_flexion"),
    ("index",  "pip_flexion"),   ("index",  "dip_flexion"),
    ("middle", "mcp_abduction"), ("middle", "mcp_flexion"),
    ("middle", "pip_flexion"),   ("middle", "dip_flexion"),
    ("ring",   "mcp_abduction"), ("ring",   "mcp_flexion"),
    ("ring",   "pip_flexion"),   ("ring",   "dip_flexion"),
]

# Thumb CMC URDF limits -- CMC must stay in [lo, hi].
_THUMB_CMC_LO  = 0.263
_THUMB_CMC_HI  = 1.396
_THUMB_CMC_MID = (_THUMB_CMC_LO + _THUMB_CMC_HI) / 2.0   # 0.829


class AllegroController:
    """Convert retargeted human-hand angles to clamped Allegro joint commands.

    Parameters
    ----------
    limits_path:
        Path to ``kinematics/joint_limits.yaml``.
    control_rate_hz:
        Control loop frequency; used with ``max_joint_delta_rad`` for velocity
        capping.
    """

    def __init__(
        self,
        limits_path: str | Path = _LIMITS_PATH,
        control_rate_hz: float = 60.0,
    ) -> None:
        with open(limits_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        safety = raw.get("safety", {})
        self._margin = float(safety.get("margin_rad", 0.087))
        self._max_delta = float(safety.get("max_joint_delta_rad", 0.175))

        self._lo = np.zeros(16, dtype=np.float64)
        self._hi = np.zeros(16, dtype=np.float64)

        fingers_cfg = raw["fingers"]

        # Joints 0-11: three fingers.
        for i, (finger, role) in enumerate(_FINGER_JOINT_META):
            jc = fingers_cfg[finger][role]
            self._lo[i] = float(jc["min"]) + self._margin
            self._hi[i] = float(jc["max"]) - self._margin

        # Joint 12: thumb CMC -- URDF lower bound is 0.263, must stay positive.
        self._lo[12] = _THUMB_CMC_LO + self._margin
        self._hi[12] = _THUMB_CMC_HI - self._margin

        # Joint 13: thumb lateral spread -- reuse CMC range as proxy.
        tc = fingers_cfg["thumb"]
        j13 = tc["cmc_flexion"]
        self._lo[13] = float(j13["min"]) + self._margin
        self._hi[13] = float(j13["max"]) - self._margin

        # Joint 14: thumb MCP flexion.
        j14 = tc["mcp_flexion"]
        self._lo[14] = float(j14["min"]) + self._margin
        self._hi[14] = float(j14["max"]) - self._margin

        # Joint 15: thumb IP flexion.
        j15 = tc["ip_flexion"]
        self._lo[15] = float(j15["min"]) + self._margin
        self._hi[15] = float(j15["max"]) - self._margin

        self._prev = np.zeros(16, dtype=np.float64)
        # Initialise thumb CMC at midpoint so it starts naturally opposed.
        self._prev[12] = _THUMB_CMC_MID

    def retargeted_to_allegro(self, retargeted: np.ndarray) -> np.ndarray:
        """Convert a 19-element retargeting flat array to 16 Allegro angles.

        Pipeline:
          1. Reindex retargeting -> Allegro joint order via _FINGER_MAP.
          2. Remap thumb CMC from [0, pi] into URDF range [0.263, 1.396].
          3. Clamp all joints to [min+margin, max-margin].
          4. Velocity cap: limit per-step delta to prevent jerk.

        Parameters
        ----------
        retargeted:
            ``JointAngles.as_flat_array()`` -- 19 float32 values.

        Returns
        -------
        np.ndarray
            Shape (16,) float64, ready for ``AllegroSimEnv.set_joint_angles()``.
        """
        cmd = np.zeros(16, dtype=np.float64)

        # Fingers (joints 0-11).
        for allegro_idx, retarget_idx in enumerate(_FINGER_MAP):
            cmd[allegro_idx] = float(retargeted[retarget_idx])

        # Thumb joint 12: CMC rotation.
        # Human CMC flexion retarget[16] is in [0, pi].
        # Remap into URDF [0.263, 1.396] so thumb stays naturally opposed.
        cmc_human = float(retargeted[16])
        cmc_norm  = cmc_human / math.pi
        cmd[12] = _THUMB_CMC_LO + cmc_norm * (_THUMB_CMC_HI - _THUMB_CMC_LO)

        # Thumb joint 13: lateral spread derived from CMC -- invert for opposition.
        cmd[13] = _THUMB_CMC_MID - (cmd[12] - _THUMB_CMC_MID) * 0.4

        # Thumb joints 14-15: direct from retargeting.
        cmd[14] = float(retargeted[17])
        cmd[15] = float(retargeted[18])

        # Clamp to soft limits.
        clamped = np.clip(cmd, self._lo, self._hi)

        # Velocity cap.
        delta   = np.clip(clamped - self._prev, -self._max_delta, self._max_delta)
        command = np.clip(self._prev + delta, self._lo, self._hi)

        self._prev = command.copy()
        return command

    def reset(self) -> None:
        """Reset to neutral (fingers open, thumb CMC at midpoint)."""
        self._prev[:] = 0.0
        self._prev[12] = _THUMB_CMC_MID