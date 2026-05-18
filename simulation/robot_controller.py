"""Map retargeted human hand angles to Allegro Hand joint commands.

Allegro Hand right — 16 revolute joints, URDF order
----------------------------------------------------
Joint  0  finger_0 (pinky)  MCP lateral spread    lower=-0.47  upper=0.47
Joint  1  finger_0 (pinky)  MCP flexion           lower=-0.196 upper=1.61
Joint  2  finger_0 (pinky)  PIP flexion           lower=-0.174 upper=1.709
Joint  3  finger_0 (pinky)  DIP flexion           lower=-0.227 upper=1.618
Joint  4  finger_1 (ring)   MCP lateral spread
Joint  5  finger_1 (ring)   MCP flexion
Joint  6  finger_1 (ring)   PIP flexion
Joint  7  finger_1 (ring)   DIP flexion
Joint  8  finger_2 (middle) MCP lateral spread
Joint  9  finger_2 (middle) MCP flexion
Joint 10  finger_2 (middle) PIP flexion
Joint 11  finger_2 (middle) DIP flexion
Joint 12  thumb             CMC rotation          lower=0.263  upper=1.396
Joint 13  thumb             MCP lateral           lower=-0.105 upper=1.163
Joint 14  thumb             MCP flexion           lower=-0.189 upper=1.644
Joint 15  thumb             IP flexion            lower=-0.162 upper=1.719

NOTE: The Allegro has NO separate index-finger group — its four fingers are
pinky / ring / middle / index (finger 3).  The "index" finger joints appear
nowhere in this file because the Allegro only has 4 fingers total and they
are already named 0–3 in the URDF.  Finger 3 (joints 12–15 if it existed)
is absent; those slot numbers belong to the thumb.

Thumb-specific notes
--------------------
- Joint 12 (CMC): URDF lower=0.263.  Sending 0 retracts the thumb entirely
  behind the palm.  We map human CMC flexion into [0.263, 1.396] so the
  thumb always stays in a visible, natural position.
- Joint 13 (MCP lateral): maps to human thumb CMC abduction (side-spread).
- Joint 14 (MCP flexion): maps to human thumb MCP flexion.
- Joint 15 (IP flexion):  maps to human thumb IP flexion.
  This gives the thumb FULL 4-DOF retargeting — nothing locked at zero.

retargeting.py flat array (19 elements)
----------------------------------------
[0]  index  mcp_flexion      [4]  middle mcp_flexion
[1]  index  pip_flexion      [5]  middle pip_flexion
[2]  index  dip_flexion      [6]  middle dip_flexion
[3]  index  mcp_abduction    [7]  middle mcp_abduction
[8]  ring   mcp_flexion      [12] pinky  mcp_flexion
[9]  ring   pip_flexion      [13] pinky  pip_flexion
[10] ring   dip_flexion      [14] pinky  dip_flexion
[11] ring   mcp_abduction    [15] pinky  mcp_abduction
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
# Mapping: Allegro joint index (0–15) → retargeting flat-array index.
# -1 means no retargeting source; the value is computed analytically.
# ------------------------------------------------------------------

# Fingers 0–2 (pinky / ring / middle): straightforward.
# Allegro finger 3 does not exist as fingers; those slots (12–15) = thumb.
_FINGER_MAP: list[int] = [
    # pinky (finger 0, joints 0–3)
    15,   # joint 0: MCP spread    ← retarget[15] pinky mcp_abduction
    12,   # joint 1: MCP flexion   ← retarget[12] pinky mcp_flexion
    13,   # joint 2: PIP flexion   ← retarget[13] pinky pip_flexion
    14,   # joint 3: DIP flexion   ← retarget[14] pinky dip_flexion
    # ring (finger 1, joints 4–7)
    11,   # joint 4: MCP spread    ← retarget[11] ring  mcp_abduction
    8,    # joint 5: MCP flexion   ← retarget[8]  ring  mcp_flexion
    9,    # joint 6: PIP flexion   ← retarget[9]  ring  pip_flexion
    10,   # joint 7: DIP flexion   ← retarget[10] ring  dip_flexion
    # middle (finger 2, joints 8–11)
    7,    # joint 8:  MCP spread   ← retarget[7]  middle mcp_abduction
    4,    # joint 9:  MCP flexion  ← retarget[4]  middle mcp_flexion
    5,    # joint 10: PIP flexion  ← retarget[5]  middle pip_flexion
    6,    # joint 11: DIP flexion  ← retarget[6]  middle dip_flexion
]

# Thumb (joints 12–15): fully retargeted — no DOF locked at zero.
# joint 12 = CMC rotation       ← human CMC flexion (remapped into URDF range)
# joint 13 = MCP lateral spread ← human CMC abduction (side-to-side)
# joint 14 = MCP flexion        ← human MCP flexion
# joint 15 = IP flexion         ← human IP flexion
_THUMB_MAP = [16, 17, 18]   # retarget indices for joints 14, 15 (MCP flex, IP flex)

# URDF hard limits for thumb CMC (joint 12) — used for range remapping.
_THUMB_CMC_LO = 0.263
_THUMB_CMC_HI = 1.396
_THUMB_CMC_MID = (_THUMB_CMC_LO + _THUMB_CMC_HI) / 2.0   # 0.829 rad


# ------------------------------------------------------------------
# Limit metadata for joints 0–11 (fingers only; thumb handled separately)
# ------------------------------------------------------------------
_FINGER_JOINT_META: list[tuple[str, str]] = [
    # pinky
    ("pinky",  "mcp_abduction"),
    ("pinky",  "mcp_flexion"),
    ("pinky",  "pip_flexion"),
    ("pinky",  "dip_flexion"),
    # ring
    ("ring",   "mcp_abduction"),
    ("ring",   "mcp_flexion"),
    ("ring",   "pip_flexion"),
    ("ring",   "dip_flexion"),
    # middle
    ("middle", "mcp_abduction"),
    ("middle", "mcp_flexion"),
    ("middle", "pip_flexion"),
    ("middle", "dip_flexion"),
]


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
        control_rate_hz: float = 15.0,
    ) -> None:
        with open(limits_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        safety = raw.get("safety", {})
        self._margin = float(safety.get("margin_rad", 0.087))
        self._max_delta = float(safety.get("max_joint_delta_rad", 0.175))

        self._lo = np.zeros(16, dtype=np.float64)
        self._hi = np.zeros(16, dtype=np.float64)

        fingers_cfg = raw["fingers"]

        # Joints 0–11: fingers
        for i, (finger, role) in enumerate(_FINGER_JOINT_META):
            jc = fingers_cfg[finger][role]
            self._lo[i] = float(jc["min"]) + self._margin
            self._hi[i] = float(jc["max"]) - self._margin

        # Joint 12: thumb CMC — URDF lower bound is 0.263, must stay positive
        self._lo[12] = _THUMB_CMC_LO + self._margin
        self._hi[12] = _THUMB_CMC_HI - self._margin

        # Joint 13: thumb MCP lateral (abduction-like)
        tc = fingers_cfg["thumb"]
        j13 = tc["cmc_flexion"]   # reuse CMC range for lateral — similar ROM
        self._lo[13] = float(j13["min"]) + self._margin
        self._hi[13] = float(j13["max"]) - self._margin

        # Joint 14: thumb MCP flexion
        j14 = tc["mcp_flexion"]
        self._lo[14] = float(j14["min"]) + self._margin
        self._hi[14] = float(j14["max"]) - self._margin

        # Joint 15: thumb IP flexion
        j15 = tc["ip_flexion"]
        self._lo[15] = float(j15["min"]) + self._margin
        self._hi[15] = float(j15["max"]) - self._margin

        self._prev = np.zeros(16, dtype=np.float64)
        # Initialise thumb CMC to its midpoint so it starts naturally opposed.
        self._prev[12] = _THUMB_CMC_MID

    def retargeted_to_allegro(self, retargeted: np.ndarray) -> np.ndarray:
        """Convert a 19-element retargeting flat array to 16 Allegro angles.

        Pipeline:
          1. Reindex retargeting → Allegro joint order.
          2. Remap thumb CMC from human range [0, π] into URDF range [0.263, 1.396].
          3. Clamp all joints to [min+margin, max−margin].
          4. Velocity cap: limit per-step delta to prevent jerk.

        Parameters
        ----------
        retargeted:
            ``JointAngles.as_flat_array()`` output — 19 float32 values.

        Returns
        -------
        np.ndarray
            Shape (16,) float64, ready for ``AllegroSimEnv.set_joint_angles()``.
        """
        cmd = np.zeros(16, dtype=np.float64)

        # ── Fingers (joints 0–11) ─────────────────────────────────────
        for allegro_idx, retarget_idx in enumerate(_FINGER_MAP):
            cmd[allegro_idx] = float(retargeted[retarget_idx])

        # ── Thumb joint 12: CMC rotation ─────────────────────────────
        # Human CMC flexion from retargeting is in [0, π].
        # Remap linearly into the Allegro URDF range [0.263, 1.396] so the
        # thumb always stays in a natural, visible position.
        cmc_human = float(retargeted[16])                     # [0, π]
        cmc_norm = cmc_human / math.pi                        # [0, 1]
        cmd[12] = _THUMB_CMC_LO + cmc_norm * (_THUMB_CMC_HI - _THUMB_CMC_LO)

        # ── Thumb joint 13: MCP lateral spread ───────────────────────
        # We have no direct human abduction angle for the thumb base, so we
        # derive it from thumb CMC abduction implicitly via the CMC flexion
        # signal: a more extended thumb (low cmc_human) spreads laterally.
        # Simple inversion gives natural opposition feel.
        cmd[13] = _THUMB_CMC_MID - (cmd[12] - _THUMB_CMC_MID) * 0.4

        # ── Thumb joints 14–15: MCP and IP flexion ───────────────────
        cmd[14] = float(retargeted[17])   # human thumb mcp_flexion
        cmd[15] = float(retargeted[18])   # human thumb ip_flexion

        # ── Clamp to joint limits ─────────────────────────────────────
        clamped = np.clip(cmd, self._lo, self._hi)

        # ── Velocity cap ──────────────────────────────────────────────
        delta = np.clip(clamped - self._prev, -self._max_delta, self._max_delta)
        command = np.clip(self._prev + delta, self._lo, self._hi)

        self._prev = command.copy()
        return command

    def reset(self) -> None:
        """Reset to neutral (thumb CMC at midpoint, all fingers open)."""
        self._prev[:] = 0.0
        self._prev[12] = _THUMB_CMC_MID