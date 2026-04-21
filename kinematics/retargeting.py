"""Retarget human hand landmarks to robot joint angles.

Converts 21 MediaPipe landmarks into per-joint flexion and abduction angles
(radians) suitable for direct consumption by a robot hand controller.

Coordinate convention
---------------------
All landmark positions are made wrist-relative before any angle computation:
landmark[i] -= landmark[0].  This makes the angles invariant to where the
hand sits in the camera frame and to overall hand scale.

Output format
-------------
A ``JointAngles`` dataclass with one field per finger.  Each finger holds a
``FingerAngles`` with:

  mcp_flexion   – knuckle bend (+ = curl toward palm)
  pip_flexion   – middle-joint bend
  dip_flexion   – tip-joint bend  (index/middle/ring/pinky only)
  mcp_abduction – lateral spread at knuckle (index/middle/ring/pinky only)

Thumb uses CMC_flexion / MCP_flexion / IP_flexion instead (no DIP, no
abduction field – thumb opposition is a separate concern).

Angle sign convention
---------------------
All flexion angles are non-negative (0 = fully extended, π = fully curled).
Abduction is signed: positive = spreading away from middle finger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cv.hand_tracker import HandLandmarks, Landmark

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------

_WRIST = 0

_FINGER_CHAINS: dict[str, list[int]] = {
    #         MCP  PIP  DIP  TIP
    "index":  [5,  6,   7,   8],
    "middle": [9,  10,  11,  12],
    "ring":   [13, 14,  15,  16],
    "pinky":  [17, 18,  19,  20],
}

# Thumb chain: CMC(1) → MCP(2) → IP(3) → TIP(4)
# We treat the wrist(0)→CMC(1) vector as the "proximal bone" for CMC flexion.
_THUMB_CHAIN: list[int] = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FingerAngles:
    """Joint angles for a single non-thumb finger (radians)."""

    mcp_flexion: float   # knuckle flexion
    pip_flexion: float   # proximal interphalangeal flexion
    dip_flexion: float   # distal interphalangeal flexion
    mcp_abduction: float # lateral spread at MCP (signed)


@dataclass
class ThumbAngles:
    """Joint angles for the thumb (radians)."""

    cmc_flexion: float   # carpometacarpal flexion
    mcp_flexion: float   # metacarpophalangeal flexion
    ip_flexion: float    # interphalangeal flexion


@dataclass
class JointAngles:
    """Full set of retargeted joint angles for one hand."""

    index:  FingerAngles
    middle: FingerAngles
    ring:   FingerAngles
    pinky:  FingerAngles
    thumb:  ThumbAngles

    def as_flat_array(self) -> np.ndarray:
        """Return all angles as a 1-D numpy array (19 values).

        Order: index(4), middle(4), ring(4), pinky(4), thumb(3).
        Useful for passing directly to a controller or logging.
        """
        fingers = [self.index, self.middle, self.ring, self.pinky]
        values: list[float] = []
        for f in fingers:
            values += [f.mcp_flexion, f.pip_flexion, f.dip_flexion, f.mcp_abduction]
        values += [self.thumb.cmc_flexion, self.thumb.mcp_flexion, self.thumb.ip_flexion]
        return np.array(values, dtype=np.float32)

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Return angles as a nested dict, handy for logging / serialisation."""
        fingers = {
            "index": self.index,
            "middle": self.middle,
            "ring": self.ring,
            "pinky": self.pinky,
        }
        result: dict[str, dict[str, float]] = {}
        for name, f in fingers.items():
            result[name] = {
                "mcp_flexion": f.mcp_flexion,
                "pip_flexion": f.pip_flexion,
                "dip_flexion": f.dip_flexion,
                "mcp_abduction": f.mcp_abduction,
            }
        result["thumb"] = {
            "cmc_flexion": self.thumb.cmc_flexion,
            "mcp_flexion": self.thumb.mcp_flexion,
            "ip_flexion": self.thumb.ip_flexion,
        }
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retarget(hand: HandLandmarks) -> JointAngles:
    """Convert one detected hand into robot joint angles (radians).

    Parameters
    ----------
    hand:
        A ``HandLandmarks`` instance from ``HandTracker.detect()``.

    Returns
    -------
    JointAngles
        All finger joint angles in radians, wrist-relative.
    """
    pts = _wrist_relative(hand.landmarks)

    finger_angles = {
        name: _finger_flexion_abduction(pts, chain)
        for name, chain in _FINGER_CHAINS.items()
    }

    thumb = _thumb_flexion(pts)

    return JointAngles(
        index=finger_angles["index"],
        middle=finger_angles["middle"],
        ring=finger_angles["ring"],
        pinky=finger_angles["pinky"],
        thumb=thumb,
    )


def retarget_all(hands: list[HandLandmarks]) -> list[JointAngles]:
    """Retarget every detected hand in a frame."""
    return [retarget(h) for h in hands]


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _wrist_relative(landmarks: list[Landmark]) -> np.ndarray:
    """Return an (N, 3) array of landmarks translated so the wrist is origin.

    The coordinates remain in the same scale as the raw normalized landmarks
    (roughly 0–1 in x/y, smaller in z).  No further scaling is applied here;
    the bone-angle computation is scale-invariant by construction.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts -= pts[_WRIST]
    return pts


def _vec(pts: np.ndarray, a: int, b: int) -> np.ndarray:
    """Unit vector from landmark a to landmark b.  Returns zero vector if
    the two points coincide (degenerate detection)."""
    v = pts[b] - pts[a]
    n = np.linalg.norm(v)
    return v / n if n > 1e-7 else np.zeros(3)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors.  Safe against numerical errors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return math.acos(float(np.clip(cos_a, -1.0, 1.0)))


def _signed_abduction(
    pts: np.ndarray,
    mcp_idx: int,
    pip_idx: int,
    reference_finger_mcp: int,
    reference_finger_pip: int,
) -> float:
    """Signed MCP abduction angle relative to a reference finger.

    Positive = spreading away from the reference (middle) finger.
    The sign is determined by the cross product of the palm normal with the
    spread vector, so it is consistent regardless of hand orientation.
    """
    finger_bone = _vec(pts, mcp_idx, pip_idx)
    ref_bone = _vec(pts, reference_finger_mcp, reference_finger_pip)

    magnitude = _angle_between(finger_bone, ref_bone)

    # Determine sign: cross product gives a vector whose component along the
    # palm normal indicates which side of the reference finger we're on.
    cross = np.cross(ref_bone, finger_bone)
    # Palm normal approximated from wrist → index-MCP × wrist → pinky-MCP
    palm_normal = np.cross(
        pts[_FINGER_CHAINS["index"][0]] - pts[_WRIST],
        pts[_FINGER_CHAINS["pinky"][0]] - pts[_WRIST],
    )
    n = np.linalg.norm(palm_normal)
    if n > 1e-7:
        palm_normal /= n

    sign = math.copysign(1.0, float(np.dot(cross, palm_normal)))
    return sign * magnitude


def _finger_flexion_abduction(
    pts: np.ndarray,
    chain: list[int],
) -> FingerAngles:
    """Compute MCP/PIP/DIP flexion and MCP abduction for one finger.

    chain = [MCP, PIP, DIP, TIP] indices into pts.
    """
    mcp, pip, dip, tip = chain

    # Flexion: angle at each joint = angle between the two bones meeting there.
    # We measure the supplement so 0 rad = straight, π rad = fully curled.
    mcp_flex = math.pi - _angle_between(
        pts[mcp] - pts[_WRIST], pts[pip] - pts[mcp]
    )
    pip_flex = math.pi - _angle_between(
        pts[pip] - pts[mcp], pts[dip] - pts[pip]
    )
    dip_flex = math.pi - _angle_between(
        pts[dip] - pts[pip], pts[tip] - pts[dip]
    )

    # Clamp: floating-point noise can push just past [0, π]
    mcp_flex = float(np.clip(mcp_flex, 0.0, math.pi))
    pip_flex = float(np.clip(pip_flex, 0.0, math.pi))
    dip_flex = float(np.clip(dip_flex, 0.0, math.pi))

    # Abduction relative to middle finger (the anatomical reference)
    mid_chain = _FINGER_CHAINS["middle"]
    abduction = _signed_abduction(pts, mcp, pip, mid_chain[0], mid_chain[1])

    return FingerAngles(
        mcp_flexion=mcp_flex,
        pip_flexion=pip_flex,
        dip_flexion=dip_flex,
        mcp_abduction=abduction,
    )


def _thumb_flexion(pts: np.ndarray) -> ThumbAngles:
    """Compute CMC, MCP, and IP flexion for the thumb.

    The proximal reference bone for CMC flexion is the wrist → CMC vector,
    with the palm plane used to keep the angle in the flexion–extension plane.
    """
    cmc, mcp_idx, ip_idx, tip = _THUMB_CHAIN  # indices 1, 2, 3, 4

    cmc_flex = math.pi - _angle_between(
        pts[cmc] - pts[_WRIST], pts[mcp_idx] - pts[cmc]
    )
    mcp_flex = math.pi - _angle_between(
        pts[mcp_idx] - pts[cmc], pts[ip_idx] - pts[mcp_idx]
    )
    ip_flex = math.pi - _angle_between(
        pts[ip_idx] - pts[mcp_idx], pts[tip] - pts[ip_idx]
    )

    cmc_flex = float(np.clip(cmc_flex, 0.0, math.pi))
    mcp_flex = float(np.clip(mcp_flex, 0.0, math.pi))
    ip_flex  = float(np.clip(ip_flex,  0.0, math.pi))

    return ThumbAngles(
        cmc_flexion=cmc_flex,
        mcp_flexion=mcp_flex,
        ip_flexion=ip_flex,
    )