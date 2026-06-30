"""Simulation diagnostic for the Allegro Hand.

Usage
-----
    python diag_sim.py --mode info     # print joint names/limits and quit
    python diag_sim.py --mode step     # binary open/close on flexion joints only
    python diag_sim.py --mode sweep    # sine wave on flexion joints only
    python diag_sim.py --mode finger   # one finger group at a time
    python diag_sim.py --kp 10 --kd 5 --hz 60

ZOH stability: control rate must be well above joint natural frequency.
Natural frequency with kp=10, I_eff=0.007 kg*m^2: omega_n=37.7 rad/s.
At 15 Hz control: UNSTABLE (omega_n*T=2.5, limit is 1.0).
At 60 Hz control: STABLE  (omega_n*T=0.63).
Default is now 60 Hz.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

_NUM_JOINTS = 16
_MAX_FLEX_RAD = 1.0
PRINT_EVERY = 30   # at 60 Hz this prints every 0.5 seconds

_FLEXION_JOINTS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15]
_THUMB_CMC_NEUTRAL = 0.263


def _make_neutral() -> np.ndarray:
    cmd = np.zeros(_NUM_JOINTS, dtype=np.float64)
    cmd[12] = _THUMB_CMC_NEUTRAL
    return cmd


def _flex_cmd(t: float, active: list[int] | None = None) -> np.ndarray:
    cmd = _make_neutral()
    flex = _MAX_FLEX_RAD * max(0.0, math.sin(2.0 * math.pi * t / 4.0))
    for j in (active if active is not None else _FLEXION_JOINTS):
        cmd[j] = flex
    return cmd


def _step_flex_cmd(t: float, active: list[int] | None = None) -> np.ndarray:
    cmd = _make_neutral()
    flex = _MAX_FLEX_RAD if int(t / 2.0) % 2 == 0 else 0.0
    for j in (active if active is not None else _FLEXION_JOINTS):
        cmd[j] = flex
    return cmd


def _finger_cmd(t: float) -> tuple[np.ndarray, str]:
    groups = [
        ([1, 2, 3],   "finger0"),
        ([5, 6, 7],   "finger1"),
        ([9, 10, 11], "finger2"),
        ([14, 15],    "thumb"),
        (_FLEXION_JOINTS, "ALL"),
    ]
    seg = int(t / 5.0) % len(groups)
    active, label = groups[seg]
    return _flex_cmd(t, active), label


def print_joint_info(sim) -> None:
    try:
        import pybullet as pb
    except ImportError:
        return
    print(f"\n{'Allegro':>7s}  {'PyBullet':>8s}  {'Joint name':40s}  {'lo':>7s}  {'hi':>7s}")
    print("-" * 75)
    for allegro_idx, joint_idx in enumerate(sim._joint_indices):
        info = pb.getJointInfo(sim._hand_id, joint_idx)
        name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
        print(f"{allegro_idx:>7d}  {joint_idx:>8d}  {name:40s}  {info[8]:>7.3f}  {info[9]:>7.3f}")
    print()


def run(mode: str, hz: float, kp: float | None, kd: float | None,
        force: float | None, urdf: str) -> None:
    try:
        from simulation.sim_env import AllegroSimEnv, _KP, _KD, _MAX_FORCE
        import pybullet as pb
    except ImportError as exc:
        print(f"Import failed: {exc}")
        sys.exit(1)

    use_kp    = kp    if kp    is not None else _KP
    use_kd    = kd    if kd    is not None else _KD
    use_force = force if force is not None else _MAX_FORCE

    print(f"[diag] mode={mode}  hz={hz}  kp={use_kp}  kd={use_kd}  force={use_force}")
    print(f"[diag] sim_env defaults: kp={_KP}, kd={_KD}, force={_MAX_FORCE}")

    gui = (mode != "info")
    with AllegroSimEnv(urdf_path=urdf, gui=gui, gravity=0.0) as sim:
        print_joint_info(sim)
        if mode == "info":
            return

        if use_kp != _KP or use_kd != _KD or use_force != _MAX_FORCE:
            print(f"[diag] overriding: kp={use_kp} kd={use_kd} force={use_force}")
            neutral = _make_neutral()
            for i, joint_idx in enumerate(sim._joint_indices):
                pb.setJointMotorControl2(
                    sim._hand_id, joint_idx, pb.POSITION_CONTROL,
                    targetPosition=float(neutral[i]),
                    positionGain=use_kp, velocityGain=use_kd, force=use_force,
                )

        print(f"\n{'step':>5s}  {'t':>6s}  {'label':>8s}  {'cmd[1]':>7s}  {'act[1]':>7s}  {'err[1]':>7s}  {'maxerr':>7s}")
        print("-" * 60)

        step = 0
        t0 = time.perf_counter()
        label = "all"

        while True:
            tl = time.perf_counter()
            t = tl - t0
            if not pb.isConnected():
                break

            if mode == "finger":
                cmd, label = _finger_cmd(t)
            elif mode == "step":
                cmd = _step_flex_cmd(t)
            else:
                cmd = _flex_cmd(t)

            sim.set_joint_angles(cmd)
            # Single physics substep per control step at 60 Hz
            # = 60 Hz control ~= 60 Hz physics (timestep is 1/240 internally
            # but we call step() only once, so effective physics is 60 Hz too).
            # For smoother physics, call step() 4x at 60 Hz = 240 Hz internal.
            for _ in range(4):
                sim.step()

            actual = sim.get_joint_angles()

            if step % PRINT_EVERY == 0:
                err = np.abs(cmd - actual)
                print(
                    f"{step:5d}  {t:6.2f}  {label:>8s}  "
                    f"{cmd[1]:7.3f}  {float(actual[1]):7.3f}  "
                    f"{abs(cmd[1]-float(actual[1])):7.3f}  {float(err.max()):7.3f}"
                )

            sl = (1.0 / hz) - (time.perf_counter() - tl)
            if sl > 0:
                time.sleep(sl)
            step += 1

    print("\n[diag] done.")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",  choices=["info", "sweep", "step", "finger"], default="info")
    p.add_argument("--hz",    type=float, default=60.0)   # 60 Hz default
    p.add_argument("--kp",    type=float, default=None)
    p.add_argument("--kd",    type=float, default=None)
    p.add_argument("--force", type=float, default=None)
    p.add_argument("--urdf",  default="simulation/assets/allegro/allegro_hand_right.urdf")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    run(mode=a.mode, hz=a.hz, kp=a.kp, kd=a.kd, force=a.force, urdf=a.urdf)