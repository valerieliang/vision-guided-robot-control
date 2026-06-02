from __future__ import annotations

import numpy as np

from simulation.robot_controller import AllegroController


def test_controller_returns_16_joint_command() -> None:
    controller = AllegroController()

    command = controller.retargeted_to_allegro(np.zeros(19, dtype=np.float32))

    assert command.shape == (16,)
    assert np.all(np.isfinite(command))


def test_controller_caps_per_step_delta() -> None:
    controller = AllegroController()
    first = controller.retargeted_to_allegro(np.zeros(19, dtype=np.float32))
    second = controller.retargeted_to_allegro(np.ones(19, dtype=np.float32) * 3.0)

    assert np.max(np.abs(second - first)) <= 0.175 + 1e-9
