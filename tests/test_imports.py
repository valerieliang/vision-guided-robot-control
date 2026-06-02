"""Import smoke tests for the Milestone 1 modules."""


def test_core_modules_import() -> None:
    """Core modules should import without side effects."""
    import cv.camera  # noqa: F401
    import cv.hand_tracker  # noqa: F401
    import cv.smoothing  # noqa: F401
    import data.recorder  # noqa: F401
    import kinematics.retargeting  # noqa: F401
    import neurotech.datasets  # noqa: F401
    import neurotech.decoders  # noqa: F401
    import neurotech.intent_mapping  # noqa: F401
    import simulation.robot_controller  # noqa: F401
    import viz.overlay  # noqa: F401


def test_handedness_display_modes() -> None:
    """HUD handedness should only flip in mirrored mode."""
    from control.teleop_loop import _display_handedness

    assert _display_handedness("Left", "mediapipe") == "Left"
    assert _display_handedness("Right", "mediapipe") == "Right"
    assert _display_handedness("Left", "mirrored") == "Right"
    assert _display_handedness("Right", "mirrored") == "Left"
