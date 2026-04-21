"""Import smoke tests for the Milestone 1 modules."""


def test_core_modules_import() -> None:
    """Core modules should import without side effects."""
    import cv.camera  # noqa: F401
    import cv.hand_tracker  # noqa: F401
    import cv.smoothing  # noqa: F401
    import data.recorder  # noqa: F401
    import viz.overlay  # noqa: F401
