"""Entry point: webcam hand tracking → retargeting → Allegro Hand simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from control.teleop_loop import TeleopConfig, TeleopLoop

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> TeleopConfig:
    """Load application config from YAML, falling back to defaults."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Only pass keys that TeleopConfig actually accepts.
    valid_fields = TeleopConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    return TeleopConfig(**filtered)


def main() -> None:
    """Load config and start the teleoperation loop."""
    try:
        config = load_config()
        with TeleopLoop(config) as loop:
            loop.run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
