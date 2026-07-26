from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    seed: int = _env_int("SYNTHETIC_DATA_SEED", 42)
    device: str = os.getenv("SYNTHETIC_DATA_DEVICE", "cpu")
    default_epochs: int = _env_int("SYNTHETIC_DATA_DEFAULT_EPOCHS", 120)


settings = Settings()

