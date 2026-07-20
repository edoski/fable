"""Process environment resolution."""

import os
from pathlib import Path


def resolve_storage_root() -> Path:
    storage_root = Path(os.environ["STORAGE_ROOT"])
    if not storage_root.is_absolute():
        raise ValueError("STORAGE_ROOT must be an absolute path")
    return storage_root
