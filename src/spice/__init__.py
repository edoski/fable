"""SPICE package."""

import os
from pathlib import Path
from tempfile import gettempdir

_MPLCONFIGDIR = Path(gettempdir()) / "spice-matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

__all__ = ["__version__"]

__version__ = "0.1.0"
