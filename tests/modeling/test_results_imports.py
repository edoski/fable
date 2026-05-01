from __future__ import annotations

import subprocess
import sys


def test_results_import_does_not_load_training_pipeline() -> None:
    script = """
import importlib
import sys

importlib.import_module("spice.modeling.results")
if "spice.modeling.pipeline" in sys.modules:
    raise SystemExit("spice.modeling.results imported spice.modeling.pipeline")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
