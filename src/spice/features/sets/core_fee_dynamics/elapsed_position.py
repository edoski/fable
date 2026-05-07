"""Elapsed-position ablation catalog."""

from __future__ import annotations

from pathlib import Path

from ._family_builder import base_outputs, build_catalog, fingerprint_sources
from ._time import (
    ELAPSED_POSITION_OUTPUTS,
    elapsed_position_features,
)

CORE_FEE_DYNAMICS_OUTPUTS = base_outputs()
CORE_FEE_DYNAMICS_ELAPSED_POSITION_OUTPUTS = (
    *CORE_FEE_DYNAMICS_OUTPUTS,
    *ELAPSED_POSITION_OUTPUTS,
)
CORE_FEE_DYNAMICS_ELAPSED_POSITION_FINGERPRINT_SOURCES = fingerprint_sources(
    Path(__file__).resolve()
)

CORE_FEE_DYNAMICS_ELAPSED_POSITION = build_catalog(
    variant_module_path=Path(__file__).resolve(),
    allowed_outputs=CORE_FEE_DYNAMICS_ELAPSED_POSITION_OUTPUTS,
    extra_features=elapsed_position_features(),
)
