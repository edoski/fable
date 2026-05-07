"""Priority-fee extended core fee-dynamics catalog."""

from __future__ import annotations

from pathlib import Path

from . import _priority_fee
from ._family_builder import base_outputs, build_catalog, fingerprint_sources
from ._priority_fee import (
    PRIORITY_FEE_OUTPUTS,
    priority_fee_features,
    priority_fee_sources,
)

CORE_FEE_DYNAMICS_OUTPUTS = base_outputs()

CORE_FEE_DYNAMICS_PRIORITY_FEE_OUTPUTS = (
    *CORE_FEE_DYNAMICS_OUTPUTS,
    *PRIORITY_FEE_OUTPUTS,
)
CORE_FEE_DYNAMICS_PRIORITY_FEE_FINGERPRINT_SOURCES = fingerprint_sources(
    Path(__file__).resolve(),
    extra_modules=(_priority_fee,),
)

CORE_FEE_DYNAMICS_PRIORITY_FEE = build_catalog(
    variant_module_path=Path(__file__).resolve(),
    allowed_outputs=CORE_FEE_DYNAMICS_PRIORITY_FEE_OUTPUTS,
    extra_sources=priority_fee_sources(),
    extra_features=priority_fee_features(),
    extra_fingerprint_modules=(_priority_fee,),
)
