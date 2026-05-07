"""Safe core fee-dynamics catalog."""

from __future__ import annotations

from pathlib import Path

from ._family_builder import base_outputs, build_catalog, fingerprint_sources

CORE_FEE_DYNAMICS_FINGERPRINT_SOURCES = fingerprint_sources(Path(__file__).resolve())

CORE_FEE_DYNAMICS_OUTPUTS = base_outputs()


CORE_FEE_DYNAMICS = build_catalog(
    variant_module_path=Path(__file__).resolve(),
    allowed_outputs=CORE_FEE_DYNAMICS_OUTPUTS,
)
