from __future__ import annotations

from spice.temporal import TemporalCapability
from spice.temporal.compilers.observed_time_window import ObservedTimeWindowRuntimeMetadata


def _capability() -> TemporalCapability:
    return TemporalCapability(
        compiler_id="observed_time_window",
        max_delay_seconds=36,
        action_width=4,
        compiler_runtime_metadata=ObservedTimeWindowRuntimeMetadata(
            slot_spacing_id="nominal",
            slot_spacing_seconds=12.0,
        ),
    )


def test_temporal_capability_semantics_projects_authoritative_delay_and_width() -> None:
    semantics = _capability().semantics

    assert semantics.compiler_id == "observed_time_window"
    assert semantics.max_delay_seconds == 36
    assert semantics.action_width == 4
