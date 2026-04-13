"""Current candidate-slate prediction family."""

from __future__ import annotations

from ...contracts import CompiledPredictionContract
from ...registry import PredictionFamilySpec, register_prediction_family_spec
from .config import CandidateSlateCurrentFamilyConfig
from .metrics import (
    METRIC_DESCRIPTORS,
    best_epoch,
    compute_batch_loss_and_state,
    objective_value,
    summarize_epoch_metrics,
)
from .outputs import CANDIDATE_LOGITS_HEAD_ID, build_output_spec
from .replay import allocate_prediction_buffer, decode_into, run_replay
from .targets import prepare_candidate_slate_targets


def _compile(
    prediction_id: str,
    family: CandidateSlateCurrentFamilyConfig,
) -> CompiledPredictionContract:
    del family
    return CompiledPredictionContract(
        prediction_id=prediction_id,
        prediction_family_id="candidate_slate_current",
        objective_id="profit_over_baseline",
        metric_descriptors=METRIC_DESCRIPTORS,
        primary_metric_id="profit_over_baseline",
        direction="maximize",
        build_output_spec=build_output_spec,
        prepare_targets=prepare_candidate_slate_targets,
        compute_batch_loss_and_state=lambda outputs, targets: compute_batch_loss_and_state(
            outputs.head(CANDIDATE_LOGITS_HEAD_ID),
            targets,
        ),
        summarize_epoch_metrics=summarize_epoch_metrics,
        best_epoch=best_epoch,
        objective_value=objective_value,
        allocate_prediction_buffer=allocate_prediction_buffer,
        decode_into=decode_into,
        replay=run_replay,
        supported_workflows=frozenset(
            {
                "train",
                "tune",
                "simulate",
            }
        ),
    )


register_prediction_family_spec(
    PredictionFamilySpec(
        id="candidate_slate_current",
        config_type=CandidateSlateCurrentFamilyConfig,
        compile=_compile,
    )
)
