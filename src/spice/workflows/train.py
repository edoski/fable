"""Training workflow."""

from __future__ import annotations

from ..config.models import TrainConfig
from ..core.rendering import metric_string
from ..core.reporting import Reporter
from ..modeling.persisted_training import run_persisted_training
from ..modeling.summary import training_result_fields
from ..modeling.training_runner import TrainingEpochProgress
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.transactions import FullRootTransaction
from ..storage.workflow_roots import (
    TrainWorkflowRoots,
    TunedTrainWorkflowRoots,
)
from .preparation import prepare_train


def _workflow_facts(config: TrainConfig, roots: TrainWorkflowRoots) -> list[tuple[str, str]]:
    facts = [
        ("dataset", roots.corpus.dataset_name),
        ("dataset_id", roots.corpus.dataset_id),
        ("chain", roots.corpus.chain_name),
        ("problem", config.problem.id),
        ("prediction", config.prediction.id),
        ("model", config.model.id),
        ("variant", config.artifact.variant.value),
        ("artifact_id", roots.artifact.artifact_id),
    ]
    if isinstance(roots, TunedTrainWorkflowRoots):
        facts.append(("study", roots.study.study_name))
        facts.append(("study_id", roots.study.study_id))
    return facts


def _fit_epoch_message(
    progress: TrainingEpochProgress,
    *,
    primary_metric_id: str,
) -> str:
    fields = [f"epoch={progress.epoch}/{progress.max_epochs}"]
    if progress.objective_metric_id in progress.objective_metrics.values:
        fields.append(
            f"objective.{progress.objective_metric_id}="
            f"{metric_string(progress.objective_metrics.values[progress.objective_metric_id])}"
        )
    if primary_metric_id in progress.validation_metrics.values:
        fields.append(
            f"validation.{primary_metric_id}="
            f"{metric_string(progress.validation_metrics.values[primary_metric_id])}"
        )
    fields.append(f"best_epoch={progress.best_epoch}")
    fields.append(
        f"best.{progress.objective_metric_id}={metric_string(progress.best_objective_value)}"
    )
    return "fit " + " ".join(fields)


def run(config: TrainConfig, *, reporter: Reporter | None = None) -> None:
    active_reporter = reporter or Reporter()
    prepared = prepare_train(config)
    roots = prepared.roots
    spec = prepared.spec
    active_reporter.header("train", _workflow_facts(prepared.active_config, roots))
    artifact_dir = roots.artifact.root_path
    history_block_path = roots.corpus.history_dir
    transaction = FullRootTransaction(
        storage_root=roots.artifact.storage_root,
        destination_root=artifact_dir,
        expected_root_kind=ARTIFACT_ROOT_KIND,
        purpose="staging",
        prune_stop_at=artifact_dir.parent.parent,
    )
    with transaction.open() as artifact_stage:
        persisted = run_persisted_training(
            history_block_path,
            spec=spec,
            artifact_dir=artifact_stage.staged_root,
            on_prepare_complete=lambda prepared: active_reporter.milestone(
                f"prepare rows={prepared.n_rows_used} samples={prepared.sample_count}"
            ),
            on_fit_start=lambda: active_reporter.milestone(
                f"fit started epochs={spec.training.max_epochs}"
            ),
            on_epoch_end=lambda progress: active_reporter.milestone(
                _fit_epoch_message(
                    progress,
                    primary_metric_id=spec.prediction_contract.primary_metric_id,
                )
            ),
            on_early_stop=lambda epoch, best_epoch: active_reporter.milestone(
                f"fit early_stop epoch={epoch} best_epoch={best_epoch}"
            ),
        )
        artifact_stage.promote()
    active_reporter.result(
        "train",
        training_result_fields(
            persisted.summary,
            artifact_dir=artifact_dir,
        ),
    )
