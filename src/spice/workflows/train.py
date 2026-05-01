"""Training workflow."""

from __future__ import annotations

from typing import cast

from ..config.models import ArtifactVariant, TrainConfig
from ..core.rendering import metric_string
from ..core.reporting import Reporter
from ..corpus.coverage import training_coverage_requirement, validate_corpus_coverage
from ..modeling.persisted_training import run_persisted_training
from ..modeling.pipeline import build_artifact_training_spec
from ..modeling.summary import training_result_fields
from ..modeling.training_runner import TrainingEpochProgress
from ..modeling.tuning import apply_study_best_params
from ..storage.corpus import load_dataset_manifest
from ..storage.engine import ARTIFACT_ROOT_KIND
from ..storage.lifecycle import staged_root
from ..storage.root_consumer_handles import resolve_train_consumer_roots
from ..storage.root_handles import TrainWorkflowRoots, TunedTrainWorkflowRoots


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
    active_config: TrainConfig = config
    roots = resolve_train_consumer_roots(config)
    if config.artifact.variant is ArtifactVariant.TUNED:
        assert isinstance(roots, TunedTrainWorkflowRoots)
        applied = apply_study_best_params(
            config,
            study=roots.study,
            corpus=roots.corpus,
        )
        active_config = cast(TrainConfig, applied.config)
    corpus_manifest = load_dataset_manifest(roots.corpus.state_db_path)
    active_reporter.header("train", _workflow_facts(active_config, roots))
    spec = build_artifact_training_spec(
        active_config,
        corpus=roots.corpus,
        artifact=roots.artifact,
        corpus_manifest=corpus_manifest,
    )
    validate_corpus_coverage(
        corpus_manifest,
        contract=spec.problem_contract,
        feature_contract=spec.feature_contract,
        requirement=training_coverage_requirement(spec.problem_contract),
    )
    artifact_dir = roots.artifact.root_path
    history_block_path = roots.corpus.history_dir
    with staged_root(
        storage_root=roots.storage.root_path,
        destination_root=artifact_dir,
        expected_root_kind=ARTIFACT_ROOT_KIND,
        purpose="staging",
        prune_stop_at=artifact_dir.parent.parent,
    ) as artifact_stage:
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
