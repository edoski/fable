"""Study-root inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.rendering import mapping_fields, metric_string
from .catalog.records import CatalogStudyRecord
from .study_models import StudySummary, StudyTrialRecord
from .study_optuna import list_trial_records, load_study_summary


@dataclass(frozen=True, slots=True)
class StudyRootDescription:
    summary: StudySummary
    include_config: bool = False
    trials: list[StudyTrialRecord] | None = None


def study_list_sections(
    records: list[CatalogStudyRecord],
) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "studies",
            [
                (
                    record.study_name,
                    (
                        f"chain={record.chain_name} "
                        f"dataset={record.dataset_name} "
                        f"features={record.features_id} "
                        f"prediction={record.prediction_id} "
                        f"model={record.model_id} "
                        f"problem={record.problem_id} "
                        f"id={record.study_id}"
                    ),
                )
                for record in records
            ],
        )
    ]


def describe_study_root(root_db_path: Path, *, detail: str | None = None) -> StudyRootDescription:
    return StudyRootDescription(
        summary=load_study_summary(root_db_path),
        include_config=detail == "config",
        trials=list_trial_records(root_db_path) if detail == "trials" else None,
    )


def study_sections(
    description: StudyRootDescription,
) -> list[tuple[str, list[tuple[str, str]]]]:
    summary = description.summary
    manifest = summary.manifest
    best_trial = summary.best_trial
    sections = [
        (
            "study",
            [
                ("name", manifest.study_name),
                ("storage id", manifest.study_id),
                ("prediction", manifest.prediction_id),
                ("chain", manifest.chain_name),
                ("dataset", manifest.dataset_name),
                ("dataset id", manifest.dataset_id),
                ("problem", manifest.problem_id),
                ("features", manifest.features_id),
                ("model", manifest.model_id),
                ("sampler", f"{manifest.sampler_name} seed={manifest.sampler_seed}"),
                ("pruner", manifest.pruner_name),
                ("trials", str(summary.trial_counts.total)),
            ],
        ),
        (
            "best trial",
            [
                ("number", "n/a" if best_trial is None else str(best_trial.number + 1)),
                (
                    "value",
                    "n/a"
                    if best_trial is None or best_trial.value is None
                    else metric_string(best_trial.value),
                ),
            ],
        ),
    ]
    if description.include_config:
        sections.extend(
            [
                ("problem spec", mapping_fields(manifest.problem.model_dump(mode="json"))),
                (
                    "features config",
                    mapping_fields(manifest.features.model_dump(mode="json")),
                ),
                ("prediction config", mapping_fields(manifest.prediction.model_dump(mode="json"))),
                (
                    "model config",
                    mapping_fields(manifest.model.model_dump(mode="json", exclude_none=True)),
                ),
                ("split config", mapping_fields(manifest.split.model_dump(mode="json"))),
                ("training config", mapping_fields(manifest.training.model_dump(mode="json"))),
                (
                    "tuning config",
                    mapping_fields(
                        {
                            "sampler_name": manifest.sampler_name,
                            "sampler_seed": manifest.sampler_seed,
                            "pruner_name": manifest.pruner_name,
                            "enable_pruning": manifest.enable_pruning,
                        }
                    ),
                ),
                (
                    "tuning space",
                    mapping_fields(
                        manifest.tuning_space.model_dump(mode="json", exclude_none=True)
                    ),
                ),
            ]
        )
    if description.trials:
        sections.append(
            (
                "trials",
                [
                    (f"trial {record.number + 1}", trial_string(record))
                    for record in description.trials
                ],
            )
        )
    return sections


def trial_string(record: StudyTrialRecord) -> str:
    value = "n/a" if record.value is None else f"{record.value:.4f}"
    params = record.params.model_dump(mode="json", exclude_none=True)
    params_text = "" if not params else f" params={params}"
    epoch_text = "" if record.best_epoch is None else f" best_epoch={record.best_epoch}"
    return f"state={record.state.value} value={value}{epoch_text}{params_text}"
