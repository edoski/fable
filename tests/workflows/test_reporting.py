from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import cast

from optuna.trial import TrialState

from spice.config import TrainConfig, TuneConfig, WorkflowTask
from spice.core.reporting import Reporter
from spice.modeling.training import TrainingEpochProgress
from spice.prediction import MetricSet
from spice.workflows import train as train_workflow
from spice.workflows import tune as tune_workflow


def _load_test_train_config(
    load_workflow_config,
    tmp_path,
    *,
    override: dict[str, object] | None = None,
) -> TrainConfig:
    return cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )


def _load_test_tune_config(
    load_workflow_config,
    tmp_path,
    *,
    override: dict[str, object] | None = None,
) -> TuneConfig:
    return cast(
        TuneConfig,
        load_workflow_config(
            WorkflowTask.TUNE,
            workspace=tmp_path,
            preset="icdcs_2026",
            override=override,
        ),
    )


def test_train_workflow_emits_compact_epoch_output(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _load_test_train_config(
        load_workflow_config,
        tmp_path,
        override=model_workflow_override(),
    )
    output = StringIO()
    reporter = Reporter(stream=output)

    monkeypatch.setattr(
        train_workflow,
        "build_training_spec",
        lambda _config: SimpleNamespace(
            training=SimpleNamespace(max_epochs=3),
            prediction_contract=SimpleNamespace(primary_metric_id="profit_over_baseline"),
        ),
    )

    def fake_run_persisted_training(*args, **kwargs):
        del args
        kwargs["on_prepare_complete"](SimpleNamespace(n_rows_used=128, sample_count=24))
        kwargs["on_fit_start"]()
        kwargs["on_epoch_end"](
            TrainingEpochProgress(
                epoch=1,
                max_epochs=3,
                train_metrics=MetricSet(values={"profit_over_baseline": 0.15}),
                validation_metrics=MetricSet(values={"profit_over_baseline": 0.2}),
                objective_metrics=MetricSet(values={"profit_over_baseline": 0.2}),
                objective_metric_id="profit_over_baseline",
                direction="maximize",
                best_epoch=1,
                best_objective_value=0.2,
            )
        )
        return SimpleNamespace(summary=object())

    monkeypatch.setattr(train_workflow, "run_persisted_training", fake_run_persisted_training)
    monkeypatch.setattr(train_workflow, "promote_paths_atomic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_workflow, "reindex_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_workflow,
        "training_result_fields",
        lambda summary, artifact_dir: [("artifact", str(artifact_dir)), ("best_epoch", "1")],
    )

    train_workflow.run(config, reporter=reporter)

    rendered = output.getvalue()
    assert "train dataset=" in rendered
    assert "prepare rows=128 samples=24" in rendered
    assert "fit started epochs=3" in rendered
    assert "fit epoch=1/3 objective.profit_over_baseline=0.2000" in rendered
    assert "validation.profit_over_baseline=0.2000" in rendered
    assert "train complete artifact=" in rendered
    assert "[running]" not in rendered
    assert "batches" not in rendered


def test_tune_workflow_emits_per_trial_not_per_epoch_output(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    override = model_workflow_override() | tune_override()
    override["tuning"] = {
        "trial_count": 2,
        "timeout_seconds": None,
        "sampler_seed": 2026,
        "enable_pruning": False,
    }
    config = _load_test_tune_config(load_workflow_config, tmp_path, override=override)
    output = StringIO()
    reporter = Reporter(stream=output)

    class FakeStudy:
        def __init__(self) -> None:
            self.trials: list[SimpleNamespace] = []
            self.best_trial: SimpleNamespace | None = None

        def optimize(self, objective, *, n_trials: int, timeout, callbacks) -> None:
            del objective, timeout
            values = (0.2, 0.35)
            epochs = (2, 3)
            for number in range(n_trials):
                trial = SimpleNamespace(
                    number=number,
                    state=TrialState.COMPLETE,
                    value=values[number],
                    user_attrs={"spice_best_epoch": epochs[number]},
                )
                self.trials.append(trial)
                if self.best_trial is None or trial.value > self.best_trial.value:
                    self.best_trial = trial
                for callback in callbacks:
                    callback(self, trial)

    fake_study = FakeStudy()
    monkeypatch.setattr(
        tune_workflow,
        "open_tuning_study",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest=object(),
            study=fake_study,
            existing_trial_count=0,
            target_trial_count=2,
            remaining_trial_count=2,
        ),
    )
    monkeypatch.setattr(tune_workflow, "build_study_summary", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(tune_workflow, "reindex_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tune_workflow,
        "study_result_fields",
        lambda _summary: [
            ("complete", "2"),
            ("pruned", "0"),
            ("failed", "0"),
            ("best_trial", "2"),
            ("best_value", "0.3500"),
        ],
    )

    tune_workflow.run(config, reporter=reporter)

    rendered = output.getvalue()
    assert "tune dataset=" in rendered
    assert "study started trials=2" in rendered
    assert "trial 1/2 complete value=0.2000 best_epoch=2" in rendered
    assert "best improved trial=1 value=0.2000" in rendered
    assert "trial 2/2 complete value=0.3500 best_epoch=3" in rendered
    assert "best improved trial=2 value=0.3500" in rendered
    assert "tune complete complete=2 pruned=0 failed=0 best_trial=2 best_value=0.3500" in rendered
    assert "fit epoch=" not in rendered
    assert "[running]" not in rendered
