from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import cast

from optuna.trial import TrialState

from spice.config import EvaluateConfig, TrainConfig, TuneConfig, WorkflowTask
from spice.core.reporting import Reporter
from spice.evaluation import EvaluationSummary
from spice.modeling.training_runner import TrainingEpochProgress
from spice.prediction import MetricDescriptor, MetricSet
from spice.workflows import evaluate as evaluate_workflow
from spice.workflows import train as train_workflow
from spice.workflows import tune as tune_workflow
from tests.root_handle_helpers import (
    artifact_handle,
    baseline_train_roots,
    corpus_handle,
    evaluate_roots,
    study_handle,
    tune_roots,
    tuned_train_roots,
)


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
            surface="current_row_fee_dynamics",
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
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )


def _load_test_evaluate_config(
    load_workflow_config,
    tmp_path,
    *,
    override: dict[str, object] | None = None,
) -> EvaluateConfig:
    return cast(
        EvaluateConfig,
        load_workflow_config(
            WorkflowTask.EVALUATE,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=override,
        ),
    )


def test_evaluate_workflow_delegates_artifact_inference_preparation(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _load_test_evaluate_config(
        load_workflow_config,
        tmp_path,
        override=model_workflow_override(),
    )
    output = StringIO()
    reporter = Reporter(stream=output)
    calls: list[str] = []
    prepared = SimpleNamespace(n_history_rows=10, n_evaluation_rows=5, sample_count=2)
    evaluator_contract = SimpleNamespace(
        evaluation_id="poisson_replay_2h",
        config_payload={"id": "poisson_replay_2h"},
        metric_descriptors=(
            MetricDescriptor(
                id="profit_over_baseline",
                label="profit over baseline",
                role="primary",
            ),
        ),
    )
    context = SimpleNamespace(
        loaded_artifact=SimpleNamespace(manifest=object()),
        prepared=prepared,
        evaluator_contract=evaluator_contract,
        scoring_context=object(),
        delay_seconds=36,
    )
    evaluation = EvaluationSummary(
        metrics=MetricSet(values={"profit_over_baseline": 0.25}),
        window_metrics={},
        total_events=2,
        runs=[],
    )

    corpus = corpus_handle(
        tmp_path / "outputs",
        dataset_id=config.dataset_id,
        dataset_name="test_dataset",
    )
    artifact = artifact_handle(
        tmp_path / "outputs",
        corpus=corpus,
        artifact_id=config.artifact_id,
    )
    roots = evaluate_roots(tmp_path / "outputs", corpus=corpus, artifact=artifact)

    def fake_prepare(active_config, *, corpus, artifact):
        del corpus
        calls.append(f"prepare:{active_config.delay_seconds}:{artifact.root_path.name}")
        return context

    def fake_score(scoring_context):
        calls.append(f"score:{scoring_context is context.scoring_context}")
        return evaluation

    monkeypatch.setattr(
        evaluate_workflow,
        "prepare_artifact_inference_context",
        fake_prepare,
    )
    monkeypatch.setattr(evaluate_workflow, "score_evaluation", fake_score)
    monkeypatch.setattr(
        evaluate_workflow,
        "upsert_evaluation_state",
        lambda _path, **_kwargs: ("poisson_replay_2h-36s-test", 123),
    )
    monkeypatch.setattr(
        evaluate_workflow,
        "evaluation_result_fields",
        lambda _summary: [("evaluation", "poisson_replay_2h"), ("events", "2")],
    )
    monkeypatch.setattr(
        evaluate_workflow,
        "resolve_evaluate_consumer_roots",
        lambda _config: roots,
    )

    evaluate_workflow.run(config, reporter=reporter)

    assert calls[0].startswith(f"prepare:{config.delay_seconds}:")
    assert calls[1:] == ["score:True"]
    rendered = output.getvalue()
    assert "evaluate dataset=test_dataset dataset_id=" in rendered
    assert "prepare history_rows=10 evaluation_rows=5 samples=2" in rendered
    assert "evaluate complete evaluation=poisson_replay_2h events=2" in rendered


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
        "build_artifact_training_spec",
        lambda _config, **_kwargs: SimpleNamespace(
            training=SimpleNamespace(max_epochs=3),
            prediction_contract=SimpleNamespace(primary_metric_id="profit_over_baseline"),
            problem_contract=object(),
            feature_contract=object(),
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
    monkeypatch.setattr(
        train_workflow,
        "resolve_train_consumer_roots",
        lambda _config: baseline_train_roots(
            tmp_path / "outputs",
            corpus=corpus_handle(
                tmp_path / "outputs",
                chain_name=config.chain.name,
                dataset_id=config.dataset_id,
                dataset_name=config.dataset.name,
            ),
        ),
    )
    monkeypatch.setattr(
        train_workflow,
        "load_dataset_manifest",
        lambda *_args: SimpleNamespace(chain=SimpleNamespace(name=config.chain.name)),
    )
    monkeypatch.setattr(train_workflow, "training_coverage_requirement", lambda *_args: object())
    monkeypatch.setattr(train_workflow, "validate_corpus_coverage", lambda *_args, **_kwargs: None)

    class FakeStage:
        staged_root = tmp_path / "stage"

        def __enter__(self):
            self.staged_root.mkdir(parents=True, exist_ok=True)
            return self

        def __exit__(self, *_args):
            return None

        def promote(self) -> None:
            return None

    monkeypatch.setattr(train_workflow, "staged_root", lambda **_kwargs: FakeStage())
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
    assert "train complete artifact=" in rendered


def test_train_workflow_accepts_id_resolved_corpus_chain(
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
    reporter = Reporter(stream=StringIO())

    monkeypatch.setattr(
        train_workflow,
        "resolve_train_consumer_roots",
        lambda _config: baseline_train_roots(
            tmp_path / "outputs",
            corpus=corpus_handle(
                tmp_path / "outputs",
                chain_name="polygon",
                dataset_id=config.dataset_id,
                dataset_name="polygon_dataset",
            ),
        ),
    )
    corpus_manifest = SimpleNamespace(
        chain=SimpleNamespace(
            name="polygon",
            runtime=SimpleNamespace(),
        ),
        dataset=SimpleNamespace(name="polygon_dataset"),
    )
    monkeypatch.setattr(train_workflow, "load_dataset_manifest", lambda *_args: corpus_manifest)

    def fake_build_training_spec(active_config, *, corpus_manifest, **_kwargs):
        assert active_config.chain.name == "ethereum"
        assert corpus_manifest.chain.name == "polygon"
        return SimpleNamespace(
            training=SimpleNamespace(max_epochs=1),
            prediction_contract=SimpleNamespace(primary_metric_id="total_loss"),
            problem_contract=object(),
            feature_contract=object(),
        )

    monkeypatch.setattr(
        train_workflow,
        "build_artifact_training_spec",
        fake_build_training_spec,
    )
    monkeypatch.setattr(train_workflow, "training_coverage_requirement", lambda *_args: object())
    monkeypatch.setattr(train_workflow, "validate_corpus_coverage", lambda *_args, **_kwargs: None)

    class FakeStage:
        staged_root = tmp_path / "stage"

        def __enter__(self):
            self.staged_root.mkdir(parents=True, exist_ok=True)
            return self

        def __exit__(self, *_args):
            return None

        def promote(self) -> None:
            return None

    monkeypatch.setattr(train_workflow, "staged_root", lambda **_kwargs: FakeStage())
    monkeypatch.setattr(
        train_workflow,
        "run_persisted_training",
        lambda *_args, **_kwargs: SimpleNamespace(summary=object()),
    )
    monkeypatch.setattr(
        train_workflow,
        "training_result_fields",
        lambda _summary, *, artifact_dir: [("artifact", str(artifact_dir))],
    )

    train_workflow.run(config, reporter=reporter)


def test_tuned_train_keeps_artifact_path_stable_after_best_params(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = cast(
        TrainConfig,
        load_workflow_config(
            WorkflowTask.TRAIN,
            workspace=tmp_path,
            surface="current_row_fee_dynamics",
            override=model_workflow_override(),
            variant="tuned",
        ),
    )
    reporter = Reporter(stream=StringIO())
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id="cor_9a73b1e88edb488afb1e",
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        tmp_path / "outputs",
        corpus=corpus,
        study_id="std_test",
        study_name=config.study.name,
    )
    roots = tuned_train_roots(
        tmp_path / "outputs",
        corpus=corpus,
        study=study,
        artifact_id="art_original",
    )
    original_artifact_root = roots.artifact.root_path
    resolved_batch_sizes: list[int] = []
    captured: dict[str, object] = {}

    def fake_resolve(active_config):
        resolved_batch_sizes.append(active_config.training.batch_size)
        return roots

    tuned_training = config.training.model_copy(
        update={"batch_size": config.training.batch_size + 7}
    )
    tuned_config = config.model_copy(update={"training": tuned_training})
    monkeypatch.setattr(train_workflow, "resolve_train_consumer_roots", fake_resolve)
    monkeypatch.setattr(
        train_workflow,
        "apply_study_best_params",
        lambda *_args, **_kwargs: SimpleNamespace(config=tuned_config),
    )
    monkeypatch.setattr(
        train_workflow,
        "load_dataset_manifest",
        lambda *_args: SimpleNamespace(
            chain=SimpleNamespace(name=config.chain.name, runtime=SimpleNamespace()),
            dataset=SimpleNamespace(name=config.dataset.name),
        ),
    )

    def fake_build_training_spec(active_config, *, artifact, **_kwargs):
        captured["training_batch_size"] = active_config.training.batch_size
        captured["artifact_root"] = artifact.root_path
        return SimpleNamespace(
            training=SimpleNamespace(max_epochs=1),
            prediction_contract=SimpleNamespace(primary_metric_id="total_loss"),
            problem_contract=object(),
            feature_contract=object(),
        )

    monkeypatch.setattr(
        train_workflow,
        "build_artifact_training_spec",
        fake_build_training_spec,
    )
    monkeypatch.setattr(train_workflow, "training_coverage_requirement", lambda *_args: object())
    monkeypatch.setattr(train_workflow, "validate_corpus_coverage", lambda *_args, **_kwargs: None)

    class FakeStage:
        staged_root = tmp_path / "stage"

        def __enter__(self):
            self.staged_root.mkdir(parents=True, exist_ok=True)
            return self

        def __exit__(self, *_args):
            return None

        def promote(self) -> None:
            return None

    monkeypatch.setattr(train_workflow, "staged_root", lambda **_kwargs: FakeStage())
    monkeypatch.setattr(
        train_workflow,
        "run_persisted_training",
        lambda *_args, **_kwargs: SimpleNamespace(summary=object()),
    )
    monkeypatch.setattr(
        train_workflow,
        "training_result_fields",
        lambda _summary, *, artifact_dir: [("artifact", str(artifact_dir))],
    )

    train_workflow.run(config, reporter=reporter)

    assert resolved_batch_sizes == [config.training.batch_size]
    assert captured == {
        "training_batch_size": tuned_training.batch_size,
        "artifact_root": original_artifact_root,
    }


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
    corpus = corpus_handle(
        tmp_path / "outputs",
        chain_name=config.chain.name,
        dataset_id=config.dataset_id,
        dataset_name=config.dataset.name,
    )
    study = study_handle(
        tmp_path / "outputs",
        corpus=corpus,
        study_id="std_test",
        study_name=config.study.name,
    )
    roots = tune_roots(tmp_path / "outputs", corpus=corpus, study=study)
    monkeypatch.setattr(
        tune_workflow,
        "resolve_tune_consumer_roots",
        lambda _config: roots,
    )
    monkeypatch.setattr(
        tune_workflow,
        "load_dataset_manifest",
        lambda *_args: SimpleNamespace(
            chain=SimpleNamespace(name=config.chain.name),
            dataset=SimpleNamespace(name=config.dataset.name),
        ),
    )
    monkeypatch.setattr(
        tune_workflow,
        "_coverage_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            problem_contract=object(),
            feature_contract=object(),
        ),
    )
    monkeypatch.setattr(tune_workflow, "training_coverage_requirement", lambda *_args: object())
    monkeypatch.setattr(tune_workflow, "validate_corpus_coverage", lambda *_args, **_kwargs: None)
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
    monkeypatch.setattr(tune_workflow, "reindex_catalog_root", lambda *_args, **_kwargs: None)
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
    assert "trial 2/2 complete value=0.3500 best_epoch=3" in rendered
    assert "tune complete" in rendered
    assert "fit epoch=" not in rendered


def test_tune_workflow_accepts_id_resolved_corpus_chain(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
    tune_override,
) -> None:
    override = model_workflow_override() | tune_override()
    config = _load_test_tune_config(load_workflow_config, tmp_path, override=override)
    reporter = Reporter(stream=StringIO())

    monkeypatch.setattr(
        tune_workflow,
        "resolve_tune_consumer_roots",
        lambda _config: tune_roots(
            tmp_path / "outputs",
            corpus=corpus_handle(
                tmp_path / "outputs",
                chain_name="polygon",
                dataset_id=config.dataset_id,
                dataset_name="polygon_dataset",
            ),
            study=study_handle(
                tmp_path / "outputs",
                corpus=corpus_handle(
                    tmp_path / "outputs",
                    chain_name="polygon",
                    dataset_id=config.dataset_id,
                    dataset_name="polygon_dataset",
                ),
                study_id="std_test",
                study_name=config.study.name,
            ),
        ),
    )
    corpus_manifest = SimpleNamespace(
        chain=SimpleNamespace(
            name="polygon",
            runtime=SimpleNamespace(),
        ),
        dataset=SimpleNamespace(name="polygon_dataset"),
    )
    monkeypatch.setattr(tune_workflow, "load_dataset_manifest", lambda *_args: corpus_manifest)

    def fake_coverage_spec(active_config, *, corpus_manifest, **_kwargs):
        assert active_config.chain.name == "ethereum"
        assert corpus_manifest.chain.name == "polygon"
        return SimpleNamespace(problem_contract=object(), feature_contract=object())

    monkeypatch.setattr(tune_workflow, "_coverage_spec", fake_coverage_spec)
    monkeypatch.setattr(tune_workflow, "training_coverage_requirement", lambda *_args: object())
    monkeypatch.setattr(tune_workflow, "validate_corpus_coverage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tune_workflow, "reindex_catalog_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tune_workflow,
        "open_tuning_study",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest=object(),
            study=SimpleNamespace(trials=[]),
            existing_trial_count=0,
            target_trial_count=0,
            remaining_trial_count=0,
        ),
    )
    monkeypatch.setattr(tune_workflow, "build_study_summary", lambda *_args: object())
    monkeypatch.setattr(
        tune_workflow,
        "study_result_fields",
        lambda _summary: [("complete", "0")],
    )

    tune_workflow.run(config, reporter=reporter)
