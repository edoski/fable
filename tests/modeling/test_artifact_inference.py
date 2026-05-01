from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from spice.config import EvaluateConfig, WorkflowTask
from spice.config.models import ChainRuntimeSpec
from spice.core.errors import ConfigResolutionError
from spice.modeling.artifact_inference import prepare_artifact_inference_context
from spice.modeling.evaluation_runtime import EvaluationScoringRuntimePlan
from spice.modeling.representations import RepresentationRuntimeContext
from spice.storage.root_handles import EvaluateWorkflowRoots
from tests.root_handle_helpers import artifact_handle, corpus_handle, evaluate_roots


def _evaluate_config(load_workflow_config, tmp_path, model_workflow_override) -> EvaluateConfig:
    del model_workflow_override
    return cast(
        EvaluateConfig,
        load_workflow_config(
            WorkflowTask.EVALUATE,
            workspace=tmp_path,
        ),
    )


def _roots(config: EvaluateConfig) -> EvaluateWorkflowRoots:
    corpus = corpus_handle(
        config.storage.root,
        chain_name="ethereum",
        dataset_id=config.dataset_id,
        dataset_name="test_dataset",
    )
    artifact = artifact_handle(
        config.storage.root,
        corpus=corpus,
        artifact_id=config.artifact_id,
    )
    return evaluate_roots(
        config.storage.root,
        corpus=corpus,
        artifact=artifact,
    )


def _install_artifact_context_fakes(monkeypatch, config: EvaluateConfig, *, max_delay: int = 36):
    calls: list[str] = []
    runtime_metadata = SimpleNamespace(compiler_runtime_metadata={"compiler": "payload"})
    loaded_artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            chain_name="ethereum",
            max_delay_seconds=max_delay,
            dataset_builder=SimpleNamespace(id="fixed_sequence_temporal"),
            builder_runtime_metadata=runtime_metadata,
            scaler=object(),
            max_candidate_slots=4,
            model=object(),
            features=object(),
            feature_graph_fingerprint="feature-fp",
            feature_prerequisites=("base_fee",),
            problem=object(),
            prediction=SimpleNamespace(id="icdcs_2026", family_id="fee_offset"),
        ),
        model=object(),
        dataset_builder_contract=None,
        representation_contract=object(),
    )
    feature_contract = SimpleNamespace(
        feature_graph_fingerprint="feature-fp",
        feature_prerequisites=("base_fee",),
    )
    problem_contract = SimpleNamespace(
        compiler_id="observed_time_window",
        max_delay_seconds=max_delay,
        required_history_seconds=120,
        warmup_rows=0,
    )
    prediction_contract = SimpleNamespace(decoded_result_id="offsets")
    evaluator_contract = SimpleNamespace(
        evaluation_id="poisson_replay_2h",
        config_payload={"id": "poisson_replay_2h"},
        metric_descriptors=(),
    )
    prepared = SimpleNamespace(
        n_history_rows=10,
        n_evaluation_rows=5,
        sample_count=2,
        execution_policy=object(),
        store=object(),
        sample_indices=np.array([0, 1], dtype=np.int64),
    )
    corpus_manifest = SimpleNamespace(
        chain=SimpleNamespace(
            name="ethereum",
            runtime=ChainRuntimeSpec(
                chain_id=1,
                uses_poa_extra_data=False,
                nominal_block_time_seconds=12.0,
            ),
        ),
        coverage=SimpleNamespace(
            evaluation=SimpleNamespace(start_timestamp=1000, end_timestamp=2000)
        ),
    )

    monkeypatch.setattr(
        "spice.modeling.artifact_inference.load_training_artifact",
        lambda artifact_root: (
            calls.append(f"load_artifact:{artifact_root.name}") or loaded_artifact
        ),
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.compile_feature_contract",
        lambda **_kwargs: calls.append("compile_features") or feature_contract,
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.compile_problem_contract",
        lambda **_kwargs: calls.append("compile_problem") or problem_contract,
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.compile_prediction_contract",
        lambda **_kwargs: calls.append("compile_prediction") or prediction_contract,
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.compile_evaluator_contract",
        lambda _config: calls.append("compile_evaluator") or evaluator_contract,
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.load_dataset_manifest",
        lambda _path: calls.append("load_dataset_manifest") or corpus_manifest,
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.validate_corpus_coverage",
        lambda *_args, **_kwargs: calls.append("validate_coverage"),
    )
    monkeypatch.setattr(
        "spice.modeling.artifact_inference.load_block_frame",
        lambda path: calls.append(f"load_blocks:{path.name}") or object(),
    )
    monkeypatch.setattr(
        "spice.modeling.scoring.build_evaluation_scoring_runtime_plan",
        lambda **kwargs: calls.append(f"build_runtime:{kwargs['batch_size']}")
        or EvaluationScoringRuntimePlan(
            resolved_device=torch.device("cpu"),
            precision="fp32",
            representation_runtime_context=RepresentationRuntimeContext(
                batch_size=kwargs["batch_size"],
                available_host_memory_bytes=1024,
            ),
            deterministic=None,
            seed=0,
        ),
    )

    def fake_prepare_inference_dataset(*_args, spec):
        calls.append("prepare_inference")
        assert spec.builder_runtime_metadata is runtime_metadata
        assert spec.max_candidate_slots == loaded_artifact.manifest.max_candidate_slots
        return prepared

    loaded_artifact.dataset_builder_contract = SimpleNamespace(
        prepare_inference_dataset=fake_prepare_inference_dataset,
    )
    return calls, loaded_artifact, prepared, evaluator_contract, prediction_contract


def test_artifact_inference_context_prepares_scoring_inputs(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _evaluate_config(load_workflow_config, tmp_path, model_workflow_override)
    roots = _roots(config)
    calls, loaded_artifact, prepared, evaluator_contract, prediction_contract = (
        _install_artifact_context_fakes(monkeypatch, config)
    )

    context = prepare_artifact_inference_context(
        config,
        corpus=roots.corpus,
        artifact=roots.artifact,
    )

    assert context.loaded_artifact is loaded_artifact
    assert context.prepared is prepared
    assert context.evaluator_contract is evaluator_contract
    assert context.scoring_context.model_input.model is loaded_artifact.model
    assert context.scoring_context.model_input.prediction_contract is prediction_contract
    assert context.scoring_context.model_input.sample_indices.tolist() == [0, 1]
    assert calls == [
        f"load_artifact:{roots.artifact.root_path.name}",
        "load_dataset_manifest",
        "compile_features",
        "compile_problem",
        "compile_prediction",
        "compile_evaluator",
        "validate_coverage",
        "load_blocks:history",
        "load_blocks:evaluation",
        "prepare_inference",
        f"build_runtime:{config.batch_size}",
    ]


def test_artifact_inference_passes_inclusive_corpus_evaluation_window(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _evaluate_config(load_workflow_config, tmp_path, model_workflow_override)
    roots = _roots(config)
    captured: dict[str, int] = {}

    def fake_prepare_inference_dataset(*_args, spec):
        captured["evaluation_start_timestamp"] = spec.evaluation_start_timestamp
        captured["evaluation_end_timestamp"] = spec.evaluation_end_timestamp
        return SimpleNamespace(
            n_history_rows=10,
            n_evaluation_rows=5,
            sample_count=1,
            execution_policy=object(),
            store=object(),
            sample_indices=np.array([0], dtype=np.int64),
        )

    loaded_artifact = _install_artifact_context_fakes(monkeypatch, config)[1]
    loaded_artifact.dataset_builder_contract = SimpleNamespace(
        prepare_inference_dataset=fake_prepare_inference_dataset,
    )

    prepare_artifact_inference_context(
        config,
        corpus=roots.corpus,
        artifact=roots.artifact,
    )

    assert captured == {
        "evaluation_start_timestamp": 1000,
        "evaluation_end_timestamp": 2000,
    }


def test_artifact_inference_context_rejects_artifact_semantic_mismatch(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _evaluate_config(load_workflow_config, tmp_path, model_workflow_override)
    roots = _roots(config)
    calls, loaded_artifact, *_ = _install_artifact_context_fakes(monkeypatch, config)

    loaded_artifact.manifest.feature_graph_fingerprint = "other"

    with pytest.raises(ConfigResolutionError, match="feature graph"):
        prepare_artifact_inference_context(
            config,
            corpus=roots.corpus,
            artifact=roots.artifact,
        )


def test_artifact_inference_context_rejects_delay_beyond_artifact_capability(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _evaluate_config(
        load_workflow_config,
        tmp_path,
        lambda: model_workflow_override(max_delay_seconds=36, delay_seconds=36),
    )
    config = config.model_copy(update={"delay_seconds": 36})
    roots = _roots(config)
    _install_artifact_context_fakes(monkeypatch, config, max_delay=12)

    with pytest.raises(ConfigResolutionError, match="delay_seconds exceeds artifact capability"):
        prepare_artifact_inference_context(
            config,
            corpus=roots.corpus,
            artifact=roots.artifact,
        )


def test_artifact_inference_context_validates_coverage_before_inference_preparation(
    tmp_path,
    monkeypatch,
    load_workflow_config,
    model_workflow_override,
) -> None:
    config = _evaluate_config(load_workflow_config, tmp_path, model_workflow_override)
    roots = _roots(config)
    calls, *_ = _install_artifact_context_fakes(monkeypatch, config)

    def fail_coverage(*_args, **_kwargs):
        calls.append("validate_coverage")
        raise ConfigResolutionError("coverage mismatch")

    monkeypatch.setattr(
        "spice.modeling.artifact_inference.validate_corpus_coverage",
        fail_coverage,
    )

    with pytest.raises(ConfigResolutionError, match="coverage mismatch"):
        prepare_artifact_inference_context(
            config,
            corpus=roots.corpus,
            artifact=roots.artifact,
        )

    assert "prepare_inference" not in calls
