# Modeling Architecture

## Purpose

`modeling` owns tensorization, model-family construction, training loops, inference, scoring, trained artifact assembly, and model result payload codecs.

It is the bridge between generic temporal examples and neural network execution.

## Training Flow

```text
TrainConfig
  |
  v
build_artifact_training_spec() / build_trial_training_spec()
  |
  +--> feature contract
  +--> problem contract
  +--> prediction contract
  +--> objective runtime
  +--> dataset-builder contract
  +--> input-normalization contract
  |
  v
Temporal Dataset Preparation Interface
  |
  v
build model family
  |
  v
run_training_fit()
  |
  v
persist artifact state
```

The model family is only one part of learning. The same model architecture can learn a different task if the prediction family changes. The same prediction task can use a different model family if model config changes.

## Evaluation/Scoring Flow

```text
loaded artifact
  |
  v
Artifact Inference Context
  |
  +--> validate artifact semantics
  +--> validate corpus coverage
  +--> call Temporal Dataset Preparation Interface
  |
  v
ModelScoringInput
  +--> prepared store
  +--> representation/prediction/execution/evaluation contracts
  +--> scoring runtime plan
  |
  v
score_evaluation(model_input, evaluator_contract)
  |
  +--> evaluator checks accepted decoded-result id
  +--> predict_with_model()
  +--> evaluator.run()
  |
  v
EvaluationSummary
```

Artifact inference validation is centralized in `modeling.artifact_inference`. It turns an active evaluate workflow config plus a trained artifact into trusted scoring inputs, then delegates tensorization state reconstruction to the Temporal Dataset Preparation Interface. Evaluator execution stays in `modeling.scoring` when model inference is involved.

Artifact training and temporary tuning trials use separate training-spec entrypoints so artifact identity and study identity stay explicit at the workflow seam.

## Dataset Builders

Dataset builders adapt canonical block frames into model-ready training and inference datasets. They own tensorization policy, split behavior, runtime metadata, scaler fitting inputs, and inference-time reconstruction of builder/compiler runtime state. They do not own corpus IO, feature semantics, prediction losses, or evaluator metrics.

```text
raw/canonical blocks
    |
    v
builder-local frame preparation
    |
    v
feature table
    |
    v
problem store
    |
    v
prepared dataset + runtime metadata
```

Shared dataset-builder abstractions should remove real duplication without hiding sample-ordering, split, candidate-window, or empty-frame policy.

## Module Map

```text
modeling/
  pipeline.py             training spec construction and fit orchestration
  dataset_builders/       temporal dataset preparation and tensorization strategies
  families/               neural network family configs/builders/tuning hooks
  batch_plan.py           training/inference batch planning
  forward_runtime.py      forward-only warmup, memory measurement, and execution
  training_runtime.py     training warmup, budget planning, and prediction state fitting
  training_runner.py      fit and split metric execution
  training_run.py         neutral training run result envelope
  _epoch_execution.py     private train/validation batch mechanics
  _fit_policy.py          private best-state and early-stop policy
  objective_runtime.py    objective contract and metric production during training
  inference.py            model prediction over prepared stores
  scoring.py              model inference -> evaluator bridge
  artifacts.py            artifact loading/validation helpers
  artifact_inference.py   artifact validation -> inference scoring context
  persisted_training.py   training artifact write path
```

## Extension Points

Add a model family for a new neural architecture. Add a dataset builder for a new tensorization strategy. Add scoring behavior only when it is generic model-to-evaluator bridging; evaluator-specific scoring belongs in `evaluation`.

Runtime planning is intentionally split from fit policy. `_runtime.py` owns CUDA memory discovery, budget arithmetic, and shared model-forward execution. `batch_plan.py` consumes a supplied representation runtime context to choose sample order, target binding, host/device storage, and OOM fallback; it does not probe CUDA memory. `forward_runtime.py` owns forward-only host warmup and measured final batch planning for inference and split metrics. `training_runtime.py` owns the destructive gradient-bearing probe, restores model state, clears CUDA cache after that probe, and returns final train/validation batch plans plus one reusable prediction training state. `training_runner.py` remains the public fit interface and keeps callback, best-state, and result assembly ownership.
