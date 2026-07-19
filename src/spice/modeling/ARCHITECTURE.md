# Modeling Architecture

## Purpose

`modeling` owns sequence preparation, model-family construction, CUDA training,
forward inference, model-bound scoring, artifact assembly, and runtime result
objects. It bridges temporal examples and neural network execution.

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
  |
  v
fixed sequence preparation
  |
  v
build model via model-family registry
  |
  v
run_training_fit()
  |
  v
write artifact payloads into storage-provided staged root
```

Training is CUDA-only. `training_runner.run_training_fit()` keeps SPICE's fit
policy and delegates the epoch host loop to Lightning. Lightning owns `fit`; SPICE
still owns optimizer settings, loss computation, metric accumulation, validation
total-loss selection, finite-metric policy, best-state restore, callbacks, and
split metrics.

## Evaluation Flow

```text
loaded artifact
  |
  v
Artifact Inference Context
  |
  +--> validate artifact semantics
  +--> validate corpus coverage
  +--> reconstruct fixed sequence preparation from artifact metadata
  |
  v
EvaluationScoringRuntimePlan
  |
  v
score_evaluation(scoring_plan, evaluator_contract)
  |
  +--> score model into decoded result
  +--> evaluator.run()
  |
  v
EvaluationSummary
```

Serving can use a CPU runtime plan. Training, tuning, and evaluation use CUDA.

## Module Map

```text
modeling/
  pipeline.py             training spec construction and fit orchestration
  dataset_builders/       fixed sequence corpus preparation
  families/               neural network family configs/builders/tuning hooks
  representations/        fixed sequence model-input tensorization
  batch_plan.py           streaming DataLoader batch planning
  runtime_planning.py     device, backend, seed, precision, and model placement
  training_runtime.py     training batch plans and prediction state fitting
  lightning_module.py     thin LightningModule hosting SPICE training steps
  training_runner.py      public fit entrypoint and result assembly
  _fit_policy.py          best-state and early-stop policy
  scoring.py              model inference to evaluator bridge
  artifacts.py            artifact loading/validation helpers
  artifact_inference.py   artifact validation to inference scoring context
  persisted_training.py   training artifact write path
  ../tuning.py            one typed Study candidate execution and retention
```

## Runtime Model

`ModelingRuntimePlan` carries resolved device, precision, batch runtime context,
determinism, and seed. Precision is fixed to `32-true`. Batch planning always uses
streaming host `DataLoader`s, deterministic position grouping, optional batch-level
shuffle for training, pinned host memory when CUDA is available, and explicit
per-batch device transfer.

There is no runtime memory probing, CUDA-resident batch storage, CPU train/eval
fallback, or `torch.compile` branch. Those policies were removed to keep the
runtime direct and predictable on the intended L40 CUDA environment.

## Extension Points

Add a model family for a new neural architecture. Add prediction or evaluator
behavior in their owning packages. Do not add a new public sequence builder,
representation registry, or input-normalization selector without a new design pass.
