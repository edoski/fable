# Modeling Architecture

## Purpose

`modeling` owns tensorization, model-family construction, training loops, inference, scoring, trained artifact assembly, runtime training/evaluation result objects, and model-bound scoring.

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
write artifact payloads into storage-provided staged root
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
EvaluationScoringRuntimePlan
  +--> prepared store
  +--> representation/prediction/execution/evaluation contracts
  +--> scoring runtime plan
  |
  v
score_evaluation(scoring_plan, evaluator_contract)
  |
  +--> evaluator checks accepted decoded-result id
  +--> score model into a decoded result
  +--> evaluator.run()
  |
  v
EvaluationSummary
```

Artifact inference validation is centralized in `modeling.artifact_inference`. It turns an active evaluate workflow config plus a trained artifact into trusted scoring inputs, then delegates tensorization state reconstruction to the Temporal Dataset Preparation Interface. Evaluator execution stays in `modeling.scoring` when model inference is involved.

Artifact training and tuning trials use separate training-spec entrypoints so artifact identity and study identity stay explicit at the workflow seam.

Persisted training writes model files and artifact state into the directory supplied by the workflow storage effect. Modeling owns payload production and split-metric evaluation; storage owns whether that directory becomes the committed artifact root. Tuning trials run through the non-persisted trial-training path and keep model selection in memory.

Tuning Execution is centralized in `modeling.tuning_execution`. It opens compatible study state, validates resume counts, runs Optuna trials, records trial metadata, and returns storage-owned study summaries. Workflows resolve roots, validate coverage, attach reporter callbacks, and delegate study reindex effects to storage.

During training, `Objective Runtime` owns objective metric production. Validation objectives return validation metrics directly. Evaluation objectives receive the `EvaluationScoringRuntimePlan` built by the Training Runner and call the generic model-to-evaluator scoring bridge; the training loop still does not know evaluator internals.

## Dataset Builders

Dataset builders adapt canonical block frames into model-ready training and inference datasets. They own tensorization policy, split behavior, builder runtime metadata, scaler fitting inputs, and inference-time reconstruction from the artifact Temporal Capability. They do not own corpus IO, feature semantics, prediction losses, or evaluator metrics.
Callers provide domain facts such as split, delay, and evaluation window plus compiled/trusted context; builders decide how those facts become samples and tensors.

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
problem store + Temporal Capability
    |
    v
prepared dataset + builder runtime metadata
```

Shared dataset-builder abstractions should remove real duplication without hiding sample-ordering, split, candidate-window, or empty-frame policy.

## Module Map

```text
modeling/
  pipeline.py             training spec construction and fit orchestration
  dataset_builders/       temporal dataset preparation and tensorization strategies
  families/               neural network family configs/builders/tuning hooks
  models.py               model protocols and model-output envelope
  representations.py      representation contracts and model-input preparation
  results.py              runtime training/evaluation result records
  summary.py              rendered model/training/evaluation summaries
  tuned_config.py         tuned artifact config alignment
  tuning.py               tuned artifact parameter application
  batch_plan.py           training/inference batch planning and loader/storage policy
  _runtime.py             shared PyTorch/CUDA runtime helpers
  _runtime_probe.py       private host-warmup and measured-budget helpers
  forward_runtime.py      forward-only warmup, memory measurement, and execution
  runtime_planning.py     device, backend, seed, precision, and model preparation plans
  training_runtime.py     training warmup, budget planning, and prediction state fitting
  training_runner.py      fit and split metric execution
  training_run.py         neutral training run result envelope
  _epoch_execution.py     private train/validation batch mechanics
  _fit_policy.py          private best-state and early-stop policy
  objective_runtime.py    objective contract and metric production during training
  scoring.py              model inference -> evaluator bridge
  artifacts.py            artifact loading/validation helpers
  artifact_inference.py   artifact validation -> inference scoring context
  persisted_training.py   training artifact write path
  tuning_execution.py     study opening, trial execution, and summary production
```

## Extension Points

Add a model family for a new neural architecture. Add a dataset builder for a new tensorization strategy. Add scoring behavior only when it is generic model-to-evaluator bridging; evaluator-specific scoring belongs in `evaluation`.

Runtime planning is intentionally split from fit policy. `ModelingRuntimePlan` is the unit callers pass into training, inference, split-metric forward passes, and model-bound evaluator scoring; callers do not pass device, precision, seed, or runtime-context fragments around the seam. `BatchRuntimeContext` carries the batch size, host-memory facts, host loader policy, and a `DeviceStorageBudget`, which names whether CUDA-resident batch storage is disabled, coarse, or measured. Batch Plan adapts that context into the smaller `RepresentationRuntimeContext` when it asks the Representation seam to prepare model inputs. `_runtime_probe.py` owns the shared host-warmup to measured-plan transition. Forward scoring uses a no-grad eval probe and is non-destructive. Training owns the destructive gradient-bearing probe, temporary optimizer, model-state restoration, and CUDA cache cleanup. Batch planning consumes the budget and caller-prepared temporal facts or Action Space; it does not prepare policy facts, measure CUDA memory, or revalidate selected-sample alignment. `training_runner.py` remains the public fit interface and keeps callback, best-state, and result assembly ownership.
