# Input Normalization Architecture

## Purpose

`temporal.input_normalization` owns scaler fitting and application policies for temporal problem stores. It decides how normalization statistics are learned from training data and how fitted scaler facts transform feature matrices.

## Flow

```text
TrainingConfig.input_normalization
        |
        v
coerce_input_normalization_config()
        |
        v
compile_input_normalization_contract()
        |
        v
fit_scaler(problem_store, train_sample_indices)
        |
        v
ScalerStats
        |
        v
transform_problem_store_features(problem_store, scaler)
```

The same saved `ScalerStats` is used later during inference/evaluation so that evaluation data is transformed with training-time statistics.

## Registry Pattern

Input-normalization configs use `id` as the implementation selector. The local registry maps ids to concrete config types and compile hooks. `core.specs` supplies the mechanical owner-spec helper for payload coercion and compile-time type assertions; policy ids and fit hooks stay local.

```text
id -> config_type -> concrete compile hook -> CompiledInputNormalizationContract
```

Config-facing input-normalization coercion normalizes invalid payload envelopes to `ConfigResolutionError` and returns already typed configs unchanged.

## Theory

Normalization stabilizes model training by putting features on comparable scales. Temporal normalization must avoid leakage: statistics should be fit from allowed training rows, not future evaluation rows.

Different policies can choose different fitting windows or weights. Dataset builders call the compiled contract and do not need to know policy internals.

## Extension Points

Add a normalization policy when the fitting rule changes. Keep scaler application in this package so builders can apply any fitted scaler uniformly without knowing scaler internals.
