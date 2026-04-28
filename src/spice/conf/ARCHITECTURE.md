# Checked-In Config Architecture

## Purpose

`conf` stores the package's checked-in YAML specs. These files are declarative inputs to config resolution, not executable code. They describe reusable surfaces, named implementation configs, and benchmark matrices.

## Theory

Good experiment config separates durable concepts from run controls. A surface should describe the research context: dataset, chain, problem, features, model family, objective, and evaluator. A workflow request or benchmark should vary run-specific selections such as delay, trial count, study name, and artifact variant.

## Invariants

YAML specs should validate through their owning package model. Config ids must match named files when the group has an identity field. Implementation-selection fields, such as evaluator `engine`, must be explicit when the implementation is not implied by the group.

## Extension Points

Add a new YAML file when a reusable named concept exists. Add a new group only after adding typed validation in `config.registry` and owner-package coercion.

## Group Map

```text
conf/
  surface/          reusable workflow compositions
  benchmark/        experiment matrices that expand into workflow DAGs
  chain/            chain identity and timing assumptions
  dataset/          dataset date/window identity
  provider/         RPC endpoint/provider specs
  features/      requested feature compositions
  problem/          temporal problem specs
  dataset_builder/  tensorization strategy specs
  prediction/       prediction-family configs
  model/            model-family configs
  objective/        optimization objective configs
  evaluation/       evaluator configs
  training/         optimizer and loop settings
  split/            train/validation/test split settings
  tuning/           tuning run controls
  tuning_space/     hyperparameter search spaces
  execution/        remote target specs
```

## YAML Identity Rule

Some groups have file-name identity:

```text
conf/problem/my_problem.yaml
  id: my_problem
```

The registry enforces this for groups with identity fields. This prevents one file from silently pretending to be another spec.
