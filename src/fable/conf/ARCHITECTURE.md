# Checked-In Config Architecture

## Purpose

`conf` stores the package's checked-in YAML specs. These files are declarative inputs to config resolution, not executable code. They describe reusable surfaces, named implementation configs, and benchmark matrices.

## Theory

Good experiment config separates durable concepts from run controls. A surface should describe the research context: dataset, chain, problem, features, model family, and prediction family. A workflow selection or benchmark should vary run-specific selections such as delay, trial count, study name, and artifact variant.

## Invariants

YAML specs should validate through their owning package model. Config ids must match named files when the group has an identity field. Implementation-selection fields, such as evaluator `id`, must be explicit when the implementation is not implied by the group.

## Extension Points

Add a new YAML file when a reusable named concept exists. Add a new group only after adding catalog metadata in `config.group_catalog`, raw access through `config.groups`, typed loading through `config.typed_groups`, and owner-package coercion when needed.

## Group Map

```text
conf/
  surface/          reusable workflow compositions
  benchmark/        experiment matrices that expand into workflow DAGs
  chain/            chain identity and timing assumptions
  dataset/          corpus date/window identity
  provider/         RPC endpoint/provider specs
  features/         feature-composition specs
  problem/          temporal problem specs
  prediction/       prediction-family configs
  model/            model-family configs
  evaluator/        evaluator configs
  evaluations/      reusable evaluation-window suites
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

`config.groups` enforces this for groups with identity fields. This prevents one file from silently pretending to be another spec.
