# Concrete CLI Commands

CLI commands are the user-facing edge. They resolve named configs, own command defaults, submit remote jobs, inspect storage, and start transfers.

## Mental Model

```text
terminal command
  -> Typer command parser
  -> config/surface resolution
  -> local acquire or remote submission
```

The CLI should be ergonomic. Lower layers should stay explicit and typed.

Workflow commands keep explicit Typer signatures, build typed **Workflow Selections** from CLI values in-place, then call config resolution before local acquire or remote submission.

## Workflow Commands

Commands:

| Command | Behavior |
| --- | --- |
| `acquire` | Run acquisition locally. |
| `train` | Submit training remotely. |
| `tune` | Submit tuning remotely. |
| `evaluate` | Submit evaluation remotely. |

`train`, `tune`, and `evaluate` support `--dependency`, `--target`, and `--detach`. They do not expose `--submit` or `--storage-root`; remote execution rewrites storage root from the selected target config.

Default remote target is `disi_l40` at the CLI layer.

## Storage Show Commands

`show dataset`, `show study`, and `show artifact` query the catalog. With broad filters, they list matches. With one exact match and detail options, they render root-specific state.

Detail panels:

| Root | Detail views |
| --- | --- |
| Dataset | Manifest and acquire runs. |
| Study | Manifest, trials, config. |
| Artifact | Manifest, training epochs, evaluation runs. |

## Delete Commands

Delete commands require one resolved root. Dataset and study deletes protect dependent roots unless cascade is explicitly requested.

```text
selector
  -> exactly one catalog match
  -> dependency check
  -> root-kind validation
  -> remove root
  -> refresh/reindex catalog
```

## Refresh Command

`refresh catalog` rebuilds the derived catalog from root state DBs. This is useful after manual file movement or transfer completion.

## Transfer Commands

Supported commands:

| Command | Direction |
| --- | --- |
| `transfer push dataset` | Local acquired dataset/corpus root to cluster. |
| `transfer pull artifact` | Cluster artifact root to local. |

Study transfer is not operator-facing because train and tune run remotely. `pull artifact` exists for inspection, archive, and benchmark collection. `--replace` controls destination replacement.

## Config And Benchmark Commands

Config commands inspect resolved config and available specs.

Benchmark commands use the same config resolution stack:

| Command | Behavior |
| --- | --- |
| `benchmark plan <name> --target <target>` | Create a durable run dir with metadata and resolved plan JSONL. |
| `benchmark submit <run-dir>` | Submit exactly the persisted plan using the target in run metadata. |
| `benchmark collect <run-dir>` | Pull every expected evaluate artifact, refuse partial collection, then replace `collection.json` and upsert `results.sqlite`. |
| `benchmark index export --output <csv>` | Export CSV from the selected result index/query, overwriting the destination. |
| `benchmark show <run-dir>` | Print read-only run state. |
| `benchmark index rebuild/show/list` | Rebuild and inspect the SQLite benchmark result projection. |

Run dirs plus `collection.json` are source of truth. `benchmarks/results.sqlite` is rebuildable query state. `benchmarks/results.csv` is export-only.

## Invariants

| Rule | Why |
| --- | --- |
| Remote target fallback lives only here. | Execution and transfer APIs stay explicit. |
| Dependency and detach only exist on remote workflows. | They only make sense for scheduled jobs. |
| Detail views require one match. | Avoids rendering mixed root state. |
| Delete validates root kind. | Protects storage layout. |

## Extension Pattern

New commands should parse user intent, resolve typed config or selectors, then delegate to workflow/storage/execution functions. Keep business logic in owner packages.
