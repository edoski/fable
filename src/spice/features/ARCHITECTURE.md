# Features Architecture

`features` turns canonical block rows into finite numeric model inputs. It owns feature configs, source availability, feature formulas, prerequisites, fingerprints, and table execution.

Features answer what information is observable at the decision row. They do not define labels, losses, model architecture, evaluator metrics, storage layout, or workflow specs.

## Flow

```text
FeaturesConfig(id, outputs)
        |
        v
feature catalog lookup
        |
        v
CompiledFeatureContract executable plan
        |
        v
SourceSpec availability + lag/null policy
        |
        v
FeatureSpec formulas
        |
        v
ResolvedFeatureTable
```

`CompiledFeatureContract` directly owns the executable feature plan for a selected catalog: dependency order, source names, required canonical columns, acquisition enrichments, prerequisites, fingerprint facts, and table execution. Callers consume the compiled contract instead of recomputing catalog graph facts.

`SourceSpec` owns causality, availability, required canonical columns, and optional acquisition enrichments. Current base fee is allowed as `base_fee_per_gas[t]` because EIP-1559 base fee for block `t` is deterministic from parent state and observable before block `t` execution. Canonical finalized block facts such as gas used and tx count are exposed only through lagged sources.

`FeatureSpec` owns formulas over source and feature dependencies. The default catalog is `core_fee_dynamics`.

## Catalog Family

`core_fee_dynamics` is a catalog family with separate registered catalogs for the safe baseline, priority-fee extension, and elapsed-position ablation. Each registered catalog owns its selectable outputs and fingerprint sources. Formula helpers are split by behavior owner: transforms, time/cadence, base fee, block facts, fee context, and priority fee. The registry does not slice one broad catalog with external allow-lists.

The safe `core_fee_dynamics` catalog is protocol-first and includes the safe local-trend signals that improved the 1M A/B grid. Output sets are composed from explicit Python groups so YAML specs can stay fully expanded while tests verify they match the canonical composition.

| Group | Outputs | Reason |
| --- | --- | --- |
| Current base fee | `log_base_fee_per_gas` | Current base fee is the direct fee level the decision must reason about. |
| Previous block pressure | `log_prev_gas_used`, `log_prev_gas_limit`, `prev_gas_utilization` | Finalized demand/pressure from the previous block is high-signal and safe. |
| Transaction count | `log_prev_tx_count` | Captures recent activity density without receipt/log ingestion. |
| Cadence/calendar | `seconds_since_previous_block`, `hour_*`, `dow_*` | Captures timing irregularity and coarse daily/weekly effects. |
| Rolling fee context | `roll25_*_logfee`, `roll100_*_logfee` | Captures local fee level, volatility, and recent minima. |
| Base-fee local trends | `dlog_base_fee`, `base_fee_trend`, `dlog_base_fee_lag1..6` | Captures short-term base-fee direction and persistence from known current/past base fees. |
| Gas-utilization local trends | `prev_gas_utilization_lag1..6`, `roll10/50/200_*_prev_gas_utilization` | Captures safe pressure history using lagged finalized gas facts. |
| Additional rolling fee context | `roll10/50/200_*_logfee` | Captures shorter and longer fee regimes than the original 25/100 windows. |

All previous-block facts are lagged inside their `SourceSpec`, not ad hoc in dataset builders or models. That keeps causality local to the source that owns availability.

`core_fee_dynamics_with_priority_fee` is a separate catalog that extends canonical `core_fee_dynamics` with lagged public priority-fee p10/p50/p90/spread scalars plus p50/spread local trend features. Its priority-fee sources declare the `priority_fee_percentiles` acquisition enrichment; corpus planning does not infer that provider fact from column names.

`elapsed_seconds` remains implemented only for the separate `core_fee_dynamics_elapsed_position` ablation catalog. It measures timestamp distance from the first row in the materialized feature table. It is not part of the default catalog because it can encode corpus position, long-term regime, or split-specific trends rather than reusable fee dynamics.

## Invariants

Feature matrices are `float32` and finite. Warmup rows use finite placeholders to preserve row alignment, and temporal compilers exclude invalid pre-warmup anchors before splitting. After warmup, required source values must be finite or feature construction fails.

Every compiled feature contract requires the canonical series columns `block_number`, `timestamp`, and `base_fee_per_gas`; feature-specific sources add their own required columns. `core_fee_dynamics` requires finite `base_fee_per_gas` after warmup. The global corpus schema may keep that column nullable for generic chain truth, but this feature catalog is only valid where base fee exists.

## Fingerprints

Fingerprints hash the features id, selected output names, and each catalog's implementation source files. They do not hash output paths. A catalog fingerprints only modules that can change its behavior; the priority-fee and elapsed-position catalogs fingerprint the safe helper modules they reuse, and the safe catalog does not fingerprint priority-only or elapsed-only code.
