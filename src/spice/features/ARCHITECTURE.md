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
SourceSpec availability + lag/null policy
        |
        v
FeatureSpec formulas
        |
        v
ResolvedFeatureTable
```

`SourceSpec` owns causality and availability. Current base fee is allowed as `base_fee_per_gas[t]` because EIP-1559 base fee for block `t` is deterministic from parent state and observable before block `t` execution. Finalized current-block facts such as gas used, tx count, priority-fee percentiles, and fee-history gas-used ratio are exposed only through lagged sources.

`FeatureSpec` owns formulas over source and feature dependencies. The default catalog is `core_fee_dynamics`.

## `core_fee_dynamics`

The current catalog is protocol-first and includes the safe local-trend signals that improved the 1M A/B grid:

| Group | Outputs | Reason |
| --- | --- | --- |
| Current base fee | `log_base_fee_per_gas` | Current base fee is the direct fee level the decision must reason about. |
| Previous block pressure | `log_prev_gas_used`, `log_prev_gas_limit`, `prev_gas_utilization`, `prev_eip1559_pressure` | Finalized demand/pressure from the previous block is high-signal and safe. |
| Transaction count | `log_prev_tx_count` | Captures recent activity density without receipt/log ingestion. |
| Cadence/calendar | `seconds_since_previous_block`, `hour_*`, `dow_*` | Captures timing irregularity and coarse daily/weekly effects. |
| Rolling fee context | `roll25_*_logfee`, `roll100_*_logfee` | Captures local fee level, volatility, and recent minima. |
| Fee history | `prev_priority_fee_p10/p50/p90/spread`, `prev_fee_history_gas_used_ratio` | Captures recent priority-fee market pressure from RPC `eth_feeHistory`. |
| Base-fee local trends | `dlog_base_fee`, `base_fee_trend`, `dlog_base_fee_lag1..6` | Captures short-term base-fee direction and persistence from known current/past base fees. |
| Gas-utilization local trends | `prev_gas_utilization_lag1..6`, `roll10/50/200_*_prev_gas_utilization` | Captures safe pressure history using lagged finalized gas facts. |
| Additional rolling fee context | `roll10/50/200_*_logfee` | Captures shorter and longer fee regimes than the original 25/100 windows. |

All previous-block facts are lagged inside their `SourceSpec`, not ad hoc in dataset builders or models. That keeps causality local to the source that owns availability.

`elapsed_seconds` remains implemented only for the `core_fee_dynamics_elapsed_position` ablation spec. It measures timestamp distance from the first row in the materialized feature table. It is not part of the default catalog because it can encode corpus position, long-term regime, or split-specific trends rather than reusable fee dynamics.

## Invariants

Feature matrices are `float32` and finite. Warmup rows use finite placeholders to preserve row alignment, and temporal compilers exclude invalid pre-warmup anchors before splitting. After warmup, required source values must be finite or feature construction fails.

`core_fee_dynamics` requires finite `base_fee_per_gas` after warmup. The global corpus schema may keep that column nullable for generic chain truth, but this feature catalog is only valid where base fee exists.

## Fingerprints

Fingerprints hash the features id, selected output names, and implementation source files. They do not hash output paths. This makes artifacts sensitive to feature logic while keeping storage location irrelevant.
