# Concrete Corpus Implementation

The corpus layer owns canonical blockchain data: how raw RPC block payloads become parquet rows, how rows are validated, and how enough history is proven before modeling.

## Mental Model

A corpus is split into history and evaluation data:

```text
history rows
  -> fit features, build training/validation/test examples

evaluation rows
  -> run diagnostic replay for the evaluation day
```

The dataset identity is based on dataset name, chain, and evaluation date. It is not based on the exact acquisition window.

## Canonical Block Schema

Every row uses the same columns:

| Column | Meaning |
| --- | --- |
| `block_number` | Chain block height. |
| `timestamp` | Block timestamp in seconds. |
| `base_fee_per_gas` | EIP-1559 base fee value. |
| `gas_used` | Gas consumed in the block. |
| `chain_id` | Numeric chain id. |
| `gas_limit` | Block gas limit. |
| `tx_count` | Number of transactions in the block. |
| `block_size_bytes` | Block size when the RPC payload includes it. |
| `blob_gas_used` | Blob gas used when available. |
| `excess_blob_gas` | Excess blob gas when available. |
| `priority_fee_p10` | Optional priority-fee percentile enrichment. |
| `priority_fee_p50` | Optional priority-fee percentile enrichment. |
| `priority_fee_p90` | Optional priority-fee percentile enrichment. |
| `priority_fee_spread` | Optional p90 minus p10 priority-fee enrichment. |

RPC payload conversion builds this schema. Missing required block fields fail during conversion. Missing `base_fee_per_gas` is stored as `None`, which keeps canonical rows rectangular for chains or blocks where the field is absent.

## Parquet IO

Corpus IO is parquet-only. Data is stored as chunk files under history and evaluation directories:

```text
corpus root/
  history/
    ethereum__blocks__START_to_END.parquet
  evaluation/
    ethereum__blocks__START_to_END.parquet
```

Loads scan recursively, skip hidden paths, combine chunks, and sort by `block_number`. Sorting gives deterministic row order even when chunks were written in several parts.

## Corpus Assembly

Corpus acquisition has two public Interfaces: `acquisition_source_requirements()` for configuring the acquisition Adapter, and `assemble_corpus()` for dry-run planning or committed corpus publication. Corpus Capability Planning owns history sizing and refill decisions. Corpus Acquisition Stage owns staging roots, split sequencing, dataset provenance publication, state DB staging, selected-path commit, cleanup, and preserve-on-failure behavior. Corpus Split Materialization owns history/evaluation dataset reuse, extension, rebuild, validation, and parquet IO. Its internal policy assesses staged and committed split candidates against the active Split Intent, then returns an executable plan for staged reuse, committed reuse, extension, full materialization, or invalid staged rejection. Private implementation modules keep effects local: candidate state loading, parquet chunk IO, source reuse/copying, acquisition pulls, target invariants, and executable plan execution. When extending a split, plan execution computes missing pull ranges and reusable overlap, copies whole reusable parquet chunks, and rewrites only edge or newly pulled ranges.

| Materialization outcome | Behavior |
| --- | --- |
| Staged reuse | Reuse a clean staged split only when it validates against the current Split Intent. Invalid staged data is fatal; stale clean staged data is ignored. |
| Committed history reuse | Reuse a clean committed history split when it ends at the requested boundary and starts at or before the requested history start. |
| Committed evaluation reuse | Reuse a clean committed evaluation split only when block range and exact timestamp-window validation match. |
| Extension | Pull missing prefix/suffix ranges and reuse overlapping clean parquet chunks or trimmed edge frames. |
| Full materialization | Create when no committed split exists; rebuild when committed data exists but cannot satisfy the target. |
| Completed-prefix resume | Pull sinks resume only from a clean contiguous partial dataset starting at the plan start. |

History refill is bounded. Assembly first requests an estimated history window from planning, asks planning to count valid capability samples, and expands the window up to a small internal cap if observed block cadence under-requested usable rows.

## Source Requirements

Corpus planning derives source requirements from feature and problem contracts before any provider client is created. Requirements are generic:

| Field | Meaning |
| --- | --- |
| `required_columns` | Canonical source columns needed by corpus, features, and temporal contracts. |
| `optional_enrichments` | Named acquisition enrichments that a concrete adapter may support. |
| `temporal_unit` | Unit represented by source rows, currently `block`. |
| `ordering_key` | Monotonic row key, currently `block_number`. |
| `partition_key` | Optional source partition, currently `chain_id`. |

Priority-fee features add the `priority_fee_percentiles` enrichment and require priority-fee source columns. The RPC adapter maps that enrichment to `eth_feeHistory`; other future adapters can satisfy the same requirement differently.

## Split Manifest

Corpus state stores split-level provenance:

```text
DatasetManifest
  dataset
  chain
  source_requirements
  splits.history
    request
    coverage
    validation
    materialization
  splits.evaluation
    request
    coverage
    validation
    materialization
```

Request records the intended timestamp and block ranges. Coverage records the observed first/last timestamp, first/last block, and row count. Validation records compact validation status and issues. Materialization records whether the split was created, reused, rebuilt, or extended and how many files back it.

## Validation

Validation answers two questions:

1. Is this a clean contiguous block range?
2. Does this range cover the requested timestamps?

Contiguous validation checks:

| Check | Failure meaning |
| --- | --- |
| Non-empty frame | No block data exists. |
| Required columns | Data is not canonical. |
| One chain id | Mixed chain rows. |
| Adjacent block numbers differ by `1` | Gap or duplicate block range. |
| Timestamps can be coerced | Invalid time data. |

Exact-window validation also counts rows before `start` and rows at or after `end`.

## Coverage

Coverage connects corpus rows to modeling needs. A clean block range is not enough; the model also needs enough warmup and enough future horizon.

```text
feature prerequisites
  + problem prerequisites
  + sample count
  + max delay
  -> required corpus coverage
```

Training coverage requires history seconds plus max delay seconds, and warmup rows plus requested sample count. Evaluation coverage requires enough history before the evaluation window and enough evaluation span for the delay.

## Failure Modes

| Failure | Typical cause |
| --- | --- |
| Empty frame | Acquisition produced no rows. |
| Chain mismatch | Wrong provider or mixed files. |
| Duplicate/gapped blocks | Interrupted or corrupt corpus materialization. |
| Evaluation rows outside window | Wrong date or wrong range resolution. |
| Insufficient history seconds | Feature/problem warmup cannot be met. |
| Insufficient history rows | Requested sample count cannot be built. |
| Delay exceeds capability | Evaluation asks for a horizon the trained artifact cannot support. |

## Extension Pattern

New corpus sources should still emit the canonical block schema and pass the same validation reports. Acquisition method can vary; corpus semantics should not.
