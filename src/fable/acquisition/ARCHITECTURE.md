# Acquisition Architecture

## Purpose

`acquisition` owns data collection mechanics. It turns block-window intent into ordered canonical block rows for caller-provided sinks. Corpus Split Materialization writes parquet; Corpus Assembly orchestrates and publishes corpus roots.

## Theory

Training quality begins with data provenance. Acquisition records provider and runtime mechanics; corpus state records which chain, time windows, blocks, split outcomes, and validation status produced the corpus.

## Boundaries

Acquisition does not define ML targets, train models, score predictions, write parquet datasets, or publish corpus roots. It schedules block-source pulls and emits ordered canonical rows to a caller-provided sink. Corpus Assembly decides how fetched rows become a corpus root.

Corpus planning owns source requirements. Acquisition adapters receive those requirements at construction and decide whether they can materially produce the requested source facts. For the RPC adapter, the generic `priority_fee_percentiles` enrichment maps to `eth_feeHistory`; unsupported enrichments fail before acquisition starts. The acquisition package still emits canonical rows and does not know why a feature set needed them.

## Invariants

Raw block rows must be canonical before persistence. Provider-specific transport details stay in provider-specific modules. Storage commit mechanics stay behind Corpus Assembly and storage lifecycle primitives.

## Extension Points

Add a new acquisition Adapter by keeping transport, provider client, and canonical row conversion separate. Preserve the same corpus contract so features and temporal compilers remain unchanged.

## Data Flow

```text
AcquireConfig
    |
    v
provider/chain endpoint
    |
    v
adapter requirement binding
    |
    v
block range plans
    |
    v
block source fetch
    |
    v
canonical block rows
    |
    v
Corpus Split Materialization
    |
    v
Corpus Acquisition Stage
    |
    v
Corpus Assembly
```

## Beginner Context

The model eventually trains on examples, but examples are only as trustworthy as the raw data underneath them. Acquisition is where external uncertainty enters the system: RPC latency, provider failures, missing blocks, and chain-specific payload shape. That uncertainty should be resolved before feature construction starts.

## Separation From Workflows

The acquisition package should answer "how do I fetch, retry, order, and canonicalize blocks?" Corpus Split Materialization answers "how do I resume, write, and validate parquet chunks?" Corpus Assembly answers "which windows do I fetch, how do I validate capability, and how do I commit the result?" The acquire workflow stays a thin caller.
