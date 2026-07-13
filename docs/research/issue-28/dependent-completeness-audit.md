# Issue 28 dependent completeness audit

Status: Decisions 1–6 and the corrected whole-contract recap are owner-approved. This audit was
completed locally before the later, separate authorization to publish these four evidence files.
It is not a Resolution and grants no closure or map authority.

## Verdict

The approved contract is complete and dependency-safe. No additional consequential owner choice
is required. The audit found three implementation clarifications already implied by the approved
fail-closed and separate-state contract:

1. Feature-state arrays and target-state scalars remain actual float64 values. Preparation derives
   expected provenance from canonical corpus/chain/regime, blocks, the direct feature-support
   interval, training origins, and `K`; it rejects dtype coercion, nonfinite or nonpositive scales,
   or any state/fact mismatch.
2. The disposable probe fits one declared contiguous unique physical training interval and rejects
   duplicated support. A separate multiplicity-weighted calculation demonstrates why repeated
   context membership must not enter fitting. Production must likewise reject malformed order or
   provenance; it must not sort, deduplicate, sanitize, or repair post-seal rows.
3. The shared runtime backing object may reference both separate frozen states to validate and
   serve items. It is private transient storage, not a combined persisted `PreparationState`.

These clarifications harden the prototype without changing the approved interfaces, deletion
list, behavior fixtures, host boundary, or handoffs.

## Complete retained contract

Retain one lazy map-style `HistoricalDataset` over canonical prepared rows, positive signed-int64
raw fees, consecutive block numbers, and per-role origin positions. Three role datasets share one
immutable runtime store and the two separate frozen states. An item is exactly:

```python
{
    "inputs": Tensor[C, F],       # float32 CPU
    "label": Tensor[],            # int64 CPU
    "target": Tensor[],           # float32 CPU
    "base_fees": Tensor[K],       # int64 CPU
    "origin_block": Tensor[],     # int64 CPU
}
```

Feature fitting uses float64 population statistics from unique training-visible physical rows.
Target fitting separately uses float64 population statistics from the log of exact training-origin
raw minima. Both paths fail closed on value, scale, geometry, dtype, or provenance errors. The
historical and live paths share only the strict feature transform; live accepts exactly the final
`C` closed rows and returns `[1,C,F]` without fabricated historical fields.

Construct ordinary `DataLoader`s directly. Training alone uses seeded CPU-generator shuffle;
validation and testing are sequential. Default mapping collation and `drop_last=False` retain full
and tail batches. Worker, prefetch, persistence, and pinning values remain ephemeral host
arguments. Dataset workers and loader output remain ordinary CPU tensor mappings. Issues 26 and 55
choose native host transfer and CUDA pinning, including Lightning recursive transfer if selected;
Issue 28 requires neither field-selective manual transfer nor a custom batch/device seam.

## Dependent cleanup and ownership

The repository inventory exposes no hidden compatibility requirement. Future production work must
remove every caller of the approved deleted surfaces in one clean break:

- Issue 47 owns canonical consecutive rows, direct unique-row float64 feature fitting, emitted
  float32 rows, feature provenance, and the shared offline/live transform. It replaces the current
  sklearn/fallback normalization path rather than wrapping it.
- Issue 58 supplies the separate exact-minimum target coordinate/state; Issue 23 consumes the plain
  integer label and standardized scalar target and removes target-batch/action-mask/device methods,
  masked logits/argmax, action-mask loss inputs, and masked model signatures.
- Issue 26 builds the three ordinary loaders and removes `BatchPlan`, source/runtime/payload
  protocols, signature sorting, custom sampling/collation, worker policy, `ForwardBatch`, the no-op
  Lightning transfer override, and the duplicate manual transfer path. Issue 55 selects only
  direct host arguments and native placement after target CUDA evidence.
- Issue 21 consumes plain labels/targets through batch-partition-invariant predictive reducers. Raw
  `base_fees` and `origin_block` remain available to the separate evaluation/economic-accounting
  owner; scoring does not enter the dataset.
- Issue 34 persists the separate states and direct `C`, `K`, feature-order, tensor, head, and
  provenance facts, including the minimal observed software/runtime/device provenance it owns. It
  persists no host loader/device tuning as Issue 28 state and no combined preparation state.

This removes the remaining sequence/target batch wrappers, variable padding, input/action masks,
`.to_device()`/`.pin_memory()` methods, dataset-builder hierarchy, seconds-derived context logic,
post-seal sort/dedup/repair, and their configs and tests. No alias, shim, converter, migration,
registry, dual path, project version marker, or architecture-transition test is needed.

## Lean verification boundary

Production verification remains exactly two behavior fixtures: one canonical preprocessing,
dataset, and default-loader fixture; and one offline/live feature-parity fixture. The first covers
separate fits, an exact first-tie label, five item fields, full/tail collation, role order,
deterministic first-epoch training shuffle, and representative value/scale/provenance failures.
The second compares the same frozen history, ordered raw rows, feature state, and final `[C,F]`
values without future fields on the live side.

Do not add tests for deleted architecture, compatibility, storage internals, worker heuristics,
exact stochastic resume, codecs, registries, or caller arithmetic owned elsewhere. The disposable
CPU/MPS probes choose no training host, transfer strategy, or CUDA tuning.

## Authorization boundary

At audit completion, the prototype and audit were untracked and unpublished. Edo later authorized
publication of exactly this directory's four evidence files. That publication authorization does
not authorize posting `## Resolution`, closing Issue 28, or updating map Issue 1; completion still
requires separate explicit authorization.
