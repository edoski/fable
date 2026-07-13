# Issue 28 fixed-context dataset prototype

Status: disposable HITL prototype; Decisions 1–6 and the corrected whole-contract recap are
owner-approved, and the dependent completeness audit found no new consequential choice. This
directory is the separately authorized evidence publication. No production code, tests, config,
corpora, training, or data were changed.

Question: does one lazy `C=200` map-style `HistoricalDataset`, separate strict feature/target
states, and ordinary `DataLoader` behavior cover the fixed causal contract without the existing
batch, padding, mask, transfer, or worker-policy machinery?

Run the interactive prototype:

```bash
uv run python docs/research/issue-28/prototype.py
```

Run every bounded probe as JSON:

```bash
uv run python docs/research/issue-28/prototype.py --all
```

The predeclared budget was synthetic in-memory fixtures only, one local CPU run plus an available
MPS smoke, under ten minutes, with no training. The stop condition was reached: all observations
below preserve one interface and no further local measurement can choose target CUDA host tuning.

## Observations

The synthetic fixture uses `C=200`, `K=5`, three ordered features, nine training origins, four
validation origins, and three testing origins. It includes one exact raw-fee tie.

- Three role datasets share one private runtime store containing a `[260,3]` CPU float32 input
  tensor, one `[260]` CPU int64 raw-fee tensor, block numbers, and references to the same separate
  frozen states. They do not retain `[N,C,F]` contexts or `[N,K]` outcomes. The fixture's shared
  storage is 7,280 bytes; the avoided persistent context and outcome tables would be 38,400 and 640
  bytes. Raw tensor handles are not public, and transient item clones isolate caller mutation.
- Each item is exactly `inputs [200,3] float32`, `label [] int64`, `target [] float32`,
  `base_fees [5] int64`, and `origin_block [] int64`, all on CPU. Default mapping collation yields
  `[B,200,3]`, `[B]`, `[B]`, `[B,5]`, and `[B]`. Batch size four produces full/tail sizes
  `[4,4,1]` with `drop_last=False`.
- Consecutive row slices and default-collated batches were contiguous. No explicit
  `.contiguous()` or custom collator was needed.
- Feature fitting consumed one direct contiguous interval of unique physical rows and rejected
  repeated context-membership positions. A deliberately multiplicity-weighted calculation differed
  from the fitted state, proving sample exposure cannot silently weight the fit. Offline and live
  feature transformation was bit-identical for the same final `C` rows and frozen feature state.
- Bounded `[chunk,K]` raw-int processing returned the first index for the exact tie. It retained
  only `[N]` labels and `[N]` standardized targets, not an outcome table.
- A seeded training generator reproduced the same first-epoch order when reconstructed. Reusing
  that generator advanced to a different second-epoch order. Reconstructing at an approved
  completed-validation resume boundary therefore restarts the permutation sequence; it is not an
  uninterrupted stochastic continuation.
- `num_workers=2`, `prefetch_factor=2`, and `persistent_workers=True` preserved validation order,
  the tail batch, and CPU-only tensors. These values prove ordinary worker operation only; they do
  not select host tuning.
- On the local MPS Mac, one direct-PyTorch smoke moved `inputs`, `label`, and `target` while raw
  outcomes and origin blocks remained on CPU. This is an observation, not a required transfer
  strategy. `pin_memory=True` warned that MPS does not support it and returned unpinned tensors.
  Neither result chooses Lightning versus direct-PyTorch transfer or target CUDA pinning.
- Nonfinite feature rows, duplicate support positions, zero feature scale, zero target scale, wrong
  state dtypes, float32 target overflow, mismatched canonical provenance, incomplete outcomes, and
  nonpositive fees all failed. No value was filled, clipped, dropped, sanitized, coerced, sorted,
  deduplicated, or given a fallback scale.
- Locked Torch supports exact first-tie `argmin` for CPU int64. CPU uint64 `argmin` is not
  implemented. The concrete prototype therefore uses raw int64 fees and rejects values outside
  that representable domain rather than adding a bigint/custom-collation path.

## Owner decision batch

These choices are independent after the closed causal/state contracts. Recommendation is the first
choice in each item.

1. **Representation and item interface.** Retain one concrete `HistoricalDataset`, backed by one
   shared immutable runtime storage object and the two separate frozen states. The five item keys
   and dtypes are exactly those measured above. Use signed int64 for raw fees/origin blocks with
   fail-closed representability. Reject unsigned/bigint alternatives: uint64 loses ordinary Torch
   reductions/device support, while Python integers lose default tensor collation.
2. **Lazy safety and contiguity.** Slice context/outcomes on access and clone those slices
   transiently so an item cannot mutate shared role storage. Retain only shared `[R,F]`, `[R]`, and
   block arrays plus per-role `[N]` origins, labels, and targets. Require shared arrays contiguous;
   add no item/batch `.contiguous()` because the measured direct slices and default collation are
   already contiguous.
3. **State ownership.** Keep `FeatureState` and `TargetState` as unrelated frozen records. The
   feature owner fits float64 population statistics from unique training-visible physical rows and
   emits float32 rows. The target owner computes exact raw minima/labels in bounded chunks, then
   fits its separate float64 log-target population state from training origins. Preparation only
   validates exact provenance and aligns these facts. There is no combined `PreparationState`.
4. **Loader and order ownership.** Construct ordinary `DataLoader`s directly. Training alone uses
   `shuffle=True` and one explicitly seeded CPU `torch.Generator`; validation/testing use
   `shuffle=False` and no generator. Keep `drop_last=False` for complete full/tail coverage. Do not
   persist generator/sampler/loader state or claim exact continuation: a boundary resume
   reconstructs and reseeds, as already permitted by Issue 16.
5. **Workers, pinning, and transfer.** Keep `num_workers`, `prefetch_factor`,
   `persistent_workers`, and `pin_memory` as direct ephemeral host arguments. With zero workers,
   use no prefetch and no persistence; with workers, use ordinary framework values selected only by
   the final host evidence. `HistoricalDataset` and `DataLoader` emit ordinary CPU tensor mappings,
   and dataset workers remain CPU-only. Issues 26 and 55 choose exact transfer and CUDA pinning
   after target evidence, including Lightning's native recursive batch transfer if Lightning wins.
   Add no custom batch/device adapter, transfer or pin methods, worker policy, no-op Lightning
   transfer override, manual duplicate transfer path, or persisted tuning.
6. **Offline/live parity.** Historical items and the separate live preparation function call the
   same strict feature transform. Live preparation accepts exactly the final `C` closed rows and
   returns `[1,C,F]`; it never fabricates labels, targets, outcomes, masks, or a historical dataset.

Edo approved all six choices and the complete corrected whole-contract recap exactly as written.
The [dependent completeness audit](dependent-completeness-audit.md) found no new consequential
owner choice. The decision approval did not itself authorize publication; Edo later authorized
only this four-file evidence publication. Neither approval authorizes posting Resolution, closing
Issue 28, or updating the map.

## Exact interface disposition

Retain:

```python
FeatureState(names, means_float64, scales_float64, provenance)
TargetState(mean_float64, scale_float64, provenance)

HistoricalDataset(shared_arrays, origin_positions, *, c=200, k=K)
# item -> {
#   "inputs": Tensor[C,F] float32 CPU,
#   "label": Tensor[] int64 CPU,
#   "target": Tensor[] float32 CPU,
#   "base_fees": Tensor[K] int64 CPU,
#   "origin_block": Tensor[] int64 CPU,
# }

DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=is_training,
    generator=training_generator_or_none,
    num_workers=num_workers,
    prefetch_factor=host_prefetch_factor if num_workers > 0 else None,
    persistent_workers=host_requests_persistence if num_workers > 0 else False,
    pin_memory=host_requests_cuda_pinning,
    drop_last=False,
)
```

`earliest_minimum(raw_fees, origins, K, chunk_size)` remains a direct temporal function returning
`(labels [N] int64, raw_minima [N] int64)`. The prototype function accepts origin positions and
creates only a bounded `[chunk,K]` temporary.

Delete without replacement:

- `BatchSource`, `BatchPlan`, `BatchRuntimeContext`, `PreparedBatchPayload`,
  `_PositionBatchSampler`, `_HostBatchCollator`, batch-signature sorting, host-loader policy, and
  SPICE-specific worker environment resolution from `modeling/batch_plan.py`;
- `SequenceInputBatch`, `PreparedSequenceInputBatches`, row-by-row padding/fill, input masks,
  action masks, max-context signatures, `.to_device()`, `.pin_memory()`, and `model_kwargs()` from
  `modeling/representations/sequence_inputs.py`;
- the generic dataset-builder hierarchy, seconds-derived context calibration, post-seal
  sort/dedup/repair, delayed-store reconstruction, and combined training/inference preparation;
- `PredictionBatch`/target-batch transfer and pin protocols, `MinBlockFeeTargetBatch`, prepared
  action-mask targets, masked logits/argmax, and action-mask loss inputs;
- the no-op Lightning `transfer_batch_to_device` override plus manual duplicate batch transfer;
- custom sampler/collator protocols, padding/mask tests, worker-policy tests, compatibility tests,
  and every host-tuning config/artifact field for loader workers, prefetch, persistence, pinning, or
  device selection.

No alias, shim, migration, converter, dual path, registry, project version marker, or architecture
transition test survives.

## Lean implementation verification

Keep two behavior fixtures after production implementation:

1. One canonical preprocessing/dataset/loader fixture covers unique-row feature fitting, separate
   training-origin target fitting, exact tie label, five item fields, full/tail default collation,
   sequential validation/testing, deterministic first-epoch training shuffle, and representative
   nonfinite/zero-scale/provenance failures.
2. One offline/live parity fixture compares the same frozen `h`, ordered raw rows, feature state,
   and `[C,F]` values. It contains no future fields on the live side.

Do not test deleted architecture, internal tensor storage shape, old/new parity, migrations,
registries, codecs, worker-count heuristics, exact resume, or caller arithmetic already owned by
another module.

## Handoffs

- [Choose predictive diagnostics and exact loss/scorer semantics](https://github.com/edoski/spice/issues/21):
  consume `label`/`target` in batch-partition-invariant predictive reducers. Keep `base_fees` and
  `origin_block` available to the separate evaluation/economic-accounting owner. Do not move
  scoring or transfer policy into the dataset; Issues 26/55 may choose native recursive transfer of
  the whole ordinary mapping.
- [Choose and prototype the minimum justified Min-Block-Fee task](https://github.com/edoski/spice/issues/23):
  consume plain `label: int64` and `target: float32`; remove target-batch/action-mask/device methods.
  Class weights and task loss remain outside representation.
- [Prototype and choose the lean training host](https://github.com/edoski/spice/issues/26): build
  the three direct loaders, choose Lightning-native recursive or direct-PyTorch transfer, preserve
  tail batches, and state that approved boundary resume reconstructs the seeded generator and is
  non-exact. Do not restore a custom transfer override or adapter.
- [Freeze durable ML, evaluation, weight-ABI, and provenance contracts](https://github.com/edoski/spice/issues/34):
  persist the two state records and their direct provenance separately. Persist `C`, `K`, feature
  order, and tensor/head facts; never persist host worker/prefetch/persistence/pinning/device tuning
  as Issue 28 state or create a combined state.
- [Choose causal preprocessing, split, feature, and context semantics](https://github.com/edoski/spice/issues/47):
  emit the one contiguous float32 prepared row array after strict unique-row float64 fitting. Own
  the shared offline/live transform and exact feature provenance.
- [Research the lean single-GPU batch-placement alternatives](https://github.com/edoski/spice/issues/55):
  measure standard DataLoader worker/prefetch/persistence/pinning and the selected host's native
  transfer on the actual CUDA hosts. Change host arguments only; do not create another dataset,
  custom transfer path, or placement seam.
- [Choose the auxiliary hindsight-minimum-fee target coordinate](https://github.com/edoski/spice/issues/58):
  provide the separate frozen target state; standardize one exact raw minimum per origin. No
  inverse/native prediction or float-log reconstruction enters the dataset.
