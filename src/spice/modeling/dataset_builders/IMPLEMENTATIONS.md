# Concrete Dataset Builders

Dataset builders decide which temporal samples become train, validation, test, and inference examples. They also own builder-specific runtime metadata that must be reused when evaluating a persisted artifact.

## Mental Model

The temporal compiler creates all valid samples. A dataset builder selects and splits those samples.

```text
canonical rows
  -> feature contract
  -> problem contract
  -> compiled problem store
  -> dataset builder
  -> prepared train/validation/test sample roles
```

The builder must preserve temporal ordering. Random train/test splitting would leak future data into training.

Training preparation ends by binding each split to `PreparedTemporalFacts`. Inference preparation ends by binding selected evaluation samples to a `PreparedActionSpace`. Raw split indices stay builder-local; callers receive prepared selections that already encode selected-sample alignment, action masks, and target facts.

## `fixed_sequence_temporal`

`fixed_sequence_temporal` derives a fixed sequence length from training data and stores it in runtime metadata.

Training preparation:

```text
sort by timestamp
  -> dedupe block_number
  -> sort by block_number
  -> compile feature/problem store
  -> take tail valid samples
  -> chronological split
  -> derive sequence length from train sample row span
  -> apply fixed context length
  -> reselect tail valid samples
  -> chronological split
```

Sequence length:

```text
median_dt = median positive timestamp delta in train sample row span
raw_length = round(lookback_seconds / median_dt)
sequence_length = clip(raw_length, min_sequence_length, max_sequence_length)
```

The calibration span is derived from selected training samples, not the raw corpus tail. Builder runtime metadata stores sequence length and median delta. Compiler runtime metadata travels through the artifact Temporal Capability. Inference requires both values and reuses the trained sequence length.

## Comparison

| Builder | Selection unit | Context length | Runtime metadata |
| --- | --- | --- | --- |
| `fixed_sequence_temporal` | Tail valid samples | Fixed from train sample median delta | Sequence length and median delta. |

## Scaler Ownership

The builder fits input normalization on training windows only. The scaler is persisted in the artifact and reused at inference.

```text
train indices
  -> scaler fit
  -> transform all splits with same scaler
  -> prepare split temporal facts on scaled store
```

## Inference Dataset Construction

Inference must recreate the same sample geometry used during training. The artifact supplies builder runtime metadata and Temporal Capability. The requested delay can be shorter than the trained max delay but cannot exceed it.

Inference callers provide observed evaluation coverage as inclusive first/last timestamps. The dataset-builder contract converts that to a half-open sample timestamp window `[first_timestamp, last_timestamp + 1)`; concrete builders consume only the normalized window.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Too few valid samples | Corpus/history does not satisfy requested sample count. |
| Empty train/validation/test split | Split fractions or sample count produce no rows. |
| No positive timestamp deltas | Fixed-context length cannot be derived. |
| Missing runtime metadata | Persisted artifact cannot recreate inference geometry. |
| Delay exceeds trained capability | Evaluation asks for unsupported action horizon. |

## Extension Pattern

A new builder should own all sample selection and runtime metadata for its strategy. Model families should receive only batch tensors, not builder-specific logic.
