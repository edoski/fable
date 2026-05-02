# Concrete Temporal Problem Compilers

Temporal compilers turn feature rows into supervised decision examples. A problem defines which past rows the model may see, which future rows it may choose, and how many candidate actions exist.

## `observed_time_window`

`observed_time_window` is the current compiler. It builds context and candidate windows with timestamp search, not estimated block counts.

```text
context_start = first row with timestamp >= anchor_ts - lookback_seconds
candidate_start = anchor row
candidate_end = first row with timestamp > anchor_ts + delay_seconds
```

`slot_spacing` converts the maximum delay into the artifact action width:

```text
action_width = floor(max_delay_seconds / slot_spacing_seconds) + 1
```

The extra slot includes offset `0`.

Slot-spacing ids:

- `nominal`: chain runtime nominal block time.
- `recent_median`: median positive timestamp delta from the feature table.

The compiler returns a Temporal Capability with `compiler_id`, `max_delay_seconds`, `action_width`, and typed runtime metadata. For `observed_time_window`, runtime metadata contains `slot_spacing_id` and `slot_spacing_seconds`.

This compiler is online-safe because context rows end at the anchor row, candidate rows are future outcome rows only, and feature prerequisites/warmup filtering happens before train/validation/test splitting. It does not reveal a future row count from evaluation timestamps to the model; `slot_spacing` only fixes the prediction head width.

## Shared Builder

The shared timestamp builder keeps rows aligned and filters invalid sample anchors:

| Rule | Why |
| --- | --- |
| Enough context history exists. | Inputs must satisfy lookback and feature prerequisites. |
| Candidate window is non-empty. | The model needs at least one fee choice. |
| Post-window row exists when required. | Overflow execution needs a concrete row. |
| Warmup rows are skipped. | Placeholder feature rows must not become training anchors. |

The feature table remains row-aligned with the corpus. Warmup placeholders exist only so arrays are stable; stores exclude anchors whose context would include invalid warmup state.

## Invariants

Timestamps and arrays align by row. Candidate end is exclusive. Evaluation delay cannot exceed trained capability. Runtime compiler metadata must round-trip through the Temporal Capability in artifact manifests.
