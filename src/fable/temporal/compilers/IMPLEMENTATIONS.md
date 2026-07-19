# Concrete Temporal Problem Compilers

Temporal compilers turn feature rows into temporal problem stores. A problem defines which past rows the model may see, where future candidate windows sit, and the maximum action width carried into execution-policy preparation.

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

This compiler is online-safe because context rows end at the anchor row, candidate-window rows feed execution-policy outcome facts only, and feature prerequisites/warmup filtering happens before train/validation/test splitting. It does not reveal a future row count from evaluation timestamps to the model; `slot_spacing` only fixes artifact action width.

## Timestamp Window Store

`observed_time_window` builds the timestamp-window store locally. It keeps rows aligned and filters invalid sample anchors:

| Rule | Why |
| --- | --- |
| Feature table timestamps are nondecreasing. | Timestamp search requires sorted temporal rows. |
| Enough context history exists. | Inputs must satisfy lookback and feature prerequisites. |
| Candidate window is non-empty. | The execution policy needs at least one reachable outcome row. |
| Post-window row exists when required. | Overflow execution needs a concrete row. |
| Warmup rows are skipped. | Placeholder feature rows must not become training anchors. |

`CompiledProblemStore` validates the resulting generic geometry: feature, fee, and timestamp rows align; sample arrays align; context rows precede anchors; candidate starts are at or after anchors; candidate ends are exclusive and stay inside the store; and `max_candidate_slots` is positive. Warmup placeholders exist only so arrays are stable; stores exclude anchors whose context would include invalid warmup state.

## Invariants

Evaluation delay cannot exceed trained capability. Runtime compiler metadata must round-trip through the Temporal Capability in artifact manifests. A capability store must pair a store whose `max_candidate_slots` matches the Temporal Capability action width.
