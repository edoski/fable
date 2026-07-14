# Issue 17 typed model construction prototype

Status: final complete contract explicitly approved by Edo on 2026-07-14. This is disposable
planning evidence only. Approval authorizes ticket-scoped publication, one Resolution, closure of
Issue 17, and one named Wayfinder-map pointer. It authorizes no production code, configuration,
tests, dependency, data, storage, training, evaluation, job, archive, or sibling-issue mutation.

## Question and bound

Can strict three-family values replace the current `ModelSpec`/registry/lazy-loader/Optuna/partial-
overlay stack while leaving one lean owner for Method approval and one concrete model constructor?
Is the earlier free `allowed_by: MethodSpace` parameter necessary?

The cheapest discriminating observation is to hydrate one synthetic `TuneRequest` and complete
`Method` per approved family, apply each Method through the request-owned MethodSpace, then force
unknown-field, incomplete-value, family-mismatch, out-of-space, and Transformer cross-field
failures. The probe has a two-minute local CPU budget, imports no Torch, and stops once all three
branches derive their input/action dimensions and invalid cases fail before construction. Issue
23 already proves common Torch output/full-tail/task behavior; Issue 26 proves the native Lightning
fit/checkpoint lifecycle. Repeating either would add no decision evidence.

```bash
uv run python docs/research/issue-17/prototype.py --all
```

The interactive view is `uv run python docs/research/issue-17/prototype.py`.

## Binding contracts

This proposal consumes the closed request/model algebra in Issue 10, architecture-neutral task and
three concrete models in Issue 23, complete direct-HPO `Method`/`MethodSpace` contract in Issue 29,
Issue 49's amended fixed three-family protocol, Issue 26's approved Lightning-only fit host, and
Issue 78's single-operator/package-trust simplification.

Those later contracts change the original Issue 17 framing in three important ways. HPO accepts a
complete typed Method rather than a partial patch. The `TuneRequest` already owns the exact
MethodSpace and experiment semantics. Model construction happens once inside the Lightning
`FitModule`; artifact reload delegates to `FitModule.load_from_checkpoint`, not a second builder or
custom state inspection.

Current `CONTEXT.md`, ADR 0002, and model/prediction architecture documents are superseded where
they retain `Workflow Config`, `SerializeAsAny`, owner coercers, generic registries, masks, tuned
overlays, framework-neutral hosts, or custom artifact loaders. Their rewrite belongs to the later
documentation ticket.

## Corrected seam

The free `allowed_by` parameter should be deleted. A caller could pair experiment semantics from
one study with a MethodSpace from another, creating a precondition the function itself could not
prove. The parameter also exposes approval ownership that already belongs to `TuneRequest`.

A MethodSpace receiver such as `space.apply(experiment, method)` is smaller but still wrong: the
space does not own the experiment, corpus, study identity, or their association. It would put HPO
workflow behavior on a Pydantic data record and leave the same mismatch possible.

The pure HPO interface is one function:

```python
def apply_method(request: TuneRequest, method: Method) -> TrainingDefinition: ...
```

It strictly revalidates the complete request and Method, reads only
`request.study_definition.method_space`, performs one exhaustive `(MethodSpace, Method)` match,
checks exact membership, explicitly constructs the family `ModelDefinition`, composes the complete
`TrainingDefinition`, and validates the whole result. It accepts no mapping, base object, optional
parameter group, path, reflection target, mutation callback, or separate approval authority.

Do not merge this pure function into stochastic model construction. Selected-study materialization
needs the effective definition without running a fit, and Issue 26 requires seed-before-
construction ordering. The implementation may share a private composition helper inside the HPO
module; it exposes no generic application framework.

There is no public `build_model` interface. Issue 26's one concrete Lightning `FitModule`
constructor owns the sole exhaustive `ModelDefinition` match and creates LSTM, Transformer, or
Transformer-LSTM directly after validation and seeding. Train and Tune-private fits use that same
constructor. Native `load_from_checkpoint` replays it for artifact loading. Add no constructor
table, factory object, registry, loader, model protocol/base, alternate builder, or artifact reload
path. Private shared heads and Transformer mechanics remain implementation sharing only.

This produces two deep modules at their real seams: pure HPO definition materialization and the
Lightning fit host. Both depend only on in-process values/libraries, so no adapter or port is
earned.

## What a Method means

A Method is one complete candidate recipe. It is also the full candidate value retained in HPO
progress/study evidence and later promoted into fresh final-K training. Nothing is inferred from a
parameter-name map.

| Method fact | Concrete values | Why it survives |
| --- | --- | --- |
| `family` | `lstm`, `transformer`, or `transformer_lstm` | Real approved alternatives; required tagged-union discriminator. Family is fixed per study, never a mixed-study axis. |
| `capacity` | One typed family-specific width/layer bundle | The approved choices couple dimensions. One atomic value prevents invalid Cartesian combinations and supports exact artifact reconstruction. |
| `dropout` | `0.1`, `0.2`, or `0.3` | Changes model regularization and stochastic behavior. |
| `AdamWMethod.learning_rate` | LSTM: `1e-4, 3e-4, 1e-3`; Transformer/hybrid: `3e-5, 1e-4, 3e-4` | Changes optimizer step size; family-specific leaves are approved. |
| `AdamWMethod.weight_decay` | `0, 1e-4, 1e-3` | Changes AdamW regularization. The type name owns AdamW; no `optimizer="adamw"` selector survives. |
| `training_batch` | `32` or `64` | Changes gradient estimates and update count, so it is semantic rather than runtime-only. |
| `fit` | accumulation `1`, clip `1.0`, no scheduler, seed `2026`, 36 epochs, validation every completed epoch, patience `8`, `min_delta=0`, strict-lower improvement, earliest-best restoration, no epoch floor | These facts determine exposure, stochastic construction, stopping, and selection. They remain one complete typed record because Issue 29 requires every retained result to carry a complete Method. |

Device, precision, DataLoader workers, validation/evaluation batch size, paths, target host, and
checkpoint locations do not belong to Method. Issue 26 makes them ephemeral runtime/host facts.
Hashes, inventories, locks, receipts, retry/recovery state, job facts, and publication mechanics are
also absent under Issue 78.

## What a MethodSpace means

A MethodSpace is the ex-ante finite set one `TuneRequest` permits. All three family spaces freeze
before architecture outcomes; only the globally selected family executes. It contains no candidate
order, table, generator, sampler, PRNG seed, tier, count, budget, cap, coverage claim, pruning, or
completion rule.

| MethodSpace fact | Why it survives |
| --- | --- |
| `family` | Binds the space to exactly one concrete family and discriminates the strict union. |
| `capacities` | Stores the three approved coupled capacity bundles rather than hiding them in a registry. |
| `dropouts` | Freezes the allowed dropout leaves. |
| `learning_rates` | Freezes the family-specific optimizer leaves. |
| `weight_decays` | Freezes the allowed AdamW regularization leaves. |
| `training_batches` | Freezes the two semantic batch choices. |

The fixed fit record is not duplicated in MethodSpace. Its exact values are already enforced by
the complete Method type. Repeating them in the space adds defensive equality machinery without a
second legitimate choice. Collection order is canonical encoding only, never candidate order.

MethodSpace cannot collapse to `family` plus an in-code lookup. That would recreate the forbidden
registry and hide the frozen scientific scope outside the durable TuneRequest. Explicit values are
verbose because they are the study contract, not because the engine needs a generic search system.

## Names, dimensions, and validation

Stable names are schema leaf names only. Candidate input is one complete Method JSON/YAML value.
There is no namespace such as `model.hidden_size`, `ParameterName`, tunable-field descriptor,
dotted lookup, alias, or legacy spelling map. Nested paths may appear in Pydantic diagnostics; they
are never executable instructions.

The family records use the amended protocol names: `projection`, `hidden`, `layers`,
`head_hidden`, `model_width`, `attention_heads`, `transformer_layers`, `feedforward_width`,
`lstm_hidden`, and `lstm_layers`. `feedforward_width` is a complete approved capacity fact.
`feedforward_multiplier` was an Optuna sampling convenience and is deleted, not translated.

Positive dimensions and dropout bounds are field-owned. Transformer/hybrid capacities require an
even `model_width` divisible by `attention_heads`. The Method is revalidated on hydration,
membership, `ModelDefinition` construction, and full `TrainingDefinition` composition—not through
extra defensive scans. Input width derives from the ordered feature tuple, context from `C`, and
action-head width from `K` inside the FitModule constructor. These are not Method parameters.

## Synthetic result and clean break

The corrected probe passes complete request-owned Method application for all three families,
purity, the batch-32 axis, whole-definition validation, and the sole FitModule construction match.
Changing the request from three to four features and K from 5 to 10 derives input width 4 and action
width 10 without changing Method. Unknown keys, missing fixed facts, invalid Transformer geometry,
wrong-family Methods, and valid-but-unapproved capacities fail before construction. There are zero
free MethodSpace arguments, `feedforward_multiplier` fields, or parameter-path fields.

Later implementation replaces the current cluster. Delete generic
`ModelConfig`/`ModelTuningSpaceConfig`/`TunedModelParams`, `TunableFieldSpec`, `ModelSpec`, loader
tables, `MODEL_SPEC` back-references, registry coercion/construction, family tuning-space and
partial-param records, `feedforward_multiplier`, `modeling/tuned_config.py`, partial application in
`modeling/tuning.py`, and all Optuna sampling/execution/storage/configuration/dependency code and
old tests/docs. Issue 23 removes generic prediction output/head maps and `TemporalModel`. Issue 26
removes framework-neutral fit/build/load paths. Issue 78 forbids replacement locks, hashes,
inventories, publication helpers, or recovery engines. Add no shim, alias, migration, transition
mode, parallel path, or deletion test.

Full-code-first order remains binding: approve all contracts; create review-sized implementation
tickets; implement and independently review; integrate; run Issue 76's post-integration synthetic
L40 gate for all families; only later authorize real corpus acquisition and outcome-bearing work.

## Approval and completion boundary

Edo explicitly approved this entire corrected contract: TuneRequest owns MethodSpace;
`apply_method` takes only `(request, method)`; MethodSpace repeats no fixed fit record; AdamW has no
one-entry selector; the concrete Lightning FitModule owns the sole model-construction match and
native reload path; schema leaf names are the only parameter names; input/action dimensions derive
from request facts; and the generic/Optuna/registry/lazy-loader/path/compatibility machinery is
deleted cleanly in the later implementation.
