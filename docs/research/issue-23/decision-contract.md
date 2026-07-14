# Issue 23 minimum justified Min-Block-Fee task contract

Status: final complete contract explicitly approved by Edo on 2026-07-14. This is planning/prototype
work only. Approval authorizes only ticket-scoped research publication, one Resolution, closure of
Issue 23, its one-line map pointer, and verified frontier callback. It does not authorize production,
configuration, test, data, storage, acquisition, training, tuning, evaluation, remote work, economic
analysis, model selection, or scientific execution.

## Architecture-independent task seam

The public task seam is independent of model architecture. Its interface is:

```text
MinBlockFeeOutput(action_logits [B,K], minimum_fee_z [B])
TargetState and exact target construction
ClassificationLossState and min_block_fee_loss
decode_action
PredictiveScorer and concrete predictive result
HistoricalBatch evaluator input
decoded-k scheduling semantics
```

No model class is the task seam. Training, validation, evaluation, artifact reload, and serving use
the same named two-head output. The task module hides target transformation, loss algebra, output
validation, first-argmax decode, and predictive reduction. Deleting it would redistribute those facts
across every caller, so it earns depth and locality.

All dependencies are in-process tensor computation and frozen records. Add no prediction/task
registry, plugin, adapter, abstract task/family base, compiled callable bag, output-head spec, generic
head map, target-batch protocol, decoded-result interface, or speculative extension seam.

## Closed concrete model seam

Consume Issue 10's smallest closed tagged union:

```text
ModelDefinition =
  LstmDefinition(family="lstm", ...)
  | TransformerDefinition(family="transformer", ...)
  | TransformerLstmDefinition(family="transformer_lstm", ...)
```

The LSTM branch has only input-projection width, hidden width, layer count, dropout, and head width.
The Transformer branch has only model width, attention heads, Transformer layer count, feedforward
width, dropout, and head width. The hybrid has the Transformer fields plus LSTM hidden width and
layer count. Issue 10 owns the exact strict Pydantic schema, tags, and field spelling; Issue 23 does
not create a competing config model.

One direct exhaustive `build_model` match constructs exactly one of three concrete framework-native
`nn.Module` classes. Unknown tags fail schema hydration before construction. There is no model
registry, lazy loader, `ModelSpec`, plugin, adapter, abstract family base, back-reference, dotted-path
patch language, or open-ended constructor table.

Edo's final approved Issue-49 Scope A makes all three branches concrete current requirements rather
than speculative extension points. The Issue-49 owner still owns how later evidence compares or
selects them; this ticket preserves only the code seam needed by that work.

Every model accepts fixed unmasked `inputs [B,C,F] float32` and returns the same
`MinBlockFeeOutput`:

```text
LSTM
  input projection -> multi-layer unidirectional LSTM -> final C-1 state -> two heads

Transformer
  input projection -> fixed sinusoidal positions -> Transformer encoder stack
  -> final C-1 state -> two heads

Transformer-LSTM
  input projection -> fixed sinusoidal positions -> Transformer encoder stack
  -> multi-layer unidirectional LSTM -> final C-1 state -> two heads
```

Transformer layers use multi-head self-attention, feedforward width, GELU, and dropout. The shared
private head implementation is exactly two independent
`Linear -> GELU -> Dropout -> Linear` paths: K logits and one unconstrained scalar. Private code
sharing does not create a public head abstraction. Fixed contexts need no input mask; every exact-K
action is valid, so there is no action mask.

Model construction validates positive dimensions, even positional width, attention divisibility,
exact input shape, exact output shape, and finite outputs at the owning runtime. Architecture facts
remain concrete artifact facts. No branch is a winner or default by implication.

## Paper alignment and project ownership

The [paper/model-family audit](paper-model-family-alignment.md) and visual/semantic inspection of the
hash-matched 11-page `ICDCS_2026.pdf` support this topology. Physical page 2 names LSTM,
Transformer, and Transformer-LSTM. Pages 4-5 describe selecting a specific future min-block,
estimating its associated base fee, and scheduling at that block. Pages 7-8 define all three
architectures and their shared two-head prediction. Pages 9-10 and Figures 5-7 compare all three;
page 10 calls SPICE model-agnostic while still discussing family-specific results.

The paper does not define the project's exact `k`, tensor shapes, tie rule, target z-score, loss
coefficients, scorer, artifact schema, or serving interface. Those remain closed project contracts.
The paper's inverse-frequency CE and unspecified weighted loss are evidence, not authority for the
project's two explicit CE recipes or unit-sum loss.

The final approved [Issue-49 Scope-A amendment](https://github.com/edoski/spice/issues/49#issuecomment-4966379354)
owns scientific model/recipe topology, execution
order, validation gate, winner/HPO policy, and exact experiment counts. Issue 23 freezes code shape
only and infers no cell, count, run, winner, default, or selection rule from the paper. It does not
copy those protocol facts into the task contract.

## Recipe facts

Each authored scientific training recipe carries one concrete `ModelDefinition`, the exact ordered
feature tuple, and exactly one classification-loss value:

```text
classification_loss = "unweighted" | "corrected_inverse_frequency"
```

The bounded feature values remain those approved by Issues 47 and 49. The all-chain core order is
`log_base_fee_per_gas, gas_utilization`; Ethereum inserts
`log_exact_forming_base_fee_per_gas` immediately after that core. The indivisible candidate pairs
append, in ladder order, `log_gas_limit, log1p_tx_count` and then `hour_sin, hour_cos`. Recipes spell
out the complete tuple. Add no feature-set id, combination generator, model/feature registry,
plugin, or Cartesian task architecture. Direct feature formulas remain separately owned.

Later evidence may select an authored recipe/default only under its owning approved protocol. It
cannot add a task type, dynamically register a model, alter the shared output, or redesign code.

## Exact target and evaluator batch

For each eligible closed-parent origin `h`, temporal preparation uses raw positive int64 fees at
`h+1...h+K`. Bounded `earliest_minimum` returns the raw minimum and deterministic first index before
float conversion. The label is that `int64 k`; the auxiliary raw outcome is the minimum.

The separate frozen `TargetState` remains Issue 58's concrete record: target id
`hindsight_minimum_base_fee_per_gas_within_k`, native-wei/gas log reference, `ddof=0`, float64
`mu/sigma`, training target count, model dtype, chain, `K`, and content-bound training-origin/corpus
provenance. Fit `ell=ln(O/(1 native wei/gas))`, require finite `mu` and strictly positive finite
`sigma`, and store `z=(ell-mu)/sigma` as the float32 dataset target. Validation/testing reuse the
training state. Add no epsilon, clipping, `log1p`, native inverse, transform selector, generic
scaler, or combined feature/target state.

Every architecture consumes the ordinary default-collated mapping approved by Issue 28:

```text
inputs       [B,C,F] float32
label        [B]     int64
target       [B]     float32
base_fees    [B,K]   int64
origin_block [B]     int64
```

Training moves only tensors needed by its selected native host. No task batch class, prepared target
batch, `.to_device`, `.pin_memory`, model-kwargs map, mask, sample-position binding, or custom
collation survives.

## Exact loss and decode

Fit one concrete classification-loss state from retained training labels. It records selected mode,
`N_train`, and exact K supports. Unweighted CE passes no weight tensor. Corrected inverse frequency
requires every support positive and derives `w_k=N_train/(K*n_k)` in the active model dtype/device;
missing support fails that candidate before fit. Do not persist a duplicate weight vector. The state
is a direct artifact fact, not a loss registry or configurable formula.

For each minibatch, native unreduced PyTorch operations compute:

```text
c_i = cross_entropy(action_logits_i, label_i, weight=w, reduction="none")
r_i = smooth_l1_loss(minimum_fee_z_i, target_i, reduction="none")
L_cls   = sum_i c_i / N
L_reg   = sum_i r_i / N
L_total = (sum_i c_i + sum_i r_i) / N
```

The Smooth-L1 transition is one standardized target unit. Both components enter exactly once. There
is no beta/lambda field, coefficient, scheduler, adaptive weighting, target-weight denominator,
smoothing, clipping, fallback, or HPO loss surface. Training, checkpointing, early stopping, and HPO
use only finite complete-validation `L_total` under Issue 16.

`decode_action` validates finite `MinBlockFeeOutput`, then returns ordinary
`action_logits.argmax(dim=-1) [B] int64`. PyTorch's first-index tie behavior is canonical. The
auxiliary output is validated but does not participate in decode. Add no softmax, action mask,
overflow/fallback action, offset wrapper, allocation buffer, decoded-result id, or decode context.

## Predictive scorer and range evaluator

Keep one concrete range-scoped `PredictiveScorer`, not a protocol or reusable metric framework. It
accepts `MinBlockFeeOutput` plus plain `label`/`target`. Every update derives the same unreduced loss
terms, decoded actions, and affine log view `ellhat=mu+sigma*minimum_fee_z`; it adds `A_cls`, `A_reg`,
absolute-log-error, squared-log-error, and `N` in float64.

Create fresh phase-local
`MulticlassF1Score(K, average="macro", multidim_average="global", zero_division=0)` and
`MulticlassStatScores(K, average=None, multidim_average="global")` objects. Update each batch,
compute once, serialize one complete predictive section, then discard them. Macro-F1 uses
union-active classes; accuracy is `sum(TP)/N`. Persist only Issue 21's exact concrete
`PredictiveTotals` facts and derive scalar loss, accuracy, and log diagnostics once. Add no generic
epoch accumulator, `MetricDescriptor`, `MetricSet`, scorer registry, custom F1, Accuracy object,
stored predictions, partial result, or result protocol. Later implementation declares
`torchmetrics` directly; this ticket changes no dependency file.

The range evaluator restores one frozen artifact, iterates every declared eligible origin once,
calls its selected concrete model once per batch, and feeds the shared output/batch directly to:

1. `PredictiveScorer`, reading both model outputs plus label, target, and target state; and
2. separate temporal economic calculation, reading decoded `k`, raw `base_fees`, and origin block.

It returns separate predictive and economic sections. Exact context/range/provenance equality and
`totals.N=scored_N=eligible_N>0` are mandatory. Missing, misordered, malformed, nonfinite,
incomplete, or mismatched input fails the whole section. Validation and sealed testing differ only
by declared range. Serving creates no scorer because future targets are unavailable.

## Auxiliary output and scheduling

The auxiliary minimum-fee prediction remains in `MinBlockFeeOutput`, loss, target/artifact facts,
validation, evaluation, and predictive reporting for every architecture. It is a scientifically
real auxiliary output. Scheduling is driven only by decoded `k`; the scalar is neither an
authoritative fee quote nor a second action, and it cannot override or repair decoded `k`.

The serving/artifact-chain owner selects an approved artifact. This task seam requires its exact
tagged `ModelDefinition`, `C=200`, ordered feature tuple, input width, and requested
`K in {2,3,4,5}` to match before the parent snapshot. It does not choose source-chain policy. Live
preparation returns `[1,C,F] float32`; the concrete model returns `[1,K]` logits and `[1]`
`minimum_fee_z`; full output validation runs; `decode_action` supplies scalar `k`; temporal ownership
computes `b=h+1+k`.

Issue 23 neither requires nor forbids a later serving schema from displaying the auxiliary value as
a clearly labelled non-actionable diagnostic. Such a consumer needs its own approved semantics. Do
not claim the value is never exposed, delete it from inference, call it the selected-action fee,
present it as a calibrated public quote, or treat it as another scheduling decision.

## Artifact checks and clean deletion

Every save/load and caller validates direct factual `C`, `K`, ordered feature formulas, feature
state/provenance, target state/provenance, classification-loss mode/training support, exact tagged
`ModelDefinition`, model dtype, input width, K-wide action head, scalar auxiliary head, and selected
concrete state-dict names/shapes. Artifact serialization remains Issue 34 ownership. Add no
capability envelope, semantic selector ids, project version marker, compatibility payload, or
alternate reader.

Delete the current generic prediction machinery without replacement: prediction registry,
`CompiledPredictionContract`, output/head specs and ids, callable fields, generic model output/head
map, generic target/prepared-target/accumulator protocols, `PredictionBatch`, action/decode context,
`DecodedPredictionResult`, `DecodedOffsets`, mask helpers, task metric descriptors/sets, and generic
result interfaces. Remove their config, artifact, evaluator, training, serving, docs, exports, and
tests transitively.

Delete generic model-family registry/base/spec construction, lazy loaders, tuning hierarchy, and
open-ended family machinery. Preserve the three concrete architecture classes/modules, Issue 10's
closed tagged union, one direct constructor match, and only private implementation sharing justified
by those concrete models. Add no alias, shim, migration, dual path, legacy reader,
architecture-transition test, or registry-deletion test.

The current `CONTEXT.md`, prediction/model architecture docs, and ADR 0003 retain old durable
representation ids, adapter interfaces, compiled prediction families, model registries, and
mask-shaped runtime. They conflict with later closed clean-break contracts and are superseded for
this route. Their normative rewrite remains the later documentation/ADR owner's work; this ticket
does not edit them.

## Lean implementation verification and boundaries

Future implementation keeps two focused behavior fixtures:

1. one parameterized full/tail fixture constructs every exact `ModelDefinition` branch through the
   sole exhaustive match and checks the shared typed output, `[B,K]`/`[B]` shapes, finite values,
   unmasked fixed-context input, and full-plus-tail parity; it compares no model quality;
2. one architecture-independent task/range fixture covers earliest-minimum tie, both CE modes and
   missing weighted support, unit-sum sample-denominator loss, argmax, batch-partition-invariant
   scoring, the plain five-tensor evaluator mapping, separate economic input, and
   `[1,C,F] -> full output -> scalar k` scheduling parity.

Do not test deleted architecture, private module layout, transition behavior, registries, old/new
parity, compatibility, framework math owned by Torch/TorchMetrics, model quality, rankings,
experiment matrices, counts, or scientific outcomes.

This contract consumes Issues 10, 21, 24, 28, 46, 47, 48, and 58 without reopening them. It consumes
the final approved Issue-49 Scope-A requirement for three concrete families without absorbing its
scientific protocol. It chooses no experiment cell/count/order, recipe winner/default, training
host, batch placement, artifact serialization, HPO engine/result, real corpus, economics, serving
lifecycle, or implementation slice. The bounded synthetic prototype proves structural sufficiency
only.
