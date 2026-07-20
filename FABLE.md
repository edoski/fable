# FABLE

FABLE (Fee Analysis through Blockchain Learning and Estimation) is a closed-parent, fixed-block-horizon system for learning when a future block is likely to minimize base fee per gas. This manual is the canonical detailed account of the product's scientific contract, worked decision, architecture, interfaces, requests, durable objects, commands, operator configuration, serving surface, evaluation schemas, limitations, and sources.

FABLE derives from and extends selected temporal work from *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments*. FABLE is neither SPICE nor a reproduction of SPICE. Domain terms are defined in [CONTEXT.md](CONTEXT.md); durable-object and execution-boundary decisions remain recorded in [docs/adr/](docs/adr/).

## Contents

- [Overview](#overview)
- [One decision, end to end](#one-decision-end-to-end)
- [Scientific contract](#scientific-contract)
- [Architecture and deep interfaces](#architecture-and-deep-interfaces)
- [Exact reference](#exact-reference)
- [Limitations and sources](#limitations-and-sources)

## Overview

FABLE is organized around strict request values, direct owner functions, native library objects, and UUID-addressed durable objects. Dependencies point from operator edges toward scientific owners.

### System shape

```text
strict JSON request
        |
        v
CLI or direct Python call
        |
        +--> acquisition --> Corpus
        +--> tuning ------> Study
        +--> fitting -----> native Lightning artifact
        +--> evaluation --> observations.parquet
        |
        v
resolved evaluation facts and transient reductions
```

`fable.config` owns frozen Pydantic values and small discriminated unions. `fable.requests` mints fresh UUIDv4 instances. A boundary receiving raw JSON or durable bytes hydrates the owning typed value once; downstream code trusts that value.

### Dependency direction

```text
CLI / serving / Python callers
                |
                v
execution, acquisition, study, evaluation
                |
                v
modeling, temporal, min_block_fee, corpus
                |
                v
strict config values and canonical addresses
```

Each owner has one system seam:

- `corpus` owns canonical `BlockFrame` row truth and completed Corpus association.
- `temporal` owns causal feature state, fixed-block context/outcome geometry, and lazy historical examples.
- `min_block_fee` owns target state, classification support, loss, two-head output, and decode.
- `modeling` owns the three concrete neural definitions, Lightning fitting, and native checkpoint loading.
- `study` owns bounded candidate membership, ordered retained results, publication, and selected-result materialization.
- `evaluation` owns canonical observations, resolved evaluation facts, and transient reduction.

### Durable object flow

#### Corpus

`CorpusRequest` names an inclusive chain block range and its UUID. `acquire_corpus()` reads ordinary block RPC responses in deterministic order into an owner-local hidden sibling. It validates chain identity, links, timestamps, fee and gas domains, proves ancestry to a finalized anchor, writes the exact `corpus.json` and `blocks.parquet`, removes transport-only hashes, then renames the hidden directory to the canonical Corpus address.

#### Study and artifact

`TuneRequest` contains one `ExperimentSemantics` and a finite, family-specific `MethodSpace`. `run_candidate()` prepares training state, fits one supplied Method, and appends one successful `RetainedResult` to Study scratch. `publish_study()` renames the ordered result set to its canonical JSON file.

A baseline `TrainRequest` embeds its complete `TrainingDefinition`. A selected-Study request instead names the exact Study UUID and result index while carrying the experiment. Training loads that exact row, reconstructs the definition from its Method, fits through Lightning, and renames the native weights-only best checkpoint to the artifact UUID address. The checkpoint embeds the request, feature and target state, optional classification support, and—only for selected-Study training—the exact result index and Method.

#### Evaluation

`EvaluateRequest` names an artifact, same-source Corpus, validation or testing origin window, and evaluation UUID. Evaluation rebuilds historical examples with persisted state, runs the artifact on CUDA, writes one nonnull ordered observation per origin, and publishes `evaluation.json` with `observations.parquet`.

`resolve_evaluations()` reads explicit evaluation IDs into ordered trusted facts: request, training source and definition, Corpus, lazy observations, transient reduction, and trainable parameter count. Repeated evaluation, artifact, and Corpus IDs share resolution within the call. `reduce_evaluation()` uses the same request, artifact, observation, and reduction authority without loading a Corpus.

### Training and inference

Historical preparation produces lazy datasets over contiguous feature, fee, and block-number backing.

The model union is closed: LSTM, Transformer, or Transformer-LSTM. Every model consumes float32 `[B,C,F]` and returns action logits `[B,K]` plus a scalar standardized minimum-fee prediction `[B]`. The architecture is independent of target construction and evaluation accounting.

Live serving loads cwd-local `SERVING.yaml` once, selects an exact artifact cell, freezes the latest closed head, reads its `C-1` predecessors, applies the checkpoint's ordered feature state, runs one CPU batch, and returns the decoded target coordinate.

### External boundaries

Acquisition and serving use ordinary Web3 RPC clients supplied at their operator boundaries.

`fable.execution.submit()` is the boundary to native OpenSSH and Slurm execution. [ADR 0007](docs/adr/0007-native-external-execution-boundary.md) records that ownership decision.

Completed objects have direct canonical addresses and own their exact requests once. UUIDs provide instance identity; associations provide meaning. See [ADR 0006](docs/adr/0006-direct-durable-object-authority.md).

## One decision, end to end

FABLE makes a decision immediately after a closed parent block `h`. Every number in this hand-computable Ethereum example is a fabricated teaching value.

### 1. Fix the geometry

Suppose:

```text
h = 25,400,000
C = 200 closed context blocks
K = 5 future outcome blocks
```

The model may see exactly blocks `h-C+1 … h`, or `25,399,801 … 25,400,000`. The complete outcome is `h+1 … h+K`. Actions are zero-based:

| Action `k` | Intended target block |
| ---: | ---: |
| 0 | 25,400,001 |
| 1 | 25,400,002 |
| 2 | 25,400,003 |
| 3 | 25,400,004 |
| 4 | 25,400,005 |

The arithmetic is always `target_block = h + 1 + k`.

### 2. Build only closed-parent inputs

For this calculation only, let the request's ordered feature tuple contain three supported features. Suppose closed parent `h` has:

```text
base_fee_per_gas = 24,000,000,000 wei/gas
gas_used         = 27,000,000 gas
gas_limit        = 36,000,000 gas
```

The raw closed-row features are:

```text
log_base_fee_per_gas = ln(24,000,000,000 / (1 wei/gas))
                     = 23.901320

gas_utilization = 27,000,000 / 36,000,000
                = 0.75
```

Ethereum's forming-child fee follows the exact parent recurrence. The parent target is `36,000,000 // 2 = 18,000,000 gas`. Usage exceeds target by `9,000,000`, so ordered integer arithmetic gives:

```text
increase = 24,000,000,000 * 9,000,000 // 18,000,000 // 8
         = 1,500,000,000 wei/gas

forming_child_base_fee = 25,500,000,000 wei/gas

log_exact_forming_base_fee_per_gas
  = ln(25,500,000,000 / (1 wei/gas))
  = 23.961944
```

All feature inputs come from block `h` or earlier. The exact child fee is an Ethereum parent-state result. The other 199 rows are prepared the same way from their own closed facts.

Training-only Float64 means and population standard deviations standardize the ordered raw matrix. The one-origin input is finite float32 `[C,F] = [200,3]`; a live batch is `[1,200,3]`.

### 3. Keep outcomes on the other side of the origin

Invent these complete future base fees:

```text
h+1 ... h+5 = [25.5, 23, 21, 20, 22] gwei/gas
```

They are stored and compared as positive Int64 wei/gas:

```text
[25_500_000_000, 23_000_000_000, 21_000_000_000,
 20_000_000_000, 22_000_000_000]
```

NumPy first-index `argmin` gives label `k*=3` and raw minimum
`O=20,000,000,000 wei/gas`. A tie would select the earliest equal minimum.

The dataset item is:

| Value | Shape and dtype |
| --- | --- |
| `inputs` | `[200,3]`, float32 |
| `label` | scalar, int64 |
| `target` | scalar, float32 |
| `base_fees` | `[5]`, int64 |
| `origin_block` | scalar, int64 |

The raw minimum first enters Float64 natural-log coordinates:

```text
ell = ln(20,000,000,000 / (1 wei/gas)) = 23.718998
```

For a purely illustrative fitted `TargetState(mean=23.5, standard_deviation=0.25)`:

```text
z = (23.718998 - 23.5) / 0.25 = 0.875992
```

Real state is fitted once from all retained training-origin minima with Float64 `ddof=0`. Validation, testing, and live inference use the persisted state.

### 4. Separate the roles

Every retained origin must have its complete `K`-block outcome inside its role. If validation begins at parent block `V`, a training origin is eligible only when `h+K < V`. Testing starts only after `validation_last_parent + K`.

Training fits feature state, target state, optional class support, and weights. Validation selects epochs and retained candidate objectives. Testing measures held-out behavior.

### 5. Compute one two-head loss

For one origin, suppose the model returns:

```text
action_logits = [0.2, 1.1, -0.1, 1.7, 0.5]
minimum_fee_z = 0.7
```

For the arithmetic, let the request supply:

```text
classification_weighting = unweighted
classification_scale     = 1.0
regression_threshold     = 1.0
regression_scale         = 0.5
```

Another request may supply corrected inverse-frequency classification and other positive or nonnegative values.

With label `3`, cross-entropy is:

```text
CE = log(sum(exp(action_logits))) - action_logits[3]
   = log(exp(0.2)+exp(1.1)+exp(-0.1)+exp(1.7)+exp(0.5)) - 1.7
   ≈ 0.805777

c = 1.0 * CE ≈ 0.805777
```

The z error is `e = 0.7 - 0.875992 = -0.175992`. Because `|e| < beta=1`:

```text
SmoothL1(e) = 0.5 * e^2 / beta ≈ 0.015487
r           = 0.5 * SmoothL1(e) ≈ 0.007743
t           = c + r ≈ 0.813520
```

For this one-origin batch, `mean_total = sum(t_i)/B = t`. In a larger batch every origin contributes one scaled classification term plus one scaled regression term, with sample count `B` as the denominator.

### 6. Decode and account

Native first-index `argmax` selects `k=3`; equal maximum logits would choose the first. The intended target is block `25,400,004`.

For this outcome:

```text
B = immediate h+1 fee       = 25.5 gwei/gas
R = selected h+1+k fee      = 20.0 gwei/gas
O = hindsight minimum fee   = 20.0 gwei/gas

S = B - R = 5.5
G = B - O = 5.5
Q = R - O = 0.0
S + Q = G
```

This origin saves base fee per gas versus immediate action and captures all available hindsight opportunity. Across a declared evaluation window, FABLE first sums raw Int64 `S`, `G`, and `Q`, then forms Float64 ratio-of-sums.

The canonical observation also records `selected_action_wait_seconds = timestamp(h+3)-timestamp(h)` and `full_horizon_elapsed_seconds = timestamp(h+5)-timestamp(h)`. The selected wait is zero for `k=0`.

### 7. Carry the same contract into serving

The checkpoint fixes chain association, `C`, `K`, ordered features, feature state, target state, model definition, and weights. Live serving freezes the latest closed head, fetches exactly `C-1` predecessors, creates `[1,C,F]`, validates the action logits, and decodes the same way.

Continuing the teaching values, the API response shape is:

```json
{"head_block": 25400000, "selected_action_k": 3, "target_block": 25400004}
```

## Scientific contract

FABLE is a closed-parent, fixed-block-horizon temporal learning system. This document owns the causal information set, `C/K/k` geometry, fitted-state rules, feature and target equations, evaluation estimands, claim boundaries, sources, and limitations.

### Lineage and ownership

The manuscript *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments* describes a broader spatial, temporal, and distributed-reputation system. Its temporal experiment motivates a future minimum-block decision, an associated scalar fee prediction, the LSTM/Transformer/Transformer-LSTM comparison, chronological roles, and a weighted cross-entropy plus Smooth-L1 lineage.

FABLE specifies the current closed-parent origins, fixed block-count geometry, causal features, raw-integer target ties, training-fitted state, request-authored loss, exhaustive equal-origin evaluation, durable objects, and serving semantics.

### Closed-parent causality

A decision origin occurs immediately after block `h` closes. Facts in blocks through `h` may be inputs. Facts from `h+1` onward are outcomes and cannot influence features or fitted state available at that origin.

For context length `C` and horizon `K`:

```text
context rows:  h-C+1, ..., h
outcome rows:  h+1,   ..., h+K
actions:       k in {0, ..., K-1}
target block:  b = h+1+k
```

Block number owns geometry. Timestamp spacing may vary while the number of context and outcome rows stays fixed.

An origin is eligible only with all `C` context rows and all `K` outcome rows. At a boundary where the next role begins at parent `B`, an earlier origin must satisfy `h+K < B`. Therefore no training outcome reaches validation, and no validation outcome reaches testing.

### Role ownership and fitted populations

Training alone may fit:

- feature population means and standard deviations;
- target natural-log mean and standard deviation;
- class support for corrected inverse-frequency loss;
- neural weights.

Validation selects the earliest best epoch and supplies candidate objectives. Testing measures only. Changing a method, feature route, loss choice, horizon, context, or other scientific decision after inspecting testing would turn that measurement into selection evidence.

#### Feature state

Let raw training-support feature row `x_r ∈ R^F`. For each ordered feature `j`:

```text
mu_j    = (1/N) sum_r x_rj
sigma_j = sqrt((1/N) sum_r (x_rj - mu_j)^2)
z_rj    = (x_rj - mu_j) / sigma_j
```

Fitting uses Float64 and `ddof=0`; every `sigma_j` must be positive. Transformation returns finite float32. Training support contains each closed block row once, so overlapping model windows do not reweight feature-state fitting.

### Causal features

The request supplies a nonempty unique ordered tuple drawn from the supported names, making feature choice request-authored.

| Feature | Raw equation and unit | Domain and availability |
| --- | --- | --- |
| `log_base_fee_per_gas` | `ln(base_fee_per_gas / (1 wei/gas))` | Fee positive; closed-row header fact. |
| `gas_utilization` | `gas_used / gas_limit` | `gas_limit>0`, `0≤gas_used≤gas_limit`; known after row close. |
| `log_exact_forming_base_fee_per_gas` | `ln(exact_child_base_fee / (1 wei/gas))` | Positive; Ethereum-only parent-state recurrence. |
| `log_gas_limit` | `ln(gas_limit / (1 gas))` | Gas limit positive; closed-row header fact. |
| `log1p_tx_count` | `ln(1 + tx_count / (1 transaction))` | Transaction count nonnegative; known after row close. |
| `hour_sin` | `sin(2π hour_UTC/24)` | `hour_UTC = (timestamp//3600) mod 24`; closed timestamp. |
| `hour_cos` | `cos(2π hour_UTC/24)` | Same angle and availability. |

The exact forming-fee column implements the Ethereum parent-known recurrence. Polygon and Avalanche requests use the other supported features.

#### Ethereum forming-child recurrence

For positive parent fee `f`, parent gas used `u`, and positive gas limit `L`, use Python integers throughout:

```text
t = L // 2

if u == t:
    f_child = f
elif u > t:
    f_child = f + max(f * (u - t) // t // 8, 1)
else:
    f_child = f - f * (t - u) // t // 8
```

`t` and the final child fee must be positive. Python integers carry the recurrence through the two ordered divisions; the one-wei floor applies only upward. The completed positive integer is then logged in Float64. This follows the integer ordering in [EIP-1559](https://eips.ethereum.org/EIPS/eip-1559#specification).

### Historical tensors and targets

One lazy historical item has:

| Value | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `inputs` | `[C,F]` | float32 | Standardized closed rows `h-C+1 … h`. |
| `label` | scalar | int64 | Earliest horizon minimum action. |
| `target` | scalar | float32 | Standardized log horizon minimum. |
| `base_fees` | `[K]` | int64 | Positive fees at `h+1 … h+K`. |
| `origin_block` | scalar | int64 | Closed parent `h`. |

Collation produces `[B,C,F]`, `[B]`, `[B]`, `[B,K]`, and `[B]`.

Let the positive Int64 outcomes be `y_i0 … y_i,K-1`. Then:

```text
k_i* = first argmin_k y_ik
o_i  = y_i,k_i*
ell_i = ln(o_i / (1 wei/gas))
```

Raw integer comparison precedes floating conversion. Equal minima choose the first index, consistent with [NumPy `argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html).

Target state is fitted over retained training origins only:

```text
mu_o    = mean_Float64(ell_i)
sigma_o = std_Float64(ell_i, ddof=0)
z_i     = Float32((ell_i - mu_o) / sigma_o)
```

`sigma_o` must be positive, and standardization follows the equation above exactly.

### Targets, loss, and decode

All concrete model definitions return:

```text
action_logits: [B,K]
minimum_fee_z: [B]
```

The first head scores actions. The second predicts the standardized natural log of the same horizon minimum.

#### Classification

The request chooses `cross_entropy` and either unweighted or corrected inverse-frequency classification. For corrected weighting, training support for every class must be positive. With `N_train` labels, `K` actions, and support `n_k`:

```text
w_k = N_train / (K n_k)
```

For origin `i`, letting `a_i` be its logits and `k_i*` its label:

```text
c_i = classification_scale * CE(a_i, k_i*; optional w)
```

The corrected form weights that origin's negative log probability by `w_k`. Classification scale is finite and nonnegative.

#### Regression

The request chooses Smooth L1, a positive threshold `beta`, and a finite nonnegative regression scale. For `e_i = predicted_z_i - target_z_i`:

```text
smooth_l1_beta(e) = 0.5 e^2 / beta       if |e| < beta
                    |e| - 0.5 beta       otherwise

r_i = regression_scale * smooth_l1_beta(e_i)
```

#### Total

```text
t_i = c_i + r_i
mean_total = (sum_i t_i) / B
```

The denominator is the number of origins in the batch. Evaluation reconstructs Smooth L1 from persisted float32 z values in Float32, with the artifact's threshold and regression scale; the persisted classification contribution already includes classification weighting and scale. The operative functions match PyTorch's [`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) and [`smooth_l1_loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html).

Decode is native `argmax(action_logits, dim=-1)`. Equal maximum logits select the first index, and decode depends on the logits alone.

### Model concepts

FABLE uses a closed discriminated union of three concrete sequence models:

- LSTM recurrently summarizes the fixed context and uses its final state.
- Transformer projects each row, adds sinusoidal positions, applies self-attention, and uses the final encoded position.
- Transformer-LSTM applies the Transformer encoder, then recurrently summarizes the encoded sequence.

All three attach the same two MLP heads. Architecture capacity belongs to `ModelDefinition` or Method; target and loss meaning stays in `fable.min_block_fee`.

### Evaluation estimands

For each eligible origin, define raw Int64 fee-per-gas values:

```text
B_i = fee at immediate action k=0
R_i = fee at selected action k_i
O_i = minimum fee over the K-block horizon

S_i = B_i - R_i             savings
G_i = B_i - O_i             hindsight opportunity
Q_i = R_i - O_i             hindsight regret
```

`O_i≤B_i` and `O_i≤R_i`, so `G_i,Q_i≥0`; `S_i` is signed. Exact identity:

```text
S_i + Q_i = G_i
```

Differences are computed before Float64 casting. Over `N` equally weighted origins:

```text
savings ratio       = sum(S) / sum(B)
opportunity ratio   = sum(G) / sum(B)
regret ratio        = sum(Q) / sum(B)
captured opportunity = sum(S) / sum(G), only when exact sum(G) != 0
```

Captured opportunity is null iff exact raw-Int64 `sum(G)==0`. Every other reduction field is nonnull. Positive `B_i` and `O_i` make the per-origin views defined:

```text
mean_i(S_i / B_i)
mean_i(Q_i / O_i)
mean_i(G_i / O_i)
```

Their zero-denominator exclusion counts are therefore zero. A harmful action has `R_i>B_i`. Selected action counts have length `K` and sum to `N`.

Accuracy is the fraction `k_i=k_i*`. Macro-F1 averages over the union-active classes whose true support plus prediction count is positive; absent-from-both classes do not enter its denominator.

The canonical time descriptions are:

```text
selected_action_wait_seconds = timestamp(h+k) - timestamp(h)
full_horizon_elapsed_seconds = timestamp(h+K) - timestamp(h)
```

The first is zero at `k=0`.

### HPO interpretation

A `TuneRequest` freezes the experiment and one finite typed MethodSpace. An operator submits complete Methods from that set. Each successful fit contributes validation total loss, earliest best epoch, and completed epochs in retention order. Selected training names an exact result index.

## Architecture and deep interfaces

The sections below place each direct owner interface beside the scientific and durable-object contracts it serves. Exact public records, paths, commands, YAML fields, and schemas remain in [Exact reference](#exact-reference).

### Acquisition

Acquisition exposes one interface: `acquire_corpus(request, *, storage_root, rpc_url, poa)`. It turns one exact `CorpusRequest` into one finalized, canonical Corpus. Transport, resumable work, validation, finality, and publication stay behind that call.

#### Contract

The request fixes a UUIDv4, chain ID, and inclusive first/last block. Acquisition publishes to an initially absent destination under the explicit `storage_root`; `rpc_url` and `poa` stay at the invocation boundary.

The completed object contains:

```text
corpora/<corpus_id>/
  corpus.json
  blocks.parquet
```

`corpus.json` stores the exact request and one finalized anchor. `blocks.parquet` stores the requested contiguous rows in block-number order with the exact seven-column canonical schema documented in the [reference](#corpus-object).

#### Hidden resumable prefix

Acquisition works in `corpora/.<corpus_id>/`. A request JSON binds that scratch directory to one request. Complete deterministic checkpoint chunks cover at most 4,096 consecutive blocks, and the chunk list must form an exact prefix from the requested first block. Scratch validation enforces the request binding, expected filenames, prefix continuity, complete chunks, schema, nonnull domains, and parent links.

The scratch prefix records owner-local request binding and checkpoint progress. The completed Corpus directory is the published interface.

#### Ordered ordinary reads

The RPC endpoint's chain ID must equal the request. Within a checkpoint, acquisition issues ordinary `eth_getBlockByNumber` reads in batches of four and consumes results in requested-number order. Every block must provide:

- its requested number and normalized block/parent hashes;
- a nonnegative, nondecreasing timestamp;
- positive base fee and gas limit;
- gas used in `[0, gas_limit]`;
- a transaction sequence whose length becomes `tx_count`.

Parent hashes must link across every read and checkpoint boundary. Rows enter the canonical object after all block facts pass validation.

#### Finality proof

After the exact range is present, acquisition reads the provider's `finalized` tag. The finalized height must not precede the requested last block. If it is later, numbered headers must prove that the staged last block is its ancestor. The tagged anchor is then reread by number; number, hash, and parent hash must match the tagged response.

Block and parent hashes are proof-only acquisition facts. The completed parquet drops them. The finalized anchor keeps only its number and normalized hash.

#### Publication

Checkpoint rows stream into one canonical parquet file. The exact Corpus candidate is reloaded: `BlockFrame` validates schema, nonnull domains, row count, requested range, chain ID, timestamp order, and fee/gas/transaction domains; `Corpus` validates request association and finalized coverage. Publication then removes checkpoint metadata and renames the hidden sibling to the canonical directory.

`BlockFrame(frame, definition)` is the public canonical-row boundary. Its `definition` identifies the exact owned range, `select_range(first_block, last_block)` returns an inclusive trusted subrange, and `to_polars()` returns an isolated native frame. Construction and native access isolate caller mutation. Range selection is positional after the full frame has been validated; it does not rescan rows. The value carries neither hashes nor finality provenance.

### Temporal preparation

Temporal preparation has two direct paths: historical fixed-block examples and live closed-head inference. Both use the same ordered feature functions and persisted training-only feature state.

#### Historical interface

`prepare_fit_history(corpus, experiment)` validates complete context and outcome support, fits state from training support only, and returns:

```text
HistoricalPreparation
  training: HistoricalDataset
  validation: HistoricalDataset
  feature_state: FeatureState
  target_state: TargetState
  classification_state: ClassificationLossState | None
```

`prepare_historical_window(corpus, experiment, window, *, feature_state, target_state)` prepares an exact validation or testing window with persisted state. A validation window must equal the experiment's authored validation window. A testing window must begin after all validation outcomes are complete.

For origin `h`, support is exact by block number:

```text
context:  h-C+1 ... h
outcome:  h+1   ... h+K
action k: target h+1+k
```

The Corpus must include every context and outcome block named by this geometry.

#### Lazy dataset

Preparation builds one contiguous CPU backing over the needed range:

- transformed feature rows: float32 `[rows,F]`;
- raw base fees: int64 `[rows]`;
- block numbers: int64 `[rows]`.

It stores per-origin row positions and first-argmin labels as int64 vectors and standardized targets as a float32 vector. `HistoricalDataset.__getitem__()` slices one float32 `[C,F]` input and one int64 `[K]` raw fee outcome on demand, plus scalar int64 label and origin block and scalar float32 target.

#### Feature state

The ordered feature tuple is request authority. Names must be unique and supported by the direct implementation. Raw features are assembled in exactly that order as Float64. Training-support population means and standard deviations use `ddof=0`; a constant feature is invalid. Transform applies those values and returns finite C-contiguous float32 rows.

Exact formulas, units, causal availability, and the Ethereum forming-fee recurrence belong to the [theory](#causal-features).

#### Outcome preparation

Historical outcomes remain positive int64 fees. For each origin, NumPy first-index `argmin` over `h+1 … h+K` produces the label; the selected raw minimum feeds the fitted target state. Training fits classification support only when the request asks for corrected inverse-frequency weighting.

Role boundaries are complete-outcome boundaries. The training last parent plus `K` must be strictly before the first validation parent; an authored testing window obeys the same rule after validation. Training alone fits feature state, target state, classification support, and model weights.

#### Live interface

Serving freezes one latest closed head `h`, reads exactly `C-1` predecessors, decodes untrusted RPC quantities, constructs one live `BlockFrame`, transforms the ordered features with the artifact's `FeatureState`, and constructs float32 `[1,C,F]`. Historical preparation owns outcomes, labels, and target values.

The artifact fixes `C`, `K`, feature order, and fitted states. Decoding returns `k`, and serving reports `h+1+k` as the target block coordinate.

### Minimum-block-fee task

Top-level `fable.min_block_fee` keeps the architecture-neutral target, loss, and decode contract. Temporal preparation supplies its targets, model families return its output, and evaluation consumes the result.

#### Owned values

`TargetState` contains the Float64 population mean and positive population standard deviation of `ln(raw horizon minimum)` over retained training origins.

`ClassificationLossState` contains one positive support count per action for corrected inverse-frequency classification. Unweighted classification carries `None`.

`MinBlockFeeOutput` has two tensors:

```text
action_logits:  [B,K]
minimum_fee_z:  [B]
```

The scalar head predicts the standardized natural log of the horizon minimum. Its scientific interpretation is defined in the [theory](#targets-loss-and-decode).

#### Direct functions

- `fit_target_state(raw_minima)` requires a nonempty positive int64 vector, computes Float64 `ln`, mean, and `ddof=0` standard deviation, and rejects constant targets.
- `standardize_target(raw_minima, state)` returns finite contiguous float32 z values.
- `fit_classification_loss_state(labels, *, horizon_blocks, loss_definition)` validates label range and, for corrected weighting, requires positive training support for every action.
- `min_block_fee_loss(...)` validates both heads and targets, computes scaled per-origin classification and regression contributions, and returns their per-origin sum plus the sample-denominator mean.
- `decode_action(output)` applies native first-index `argmax` along the action dimension.

The exact equations and weighting alternatives are in the [theory](#targets-loss-and-decode).

#### Boundaries

Temporal preparation owns raw `[K]` outcomes, first-argmin labels, and standardized targets. Model code owns the sequence encoder and the two concrete heads. Evaluation owns observation publication and economic accounting.

### Study

Tuning is a bounded question over a finite typed MethodSpace. A Study contains the exact `TuneRequest` and its ordered successful results.

#### Request and membership

`TuneRequest` fixes a Study UUID, Corpus UUID, `ExperimentSemantics`, and one family-specific nonempty tuple of unique Methods. Each Method is complete: architecture capacity, dropout, AdamW values, training batch size, and fit policy.

`apply_method(request, method)` requires exact membership in the request's MethodSpace, then composes one `TrainingDefinition`. `training_definition_from_method(experiment, method)` performs the same family-specific composition for a Method supplied by an authoritative association.

#### Candidate run

`run_candidate(storage_root, request, method, deployment)` loads the request's Corpus, prepares training history and state, fits the exact Method through native Lightning, and retains one successful result. Candidate checkpoints stay in Study scratch; training publishes artifacts.

`RetainedResult` has four fields:

- the exact Method;
- finite complete-validation total-loss objective;
- one-based earliest selected epoch;
- one-based completed epoch count.

The selected epoch cannot exceed completed epochs, and completed epochs cannot exceed the Method maximum.

#### Ordered progress and publication

Candidate success appends to `studies/.<study_id>/progress.json`. Existing progress must contain the identical request. Appends preserve caller completion order and directly replace the progress file through one hidden temporary sibling.

`publish_study(storage_root, study_id)` validates progress and renames it to `studies/<study_id>.json`, preserving completion order. An existing canonical Study is an error.

#### Selected training

A selected-Study `TrainRequest` supplies the exact Study UUID and zero-based `study_result_index`. `materialize_selected_training()` loads the canonical Study, verifies Study and Corpus associations, selects that ordered row, and reconstructs its `TrainingDefinition` from the embedded experiment and Method.

The resulting native artifact embeds the same result index and Method for later loading and evaluation.

### Evaluation

Evaluation separates canonical observations, resolved evaluation facts, and transient reductions. Explicit UUIDs connect durable objects to their trusted derived facts.

#### Canonical evaluation

`evaluate(request, storage_root, deployment)` loads the exact Corpus and native artifact, requires the artifact's source Corpus to equal the evaluation Corpus, prepares the authored validation or testing origin window with persisted state, and performs CUDA inference.

For every eligible origin it writes one ordered, nonnull observation containing the decision coordinate, target and decoded actions, scaled classification contribution, auxiliary z prediction, raw fee facts, and elapsed-time descriptions. Work is written under `evaluations/.<evaluation_id>/` and renamed to:

```text
evaluations/<evaluation_id>/
  evaluation.json
  observations.parquet
```

The JSON is exactly the `EvaluateRequest`. The parquet schema is the canonical 13-column contract in the [reference](#canonical-observations).

#### Resolved evaluation and reduction

`resolve_evaluations(storage_root, evaluation_ids)` resolves each first occurrence in caller order. It strictly hydrates and checks the request ID, loads and validates one artifact association per artifact ID, derives the baseline or selected training definition, checks Corpus association and evaluation-window geometry, validates and reduces the canonical observations, then loads one Corpus per Corpus ID. Empty input returns an empty tuple; duplicates retain their caller positions and share the same resolved value.

`ResolvedEvaluation` carries only the typed request, training source, training definition, Corpus, lazy canonical observations, 43-field reduction, and trainable parameter count. Neural modules and fitted-state internals do not cross this interface. The observation validation requires exact origin coverage, nonnegative origin timestamps, nonnull inputs, action bounds, positive previous/closed/target fees, wait bounds, finite values, and scientific identities.

`reduce_evaluation(storage_root, evaluation_id) -> polars.DataFrame` uses the same request, artifact, observation, and scientific-reduction core without acquiring a Corpus. Regression target and Smooth-L1 use the artifact's `TargetState` and authored loss. Economic differences begin in raw Int64 before Float64 aggregation. The sole nullable result is captured opportunity when exact total opportunity is zero.

## Exact reference

This reference defines FABLE's strict requests, completed objects, direct addresses, commands, operator YAML, serving/mobile surfaces, and evaluation schemas.

### Scalar conventions

- Object IDs are UUIDv4.
- `PositiveInt` means strict integer `>0`; `NonNegativeInt` means strict integer `≥0`. Booleans are not integers.
- Scientific floats are finite. Positive/nonnegative bounds are stated per field.
- Block ranges and origin windows are inclusive.
- Base fees are positive Int64 wei/gas unless a field explicitly says Float64 aggregation.
- Timestamps and elapsed values are integer seconds.
- Strict records reject unknown fields and revalidate nested instances.

Distribution name, import root, and installed executable are `fable`; the static distribution version is `0.1.0`.

### Requests and definitions

#### Corpus

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `CorpusDefinition` | `chain_id` | PositiveInt |
|  | `first_block` | NonNegativeInt |
|  | `last_block` | NonNegativeInt, `last_block≥first_block` |
| `CorpusRequest` | `corpus_id` | UUIDv4 |
|  | `definition` | `CorpusDefinition` |

Fresh constructor:

```python
fresh_corpus_request(definition: CorpusDefinition) -> CorpusRequest
```

#### Scientific semantics

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `OriginWindow` | `role` | `"training" | "validation" | "testing"` |
|  | `first_parent_block` | NonNegativeInt |
|  | `last_parent_block` | NonNegativeInt, not before first |
| `LossDefinition` | `classification_algorithm` | exactly `"cross_entropy"` |
|  | `classification_weighting` | `"unweighted" | "corrected_inverse_frequency"` |
|  | `regression_algorithm` | exactly `"smooth_l1"` |
|  | `regression_threshold` | finite float `>0` |
|  | `classification_scale` | finite float `≥0` |
|  | `regression_scale` | finite float `≥0` |
| `ExperimentSemantics` | `training_window` | `OriginWindow(role="training")` |
|  | `validation_window` | `OriginWindow(role="validation")` |
|  | `context_blocks` | PositiveInt `C` |
|  | `horizon_blocks` | PositiveInt `K` |
|  | `ordered_features` | nonempty unique tuple of nonempty strings |
|  | `loss` | `LossDefinition` |

The training last parent plus `K` must be strictly less than the validation first parent.

#### Model definitions

`ModelDefinition` is a discriminated union on `family`:

| Family | Ordered fields after `family` |
| --- | --- |
| `lstm` | `hidden: PositiveInt`; `layers: PositiveInt`; `head_hidden: PositiveInt`; `dropout: 0≤float<1` |
| `transformer` | `model_width`; `attention_heads`; `transformer_layers`; `feedforward_width`; `head_hidden`: PositiveInt; `dropout: 0≤float<1` |
| `transformer_lstm` | `model_width`; `attention_heads`; `transformer_layers`; `feedforward_width`; `lstm_hidden`; `lstm_layers`; `head_hidden`: PositiveInt; `dropout: 0≤float<1` |

Transformer widths must be even and divisible by `attention_heads`.

#### Method and MethodSpace

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `AdamWMethod` | `learning_rate` | finite float `>0` |
|  | `weight_decay` | finite float `≥0` |
| `FitMethod` | `accumulation` | PositiveInt |
|  | `gradient_clip_norm` | finite float `≥0` |
|  | `scheduler` | exactly `"none"` |
|  | `seed` | NonNegativeInt |
|  | `max_epochs` | PositiveInt |
|  | `validate_every_completed_epoch` | PositiveInt |
|  | `patience` | NonNegativeInt |
|  | `min_delta` | finite float `≥0` |
|  | `improvement` | exactly `"strict_lower"` |
|  | `restore` | exactly `"earliest_best"` |

Every serialized Method has ordered fields `dropout`, `optimizer`, `training_batch`, `fit`, `family`, and `capacity`. `dropout` is finite in `[0,1)`, `optimizer` is `AdamWMethod`, `training_batch` is PositiveInt, and `fit` is `FitMethod`.

| Method family | Capacity fields |
| --- | --- |
| `lstm` | `hidden`, `layers`, `head_hidden`: PositiveInt |
| `transformer` | `model_width`, `attention_heads`, `transformer_layers`, `feedforward_width`, `head_hidden`: PositiveInt |
| `transformer_lstm` | `model_width`, `attention_heads`, `transformer_layers`, `feedforward_width`, `lstm_hidden`, `lstm_layers`, `head_hidden`: PositiveInt |

Transformer capacity obeys the same even/divisible width constraints. A `MethodSpace` has `family` plus a nonempty tuple `methods` of unique Methods from exactly that family.

#### Study, training, and workflow requests

| Record | Ordered field | Type and rule |
| --- | --- | --- |
| `StudyDefinition` | `experiment` | `ExperimentSemantics` |
|  | `method_space` | family-discriminated `MethodSpace` |
| `TrainingDefinition` | `experiment` | `ExperimentSemantics` |
|  | `model` | `ModelDefinition` |
|  | `optimizer` | `AdamWMethod` |
|  | `training_batch` | PositiveInt |
|  | `fit` | `FitMethod` |
| `BaselineSource` | `kind` | exactly `"baseline"` |
|  | `corpus_id` | UUIDv4 |
|  | `training_definition` | `TrainingDefinition` |
| `SelectedStudySource` | `kind` | exactly `"selected_study"` |
|  | `corpus_id` | UUIDv4 |
|  | `study_id` | UUIDv4 |
|  | `study_result_index` | NonNegativeInt |
|  | `experiment` | `ExperimentSemantics` |
| `TrainRequest` | `workflow` | exactly `"train"` |
|  | `artifact_id` | UUIDv4 |
|  | `source` | `BaselineSource | SelectedStudySource` |
| `TuneRequest` | `workflow` | exactly `"tune"` |
|  | `study_id` | UUIDv4 |
|  | `corpus_id` | UUIDv4 |
|  | `study_definition` | `StudyDefinition` |
| `EvaluateRequest` | `workflow` | exactly `"evaluate"` |
|  | `evaluation_id` | UUIDv4 |
|  | `artifact_id` | UUIDv4 |
|  | `corpus_id` | UUIDv4 |
|  | `window` | validation or testing `OriginWindow`; training is rejected |

`WorkflowRequest` is exactly `TrainRequest | EvaluateRequest`. `TuneRequest` is intentionally separate.

Fresh constructors:

```python
fresh_train_request(source: TrainingSource) -> TrainRequest
fresh_tune_request(corpus_id: UUID, study_definition: StudyDefinition) -> TuneRequest
fresh_evaluate_request(
    artifact_id: UUID,
    corpus_id: UUID,
    window: OriginWindow,
) -> EvaluateRequest
```

### Durable addresses and objects

Given an explicit `storage_root`:

```text
corpora/<corpus_id>/corpus.json
corpora/<corpus_id>/blocks.parquet
studies/<study_id>.json
artifacts/<artifact_id>.ckpt
evaluations/<evaluation_id>/evaluation.json
evaluations/<evaluation_id>/observations.parquet
```

IDs are lowercase UUID strings produced by `str(UUID)` and appear directly in the paths above.

#### Corpus object

`corpus.json` has exactly:

```text
request: CorpusRequest
finalized_anchor:
  block_number: integer >= 0
  block_hash: 64 lowercase hexadecimal characters
```

`blocks.parquet` has this exact ordered, nonnull schema:

| # | Column | Type | Unit/rule |
| ---: | --- | --- | --- |
| 1 | `block_number` | Int64 | contiguous inclusive request range |
| 2 | `timestamp` | Int64 | nonnegative seconds; nondecreasing |
| 3 | `chain_id` | Int64 | equals request chain |
| 4 | `base_fee_per_gas` | Int64 | positive wei/gas |
| 5 | `gas_used` | Int64 | gas, `0≤used≤limit` |
| 6 | `gas_limit` | Int64 | positive gas |
| 7 | `tx_count` | Int64 | nonnegative transaction count |

Direct loader:

```python
load_corpus(storage_root: Path, corpus_id: UUID4) -> Corpus
```

`Corpus.blocks` is a `BlockFrame` whose definition equals the request definition. The finalized anchor covers its last block. These in-memory ownership facts do not change the durable JSON or Parquet formats above.

#### Study object

`studies/<study_id>.json` is a strict `Study`:

```text
request: TuneRequest
trials: nonempty ordered tuple[RetainedResult, ...]
```

Each `RetainedResult` has exact ordered fields:

| Field | Type/rule |
| --- | --- |
| `method` | exact Method contained in the request MethodSpace |
| `objective` | finite float validation total loss |
| `selected_epoch` | integer `≥1` |
| `completed_epochs` | integer `≥selected_epoch` and `≤method.fit.max_epochs` |

#### Native Lightning artifact

`artifacts/<artifact_id>.ckpt` is the native Lightning weights-only best checkpoint. Its `ArtifactAssociation` contains:

| Ordered field | Type/rule |
| --- | --- |
| `request` | exact `TrainRequest`; embedded artifact UUID must match path |
| `feature_state` | Float64 means and positive standard deviations, equal feature width |
| `target_state` | Float64 finite mean and positive standard deviation |
| `classification_state` | null for unweighted; positive class-support tuple of length `K` for corrected weighting |
| `study_result_index` | absent/null for baseline; exact nonnegative source index for selected Study |
| `method` | absent/null for baseline; exact selected Method for selected Study |

Direct loader:

```python
load_artifact(
    storage_root: Path,
    artifact_id: UUID,
) -> tuple[ArtifactAssociation, torch.nn.Module]
```

#### Evaluation object

`evaluation.json` is exactly the `EvaluateRequest`. `observations.parquet` is the canonical schema below. Reductions are transient views over this directory.

### CLI

Four public command leaves:

```text
fable submit REQUEST.json [REQUEST.json ...]
fable corpus acquire REQUEST.json --rpc-url URL --poa|--no-poa
fable study run TUNE_REQUEST.json METHOD.json
fable study finalize STUDY_ID
```

- `submit` accepts one or more WorkflowRequest files and prints one positive Slurm job ID per request.
- `corpus acquire` reads `STORAGE_ROOT`, requires an absolute path, and requires an explicit PoA choice.
- `study run` validates one strict TuneRequest and one strict Method, then prints the candidate Slurm job ID.
- `study finalize` requires UUIDv4, reads absolute `STORAGE_ROOT`, and publishes existing progress.

Two help-hidden generated-job leaves:

```text
fable remote workflow
fable remote candidate
```

Generated Slurm scripts call these leaves with strict JSON envelopes on standard input.

### Remote submission

`from fable.execution import submit` is the public execution seam:

```python
submit(request: WorkflowRequest) -> int
```

It reads cwd-local `REMOTE.yaml` with this exact strict schema:

| Section | Ordered field | Type/rule |
| --- | --- | --- |
| root | `ssh` | nonempty string passed to OpenSSH |
|  | `executable` | nonempty absolute installed-executable path |
|  | `storage_root` | nonempty absolute path |
|  | `log_root` | nonempty absolute path |
| `resources` | `partition` | nonempty string |
|  | `gres` | string |
|  | `cpus_per_task` | PositiveInt |
|  | `memory_gb` | PositiveInt, rendered as `--mem=<n>G` |
|  | `time_limit` | nonempty Slurm time string |
| `deployment` | `evaluation_batch_size` | PositiveInt |
|  | `num_workers` | NonNegativeInt |
|  | `pin_memory` | bool |
|  | `prefetch_factor` | PositiveInt or null |
|  | `persistent_workers` | bool |
|  | `deterministic` | bool or exactly `"warn"` |
|  | `benchmark` | bool |
|  | `float32_matmul_precision` | `"highest" | "high"` |
|  | `cuda_matmul_allow_tf32` | bool |
|  | `cudnn_allow_tf32` | bool |

The generated script requests one node/task, writes `%j.out` under `log_root`, exports `STORAGE_ROOT`, and executes `fable remote workflow` or `fable remote candidate` with a stdin heredoc. Submission is one `ssh -T -o BatchMode=yes … sbatch --parsable` call.

`STORAGE_ROOT`, `ETHEREUM_RPC_URL`, `POLYGON_RPC_URL`, and `AVALANCHE_RPC_URL` are the neutral operator spellings. `STORAGE_ROOT` is the implicit environment input to current CLI/remote Python paths. RPC endpoints arrive through explicit invocation or YAML values.

### Serving and mobile

The serving factory is `fable.serving:create_app`; its display title is `FABLE Inference API`. It reads cwd-local `SERVING.yaml` once during application lifespan and uses its values literally.

The strict serving root contains an absolute `storage_root` and exactly three records:
`ethereum`, `polygon`, and `avalanche`. Each chain record contains a nonempty `rpc_url` and
exactly four UUIDv4 fields: `k2_artifact_id`, `k3_artifact_id`, `k4_artifact_id`, and
`k5_artifact_id`. All fields are required; extra fields are rejected at both levels.

Serving expects Ethereum chain ID `1`, Polygon `137`, and Avalanche C-Chain `43114`; the Polygon and Avalanche clients install the PoA extra-data middleware.

The strict request is:

```text
chain: "ethereum" | "polygon" | "avalanche"
K: 2 | 3 | 4 | 5
```

`POST /inference` returns nonnegative integers:

```text
head_block
selected_action_k
target_block = head_block + 1 + selected_action_k
```

OpenAPI and interactive documentation routes are disabled. The server verifies provider chain ID, loads the exact selected-Study artifact, requires request `K` to equal artifact `K`, reads the latest head plus `C-1` predecessors, prepares float32 `[1,C,F]`, and runs CPU inference.

The private Expo app manifest is fixed at:

| Fact | Value |
| --- | --- |
| display | `FABLE Demo` |
| package | `fable-mobile-demo` |
| Expo slug/scheme | `fable-demo` |
| iOS bundle ID | `dev.edoski.fable.demo` |
| Android package | `dev.edoski.fable.demo` |
| entry | `expo/AppEntry` |

Its only backend variable is `EXPO_PUBLIC_FABLE_BACKEND_URL`. It posts the strict request to `/inference` and displays only the three response fields.

### Evaluation API

Public exports from `fable.evaluation`:

```python
class EvaluationDeployment:
    batch_size: PositiveInt
    num_workers: NonNegativeInt
    pin_memory: bool
    prefetch_factor: PositiveInt | None
    persistent_workers: bool
    deterministic: bool | Literal["warn"]
    benchmark: bool
    float32_matmul_precision: Literal["highest", "high"]
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool

@dataclass(frozen=True, slots=True)
class ResolvedEvaluation:
    request: EvaluateRequest
    training_source: TrainingSource
    training_definition: TrainingDefinition
    corpus: Corpus
    observations: polars.LazyFrame
    reduction: polars.DataFrame
    trainable_parameter_count: int

evaluate(
    request: EvaluateRequest,
    storage_root: Path,
    deployment: EvaluationDeployment,
) -> None

reduce_evaluation(
    storage_root: Path,
    evaluation_id: UUID,
) -> polars.DataFrame

resolve_evaluations(
    storage_root: Path,
    evaluation_ids: tuple[UUID, ...],
) -> tuple[ResolvedEvaluation, ...]
```

#### Canonical observations

Destination: `evaluations/<evaluation_id>/observations.parquet`. Status: canonical, ordered, nonnull, one row per inclusive origin in ascending block order.

| # | Field | Type | Unit/meaning |
| ---: | --- | --- | --- |
| 1 | `origin_block` | Int64 | closed parent `h` |
| 2 | `origin_timestamp` | Int64 | `timestamp(h)`, seconds |
| 3 | `selected_action_k` | Int64 | decoded `k∈[0,K)` |
| 4 | `earliest_hindsight_action_k` | Int64 | first raw-fee argmin |
| 5 | `classification_loss_contribution` | Float64 | per-origin CE after weighting and classification scale |
| 6 | `predicted_hindsight_minimum_base_fee_z` | Float32 | auxiliary standardized log-minimum prediction |
| 7 | `previous_closed_parent_base_fee_per_gas` | Int64 | fee at `h-1`, wei/gas |
| 8 | `closed_parent_base_fee_per_gas` | Int64 | fee at `h`, wei/gas |
| 9 | `immediate_k0_base_fee_per_gas` | Int64 | fee at `h+1`, wei/gas |
| 10 | `selected_target_base_fee_per_gas` | Int64 | fee at `h+1+k`, wei/gas |
| 11 | `hindsight_minimum_base_fee_per_gas` | Int64 | raw minimum over `h+1…h+K`, wei/gas |
| 12 | `selected_action_wait_seconds` | Int64 | `timestamp(h+k)-timestamp(h)`; zero for `k=0` |
| 13 | `full_horizon_elapsed_seconds` | Int64 | `timestamp(h+K)-timestamp(h)` |

#### Transient reduction

Destination: none. `reduce_evaluation()` returns a one-row DataFrame. Status: derived, transient, noncanonical. Every field is nonnull except field 22, whose null rule is exact `sum(G)==0`. `selected_action_count_by_k` is a native `List(Int64)` of length `K` whose values sum to the eligible count.

| # | Field | Type |
| ---: | --- | --- |
| 1 | `evaluation_id` | String |
| 2 | `eligible_origin_count` | Int64 |
| 3 | `earliest_hindsight_label_correct_count` | Int64 |
| 4 | `earliest_hindsight_label_cross_entropy_loss_sum` | Float64 |
| 5 | `hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss_sum` | Float64 |
| 6 | `hindsight_minimum_base_fee_per_gas_within_k_natural_log_absolute_error_sum` | Float64 |
| 7 | `hindsight_minimum_base_fee_per_gas_within_k_natural_log_squared_error_sum` | Float64 |
| 8 | `earliest_hindsight_label_cross_entropy_loss` | Float64 |
| 9 | `hindsight_minimum_base_fee_per_gas_within_k_smooth_l1_loss` | Float64 |
| 10 | `hindsight_minimum_base_fee_per_gas_within_k_natural_log_mae` | Float64 |
| 11 | `hindsight_minimum_base_fee_per_gas_within_k_natural_log_mse` | Float64 |
| 12 | `multitask_total_loss` | Float64 |
| 13 | `earliest_hindsight_label_accuracy` | Float64 |
| 14 | `earliest_hindsight_label_macro_f1` | Float64 |
| 15 | `immediate_k0_base_fee_per_gas_sum` | Float64 |
| 16 | `finite_target_base_fee_per_gas_savings_sum` | Float64 |
| 17 | `finite_target_base_fee_per_gas_hindsight_opportunity_sum` | Float64 |
| 18 | `finite_target_base_fee_per_gas_hindsight_regret_sum` | Float64 |
| 19 | `finite_target_base_fee_per_gas_savings_ratio_vs_immediate_k0` | Float64 |
| 20 | `finite_target_base_fee_per_gas_hindsight_opportunity_ratio_vs_immediate_k0` | Float64 |
| 21 | `finite_target_base_fee_per_gas_hindsight_regret_ratio_vs_immediate_k0` | Float64 |
| 22 | `signed_captured_hindsight_opportunity_ratio` | nullable Float64 |
| 23 | `target_base_fee_per_gas_savings_fraction_vs_immediate_k0_sum` | Float64 |
| 24 | `target_base_fee_per_gas_savings_fraction_vs_immediate_k0_defined_origin_count` | Int64 |
| 25 | `target_base_fee_per_gas_savings_fraction_vs_immediate_k0_zero_denominator_exclusion_count` | Int64 |
| 26 | `mean_origin_target_base_fee_per_gas_savings_fraction_vs_immediate_k0` | Float64 |
| 27 | `selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum` | Float64 |
| 28 | `selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count` | Int64 |
| 29 | `selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count` | Int64 |
| 30 | `mean_origin_selected_target_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k` | Float64 |
| 31 | `immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_sum` | Float64 |
| 32 | `immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_defined_origin_count` | Int64 |
| 33 | `immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k_zero_denominator_exclusion_count` | Int64 |
| 34 | `mean_origin_immediate_k0_base_fee_per_gas_increase_fraction_vs_hindsight_best_within_k` | Float64 |
| 35 | `harmful_action_count` | Int64 |
| 36 | `harmful_action_rate` | Float64 |
| 37 | `selected_action_count_by_k` | List(Int64) |
| 38 | `extra_wait_block_opportunities_vs_immediate_k0_sum` | Float64 |
| 39 | `mean_extra_wait_block_opportunities_vs_immediate_k0` | Float64 |
| 40 | `selected_action_wait_seconds_sum` | Float64 |
| 41 | `mean_selected_action_wait_seconds` | Float64 |
| 42 | `full_horizon_elapsed_seconds_sum` | Float64 |
| 43 | `mean_full_horizon_elapsed_seconds` | Float64 |

Fields 15–18 and 38, 40, 42 are sums in their named units; their ratios/means use one eligible origin as the unit. Fee sums are wei/gas represented as Float64 after exact Int64 differences.

## Limitations and sources

### Claim boundary and limitations

Evaluation describes target block base fee per gas over every eligible origin in one declared historical window. Its claims are bounded as follows:

- Base fee per gas omits priority fee and transaction gas use.
- Target-block intent does not guarantee inclusion at that block.
- The auxiliary head is not calibrated uncertainty or a quote.
- One seed or one time range does not establish seed, regime, or future robustness.
- Different `K` values are different classification problems; testing cannot choose a best `K`.
- Native assets, fee levels, protocol rules, and ranges differ by chain; totals are never pooled across chains.
- Exhaustive origins remove sampling within the declared range, not temporal dependence or selection bias outside it.

### Sources

- [EIP-1559 specification](https://eips.ethereum.org/EIPS/eip-1559)
- [Reference temporal-model repository at the frozen commit](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/tree/bcf80b92877941e3b05a7dc5138560ffe41df27e)
- [Hochreiter and Schmidhuber, “Long Short-Term Memory”](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)
- [Vaswani et al., “Attention Is All You Need”](https://arxiv.org/abs/1706.03762)
- [Caruana, “Multitask Learning”](https://doi.org/10.1023/A:1007379606734)
- [NumPy `argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
- [PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch Smooth L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html)
