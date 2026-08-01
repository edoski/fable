# Inference benchmark slice research

Date: 2026-08-01

## Implementation authority

This note records the source investigation, including stricter alternatives considered during
planning. GitHub issue #148 and the implementation-review ledger own the final contract. The
approved Slice 1 implementation trusts canonical FABLE publishers and loaders, derives chain and
family groups from the manifests, reuses evaluator-owned `ROLLING_HORIZONS`, and keeps only the
evaluation-to-artifact join, exact nine-group/36-label campaign cardinality, workload-origin, and
resume-protocol checks at the benchmark boundary. It intentionally omits host probes, observation
parity, artifact hashes, and repeated canonical association validation.

## Verdict

All four slices are implementable from the current repository. The CPU study should remain the
required thesis experiment; energy and optional MPS also belong on the reference Mac. The
configured Slurm CUDA host should continue to produce the canonical artifacts and held-out
evaluations, but a separate CUDA benchmark would answer a different hardware question and adds no
necessary evidence to the approved consumer-hardware claim.

The source review identified five corrections that are now incorporated in the issue and ledger:

1. The latency stress cascade and the economic rolling reducer are different workloads. The stress
   cascade runs all four models at one unchanged origin; the economic reducer can advance its origin
   after a terminal action.
2. PyTorch `torch.utils.benchmark.Timer` is not the correct primary recorder for chronological
   per-origin tails. Use `time.perf_counter_ns()` after a fixed warmup.
3. `powermetrics` supports an estimated combined CPU+GPU+ANE quantity. Calling it total SoC energy
   is too broad.
4. Energy must use actual sample and phase durations. Multiplying a retained-sample mean by an
   assumed 60 seconds is invalid after boundary samples are removed.
5. Same-second block timestamps are resolution-limited observations, not proven zero-duration
   block intervals. Feasibility statistics must separate them from positive intervals.

The implementation should stay experiment-private. A suitable shape is one operator entry point,
`experiments/inference_benchmark.py`, with private helpers under an experiment-owned module and
focused tests under `tests/experiments/`. It should not extend the installed `fable` CLI or create a
generic benchmark database.

## Shared input and output contract

The operator supplies the completed K-study experiment UUID, completed held-out experiment UUID,
an explicit output directory, and later the frozen economic-assumption file. The implementation
must fail rather than infer “latest” objects.

The completed K-study manifest is the artifact roster. K-study preparation creates names of the
form `chain.family.K<horizon>` and the completed manifest maps each name directly to its artifact
UUID ([`experiments/k_study.py:18-46`](../../experiments/k_study.py#L18),
[`src/fable/experiments.py:21-40`](../../src/fable/experiments.py#L21)). The completed held-out
manifest uses the same names but maps them to evaluation UUIDs
([`experiments/held_out.py:21-54`](../../experiments/held_out.py#L21)). There is no separate canonical
“final model” manifest joining these two objects.

The final lean resolution therefore:

1. load both manifests with `load_experiment_manifest`;
2. groups rolling cells from canonical labels using evaluator-owned `ROLLING_HORIZONS`;
3. requires exactly nine complete groups and 36 selected labels in both manifests;
4. parses each selected canonical `evaluation.json` as a strict `EvaluateRequest`; and
5. requires the evaluation artifact ID to equal the K-study artifact ID for that cell.

Canonical request, artifact, Corpus, Study, Method, feature, model-state, and observation invariants
remain owned by their existing publishers and loaders. Repeating them here would weaken ADR 0006's
direct-object authority rather than strengthen it
([`docs/adr/0006-direct-durable-object-authority.md:13-30`](../adr/0006-direct-durable-object-authority.md#L13)).

Scientific output should be immutable and explicit, for example:

```text
<output-directory>/
  protocol.json
  latency/
    <cell>/sweep-<n>.parquet
  energy/
    <cell>/powermetrics.plist
    <cell>/phases.json
    <cell>/pairs.parquet
  report.json
  tables/
```

Create each result through a hidden sibling or temporary file and rename it only after validation.
Never overwrite an existing scientific output. Setup output uses a separate disposable directory
and never enters `report.json`. Slice 1 `protocol.json` records only the two experiment IDs, derived
rolling horizons, selected artifact/evaluation UUID roster, warmup count, and sweep count. Host and
runtime conditions are controlled manually and stated once in the thesis methodology.

## Slice 1: CPU latency

### Canonical tensors

For each evaluation, reconstruct inputs exactly as evaluation did:

1. load its canonical Corpus with `load_corpus_blocks`;
2. use the artifact's embedded `ExperimentSemantics`, `FeatureState`, and `TargetState`; and
3. call `prepare_historical_window` with the exact testing window from `evaluation.json`.

This reuses the same training-fitted normalization state and causal feature construction as the
canonical evaluator ([`src/fable/evaluation.py:41-76`](../../src/fable/evaluation.py#L41),
[`src/fable/temporal.py:232-254`](../../src/fable/temporal.py#L232)). A dataset item contains an
unbatched `[C,F]` input view and its exact origin block; batch size one is obtained outside the timed
region by adding the leading dimension
([`src/fable/temporal.py:151-183`](../../src/fable/temporal.py#L151)). Do not use the repository's
evaluation `DataLoader`: it is fixed at batch size 512, four workers, and pinned memory for CUDA
evaluation ([`src/fable/_runtime.py:10-25`](../../src/fable/_runtime.py#L10)).

The four testing windows are not equal. Held-out preparation gives shorter-horizon evaluations
extra trailing origins so a root can advance up to three times; tests demonstrate the resulting
`K=2,3,4,5` last-parent offsets
([`experiments/held_out.py:30-46`](../../experiments/held_out.py#L30),
[`tests/experiments/test_k_study.py:228-246`](../../tests/experiments/test_k_study.py#L228)). Prepare
each horizon's full `HistoricalDataset` and contiguous backing once. Do not duplicate every
overlapping C-block window into a new tensor bank. Resolve an item by origin and obtain its batched
view before starting that call's clock. Never zip the four datasets by row index.

Only four models for the current architecture-chain cell need to be resident simultaneously. Run
cells sequentially; “36 artifacts” means campaign coverage, not loading all 36 models at once.

### Stress workload versus economic rolling policy

The primary feasibility and energy workload is a same-origin stress cascade. For every `K=5`
testing origin `h`, obtain the `K=5`, `K=4`, `K=3`, and `K=2` batch-one input views at that same
origin before starting the clock, then execute all four model forwards and decodes back to back.
Every held-out K window contains the K5 root range, so these four inputs are defined. This workload
answers the approved question: whether the maximum four-forward model-compute burst could fit
inside one observed block opportunity.

This stress cascade is deliberately not the economic rolling reducer. The sealed economic policy
has one `K=5` root `h` and constructs its route as follows:

```text
t = h
for K in 5, 4, 3, 2:
    action = predicted action for the K evaluation at origin t
    route includes the K input at origin t
    if K > 2 and action == K - 1:
        t = t + 1
selected block = t + 1 + K2 action
```

This is the canonical evaluator's rule: actions are range-checked, required origins are resolved by
block number, and only a terminal action advances the origin
([`src/fable/evaluation.py:201-248`](../../src/fable/evaluation.py#L201)). The manual gives the same
fixed-deadline contract ([`docs/FABLE.md:504-510`](../FABLE.md#L504)). It is a **same-root rolling
episode** whose later decisions can belong to `h+1`, `h+2`, or `h+3`. It is reconstructed in Slice 3
for economic savings; it is not compared as one physical same-block latency event.

The timed stress cascade still executes `decode_action` after every forward and discards the
result. It does not use actions to change its fixed same-origin inputs. Canonical
`observations.parquet` already owns evaluated predictions, so Slice 1 performs no separate parity
pass and writes no parity output.

### Timing primitive and boundaries

Use each model in `eval()` under one `torch.inference_mode()` scope. PyTorch documents that
evaluation mode and inference mode are orthogonal: inference mode removes autograd overhead, while
`eval()` controls modules such as Dropout
([PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.7/notes/autograd.html#evaluation-mode-nn-module-eval)).

Use `time.perf_counter_ns()` immediately before and after the measured workload. Python defines it
as the highest-resolution performance counter for short durations; on macOS it is monotonic and
shared across processes ([Python 3.11 `time` documentation](https://docs.python.org/3.11/library/time.html#time.perf_counter_ns)).
Measure the empty clock-pair overhead during preflight, preserve it as a diagnostic, and do not
subtract it from results. Leaving a tens-of-nanoseconds clock cost in a millisecond-scale result is
conservative and avoids noise amplification.

`torch.utils.benchmark.Timer` should not own the reported sweep. PyTorch 2.7 documents that it
repeats one statement in blocks, chooses a block size to amortize timer overhead, performs hidden
warmups, and defaults to one thread
([PyTorch 2.7 benchmark API](https://docs.pytorch.org/docs/2.7/benchmark_utils.html)). Those semantics
are useful for stationary microbenchmarks but cannot produce chronological individual-origin p95
or p99 values. `timeit(number=1)` also performs at least two hidden warmups in the pinned
implementation, and the current repository cannot import `torch.utils.benchmark` because
`setuptools` is not installed by [`pyproject.toml:14-40`](../../pyproject.toml#L14). Do not add a
dependency solely to force this tool into an unsuitable protocol.

Preserve the ordinary CPU configuration by not changing either PyTorch thread pool. The reference
thread configuration is recorded manually with the other host conditions in the thesis method.
Naive use of `Timer` would silently replace the intra-op count with its one-thread default.

Use distinct passes:

- four standalone passes time `model(input) + decode_action(output)` for each horizon;
- one same-origin stress pass times the four forwards and four decodes with one outer clock pair.

Do not place per-stage clocks inside the cascade. Their overhead would change the quantity being
measured. Standalone horizon timing supplies the per-horizon distributions; the outer cascade
supplies the feasibility quantity.

### Repetition and rows

An excluded warmup uses a fixed count or duration per resident model and cascade. Preflight chooses
and freezes that rule; the reported runner never warms “until it looks stable.” Each of ten main
sweeps traverses every selected origin chronologically at batch size one. A complete sweep means
all four standalone workloads plus the cascade for one cell. Rotate their pass order by a fixed
schedule across sweeps so one workload is not always measured hottest.

Each raw row needs at least:

```text
cell, sweep, pass_order, workload, origin_block, elapsed_ns
```

Economic rolling routes remain separate Slice 3 derived records. Preserve nanoseconds; convert to
milliseconds only during reduction. Record the observed maximum as descriptive evidence, never as
a theoretical bound.

### Slice 1 tests and gates

Tests cover manifest-derived exact nine-group/36-label resolution, the evaluation/artifact join,
chronological batch views, model residency, same-origin stress inputs, decode inclusion, warmup
exclusion, outer cascade clocks, deterministic rotation, compact output, atomic publication, and
protocol-matched resume. Fixture timing values test mechanics only.

The full-corpus fallback still lacks an objective definition of “impractical.” Do not implement an
automatic cutoff. Run one complete timing pilot, report projected wall time and output size, then
freeze either full coverage or the approved deterministic stratified origin list before the main
campaign. Only the final choice belongs in the thesis methodology.

## Slice 2: `powermetrics` energy

### Collector boundary

The current Mac's installed Apple manual and `/usr/bin/powermetrics --help` establish that:

- the tool requires superuser privileges;
- `--sample-rate` is expressed in milliseconds;
- plist output is machine-readable and NUL-separated;
- `cpu_power`, `gpu_power`, `ane_power`, and `thermal` samplers are available;
- reported average powers are model-derived estimates and may be inaccurate; and
- per-process Energy Impact is only a rough platform-specific proxy.

Apple also documents `powermetrics` as a system energy diagnostic
([Apple Energy Efficiency Guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html)).
The supported thesis name is therefore:

> `powermetrics`-estimated incremental CPU+GPU+ANE energy on the reference Mac.

Do not call it total SoC, whole-machine, battery, or wall-plug energy. The three named estimates do
not claim coverage of memory, storage, display, fans, other board loads, or charger losses.

Use one continuous collector across all phases for a cell, not a new collector for every window:

```sh
sudo /usr/bin/powermetrics \
  --samplers cpu_power,gpu_power,ane_power,thermal \
  --sample-rate 1000 \
  --poweravg 0 \
  --format plist
```

Authenticate interactively before the campaign, then elevate only the collector. The unprivileged
Python runner opens the output file and passes it as collector stdout, so the raw capture remains
user-owned. Omit `--sample-count`, supervise the process, and terminate it with SIGTERM, which the
manual defines as a clean stop. Preserve the raw NUL-separated bytes.

The 2026-08-01 supervised capture at `/tmp/fable-powermetrics.DBpLW8` resolved the local schema.
Each NUL-separated plist record contains a whole-second UTC `timestamp`, nanosecond `elapsed_ns`,
top-level `thermal_pressure`, and `processor.cpu_power`, `gpu_power`, `ane_power`, and
`combined_power`. The combined value equals the three rails within printed rounding, and the paired
text capture labels these power values in milliwatts. Store one sanitized XML sample containing
only these fields; do not copy host, OS, boot, or unrelated sampler data into the repository.

### Phase protocol

Process one architecture-chain cell at a time with its four warm resident models and input routes.
For every pair:

1. keep AC power connected and Low Power Mode disabled;
2. wait the frozen recovery delay;
3. run a 60-second loaded-idle phase with no forwards;
4. run a 60-second active phase that continuously completes batch-one same-origin stress cascades;
   and
5. retain any non-nominal pair as invalid for a later cell rerun.

Always using idle then active is cleaner than alternating order here. An active-then-idle pair can
raise its own baseline through residual heat. If alternating order is retained, a validated recovery
must occur between its two measured phases; otherwise that design is biased. The simpler correction
is 20 idle-to-active pairs with recovery between pairs.

Cycle through the frozen origin list and rotate its starting offset by pair. Feature construction,
dataset preparation, and disk access remain outside the active phase. The active phase necessarily
includes in-memory origin lookup, creation of four batch views, the four forwards and decodes, and
minimal loop/count bookkeeping. Name this as workload energy rather than pretending it is isolated
kernel energy. Record wall-clock phase boundaries for joining plist samples and monotonic elapsed
time for throughput. Count only fully completed cascades. The nominal duration is 60 seconds, but
calculations use the measured duration.

### Parsing and energy calculation

Split the capture on NUL and parse every nonempty record with Python `plistlib`. Each power sample
describes an interval ending within the whole UTC second named by `timestamp`. Accept it only when
the entire possible interval lies inside the recorded phase; this conservatively omits boundary
samples despite the timestamp's one-second precision. No rejection ledger or cadence framework is
needed because the raw trace remains the recomputation authority.

For accepted sample `j`, after preflight confirms milliwatts and nanoseconds:

\[
E_j[\mathrm J]
=
P_j[\mathrm{mW}]\frac{\Delta t_j[\mathrm{ns}]}{10^{12}}.
\]

For pair `r`, time-weight the accepted active and idle power samples:

\[
\bar P_{a,r}
=
\frac{\sum_j P_{a,r,j}\Delta t_{a,r,j}}
     {\sum_j\Delta t_{a,r,j}},
\qquad
\bar P_{i,r}
=
\frac{\sum_j P_{i,r,j}\Delta t_{i,r,j}}
     {\sum_j\Delta t_{i,r,j}}.
\]

Let `N_r` be completed cascades and `T_{a,r}` the exact active-phase duration. Then:

\[
e_r
=
\frac{(\bar P_{a,r}-\bar P_{i,r})/1000}
     {N_r/T_{a,r}}
\quad \mathrm{J/cascade}.
\]

This estimates phase-average incremental watts divided by measured cascades per second. It remains
valid when boundary samples are discarded and avoids mixing retained-sample coverage with the
nominal 60-second phase duration.

Retain negative `e_r` values. They are possible measurement noise, not values to clip. Report the
three rail powers as diagnostics and combined power as the primary input. Thermal state is an
instantaneous control, not an energy integral. A non-nominal trial is retained and marked invalid by
the prespecified environmental rule, then rerun; it is not silently deleted.

### Slice 2 tests and gate

Use the sanitized real plist fixture for parser acceptance and compose NUL framing from repeated
fixture bytes. Focused synthetic cases cover conservative boundary inclusion, thermal invalidation,
incomplete cascades, collector failure, actual-duration throughput, negative estimates, and a
hand-calculated time-weighted pair. Do not run `sudo` in tests.

The schema gate is complete. Scientific collection remains blocked until the final artifacts are
available and a short active-idle run demonstrates a measurable signal. If 60 seconds cannot
resolve it, lengthen the phase before increasing pair count; changing duration after inspecting main
results is not permitted.

## Slice 3: statistical, deadline, and economic reduction

### Statistical units and confidence intervals

For each architecture-chain cell and timed workload, compute the latency mean CI from the ten
**sweep means** and the energy mean CI from the twenty **pair estimates**. Repeated origins inside
a sweep, cascades inside an active loop, and one-second power samples are not independent
experiments.

Use the two-sided Student-t interval:

\[
\bar x \pm t_{0.975,n-1}\frac{s}{\sqrt n}.
\]

NIST gives this interval for an unknown population standard deviation and explains that uncertainty
shrinks with `sqrt(n)` and grows with sample noise
([NIST confidence limits for a mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)).
Its applicability assumes one stable process, approximate normality, and no time correlation
([NIST process assumptions](https://www.itl.nist.gov/div898/handbook/prc/section1/prc12.htm)). Preserve
and inspect the ten or twenty values in execution order. A visible thermal trend, step change, or
heavy outlier means the fixed-condition experiment failed; it is not repaired by treating thousands
of origins as independent.

The preflight chooses the smallest integer `n` at or above 10 for latency or 20 for energy satisfying

\[
t_{0.975,n-1}\frac{s_{pilot}}{\sqrt n}
\leq \epsilon |\bar x_{pilot}|,
\]

where `epsilon` is 5% or 10%. Freeze that `n` before the main run. If the pilot energy mean is near
zero, relative precision is undefined or unstable; require an explicit absolute-precision decision
or a longer phase rather than dividing by a near-zero mean.

Neither NumPy nor the pinned PyTorch environment supplies the required dynamic Student-t critical
value through the APIs already used here. Add SciPy to an experiment-only dependency group and use
`scipy.stats.t.ppf`; do not hand-code a t table
([SciPy Student-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)).
PyTorch `Measurement.significant_figures` is a display heuristic using an interquartile estimate and
`z=1.645`, not the required 95% mean CI
([PyTorch 2.7 benchmark API](https://docs.pytorch.org/docs/2.7/benchmark_utils.html)).

Report latency mean and 95% CI, median, p95, p99, and observed maximum. The CI uses sweep means; the
other values are descriptive quantiles over raw individual calls. Freeze the NumPy quantile method
in `protocol.json`.

### Block intervals and feasibility

For each chain, use the `K=5` testing window as the root-opportunity set. Read Corpus timestamps for
each root `h` and its next block and compute `timestamp[h+1]-timestamp[h]`. This produces one
chain-level interval set shared by all three architectures. Validate exact expected block numbers
before positional slicing because `BlockFrame` enforces schema and range but does not rescan row
numbers ([`src/fable/corpus.py:29-62`](../../src/fable/corpus.py#L29)). Corpus timestamps are integer
seconds ([`src/fable/corpus.py:14-25`](../../src/fable/corpus.py#L14)).

Partition differences as follows:

- negative: hard data error; stop;
- zero: same-second, resolution-limited tie; report count and fraction separately;
- positive: supported empirical interval.

Compute `positive_block_interval_p01_seconds` using an order-statistic quantile method such as
NumPy `method="inverted_cdf"`, so the reported value remains an observed whole-second interval
([NumPy quantile methods](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)).
Compute the fraction of **positive** intervals shorter than cascade p99. Zero ties are unresolved,
not failures. Treating an integer timestamp tie as a literal zero-duration physical interval would
falsely prove that inference missed a deadline.

For each cell:

```text
margin_ms = positive_block_interval_p01_seconds * 1000 - cascade_p99_ms
```

A positive margin supports the declared conservative comparison. Also report the positive-interval
shorter fraction and unresolved tie fraction. The claim remains model-compute-only because RPC,
propagation, feature construction, and submission are outside the clock.

### Energy cost and break-even gas

For each cell, convert the mean energy estimate with

\[
c_{cascade}[\mathrm{EUR}]
=
\frac{e_{cascade}[\mathrm J]}{3.6\times10^6}
p_{electricity}[\mathrm{EUR/kWh}].
\]

Report the point cost per cascade, its direct energy-CI transformation, and cost per million
cascades. Preserve a negative or zero lower CI bound as estimator uncertainty; do not describe it
as physically negative consumption.

Reconstruct the canonical dynamic-origin rolling episode directly from each horizon's sealed
observations using the route above. This is independent of the same-origin timing stress workload.
Because the public reducer exposes only aggregate metrics, the experiment-private reconstruction
must verify that its resulting three economic aggregates exactly match `reduce_rolling` before its
raw selected blocks are trusted. For root `i`, calculate the primary raw base-fee difference

\[
\Delta_i
=
B_i(0)-B_i(\hat k_{rolling}),
\]

in atomic native units per gas. This follows FABLE's canonical base-fee-only estimand; P50-inclusive
differences remain an explicitly secondary retrospective proxy
([`src/fable/evaluation.py:251-275`](../../src/fable/evaluation.py#L251),
[`docs/FABLE.md:496-510`](../FABLE.md#L496)). Do not multiply average fractional savings by average
fees.

Let `A_chain` be declared atomic native units per token and `q_chain` the dated EUR price per native
token. The expected base-fee saving for gas usage `G` is

\[
S(G)[\mathrm{EUR}]
=
G\,\overline{\Delta}\,\frac{q_{chain}}{A_{chain}}.
\]

EIP-1559 defines transaction fee-per-gas components and charges gas used times effective gas price
([EIP-1559](https://eips.ethereum.org/EIPS/eip-1559#specification)). Therefore the primary
break-even gas quantity is

\[
G_{break-even}
=
\frac{c_{cascade}A_{chain}}
     {\overline{\Delta}\,q_{chain}}.
\]

Compute `mean(Delta_i)` from raw per-origin integer differences first. If it is zero or negative,
report “no positive expected base-fee break-even” rather than a negative gas amount. Record the
electricity price, currency, date, source, token symbol, token price, atomic-unit scale, and rounding
policy in a static strict input file. No reducer performs a live price lookup.

### Slice 3 outputs and tests

`report.json` is the single reduced authority and references immutable raw input digests. Generate
TSV or LaTeX tables only from this report. Tests need hand-derived fixtures for Student-t intervals,
preflight `n`, each latency quantile, positive/tied/negative block intervals, millisecond conversion,
joule-to-kWh cost, million-cascade cost, positive/zero/negative fee differences, atomic-unit and
token-price conversion, and break-even arithmetic. Recompute-and-compare tests should prove every
table is derived from the raw records.

## Slice 4: optional MPS

The current host is built with MPS and reports MPS available. A synthetic batch-one smoke test on
the current LSTM, Transformer, and Transformer-LSTM implementations completed on `mps:0` with exact
actions and close CPU outputs. This validates the present operator seam, not the final artifacts or
scientific campaign.

Run the authorization preflight in a fresh subprocess. Set environment variables before importing
PyTorch:

```text
PYTORCH_ENABLE_MPS_FALLBACK=0
PYTORCH_MPS_LOG_PROFILE_INFO=4
```

PyTorch documents `PYTORCH_ENABLE_MPS_FALLBACK=1` as enabling CPU fallback
([PyTorch 2.7 MPS environment variables](https://docs.pytorch.org/docs/2.7/mps_environment_variables.html)).
Setting it to zero is necessary but insufficient: pinned PyTorch 2.7.1 registers some unconditional
CPU fallback operations
([MPS fallback registrations](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/mps/MPSFallback.mm#L70-L88)).
Profile bit 4 is the backend's CPU-fallback logging flag
([MPS profiler flags](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/mps/MPSProfiler.h#L234-L260)).
Run every final architecture/horizon path over the parity set, capture stderr, and reject any
CPU-fallback record. Disable profiler logging before reported timing because it changes the measured
runtime.

Also require `torch.backends.mps.is_built()` and `is_available()`, models and input tensors resident
on `mps`, exact decoded actions, the frozen numeric parity tolerance, and no unsupported-operation
exception. PyTorch's MPS guide defines the availability checks and device transfer
([PyTorch 2.7 MPS backend](https://docs.pytorch.org/docs/2.7/notes/mps.html)).

MPS execution is asynchronous. The timing boundary is:

```python
torch.mps.synchronize()
start_ns = time.perf_counter_ns()
output = workload()
torch.mps.synchronize()
elapsed_ns = time.perf_counter_ns() - start_ns
```

PyTorch defines `torch.mps.synchronize()` as waiting for queued MPS kernels
([PyTorch 2.7 MPS API](https://docs.pytorch.org/docs/2.7/mps.html)). Pinned
`torch.utils.benchmark.Timer` synchronizes CUDA, XPU, and private-use backends, but not MPS
([PyTorch 2.7.1 timer source](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/benchmark/utils/timer.py#L16-L33)).
Required finite checks and decodes during the stress cascade can also synchronize the host and must
remain inside the workload.

If the preflight passes, repeat Slice 1 as a separately named MPS configuration with the same roots,
statistics, and provenance. For optional MPS energy, repeat Slice 2 using combined CPU+GPU+ANE as
the primary quantity and retain individual rails diagnostically. Never pool CPU and MPS samples.

## Verified execution location

All benchmark slices belong on the local M2 Max:

| Slice | Location | Reason |
| --- | --- | --- |
| 1 CPU latency | local Mac | It is the declared consumer reference hardware and owns the ordinary CPU runtime being claimed. |
| 2 energy | local Mac | `/usr/bin/powermetrics`, Apple power state, and the measured CPU+GPU+ANE rails are Mac-specific. |
| 3 reduction | local Mac | It is deterministic CPU reduction over transferred canonical objects and local raw measurements. |
| 4 MPS | local Mac | MPS and its synchronization/fallback checks are Apple-GPU-specific. |

The configured remote is a heterogeneous CUDA Slurm target. Its current allocation packing and
resource values are execution details rather than a benchmark deployment contract
([`REMOTE.yaml:1-10`](../../REMOTE.yaml#L1)). ADR 0007 defines it as the narrow immutable-image
boundary for independent training and evaluation workflows, not as the thesis deployment target
([`docs/adr/0007-native-external-execution-boundary.md:9-30`](../adr/0007-native-external-execution-boundary.md#L9)).

The remote should finish canonical training and held-out evaluation. Completed artifacts,
evaluations, and corpora are then transferred through the existing boundary before local
measurement. No live remote call or scheduler mutation is needed for these slices.

A CUDA benchmark would not validate the Mac production-feasibility claim. It would introduce a
different runtime, accelerator, power instrument, and potentially three different GPU models; the
current image path also identifies a separately pinned deployed revision. A defensible CUDA study
would need an exact GPU model, exact image revision, datacenter power telemetry, and a separately
named estimand. That work adds no necessary thesis evidence unless the research question explicitly
expands to consumer-versus-datacenter deployment.

## Remaining gates

Implementation can begin with deterministic fixtures, but final scientific execution still waits
for:

- completed and transferred K-study and held-out manifests, 36 artifacts, 36 evaluations, and
  their corpora;
- one full-coverage/runtime and variance preflight, followed by frozen warmup, origin, and repetition
  settings;
- a supervised `powermetrics` text/plist schema and signal preflight;
- frozen electricity and dated native-token price inputs; and
- separate authorization for Slice 4 after final-artifact MPS parity and no-fallback validation.

The preflights are setup checks. Their raw values do not need manuscript tables. The final
methodology must state the setup they froze, and any departure from full coverage or 60-second
phases must be disclosed.
