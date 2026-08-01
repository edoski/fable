# Inference benchmark implementation-review ledger

Status: implementation active; Slices 1 and 2 green, scientific campaigns externally gated

Authority: [GitHub issue #148](https://github.com/edoski/fable/issues/148). Primary-source and
repository evidence: [inference-benchmark-slice-research.md](inference-benchmark-slice-research.md).

## Pre-run state

- Checkout: `/Users/edo/dev/python/fable`
- Branch: `main`
- Immutable planning baseline: `5364af1dbfb01aa52397adad9d9bca7f29855409`
- Last alignment audit: `64f8d7cffab87a47744f565495117d9bce74747a`
- Intervening committed work: stage-overlap experiment authoring and four-process Slurm packing.
  The audit found no changes to K-study or held-out manifest closure, artifact loading, historical
  dataset geometry, action decoding, or rolling reduction. Slice 1 must still record a fresh
  immutable baseline when implementation starts.
- Alignment verification at that head: 37 targeted experiment, execution, artifact, temporal, and
  rolling tests passed; scoped Ruff and repository Pyright passed. The host still reports Apple M2
  Max, PyTorch 2.7.1, 8 intra-op and 12 inter-op threads, MPS built/available, and the required
  `powermetrics` samplers. Final campaign metadata must be read again at execution time.
- Worktrees: one pre-existing main worktree; no run-owned branch or worktree
- Protected pre-existing changes:
  - untracked `docs/experiments/`, including `feature_ablation.md`
  - untracked `docs/research/macos-inference-energy.md`
- Run-owned planning files:
  - `docs/research/inference-benchmark-implrevloop.md`
  - `docs/research/inference-benchmark-slice-research.md`
- External mutations authorized: overwrite issue #148 only
- Checkout policy: work directly on `main`; create no branch or worktree; one writer at a time
- Authorized local mutations: scoped implementation commits and orchestration-ledger commits
- Not authorized: touching protected untracked work, pushing, opening a pull request, RPC calls,
  mobile changes, remote scheduler mutations, or reporting scientific results from fixtures

## Fixed study contract

### Execution location

- All benchmark code, preflights, CPU measurements, energy measurements, deterministic reduction,
  and optional MPS measurements run on the reference MacBook Pro M2 Max.
- The remote CUDA Slurm cluster remains responsible only for canonical training and held-out
  evaluation. Completed K-study and held-out manifests, artifacts, evaluations, and corpora cross
  the existing file-transfer boundary before local measurement.
- No CUDA comparison is part of issue #148. The configured remote is a heterogeneous CUDA
  environment with no FABLE-owned comparable energy contract; it would be a separately named
  datacenter deployment study.

### Coverage and model boundary

- The final manifests are expected to resolve the thesis roster of nine architecture-chain cells
  and 36 native PyTorch checkpoints. The benchmark does not hard-code chain or family names; it
  groups canonical manifest cells and derives each artifact horizon from its association.
- The fixed rolling horizon sequence is policy, not model metadata. Define it once as
  `ROLLING_HORIZONS` in the existing rolling evaluator and reuse that owner in held-out reporting
  and benchmarking. Standalone workloads and the descending cascade derive from this one sequence.
- Use every valid origin from each sealed testing window at true batch size one. No automatic
  sampling cutoff exists. A complete timing pilot reports projected duration and output size; any
  sampling fallback is a protocol amendment frozen before the main campaign.
- Load only the four models for the current architecture-chain cell. Keep them resident and warm
  while measuring that cell. Loading all 36 simultaneously is neither required nor desired.
- Exclude artifact loading, Corpus I/O, feature construction, dataset preparation, RPC,
  propagation, and transaction submission. Include the forward pass and `decode_action`.
- Prepare each full `HistoricalDataset` once with the artifact's training-fitted state. Obtain the
  batch-one view before the per-call clock; do not duplicate every overlapping context window into
  a second tensor bank.

### Validation ownership and leanness

- Trust canonical publishers and loaders. `ExperimentManifest`, strict `EvaluateRequest` parsing,
  `load_artifact`, `prepare_historical_window`, and `decode_action` own their existing invariants.
  The benchmark must not restate their schema, null, range, Study, Method, Corpus, feature, model
  state, device, or output-finiteness checks.
- Keep only benchmark-boundary checks: every rolling cell selected from the completed manifests is
  present in both with every canonical rolling horizon, each selected evaluation names the selected
  K-study artifact, requested batch-one origins exist for the workload, and an existing output
  protocol matches the invocation before resume.
- Do not load or validate `observations.parquet` in the latency experiment. Canonical evaluation
  publication already owns those predictions, while this experiment measures model compute.
- Host hardware, macOS, AC/power mode, and ambient run conditions are controlled manually and
  stated once in the thesis methodology. They do not require a host-metadata subsystem or runtime
  probes in experiment code.
- Record artifact UUIDs once in the protocol. Canonical UUID-addressed artifacts are immutable;
  duplicate SHA-256 fields and repeated per-row provenance are unnecessary.

### Two distinct rolling quantities

- The timing and energy stress workload runs `K=5 -> 4 -> 3 -> 2` at one unchanged origin. It is
  the maximum four-forward model-compute burst within one block opportunity. Its decoded actions
  do not alter the already selected inputs.
- Economic reduction follows the canonical fixed-deadline policy. It starts from the same K5 root
  but advances the effective origin by one only after a terminal action `k=K-1`. These later
  predictions may belong to later block opportunities and must not be presented as one physical
  same-block latency event.

### CPU timing and approved metrics

- Use `time.perf_counter_ns()` around every individual standalone forward and every complete
  four-forward stress cascade. Use separate passes; never insert stage clocks inside a cascade.
- Do not use `torch.utils.benchmark.Timer` for reported rows. It measures block averages, performs
  hidden warmups, defaults to one intra-op thread, and cannot produce the required individual-call
  tail distribution. The current environment also lacks its eager `setuptools` dependency.
- Run `model.eval()` under `torch.inference_mode()`. Use a fixed excluded warmup chosen in preflight;
  do not warm until results appear stable.
- Keep PyTorch's ordinary untuned CPU configuration. Record and assert unchanged intra-op and
  inter-op thread counts. Record Automatic power mode and require Low Power Mode disabled.
- Main campaign minimum: ten complete chronological sweeps. Report mean latency with a two-sided
  95% Student-t interval across sweep means. Report median, p95, p99, and observed maximum across
  raw individual calls. The maximum is descriptive, not theoretical.
- Feasibility per cell: positive block-interval median and p01, cascade p99, their margin, fraction
  of positive intervals shorter than cascade p99, and the count/fraction of same-second timestamp
  ties. Negative intervals are invalid; zero ties are unresolved by integer-second timestamps and
  are not counted as proven deadline failures.
- Freeze the empirical quantile method as NumPy `method="inverted_cdf"` so p01 and tail values are
  observed order statistics rather than interpolated synthetic values.

### Energy and approved metrics

- Use only Apple's `/usr/bin/powermetrics`, sampled near one second with CPU, GPU, ANE, and thermal
  samplers. Preserve the raw NUL-separated plist trace.
- Name the result **`powermetrics`-estimated incremental CPU+GPU+ANE energy on the reference Mac**.
  It is not total SoC, whole-machine, battery, or wall-plug energy.
- Run twenty 60-second loaded-idle then 60-second active pairs per architecture-chain cell, with a
  frozen recovery delay between pairs. Do not alternate active first: residual heat could
  contaminate its own idle baseline.
- The active phase continuously runs batch-one same-origin stress cascades over the frozen origin
  sequence. It includes in-memory view selection, four forwards and decodes, and minimal loop/count
  bookkeeping; it excludes I/O and feature construction.
- Parse only fields and units proven by the supervised text/plist preflight. Treat the plist's
  whole-second UTC timestamp conservatively at phase boundaries, integrate actual sample durations,
  and calculate pair energy from time-weighted power and measured active throughput:

  `e_r = ((P_active_bar - P_idle_bar) / 1000) / (N_r / T_active_r)` joules per cascade.

- Discard samples that may cross phase boundaries and retain negative pair estimates. Report mean
  joules per cascade with a 95% Student-t interval across pairs, `joules per cascade / 4` as an
  explicitly averaged per-forward value, phase power, accepted duration, cascade count,
  throughput, and thermal validity.
- No per-horizon energy campaign, CPU/GPU cycles, Energy Impact, battery-discharge inference,
  third-party meter, or undocumented plist energy counter.

### Statistics, preflight, and economics

- Independent units are sweep means for latency and paired energy estimates for energy. Origins,
  cascades within a 60-second loop, and one-second power samples are not independent replicates.
- Use `mean +/- t_(0.975,n-1) * s/sqrt(n)` through `scipy.stats.t.ppf` in an experiment-only
  dependency group. Do not hand-code a t table or use PyTorch's display heuristics.
- The setup-only preflight projects the smallest `n` at or above ten sweeps or twenty pairs that
  meets a relative CI half-width of 5% for latency or 10% for energy. Freeze the result before the
  main run. If energy is too near zero for relative precision, lengthen the phase or define an
  absolute criterion before main collection; do not divide by a near-zero pilot mean.
- Pilot data stays outside the scientific report. The thesis states only the frozen final method,
  unless the pilot forces a material departure such as sampling or longer energy phases.
- Primary economics uses the raw per-origin rolling base-fee difference in atomic native units per
  gas. P50-inclusive savings remains a secondary retrospective representative-cost proxy. Never
  multiply an average percentage saving by an average fee.
- Report energy-cost proxy per cascade and per million cascades. Primary break-even gas is
  `cost_EUR * atomic_units_per_token / (mean_raw_fee_difference * token_EUR)` when the mean raw
  difference is positive; otherwise report no positive expected break-even.
- Prices are strict static inputs: one electricity price in EUR/kWh and one dated native-token EUR
  price, symbol, atomic-unit scale, source, and rounding rule per chain. No live price lookup or
  arbitrary transaction-size example.

### Scientific claim

The strongest permitted conclusion is model-compute-only:

> On the declared MacBook Pro M2 Max, runtime, and final artifacts, FABLE's four-forward
> model-compute stress cascade fits within the positive inter-block intervals observed during the
> sealed testing period under the declared p99-versus-p01 comparison.

Same-second timestamp ties remain an explicit unresolved fraction. Live deployment also incurs
data acquisition, feature construction, propagation, and transaction-submission latency, which
depend on infrastructure and are outside this experiment.

## Shared implementation and output contract

One experiment-private operator entry point under `experiments/` owns resolution, measurement, and
reduction. Private helper modules are allowed when they keep power parsing and statistical
reduction deep; no installed `fable` CLI, public evaluator API, compatibility layer, or benchmark
database is added. Focused tests live under `tests/experiments/`.

The operator receives explicit K-study and held-out experiment UUIDs and an explicit output
directory. It never infers a latest object. Resolution must:

1. load both completed manifests;
2. derive rolling cells from their canonical labels and the evaluator-owned rolling horizons;
3. strictly parse every canonical `evaluation.json`;
4. require each selected evaluation to name the K-study artifact for that cell; and
5. load artifacts and prepare datasets through their existing canonical loaders.

The output is immutable and resumable at independent measurement-unit boundaries:

```text
<output>/
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

`protocol.json` records only experiment IDs, the derived rolling horizons, the selected
cell-to-artifact/evaluation UUID roster, warmup count, and sweep count. It has no `pilot`/`main`
branch; setup and final campaigns use separate output directories and Slice 3 accepts only the
complete ten-sweep final directory. Each unit writes through a temporary sibling and atomic rename.
A matching protocol permits existing units to be skipped on resume; outputs are never overwritten.

## Slice 1: CPU latency experiment

Status: green

- Baseline: `64f8d7cffab87a47744f565495117d9bce74747a`
- Branch/worktree: direct `main`; sole pre-existing worktree
- Pre-slice status: protected untracked `docs/experiments/` and
  `docs/research/macos-inference-energy.md`; run-owned untracked ledger and research note
- Implementer: `slice1_cpu_latency`
- Rejected head: `3ad8331f039853d886a4f9fa00794f08761c47ab`
- Review at that head was interrupted when the user rejected its 1,053-line operator and 718-line
  test suite as overvalidated and too ceremonial. The correction below is authoritative.
- Correction head: `be14e2d221aff71c9a51e184eafb88f73026d01b`; operator reduced to 309 lines
  and focused tests to 380 lines. Independent review rejected it with three remaining findings:
  resolution loaded every artifact only to discard and reload its model; `_Resolved` duplicated
  mapping/request data; and derived groups did not enforce the complete nine-cell/36-artifact
  thesis roster. The protocol and atomic publisher were accepted as distinct necessary owners;
  `bundle.py` does not own incremental sweep publication.
- Final head: `495160da6462fef82b1db76453f0ce9bdd5f9e19`; resolution-time model loading
  and `_Resolved` were removed, and the derived roster now requires exactly nine groups and 36
  rolling labels without hard-coded chain or family names. Final operator: 303 lines; focused
  tests: 385 lines.
- Review: `slice1_cpu_latency_review` returned `GREEN LIGHT` with zero Standards findings and zero
  Spec findings after two correction rounds.
- Implementer verification: 14 focused/relevant tests, 24 proportional tests, 98-test full suite,
  scoped Ruff and format, repository Pyright, Vulture with no findings, CLI help, and diff check.
- Orchestrator integration: 14 benchmark/rolling/K-study tests passed; scoped Ruff and fixed-range
  diff check passed. No scientific pilot or final measurement was run.

### Scope and algorithm

Implement the shared resolution/protocol layer and CPU workload runner.

For each cell and each sweep:

1. load the four CPU models and prepare the four exact held-out datasets through canonical APIs;
2. perform the fixed excluded warmup;
3. time four standalone chronological passes, one for each horizon's complete testing window;
4. time one chronological stress-cascade pass over every K5 root, obtaining all four same-origin
   input views before the outer clock; and
5. atomically publish the cell-sweep rows.

Use a deterministic rotating pass and cell order across sweeps so no workload is always first or
always hottest. Raw rows retain only cell, sweep, pass order, workload, origin block, and elapsed
nanoseconds. Model outputs are decoded inside the clock and discarded because canonical
observations already own predictions and later economic reduction reads them directly.

Chain and family names, base cells, horizon mappings, standalone workloads, and the cascade are
derived from the manifest groups, artifact associations, and evaluator-owned `ROLLING_HORIZONS`.
The runner accepts only warmup count and sweep count; it has no pilot/main mode or workload-rule
string.

### Non-goals

No energy collection, economics, MPS, CUDA, RPC, mobile artifacts, DataLoader batch timing,
model-load timing, dynamic rolling-route timing, observation parity pass/file, artifact hashing,
host probing, per-stage clocks inside a cascade, or generic evaluator behavior change beyond
centralizing its existing fixed horizon sequence as one shared constant.

### Expected outcome

One local command can resume and complete ten auditable full-period CPU sweeps over all nine cells,
producing individual-call horizon and four-forward stress-cascade latency records without changing
FABLE's canonical evaluation pipeline.

### Checks

- Manifest-derived rolling-cell grouping and evaluation-to-artifact join.
- View-backed chronological batch-one traversal without duplicated context banks.
- Four-model residency, fixed warmup exclusion, standalone and outer cascade clock boundaries,
  same-origin stress inputs, and decode inclusion.
- Compact raw schema, atomic publication, protocol match, interruption, and resume behavior.
- `uv run pytest tests/experiments/test_inference_benchmark.py`
- `uv run ruff check experiments tests/experiments`
- `uv run pyright`
- Proportional full suite plus `uv run vulture`, with every finding manually validated.

### Dependencies and gates

Fixture implementation can begin. Scientific execution waits for transferred final objects. One
complete pilot freezes warmup and confirms full-period duration; no automatic sampling decision is
permitted.

## Slice 2: `powermetrics` energy experiment

Status: green; scientific signal gate pending

- Baseline: `d1631d8e33aa06b9e9e664761e682aae11a3010c`
- Branch/worktree: direct `main`; sole pre-existing worktree
- Pre-slice status: only protected untracked `docs/experiments/` and
  `docs/research/macos-inference-energy.md`
- Implementer: `slice2_powermetrics`
- Initial head: `9a9c06816984ceb5126187c934497ecd045a39ed`
- First independent review: zero Standards findings; one Spec finding. A single initial sudo
  authorization would expire before later cell collectors because each cell lasts at least forty
  minutes while the implementation launched each collector with `sudo -n`.
- Correction head: `f313b398db2a06e129e9f0f91acd20e583517250`. Each unfinished cell now
  completes inherited-terminal `sudo -v` immediately before launching its collector; resumed cells
  return first. The model runner stays unprivileged and only `powermetrics` is launched elevated.
- Review: `slice2_powermetrics_review` returned `GREEN LIGHT` with zero Standards findings and zero
  Spec findings after one correction round.
- Implementer verification: 15 focused tests, 37 proportional tests, 105-test full suite, Ruff and
  format, repository Pyright, Vulture with no findings, CLI help, diff check, and successful parsing
  of all three supplied raw plist records.
- Reviewer verification: focused tests, full suite, Ruff, Pyright, Vulture, and fixed-range diff
  checks across the initial implementation and correction.
- Orchestrator integration: all 15 benchmark tests, scoped Ruff, repository Pyright, Vulture, and
  fixed-range diff check passed. No sudo collector, final-model run, signal preflight, or scientific
  measurement was run.

### Scope and algorithm

Add a compact plist reader, supervised collector, phase logger, continuous cascade loop, and pair
reducer to the existing experiment-private operator.

The benchmark runner remains unprivileged. Administrator authorization covers only
`/usr/bin/powermetrics`. One collector writes to a user-opened raw file for each cell, runs across
all its phases, and stops with SIGTERM. It requests `cpu_power,gpu_power,ane_power,thermal`, a 1000
ms interval, plist output, and no averaged display rows. Preserve the raw bytes and a compact phase
record; do not build help, version, command, host-metadata, or rejection-ledger subsystems.

Each cell runs twenty 60-second loaded-idle then 60-second active pairs, separated by the frozen
recovery delay. Phase records use wall-clock nanoseconds for joining samples and monotonic elapsed
time for active throughput. The active loop rotates through the resolved origins and counts only
fully completed same-origin four-model cascades.

Split the NUL-separated trace and read `timestamp`, `elapsed_ns`, `thermal_pressure`, and the four
`processor` power fields established by the preflight. The timestamp is a whole-second UTC sample
end label, so omit a boundary sample unless its entire possible interval fits inside the phase.
Time-weight accepted milliwatt values and divide the active-minus-idle watts by measured cascades
per second. Keep negative estimates. A pair containing a non-nominal accepted sample remains in the
output with `thermal_valid=false`; the scientific campaign reruns that cell rather than silently
filtering it.

`pairs.parquet` contains pair index, accepted sample counts and durations, idle/active CPU, GPU,
ANE, and combined mean power, active duration, completed cascade count and throughput, thermal
validity, and joules per cascade. `phases.json` contains only the phase boundaries and frozen energy
settings. Existing matching cell output is resumable; publication remains atomic.

### Non-goals

No separate K energy, CPU/GPU cycles, process Energy Impact, battery or charging telemetry,
AlDente discussion, external meter, wall-energy claim, undocumented counter units, root model
runner, or energy inference from TDP/utilization.

### Expected outcome

A supervised Mac run produces twenty auditable paired incremental CPU+GPU+ANE energy estimates for
every architecture-chain stress cascade, with raw Apple traces and enough phase evidence to
recompute every joule value.

### Checks

- A sanitized XML plist sample from the real capture establishes the parsed shape without retaining
  machine or boot metadata; tests compose NUL framing from that fixture.
- Focused tests cover parsing, conservative phase membership, a hand-calculated time-weighted pair,
  negative estimates, thermal invalidation, completed-cascade counting, collector failure, atomic
  publication, and resume.
- No test invokes `sudo`.
- Slice 1 checks plus focused energy tests and proportional full verification.

### Dependencies and gates

Schema gate passed on 2026-08-01 using `/tmp/fable-powermetrics.DBpLW8`: three NUL-separated plist
records exposed `timestamp`, nanosecond `elapsed_ns`, nominal thermal state, and CPU, GPU, ANE, and
combined processor power fields; combined power equalled the three rails within printed rounding.
The paired text capture labelled the powers in milliwatts. Raw host and boot fields will not enter
the repository fixture.

Implementation may proceed. Scientific collection remains blocked until final artifacts are
transferred and a short active-idle run confirms that 60 seconds resolves the signal. If not,
freeze a longer phase before main collection.

## Slice 3: statistical, deadline, and economic reduction

Status: pending

### Scope and algorithm

Add an experiment-only SciPy dependency for Student-t critical values and reduce only a complete
main campaign.

Latency reduction groups raw calls by cell/workload/sweep, calculates each sweep mean, then uses
the ten means for the mean and 95% CI. Median, p95, p99, and maximum come from individual raw calls
and remain descriptive. Energy reduction uses the twenty valid pair-level joule estimates.

For each chain, derive one block-interval series from K5 root `h` to `h+1`, validate exact block
numbers, reject negative differences, and separate zero ties from positive values. Calculate
positive interval median/p01 and, per architecture, margin and positive-interval shorter fraction.

Reconstruct each dynamic-origin rolling route from the four canonical observation files. Because
the public reducer exposes aggregates only, the experiment-private reconstruction must reproduce
all three `reduce_rolling` aggregates exactly before raw selections are trusted. Derive mean raw
base-fee differences and the secondary raw P50-inclusive differences directly from per-origin
integers.

Read one strict static assumptions file. Convert mean energy and its CI to the CPU+GPU+ANE
energy-cost proxy. Calculate break-even gas from mean raw fee difference, native token atomic-unit
scale, and dated price; explicitly handle zero/negative savings. Publish one `report.json`, then
generate TSV/LaTeX tables only from that report.

### Non-goals

No live price/RPC lookup, universal block-time constants, inference from zero-second ties, ratio of
fee sums, average-percentage multiplication, arbitrary gas-use examples, inclusion guarantee,
generic persisted evaluator metric, or CI over origins/power samples.

### Expected outcome

The final machine-readable report and derived tables answer typical and tail latency, conservative
model-compute feasibility, incremental energy/cost, and expected break-even gas for every declared
cell while preserving the limits of the timestamp and Apple power estimators.

### Checks

- Hand-derived Student-t interval and repetition-projection fixtures.
- Frozen empirical quantiles; positive, tied, and negative interval cases; millisecond conversion.
- Dynamic rolling terminal/nonterminal routes and exact aggregate agreement with `reduce_rolling`.
- Joule/kWh/EUR, million-cascade, atomic-token, positive/zero/negative saving, and break-even cases.
- Strict assumptions/provenance and incomplete/mixed/pilot campaign rejection.
- Recompute every table and compare it with `report.json`.
- Slice 1 and 2 checks plus proportional full verification.

### Dependencies and gates

Requires green Slices 1 and 2, completed measurements, and frozen electricity/token assumption
inputs. Pilot output never enters the reducer.

## Slice 4: optional MPS comparison

Status: deferred; separately authorized only after CPU completion

### Scope and algorithm

Run a fresh-process validation before any MPS measurement with
`PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_LOG_PROFILE_INFO=4` set before importing PyTorch.
Require MPS built and available, all final family/horizon paths successful, models and inputs on
MPS, exact action parity, frozen numeric parity, and no CPU-fallback log. PyTorch 2.7.1 contains
unconditional fallback registrations, so a successful output tensor alone is insufficient.

Disable fallback logging for measurement. Repeat Slice 1 as a distinct MPS configuration. Surround
every measured workload with `torch.mps.synchronize()` before the start clock and after the final
decode because MPS execution is asynchronous. Never pool CPU and MPS rows. If separately
authorized, repeat Slice 2 for MPS using the same combined CPU+GPU+ANE primary estimate and
individual rails as diagnostics.

### Non-goals

No CUDA cluster comparison, Core ML, ExecuTorch, backend rewrite, GPU cycles, silent CPU fallback,
mixed CPU/MPS statistics, or weakening of CPU completion.

### Expected outcome

If all final artifacts execute natively and reproducibly on MPS, the thesis gains a separately
labelled CPU-versus-MPS model-compute comparison. Any fallback, unsupported operation, or material
parity failure closes the optional slice without affecting the required CPU study.

### Checks and gates

- Fresh-process environment ordering, availability, device residency, stderr fallback detection,
  action/numeric parity, and explicit synchronization boundaries.
- Same raw/provenance/statistical checks as CPU, in a separate configuration.
- Separate authorization after CPU results; MPS energy requires another explicit authorization.

## Execution protocol

- Work directly on `main`. Create no branch or worktree, preserve the dirty checkout, never touch
  protected untracked paths, and allow only one writer at a time.
- Before every slice, record immutable baseline, branch, worktree, and status below.
- Assign each slice to a fresh implementer using the `implement` skill. The implementer commits only
  its scoped change and does not edit this ledger.
- Pin the implementation head, then assign a distinct read-only reviewer using `code-review` over
  the fixed three-dot diff. Standards and Spec must both have zero actionable findings.
- Rejected findings return to the same implementer; focused correction commits return to the same
  reviewer. Advance only after green.
- Do not push, open a pull request, run scientific measurements from fixtures, bypass a gate, or
  start optional MPS without authorization.

## Run records

| Slice | Baseline | Head | Implementer | Reviewer | Corrections | Result |
| --- | --- | --- | --- | --- | ---: | --- |
| 1 | `64f8d7cf` | `495160da` | `slice1_cpu_latency` | `slice1_cpu_latency_review` | 2 | GREEN LIGHT |
| 2 | `d1631d8e` | `f313b398` | `slice2_powermetrics` | `slice2_powermetrics_review` | 1 | GREEN LIGHT |
| 3 | pending | pending | pending | pending | 0 | gated |
| 4 | pending | pending | pending | pending | 0 | deferred |

## Run-owned branches and worktrees

None. The user requires direct work on `main`; this section must remain empty.
