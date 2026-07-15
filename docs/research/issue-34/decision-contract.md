# Issue 34 — durable ML and evaluation contract

Edo approved this complete planning contract on 2026-07-15. It coordinates the
clean-break corpus, study, artifact, evaluation, command, storage-root, and
provenance boundaries. It changes no production code, configuration, tests,
dependencies, data, storage, acquisition, training, tuning, evaluation,
scheduler, serving process, mobile code, archive, or sibling issue.

## Canonical authority

The complete canonical layout is:

```text
corpora/<corpus_id>/corpus.json
corpora/<corpus_id>/blocks.parquet
studies/<study_id>.json
artifacts/<artifact_id>.ckpt
evaluations/<evaluation_id>/evaluation.json
evaluations/<evaluation_id>/observations.parquet
```

There is no canonical request root, artifact manifest, sidecar, inventory,
summary, benchmark record, index, cache, catalog, SQL store, content-derived
identity, compatibility version, migration reader, or discovery system. A
completed corpus, study, artifact, or evaluation owns its exact request once.
Operator input and hidden work are not second authorities.

UUIDs identify instances. They are not content hashes. Cheap derived facts are
recomputed instead of being persisted twice.

## Corpus

The complete durable definition and request are:

```text
CorpusDefinition = {
  chain_id,
  first_block,
  last_block,
}

CorpusRequest = {
  corpus_id,
  definition,
}
```

The range is inclusive. `corpus.json` is exactly:

```text
{
  request: CorpusRequest,
  finalized_anchor: {
    block_number,
    block_hash,
  },
}
```

Protocol-era names and boundaries are human scientific context in thesis and
research prose only. They are not application objects or fields. No regime
name, start, object, or identity appears in a request, corpus row, checkpoint,
evaluation, or serving state.

`corpus acquire` owns acquisition, resume, validation, finalization, and
publication as one operation. Hidden acquisition work may use resumable chunks
and may retain protocol `block_hash` and `parent_hash` long enough to prove
ordering, links, ancestry, and finality. Internal finalization validates those
facts, streams the approved hash-free raw block schema into one
`blocks.parquet`, writes `corpus.json`, and publishes the complete corpus. If
finality is not available, hidden incomplete work remains and a later identical
`corpus acquire` invocation continues it.

There is no public corpus-finalize command. Canonical storage has no chunk
naming, discovery, or ordering contract, payload inventory, or project digest.

A canonical reload validates the request and instance association, raw schema,
inclusive range, row count, order, domains, and finalized-anchor shape. The
canonical anchor `block_hash` survives because it is a protocol fact. Parent
hashes do not. Reload therefore cannot re-prove ancestry or finality after
publication. Edo explicitly accepts this residual risk; the loader makes no
stronger claim and adds no replacement proof machinery.

## Study

`studies/<study_id>.json` is exactly:

```text
{
  request: TuneRequest,
  trials: [RetainedResult, ...],
}
```

The result list is nonempty. `RetainedResult` contains only:

```text
method
validation_total_loss
earliest_best_epoch
completed_epochs
```

The loss is finite and nonnegative. Epoch fields are strict integers, not
booleans, and satisfy:

```text
1 <= earliest_best_epoch <= completed_epochs <= method.max_epochs
```

The complete `Method` must belong to the request's `MethodSpace`. Duplicate
Methods and duplicate result values are valid observations. Selection is
recomputed as the zero-based minimum of
`(validation_total_loss, current_list_index)`. The study stores no selected
index, winner, status, trial identity, invocation, sampler, pruning state,
count budget, runtime history, or derived summary.

Edo manually permits one live writer per study UUID and starts neither another
candidate nor finalization while a candidate may still be running or accepted.
Distinct study UUIDs may proceed concurrently. This is a human precondition,
not application state. Add no lock, proof, queue lookup, scheduler guard,
ledger, marker, or concurrency protocol. Hidden progress uses ordinary
temporary-file replacement. A lost acknowledgement is handled manually.

## Native Lightning artifact

`artifacts/<artifact_id>.ckpt` is Lightning 2.6.5's stock monitored
weights-only best checkpoint, renamed unchanged. The public load path delegates
to native strict `load_from_checkpoint` with CPU mapping. A broad `last.ckpt`
exists only in private fit work for continuation and never becomes the final
artifact.

The embedded SPICE domain facts are only:

- the exact `TrainRequest`;
- fitted ordered feature means and standard deviations;
- fitted target mean and standard deviation;
- only for corrected inverse-frequency classification loss, one positive
  length-`K` class-support tuple;
- only for a selected-study fit, the strict nonnegative
  `study_result_index` and its exact `Method`.

Training count and class weights derive from the support tuple. They are not
stored separately.

A baseline source carries its complete `TrainingDefinition`. A selected-study
source carries `corpus_id`, `study_id`, and complete downstream
`ExperimentSemantics`. Before construction, fitting, or publication, validate
the corpus association, selected list position, exact Method equality,
MethodSpace membership, and materialization of a `TrainingDefinition` from the
Method plus downstream semantics. This complete association is sufficient to
restore the selected artifact on Edo's Mac without loading the Study. Do not
store selected loss or epochs, the Study request, a duplicate effective
definition, or a winner record.

The exact request and fitted states reconstruct the concrete model, task,
`C`, `K`, ordered features, loss, and scalar-target view. Lightning owns the
checkpoint structure, wrapper metadata, serialization, best selection, and
strict state restoration. SPICE adds no custom `state_dict` ABI or parser,
resave/reload validator, tensor key/shape/dtype inventory, sidecar, artifact
database, checksum, byte length, device, scheduler job, source-revision, or
runtime-history record.

The final compatibility observation is native loading of the transferred
checkpoint on Edo's Mac. The Mac and university environments need no source
revision equality. Issue 33 alone owns the literal twelve-entry `Chain × K`
serving map; it creates no artifact discovery metadata.

## Explicit post-fit evaluation

Every explicit post-fit `EvaluateRequest` publishes one uniform two-file
directory for validation and testing, for every `K` and every model family:

```text
evaluations/<evaluation_id>/evaluation.json
evaluations/<evaluation_id>/observations.parquet
```

Per-epoch validation inside Train and Tune candidate fits is outside this
contract.

`evaluation.json` is exactly the five-field request:

```text
workflow
evaluation_id
artifact_id
corpus_id
window
```

It has no result wrapper, count, aggregate, metric, status, runtime,
provenance, profile, or summary section.

`observations.parquet` has one ordered, non-null row per eligible origin and
exactly these columns:

| Column | Type |
| --- | --- |
| `origin_block` | `Int64` |
| `origin_timestamp` | `Int64` |
| `selected_action_k` | `Int64` |
| `earliest_hindsight_action_k` | `Int64` |
| `classification_loss_contribution` | `Float64` |
| `predicted_hindsight_minimum_base_fee_z` | `Float32` |
| `previous_closed_parent_base_fee_per_gas` | `Int64` |
| `closed_parent_base_fee_per_gas` | `Int64` |
| `immediate_k0_base_fee_per_gas` | `Int64` |
| `selected_target_base_fee_per_gas` | `Int64` |
| `hindsight_minimum_base_fee_per_gas` | `Int64` |
| `selected_action_wait_seconds` | `Int64` |
| `full_horizon_elapsed_seconds` | `Int64` |

The Parquet row count is `N`; JSON does not duplicate it. Request, corpus, and
checkpoint supply chain, role and range, `C`, `K`, feature, loss, and fitted
state facts.

One concrete column-pruned Polars reduction derives all approved predictive
and economic summaries, validation gates, the final ordered testing TSV, and
requested thesis tables and views. No second reducer or aggregate schema
survives.

`selected_action_wait_seconds` is zero for `k=0`; otherwise it is
`timestamp(h+k) - timestamp(h)`. It is an offline descriptive action offset,
not an observed broadcast, receipt, inclusion, or latency fact.

Do not store logits, probabilities, `K`-wide arrays, tensors, calibration or
confidence fields, duplicate regression errors, `S/G/Q` columns, booleans,
totals, condition profiles, bins, quartiles, plots, or additional result
files. The external thesis workbench may derive scatter, hexbin, quartile, and
other figures from the observations. Those are optional external views, not
application fields or a plotting framework.

The baseline makes no probability-calibration, confidence, abstention,
coverage, epistemic-uncertainty, or bitwise-score-reproduction claim. Adaptive
decision research is different from calibration: it requires new causal model
calls after newly closed blocks and a separately approved action and estimand
contract. Initial logits would not supply that information.

Later uncertainty or adaptive-decision work does not block this baseline and
preserves no generic seam, score sidecar, dependency, or test now. A future
fresh issue must choose one exact method, selected artifacts and windows,
calibration and assessment roles, purpose-specific prediction evidence, and
any bounded L40 re-inference from the retained request, native checkpoint,
fitted state, and corpus. It owns any fitted calibrator, additional
checkpoints, stochastic inference rule, evidence schema, dependency, or
serving confidence/abstention field that the chosen method earns.

The approved observations intentionally cannot reconstruct a complete score
vector. Retained authority permits semantic re-inference under compatible
code; it does not promise bit-identical logits across later libraries or
hardware.

Publication uses the approved hidden sibling and direct rename. Ordinary
`rsync` to a hidden sibling followed by `mv` remains an external operator
transfer primitive.

## Command, root, and target boundary

One Typer executable exposes exactly four public leaves and two help-hidden
generated-job hooks:

```text
submit REQUEST.json --remote REMOTE.yaml
corpus acquire REQUEST.json --rpc-url URL
study run TUNE_REQUEST.json METHOD.json --remote REMOTE.yaml
study finalize STUDY_ID
remote workflow       # hidden
remote candidate      # hidden
```

There is no public follow command, public corpus-finalize command, `--root`,
`--storage-root`, `--commit`, `--target`, `--dependency`, `--detach`, `--gpu`,
second executable, or `python -m` launcher. Hidden hooks trust their generated
script caller. They add no `SLURM_JOB_ID` guard, authentication, provenance,
misuse check, or Slurm defensive state.

`submit` and `study run` read no local `STORAGE_ROOT`. `corpus acquire` and
`study finalize` require the current host's single `STORAGE_ROOT` environment
value. Generated remote jobs export the selected target file's storage root.
Mac serving reads its process's `STORAGE_ROOT`. Roots are never persisted,
inferred from the working directory, loaded through a project `.env` file,
compared between hosts, or duplicated in a flag and file.

`submit` accepts exactly a `TrainRequest` or `EvaluateRequest`, selects the
matching Train or Evaluate resource stanza, and sends exact JSON to hidden
`remote workflow`. Tune dispatch exists only through `study run`, which
selects Tune resources and sends a separate `TuneRequest` plus `Method` to
hidden `remote candidate`. Each Slurm job receives one payload. One
`sbatch --parsable` call prints the numeric job ID. Edo follows work outside
SPICE with ordinary `ssh`, `squeue`, `sacct`, and `tail -F`.

Each direct target YAML contains only:

- an OpenSSH alias;
- the absolute installed `spice` executable;
- remote storage root;
- remote log root;
- concrete Train, Tune, and Evaluate Slurm resources.

There is no target identifier, registry, profile inheritance, default,
fallback, repository or checkout path, Python or virtual-environment path,
duplicated accelerator label, separate GPU knob, or follow policy. A target
file may directly describe the university L40 or another accelerator's exact
scheduler resource. Official thesis Train, Tune, validation, and Evaluate use
the L40 with native TF32 only. Another target is not silently equivalent or
outcome-authorized.

## Provenance boundary

Application and thesis authority contain no software commit or revision
identity. Requests, records, checkpoints, experiment notes, target-hardware
evidence, this research contract, and thesis requirements contain no expected
revision, informational revision, clean-tree result, host-equality result,
synchronization state, or execution-source identity. Edo manually manages the
installed university code outside the documented SPICE contract.

SPICE computes no content or integrity digest, fingerprint, inventory, repeat
hash, byte-length inventory, or content-derived ID. This does not erase
transient blockchain block and parent hashes, the canonical finalized-anchor
block hash, or historical custody checksums already frozen by their evidence
owners.

## Implementation handoff

This ticket coordinates the contract. It does not implement it. Issues 38 and
44 must specify the clean replacement and create later code-review-sized
implementation slices. Their handoff includes:

- replace root selectors, `StorageSpec`, resolved snapshots, catalogs, and
  duplicated request storage with the direct paths above;
- replace acquisition assembly/publication with private hash-bearing stage
  rows, one internal finalizer, and one canonical `blocks.parquet`;
- delete Optuna, study locks, lifecycle, history, sampler, pruning, and count
  machinery;
- replace custom training-artifact persistence, checkpoint dictionaries,
  manifest/SQLite/codecs, and disabled native checkpointing with stock
  `ModelCheckpoint` plus the strict embedded request and fitted state;
- replace generic evaluation runs/summaries/contracts, replay and Poisson
  machinery, persisted totals/provenance, and evaluation SQL transactions with
  direct observation writing and one reducer;
- delete repository/virtual-environment target paths, remote revision and
  follow state, generated provenance exports, target registries/defaults, and
  obsolete command flags;
- delete the generic benchmark package, CLI, YAML, SQL, export, and revision
  plumbing while retaining only exact named request constructors and the
  all-or-nothing report caller;
- delete transaction, analytics, and broadcast-wait serving machinery while
  retaining the approved stateless inference path.

Create no generic software-revision, regime, root, provenance, score/UQ,
discovery, or analysis subsystem or ticket.

## Lean verification

Keep only focused positive tests of current SPICE-owned behavior:

- strict request and schema hydration;
- corpus resume, internal finalization, and single-file publication;
- retained-result constraints, selection, and materialization;
- native checkpoint association and Mac load;
- one hand-computable 13-column write/reload and column-pruned reduction;
- direct target YAML, generated command, dispatch, and numeric Slurm job ID.

Delete obsolete tests with obsolete code. Add no absence, deletion, transition,
or compatibility tests. Do not retest Lightning, Polars, Pydantic, `rsync`,
Slurm, or standard-library behavior. The uncertainty clarification adds no
baseline test. Later implementation owners run focused and full tests, Ruff,
Pyright, and manually review every Vulture finding before deletion.

## Accepted limitations

The approved lean boundary deliberately accepts:

- canonical corpus reload cannot re-prove stripped parent links or finality;
- a structurally loadable same-shape checkpoint substitution may remain
  undetected under native package trust;
- source deployment and host equality are manual facts outside SPICE;
- later score regeneration is semantic, not promised bit-for-bit identical;
- one operator manually avoids concurrent writers and resolves ambiguous
  acknowledgement.

These limits do not authorize defensive hashes, inventories, locks, source
identity, compatibility machinery, generic UQ state, or operational ledgers.
