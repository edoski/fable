# Issue 41 — native-only cutover contract

Status: complete planning contract explicitly approved by Edo on 2026-07-15.
It defines eligibility, recovery, and final manual custody. It performs no code,
configuration, test, dependency, data, storage, archive, deletion, acquisition,
training, evaluation, serving, worktree, or cutover mutation.

## Superseded question

The issue body predates the clean native object model. There is no conversion,
import, remapping, collision, migration, or referential-closure problem to solve.
Old Optuna trial identities, timestamps, database state, and `RUNNING` rows do not
enter the clean design. Old checkpoints are not converted. Historical manifests,
summaries, hashes, content identities, software revisions, environments, and
database facts establish no clean identity or association.

The single operator needs no migration worksheet, acceptance ledger, inventory,
root scanner, lock, rollback system, archive service, or cleanup framework. The
cutover boundary is a new root, not a record of how old objects map into new ones.

## Native clean eligibility

Each host starts with a fresh empty `STORAGE_ROOT`. Only the clean owners create
canonical objects there, at the Issue-34 paths:

```text
corpora/<corpus_id>/corpus.json
corpora/<corpus_id>/blocks.parquet
studies/<study_id>.json
artifacts/<artifact_id>.ckpt
evaluations/<evaluation_id>/evaluation.json
evaluations/<evaluation_id>/observations.parquet
```

A corpus is the native clean corpus pair. A Study is the native finalized Study
JSON. An artifact is Lightning's native weights-only best checkpoint. An
evaluation is the native request and observation pair. Their UUIDs identify new
instances; no legacy ID is reused or mapped.

Given an exact UUID, a normal consumer addresses only its exact canonical path and
uses the owning strict loader. Loader success proves only the current typed and
domain validity owned by that loader. It does not discover candidates, confer
scientific acceptance, or prove archive equality. Scientific acceptance belongs
to the owning execution and evidence gate before an object is selected or copied.

No current result count is part of cutover eligibility. Experiment shape and
result counts remain with their scientific owner and may change without changing
this contract.

## Legacy boundary

No legacy corpus root, Study or Optuna state, checkpoint, evaluation, benchmark
collection, export, figure, metric, manifest, summary, SQLite row, or source fact
is converted, repaired, promoted, relabelled, or used to establish clean closure.
Legacy persisted bytes do not feed the fresh roots. If one exact legacy object is
later claimed to be indispensable, work stops for a new path-specific Edo
decision. Passing a normal loader would be necessary after such approval but
would not itself authorize the exception or create a reusable import rule.

Current metrics, gates, tables, and thesis views derive only from native clean
evaluation observations through the approved reducer. Frozen old results keep
their original historical semantics and provenance limits. Recomputing or
renaming them cannot make them current evidence.

Issue 14 continues to own its raw SQLite/catalog originals and sanitized neutral
export. Issue 20 continues to own its exact private historical custody set and
already-frozen checksums. Those checksums remain static archive evidence only and
are never propagated into clean storage. The old serving SQLite follows its exact
Issue-33 disposition under the later owning implementation. This ticket moves,
copies, or deletes none of them.

## Fresh-root cutover

The cutover is one-way and host-local:

1. Complete the clean implementation, independent reviews, full integration, and
   synthetic target-hardware verification before authorizing real work.
2. Stop the old university jobs and writers. Run separately authorized clean
   acquisition, Study, Train, and Evaluate work only against a fresh university
   root. Never read the old university root as fallback input.
3. Let the scientific execution and evidence owners accept the exact native
   objects needed downstream.
4. For each exact approved canonical object needed on the Mac, Edo uses ordinary
   external `rsync` or `scp` to copy that exact path into a hidden sibling under
   the fresh Mac root. Ordinary `mv` or direct rename installs it at its canonical
   path. The normal consumer loader validates it when used.
5. After the later map gates permit final serving activation and every exact
   checkpoint is present, stop the old Mac service and start clean serving against
   the fresh Mac root.

There is no SPICE transfer command, typed transfer owner or wrapper, receipt,
inventory, equality check, registry, transfer state, global transaction, barrier,
marker, pointer swap, or rollback. The two hosts need not switch together.

A failure stops only the affected host or activation. Preserve both old and new
roots. Fix, rebuild, recopy, or reinstall through the clean owner path and retry.
An already-clean host stays clean. Never point runtime back to legacy storage.
Separate old university and Mac roots remain untouched unless Edo later approves
an exact path-specific archive move or deletion.

## Incomplete clean work

Normal consumers do not scan, detect, classify, filter, or branch on hidden
directories. An unrelated hidden sibling is naturally invisible. If the required
canonical path is absent, ordinary exact-path loading fails; there is no special
`in progress` state or hidden-work interpretation.

Only the owning workflow may directly address its own exact hidden path for its
already-approved resume or rebuild behavior. It may not rename partial work into
eligibility, repair legacy bytes, overwrite a canonical object, or create a second
authority. Manual inspection preserves ambiguous bytes. Deleting an abandoned
hidden path or invalid canonical path requires Edo to approve that exact path at
the time.

Add no root enumeration, hidden-directory service, status model, recovery
registry, custom error taxonomy, verbose guard, or dedicated test proving hidden
directories are ignored. Keep only an ordinary focused owner resume fixture where
the owner behavior is genuinely SPICE-owned.

## Manual quiescence and recovery

Before activating a host, Edo manually stops and inspects the relevant old
processes. This is a human precondition, not persisted proof. There is no lock,
lease, writer registry, process ledger, quiescence record, dual reader, restore
path, compatibility mode, or rollback authority.

Canonical-load failure stops normal use. Preserve the paths and fix forward. A
clean owner may resume or rebuild only through its approved native behavior. No
archive is mounted, restored, or converted to recover runtime service.

## Final repository and checkout custody

After accepted clean replacement evidence exists, the final cutover owner moves
each exact legacy generated-output path still under either host's repository root
to an Edo-approved external archive. Destructive deletion still requires Edo to
approve the exact path. The final repository has no loose training, tuning,
evaluation, benchmark, export, or figure output tree.

The earlier implementation and documentation owners remove obsolete source,
tests, configuration, empty directories, root files, and superseded or stray
Markdown before full integration. Deliberately retained ticket-scoped research
evidence stays in its proper research location. This bounded human review creates
no scanner, file inventory, archive command, cleanup service, migration layer,
absence test, CI gate, retention policy, or ongoing governance.

The same final manual custody boundary covers local and university checkouts and
their linked worktrees:

- keep one intentional primary checkout per host;
- near final completion, use ordinary operator Git commands to identify each
  exact extra linked worktree;
- remove an unneeded worktree only after manually confirming that it contains no
  unique uncommitted work;
- worktree removal preserves its branch; branch deletion is a separate action and
  is not authorized here.

At this planning observation, `/Users/edo/dev/python/spice-fast-ab` on
`codex/fast-ab-training` is clean and is a later local removal candidate. Any
extra university worktree is checked at final cleanup. These are operator facts,
not SPICE identity or state. Add no SPICE worktree scan, registry, cleanup command,
Git-state record, provenance requirement, automation, or test. This ticket removes
no worktree and deletes no branch.

## Required order and ownership

The complete order remains:

1. freeze the complete clean specification and implementation DAG;
2. implement, review, and perform the bounded pre-integration code/document tree
   cleanup;
3. integrate the full clean codebase and verify it with synthetic evidence;
4. verify the selected host and precision on target hardware with synthetic data;
5. separately authorize and create native real corpora and outcome evidence on the
   fresh university root;
6. accept result-selected recipes, objects, and claims through their owning gates;
7. finish normative documentation, disposable cutover rehearsal, and the
   sanitized neutral export;
8. activate the Mac fresh root and complete the bounded repository-output and
   extra-worktree custody review.

Issue 41 specifies this operator procedure. It owns no implementation, execution,
archive movement, deletion, or worktree removal.

## Completion authority

Edo's approval authorizes only this ticket-scoped research contract, exactly one
Resolution on Issue 41, closing only Issue 41, one count-agnostic context pointer
or approved fog correction on the open Wayfinder map, and fresh verification. It
authorizes no production, configuration, test, dependency, data, storage,
archive, deletion, job, acquisition, training, evaluation, serving, cutover,
sibling-issue, branch, worktree, or native-graph mutation.
