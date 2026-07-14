# Issue 23 dependent completeness audit

Status: final dependent completeness audit for Edo's approved Issue-23 contract. This audit used
repository code, closed issue contracts, the final approved
[Issue-49 Scope-A amendment](https://github.com/edoski/spice/issues/49#issuecomment-4966379354), the
[paper/model-family alignment](paper-model-family-alignment.md), visual/semantic PDF inspection, and
synthetic tensors only. It changed no production code, configuration, tests, data, storage, issue,
or map state and ran no training, tuning, evaluation, HPO, economics, acquisition, or remote work.

## Verdict

The corrected contract is dependency-safe and leaves no consequential Issue-23 choice hidden. It
has two small interfaces:

- one architecture-independent task seam shared by all callers: typed two-head output, exact target,
  loss, decode, predictive scorer, evaluator input, and decoded-k scheduling semantics;
- one real closed model-construction seam: Issue 10's three tagged definitions, three concrete
  in-process models, and one direct exhaustive constructor match.

The task seam is deep because removing it spreads the same algebra across training, evaluation,
artifacts, and serving. Model variation is real, but bounded and fully expressed without a registry,
plugin, adapter, abstract family base, or generic head map.

Edo's final approved Issue-49 Scope A confirms that all three concrete architectures must survive
behind the shared seam. The Issue-49 owner retains scientific comparison topology, validation gate,
exact inventory/counts, execution order, and winner/HPO policy. Issue 23 infers none and cannot
select a family. Later evidence may change recipe/default data only; it cannot redesign this seam.

The remaining choices are outside this ticket: exact experiment lists/counts/gates, production
implementation slices, training host and batch placement, artifact serialization, HPO mechanics and
result, source-chain serving policy, serving lifecycle/schema details, real corpora, and scientific
outcomes.

## Primary-source cross-check

The local PDF SHA-256 matches the alignment note. Rendered physical pages 2, 4-5, and 7-10 show:

- LSTM, Transformer, and Transformer-LSTM are the three studied prediction techniques;
- temporal scheduling identifies the future min-block and estimates its associated base fee;
- all three use the same two task-specific prediction heads;
- Figures 5-7 compare all three, and the paper describes SPICE as model-agnostic while discussing
  family-specific behavior.

This supports a shared task plus three concrete models. It does not establish project tensor shapes,
argmax/tie behavior, target state, exact CE alternatives, unit-sum loss, scorer, artifact facts,
serving schema, experiment counts, or a family winner. Those facts come from project contracts.

## Consumer inventory

Current generic task/model knowledge crosses these consumers:

- config hydration and artifact manifests persist prediction, model, and representation selectors;
- model construction consumes output-head specs and returns a generic head dictionary;
- batch planning binds prepared target protocols, masks, sample positions, pinning, and transfer;
- direct and Lightning training call a compiled prediction callable bag for loss and accumulation;
- forward/scoring allocates and fills a generic decoded-result buffer;
- evaluator contracts narrow decoded ids to `DecodedOffsets` before replay/accounting;
- serving recompiles prediction/temporal/feature registries and unwraps the decoded buffer;
- tests and architecture docs assert those transition layers.

Issues 24 and 28 remove temporal compilers, masks, custom batching, and the generic representation
adapter. The generic prediction interface then has one fixed task implementation and is shallow.
The generic model-family interface still fronts three real architectures, but Issue 10 already gives
their smaller closed expression: one tagged union and one match. Consumer inspection supports
deleting both generic frameworks transitively while retaining every concrete model.

## Seam analysis

All task and model dependencies are in-process pure tensor operations and frozen records. They need
no port or adapter. Task callers learn one output, one target/loss/decode algebra, one concrete
scorer, one ordinary evaluator mapping, and one scheduling rule. They do not learn architecture
classes, family ids, head ids, or callable ordering.

Model construction separately learns one closed `ModelDefinition` union. Three explicit branches
are justified because three implementations differ materially. A registry/plugin interface adds no
leverage: every valid alternative is already known, schema-hydrated, and exhaustively matched.

Folding task behavior into each model is rejected because it would duplicate target/loss/decode and
scorer facts three times. Folding economic accounting into the task is rejected because it would
merge separately owned result sections and make serving depend on future outcomes. The corrected
placement maximizes locality without absorbing evaluation or serving lifecycle ownership.

## Closed-contract parity

- Models: three concrete fixed-context implementations return the same finite
  `MinBlockFeeOutput(action_logits [B,K], minimum_fee_z [B])`.
- Target: raw positive-int64 earliest within-K minimum/argmin, natural log, strict float64
  per-(chain,K) training z-score, float32 dataset target, no native inverse.
- Loss: sample-count CE plus unit-standard Smooth L1, unit-sum composition, unweighted or corrected
  inverse frequency only, complete-validation total-loss selection.
- Decode: direct first-index `argmax`; every exact-K action is valid; auxiliary output is validated
  but does not select or alter the action.
- Scorer: float64 additive totals, direct union-active TorchMetrics F1/stats, earliest accuracy,
  exact Smooth-L1 and natural-log MAE/MSE, atomic fail-closed section.
- Evaluator: one exhaustive traversal of the ordinary five-tensor mapping, with predictive and raw
  economic calculations kept separate.
- Scheduling: full two-head output survives inference; only decoded `k` drives scheduling. The
  auxiliary prediction remains in loss, artifacts, validation, evaluation, and reporting and is
  neither an authoritative fee quote nor a second action.
- Recipes: bounded ordered feature tuples, closed model definitions, and two CE values remain
  authored scientific facts rather than registries or generic task architecture.

## Synthetic stop condition

Each of the three concrete definitions was constructed through the sole direct match. Each model ran
one synthetic full batch and one tail batch, returned the same typed output shapes, produced finite
common loss, decoded through the common task, and updated the same concrete scorer. Whole-batch
output equaled concatenated full/tail output in evaluation mode. Separate hand-authored tensors
exercised target ties, both CE modes, exact loss composition, decode tie, one-batch versus `3+1`
scoring, evaluator mapping, and decoded-k scheduling.

These are structural compatibility observations only. They compare no metric value, quality,
ranking, family winner, default, experiment cell/count, or scientific outcome. More synthetic cases
would retest framework behavior or trespass into Issue-49/50 ownership, so the stop condition is
reached.
