# Issue 23 shared Min-Block-Fee task prototype

Status: final disposable planning evidence for Edo's Issue-23 contract approved on 2026-07-14. It is
not production implementation. It changes no production code, configuration, tests, corpus, data,
storage, or workflow state.

Question: can one architecture-independent Min-Block-Fee task interface cover exact targets, loss,
decode, predictive scoring, evaluator input, and scheduling semantics while three closed concrete
architectures return the same typed two-head output through one direct constructor match? Can this
replace the current prediction/model registries and protocols without erasing real model variation?

Run the lightweight terminal prototype:

```bash
uv run python docs/research/issue-23/prototype.py
```

Run every bounded probe as JSON:

```bash
uv run python docs/research/issue-23/prototype.py --all
```

The predeclared budget is one local CPU run under five minutes using four synthetic origins,
`C=4`, `K=3`, and three ordered Ethereum-like features. Each of LSTM, Transformer, and
Transformer-LSTM receives one full batch of three samples and one tail batch of one sample. The
cheapest discriminating observation is whether all three return the same finite
`MinBlockFeeOutput` shapes and pass unchanged through common loss, decode, and scorer behavior.
Stop after structural compatibility and full/tail parity. Compare no quality, metric value, ranking,
winner, default, experiment count, or scientific outcome.

## Observations

- The public task interface is architecture-independent: `MinBlockFeeOutput`, the exact target state,
  two approved CE states, `min_block_fee_loss`, `decode_action`, `PredictiveScorer`, the ordinary
  evaluator mapping, and decoded-`k` scheduling semantics. No model class is the public task seam.
- The closed `ModelDefinition` union has exactly `family="lstm" | "transformer" |
  "transformer_lstm"`. One exhaustive direct constructor match builds three concrete `nn.Module`
  classes. There is no registry, plugin, adapter, abstract family base, generic head map, or dormant
  extension point.
- Every architecture consumed fixed unmasked `[B,C,F] float32` tensors. Its full batch emitted
  `action_logits [3,K]` and `minimum_fee_z [3]`; its tail emitted `[1,K]` and `[1]`. Whole-batch
  output equaled concatenated full/tail output in evaluation mode. Common loss was finite, common
  decode returned `[3]`/`[1]`, and one common scorer consumed all four samples.
- Exact raw-int64 minima over `[B,K]` produced deterministic earliest labels, including an equal-fee
  tie, before natural-log conversion. One float64 population target state produced the scalar
  standardized target. No clipping, epsilon, native inverse, or future value entered model input.
- Unweighted CE used no weight tensor. Corrected inverse frequency derived
  `w_k=N_train/(K*n_k)` from persisted support. Both paths used unreduced CE and Smooth L1,
  divided their unit-sum total by sample count, and contained no regression coefficient.
- Plain `argmax` decoded `k in 0...K-1`; its native first-index tie behavior matched the earliest
  action rule. No decoded-result buffer, id, context, mask, or adapter was needed.
- One concrete scorer consumed the typed output plus `label`/`target`, accumulated loss and log-error
  sums in float64, and used fresh direct `MulticlassF1Score` and `MulticlassStatScores` instances.
  One batch and a `3+1` partition returned identical complete results.
- The evaluator-facing batch stayed the approved mapping: `inputs [B,C,F]`, `label [B]`,
  `target [B]`, `base_fees [B,K]`, and `origin_block [B]`. Predictive scoring reads the shared task
  output, label, and target; separate economic calculation reads decoded actions, raw fees, and
  origins during the same traversal.
- Serving retains both `[1,K]` logits and `[1] minimum_fee_z` in model output. Scheduling consumes
  only scalar decoded `k`, then temporal ownership derives `b=h+1+k`. The auxiliary prediction stays
  in loss, artifact facts, validation, evaluation, and reporting. It is neither an authoritative fee
  quote nor a second action. This ticket neither requires nor forbids a later clearly labelled
  diagnostic display owned by the serving schema.

The prototype answered the structural question. The shared task module gives locality across all
callers; the three-branch model constructor expresses real closed variation without generic family
machinery. The final approved Issue-49 Scope-A amendment owns scientific experiment topology and
exact counts. Issue 23 infers none.

The corrected complete contract is in [decision-contract.md](decision-contract.md). Its dependent
consumer/seam audit is in [dependent-completeness-audit.md](dependent-completeness-audit.md). The
primary-source model-family evidence is in
[paper-model-family-alignment.md](paper-model-family-alignment.md); the PDF hash matched and visual/
semantic checks of physical pages 2, 4-5, and 7-10 confirmed the three concrete architectures,
shared two-head prediction, scheduling role, and paper comparisons. Edo's final approved
[Issue-49 Scope-A amendment](https://github.com/edoski/spice/issues/49#issuecomment-4966379354) fixes
those three families behind the shared seam; it retains ownership of later scientific experiment
topology and exact counts.
