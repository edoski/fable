# Issue 23 paper and model-family alignment

Status: final consumed primary-source audit for Edo's approved Issue-23 contract. This note changes
no production code, configuration, test, data, artifact, issue, or map state.

## Sources

- `ICDCS_2026.pdf`, 11 pages, SHA-256
  `2afa36d5c82cc2f8be854707fad91b86562d399896b9ee163decd75f470d4b5c`, at
  `/Users/edo/Documents/Obsidian/the-vault/university/Thesis/ICDCS_2026.pdf`.
- Final [Issue 49 architecture amendment](https://github.com/edoski/spice/issues/49#issuecomment-4966379354)
  and its immutable [amended decision contract](https://github.com/edoski/spice/blob/0720ed7f597ad25b325814e27dac56a54b7c3576/docs/research/issue-49-temporal-baseline/decision-contract.md).
- Historical [Issue 49 Resolution](https://github.com/edoski/spice/issues/49#issuecomment-4960871251)
  and its immutable [original decision contract](https://github.com/edoski/spice/blob/5647d5b3525ed93b73b34aa5900c867774fb1105/docs/research/issue-49-temporal-baseline/decision-contract.md),
  retained only to identify the LSTM-only clause superseded by the amendment.
- Original paper-training repository at commit
  [`bcf80b92877941e3b05a7dc5138560ffe41df27e`](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/commit/bcf80b92877941e3b05a7dc5138560ffe41df27e).

## Serving output

The paper's externally actionable temporal decision is the future execution block. It says SPICE
recommends the target chain and specific future block (PDF p. 4), then defines temporal forecasting as
identifying the `min-block` and scheduling the transaction there (PDF p. 5). The decoded offset `k`
is therefore sufficient for the serving action once the parent block is fixed.

The auxiliary fee prediction is still scientifically real. The same passage says SPICE estimates the
base fee associated with the predicted minimum (PDF p. 5). All three paper models have two heads:
min-block logits and a scalar fee prediction (PDF p. 8). The original implementation returns both
values for [Transformer](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/blob/bcf80b92877941e3b05a7dc5138560ffe41df27e/train_model_classific.py#L259-L296),
[LSTM](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/blob/bcf80b92877941e3b05a7dc5138560ffe41df27e/train_model_classific.py#L298-L360),
and [TransformerLSTM](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/blob/bcf80b92877941e3b05a7dc5138560ffe41df27e/train_model_classific.py#L413-L476).
Training and evaluation use the scalar in Smooth-L1 and total loss, while the action is obtained from
`argmax(logits)` ([lines 487-523](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/blob/bcf80b92877941e3b05a7dc5138560ffe41df27e/train_model_classific.py#L487-L523)).
The source repository contains no serving endpoint that exposes either value.

Verdict: scheduling from decoded `k` is aligned with the paper's operational decision. The auxiliary
prediction remains in model output, loss, artifact facts, validation, evaluation, and reporting. It is
neither an authoritative fee quote nor a second action and cannot override `k`. Whether a later
serving schema displays it as a clearly labelled non-actionable diagnostic remains separate owner
work. The contract must not imply that SPICE does not predict the associated fee or that the
auxiliary head can be deleted.

## Transformer and Transformer-LSTM

The paper explicitly evaluates LSTM, Transformer, and TransformerLSTM. It names all three in the
contribution/evaluation overview (PDF p. 2), defines all three architectures and their shared two-head
task (PDF pp. 7-8), and compares all three in the temporal and spatio-temporal results (PDF pp. 9-10,
Figs. 5-7). The original source also selects among those three concrete implementations directly
([lines 528-560](https://github.com/UniBO-PRISMLab/ICDCS-Model-Training/blob/bcf80b92877941e3b05a7dc5138560ffe41df27e/train_model_classific.py#L528-L560)).

The paper's Q3 asks whether short transaction delays reduce execution cost (PDF p. 3). PDF p. 10
calls SPICE model-agnostic and presents the three models as prediction techniques, but pages 9-10 also
compare family behavior. Thus the scheduling mechanism and the bounded three-family comparison are
both real. The Issue-23 code seam must preserve all three without choosing a winner.

The earlier LSTM-only scope came from a later project decision, not from the paper. The immutable
[Issue 49 Resolution](https://github.com/edoski/spice/issues/49#issuecomment-4960871251) says “Use LSTM
only” and defers Transformer and Transformer-LSTM with no current mode or machinery. Its immutable
contract records the same choice at
[`decision-contract.md:112-119`](https://github.com/edoski/spice/blob/5647d5b3525ed93b73b34aa5900c867774fb1105/docs/research/issue-49-temporal-baseline/decision-contract.md#L112-L119).
Edo has explicitly approved the final
[Issue-49 Scope-A amendment](https://github.com/edoski/spice/issues/49#issuecomment-4966379354):
preserve and later compare the three concrete families behind one shared task seam. That approved
amendment owns experiment topology, gates, and exact counts; this note infers none.

## Smallest corrected Issue-23 boundary

Issue 23 should define the architecture-independent task algebra:

- `MinBlockFeeOutput(action_logits [B,K], minimum_fee_z [B])`;
- exact `min_block_fee_loss`, `decode_action`, and predictive scorer;
- decoded-`k` scheduling, while retaining the scalar throughout model/scientific output.

It should not make `MinBlockFeeLSTM` the public task seam, claim that only one model implementation
exists, or duplicate task behavior inside each family. The task types and functions must be used
unchanged by every concrete backbone.

Preserve exactly `LstmDefinition`, `TransformerDefinition`, and `TransformerLstmDefinition` in one
closed tagged `ModelDefinition` union. One exhaustive direct constructor match builds three concrete
classes returning `MinBlockFeeOutput`. There is no registry, plugin, adapter, abstract family base,
generic head map, dormant fourth branch, or speculative extension seam.

Decision consequence: the corrected Issue-23 contract can reach approval because its bounded
synthetic full/tail probes confirm all three concrete models satisfy the shared task interface. The
approved Issue-49 amendment owns later scientific execution structure and counts.
