# FABLE (Fee Analysis through Blockchain Learning and Estimation) Context

This glossary defines FABLE's active domain language.

## Active glossary

**UUID instance.** An identity minted for one Corpus, Study, artifact, or evaluation.

**Typed association.** An exact request/object relationship expressed by the owning schema, UUID, embedded request, or selected Study result index plus Method.

**BlockFrame.** One isolated, validated nine-column value covering an exact contiguous single-chain `CorpusDefinition`, including gas-used-weighted effective priority-fee P50 and P90; it establishes canonical row facts and range selection, not finality or provenance.

**Rolling comparison.** One transient held-out reduction that aligns completed `K=5`, `K=4`, `K=3`, and `K=2` Evaluation observations under an immutable five-block deadline, consulting each shorter-horizon prediction only after the earlier prediction waited.

**P50 fee-inclusive savings.** The arithmetic mean of per-origin savings between the next block and the base-fee-selected block after adding each outcome block's included-transaction effective-priority-fee P50. It is a retrospective representative-cost proxy, not an inclusion guarantee.

**Decision origin.** The decision point immediately after closed parent block `h`.

**Closed parent.** The latest closed block `h` visible at a decision origin.

**Context.** Exactly `C` consecutive closed blocks `h-C+1 … h` selected by block number.

**Horizon.** The exact next `K` blocks `h+1 … h+K` whose complete outcomes define eligibility.

**Action.** Zero-based offset `k` selecting target block `b = h+1+k` within the horizon.

**Role.** One of training, validation, or testing. Training fits weights and data-dependent state, validation selects, and testing measures.
