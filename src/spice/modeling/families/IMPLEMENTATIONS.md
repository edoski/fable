# Concrete Model Families

Model families turn padded temporal input tensors into prediction heads. Current models are neural sequence encoders: they read rows of features over time and produce tensors consumed by a prediction family.

## Mental Model

A model does not know economic metrics. It maps sequences to output heads.

```text
inputs + input_mask
  -> sequence encoder
  -> final valid hidden state
  -> prediction heads
```

Prediction families define the head names and target losses. Model families provide the shared sequence representation.

## Beginner Theory: What The Network Learns

A neural network is a parameterized function. The parameters are tensors called weights. During training, the model sees input examples, produces outputs, computes a loss, then backpropagation computes how each weight contributed to that loss. The optimizer updates the weights so future outputs should reduce the loss.

In this package, the input is not one independent row. It is a short time series:

```text
row t-k, row t-k+1, ..., row t
```

The model must compress that sequence into a hidden representation. The prediction family then turns the representation into concrete heads such as candidate-offset logits.

Two terms matter:

| Term | Meaning in SPICE |
| --- | --- |
| Hidden state | Learned summary of the observed block-fee context. |
| Logit | Raw, unnormalized score for an action or class. |

Logits are useful because loss functions can combine them with numerically stable softmax or cross-entropy operations. The model should emit logits, not already-normalized probabilities.

## Shared Sequence Rule

All current families use the final real context row as the sample representation. Padding exists only to batch variable-length windows.

```text
input_mask:  true true true false false
last valid:             ^
```

This lets models ignore padded rows.

Padding is a batching device, not data. A batch may contain one sample with 64 rows and another with 400 rows. The tensor needs one rectangular shape, so shorter samples receive padded rows and the mask tells the model which positions are real.

## `lstm`

The LSTM family uses an input projection, recurrent LSTM layers, and MLP prediction heads.

An LSTM is a recurrent neural network designed for sequences. Classic recurrent networks can struggle when useful evidence is far back in the sequence because gradients shrink or grow as they are repeatedly multiplied through time. LSTM addresses this with a memory cell and gates.

| Concept | Meaning |
| --- | --- |
| Cell state | Longer-lived memory path through the sequence. |
| Input gate | Controls how much new information enters memory. |
| Forget gate | Controls how much existing memory is kept. |
| Output gate | Controls how memory affects the exposed hidden state. |
| Final valid state | Summary of the observed context window. |

For block-fee data, this means the model can learn patterns such as "recent gas pressure has been rising for several rows" or "the fee spike happened earlier in the context and is decaying."

The family implementation only defines architecture and tuning fields; runtime placement and compile policy belong to modeling runtime planning.

Tunable fields: `input_projection_dim`, `hidden_size`, `num_layers`, `head_hidden_dim`, and `dropout`.

## `transformer`

The Transformer family uses input projection, sinusoidal positional encoding, and a Transformer encoder with a padding mask.

Attention lets each row compare against other rows in the same context window. For a beginner, attention is a learned lookup: each row asks which other rows matter for interpreting it. A gas spike can directly attend to earlier fee levels without waiting for information to pass step by step through a recurrent state.

Self-attention alone does not know order. If rows were shuffled, the same set of row embeddings would be present. Positional encoding adds deterministic position signals so the model can distinguish early context from late context.

Multi-head attention repeats this comparison several ways in parallel. One head can learn local fee momentum while another head tracks longer-range context.

Important constraints:

| Constraint | Why |
| --- | --- |
| `d_model` must be even | Sinusoidal position encoding uses paired dimensions. |
| `d_model` divisible by `nhead` | Multi-head attention splits hidden dimensions evenly. |
| Padding mask supplied | Attention must ignore padded rows. |

Runtime planning currently runs this family with `32-true` precision and no compile-time graph capture.

Tunable fields: `d_model`, `nhead`, `transformer_layers`, `feedforward_multiplier`, `head_hidden_dim`, and `dropout`. `feedforward_multiplier` is sampled but not stored directly; the family derives tuned `feedforward_dim` as `d_model * feedforward_multiplier`.

## `transformer_lstm`

This family runs a Transformer encoder first, then an LSTM. The Transformer builds context-aware row embeddings; the LSTM aggregates them in temporal order.

```text
project inputs
  -> positional encoding
  -> transformer encoder
  -> LSTM
  -> final valid hidden state
  -> heads
```

This combines attention-based context mixing with recurrent sequence summarization.

The practical reason to combine them is bias. The Transformer stage is good at comparing rows globally. The LSTM stage then imposes an ordered summarization step before the heads. That can help when the final decision should depend on both "which earlier rows matter" and "how the sequence evolved toward the anchor."

Tunable fields: `hidden_size`, `num_layers`, `d_model`, `nhead`, `transformer_layers`, `feedforward_multiplier`, `head_hidden_dim`, and `dropout`. Like the Transformer family, it derives tuned `feedforward_dim` from `d_model * feedforward_multiplier`.

## Output Heads

Model families do not hard-code economic targets. They ask the prediction contract for output specs and create one head per output. For example:

| Prediction family | Output heads |
| --- | --- |
| `min_block_fee_multitask` | `min_block_offset_logits`, `min_block_log_fee` |

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Input width mismatch | Feature contract and model input dimension disagree. |
| Invalid Transformer dimensions | Attention/position encoding cannot be built. |
| Missing output head | Model does not satisfy prediction contract. |
| CUDA unavailable | Current training path cannot run. |

## Extension Pattern

A new model family should consume the same `ModelInputBatch` and emit the output heads requested by the prediction contract. Keep target economics in prediction families, not in model architecture code.

## Theory References

- LSTM: Hochreiter and Schmidhuber, "Long Short-Term Memory" (Neural Computation, 1997): https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
- Transformer: Vaswani et al., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762
