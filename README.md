# FABLE (Fee Analysis through Blockchain Learning and Estimation)

FABLE learns from finalized block history to choose a low-base-fee block within a short future [horizon](CONTEXT.md). It compares LSTM, Transformer, and Transformer-LSTM models.

Its scientific lineage is the temporal experiment in *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments*: a future minimum-block decision paired with an auxiliary fee prediction. FABLE's current equations and claim limits are documented in the [manual](FABLE.md#scientific-contract). The [glossary](CONTEXT.md) defines its domain terms.

## Hosts and responsibilities

The Python system supports two explicit operating locations:

- A workstation consumes prepared block history, creates requests, submits work, publishes tuning results, and computes transient evaluation reductions.
- A GPU server fits, tunes, and evaluates through Slurm jobs.

The [manual](FABLE.md#remote-submission) defines remote submission and host configuration.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

## Quick start

Place each completed canonical Corpus pair at
`STORAGE_ROOT/corpora/<UUID>/corpus.json` and `blocks.parquet`. Corpus production is external to
FABLE. Create workflow requests from the [request reference](FABLE.md#requests-and-definitions).

Submit one or more training or evaluation requests:

```bash
fable submit REQUEST.json
```

Run one candidate configuration from a tuning request:

```bash
fable study run TUNE_REQUEST.json METHOD.json
```

Publish the collected tuning results:

```bash
STORAGE_ROOT=/absolute/storage fable study finalize STUDY_ID
```

The [CLI reference](FABLE.md#cli) defines the exact command contracts.

## Mobile demo

The private Expo app lives in `app` and owns chain acquisition, feature preparation, model inference, history, and outcomes:

```bash
cd app
npm start
```

## Read next

Read the [FABLE manual](FABLE.md) from its worked decision through the scientific contract, architecture, and exact reference. Hard-to-reverse decisions are in [docs/adr](docs/adr/).

## Where do I look?

| Question | Owner |
| --- | --- |
| How does one decision work end to end? | [Worked decision](FABLE.md#one-decision-end-to-end) |
| Why are the inputs causal, and what do the equations mean? | [Scientific contract](FABLE.md#scientific-contract) |
| Which module owns each object and seam? | [Architecture](FABLE.md#architecture-and-deep-interfaces) |
| What are the exact requests, paths, commands, and schemas? | [Exact reference](FABLE.md#exact-reference) |
| What does a domain term mean? | [Context](CONTEXT.md) |
| How does one deep interface work internally? | [Architecture and deep interfaces](FABLE.md#architecture-and-deep-interfaces) |
