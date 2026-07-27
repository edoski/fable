# FABLE (Fee Analysis through Blockchain Learning and Estimation)

FABLE learns from finalized block history to choose a low-base-fee block within a short future [horizon](docs/CONTEXT.md). It compares LSTM, Transformer, and Transformer-LSTM models.

Its scientific lineage is the temporal experiment in *SPICE: A Predictive Framework for Cost-Optimization in Multichain Environments*: a future minimum-block decision paired with an auxiliary fee prediction. FABLE's current equations and claim limits are documented in the [manual](docs/FABLE.md#scientific-contract). The [glossary](docs/CONTEXT.md) defines its domain terms.

## Hosts and responsibilities

The Python system supports two explicit operating locations:

- A workstation consumes prepared block history, creates requests, submits work, publishes tuning results, and computes transient evaluation reductions.
- A GPU server fits, tunes, and evaluates through Slurm jobs.

The [manual](docs/FABLE.md#remote-submission) defines remote submission and host configuration.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

## Quick start

Place each completed canonical Corpus pair at
`STORAGE_ROOT/corpora/<UUID>/corpus.json` and `blocks.parquet`. Corpus production is external to
FABLE. Create workflow requests from the [request reference](docs/FABLE.md#requests-and-definitions).

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

The [CLI reference](docs/FABLE.md#cli) defines the exact command contracts.

## Mobile demo

The private Expo 55 app in `app` reads the selected EVM chain directly, prepares features in
TypeScript, runs a bundled ExecuTorch model on device, and keeps history and resolved outcomes in
local storage. It has no FABLE inference server or fallback.

The app requires a generated model bundle. Once the twelve final artifact UUIDs exist, create the
strict three-chain by four-horizon `MOBILE.yaml` roster and export all assets atomically:

```bash
STORAGE_ROOT=/absolute/storage \
uv run --project tools/mobile-export --frozen \
python tools/mobile-export/export.py MOBILE.yaml app/assets/models
```

Then install dependencies, check the Expo project, and create the custom native development build:

```bash
cd app
npm ci
npx expo-doctor
npm run ios
```

ExecuTorch is a native module, so Expo Go is unsupported. With a compatible development build
already installed and native configuration unchanged, start Metro for JavaScript or asset
iteration with:

```bash
npm start
```

The repository does not yet contain `MOBILE.yaml` or generated model assets because the twelve
final artifacts do not exist. The app cannot be bundled or exercised in the simulator until that
prerequisite is satisfied. The [acceptance plan](docs/research/on-device-inference.md) separates
the implemented code from the deferred real-artifact checks.

## Where do I look?

| Question | Owner |
| --- | --- |
| How does one decision work end to end? | [Worked decision](docs/FABLE.md#one-decision-end-to-end) |
| Why are the inputs causal, and what do the equations mean? | [Scientific contract](docs/FABLE.md#scientific-contract) |
| Which module owns each object and seam? | [Architecture](docs/FABLE.md#architecture-and-deep-interfaces) |
| What are the exact requests, paths, commands, and schemas? | [Exact reference](docs/FABLE.md#exact-reference) |
| What does a domain term mean? | [Glossary](docs/CONTEXT.md) |
| Which architectural decisions remain active? | [ADR index](docs/adr/README.md) |
