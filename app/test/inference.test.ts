import { describe, expect, it, vi } from "vitest";

vi.mock("react-native-executorch", () => ({
  ExecutorchModule: vi.fn(),
  ScalarType: { FLOAT: 6 },
  initExecutorch: vi.fn(),
}));

vi.mock("react-native-executorch-expo-resource-fetcher", () => ({
  ExpoResourceFetcher: { name: "expo-resource-fetcher" },
}));

import {
  createInferenceEngine,
  type InferenceEngineDependencies,
} from "../src/inference";
import type { Horizon } from "../src/domain";
import type {
  MobileChainManifest,
  ModelCatalog,
  ModelManifest,
  ModelOutput,
  ModelRuntime,
  ModelSelection,
} from "../src/model";
import type {
  BlockRow,
  ChainOutcome,
  ChainSession,
  PreparedChainContext,
} from "../src/rpc";
import { hashOf } from "./helpers";

function block(
  number: bigint,
  baseFeePerGas = number + 10n,
): BlockRow {
  return {
    number,
    hash: hashOf(number),
    parentHash: hashOf(number - 1n),
    timestamp: 1_700_000_000n + number,
    baseFeePerGas,
    gasUsed: 100n,
    gasLimit: 200n,
    transactionCount: 0,
  };
}

function context(
  head: bigint,
  headBaseFee = head + 10n,
): PreparedChainContext {
  return {
    blocks: [
      block(head - 1n, headBaseFee - 1n),
      block(head, headBaseFee),
    ],
    p50Rewards: null,
  };
}

function modelEntry(K: Horizon): ModelManifest {
  return {
    artifact_id: `00000000-0000-4000-8000-${K.toString().padStart(12, "0")}`,
    target: { mean: Math.log(100), standard_deviation: 0.5 },
  };
}

const chainManifest: MobileChainManifest = {
  context_blocks: 2,
  features: [
    {
      name: "log_base_fee_per_gas",
      mean: 0,
      standard_deviation: 1,
    },
  ],
  models: {
    2: modelEntry(2),
    3: modelEntry(3),
    4: modelEntry(4),
    5: modelEntry(5),
  },
};

function selection(
  K: Horizon,
  manifest = chainManifest,
): ModelSelection {
  return {
    chain: "ethereum",
    K,
    source: 10 + K,
    chainManifest: manifest,
    modelManifest: manifest.models[K],
  };
}

function catalog(manifest = chainManifest): ModelCatalog {
  return {
    chainManifest: vi.fn(() => manifest),
    select: vi.fn((_chain, K) => selection(K, manifest)),
  };
}

function session(
  sync: () => Promise<PreparedChainContext> = async () => context(10n),
): ChainSession {
  return {
    sync: vi.fn(sync),
    readOutcome: vi.fn(
      async (
        _immediateBlock: bigint,
        _selectedBlock: bigint,
      ): Promise<ChainOutcome> => ({
        immediateBaseFeePerGas: 20n,
        selectedBaseFeePerGas: 18n,
      }),
    ),
    startPolling: vi.fn(() => () => undefined),
    dispose: vi.fn(),
  };
}

function runtime(
  output: ModelOutput = {
    actionLogits: new Float32Array([0, 1]),
    minimumFeeZ: 0,
  },
): ModelRuntime {
  return {
    prepare: vi.fn(async () => undefined),
    execute: vi.fn(async () => output),
    dispose: vi.fn(async () => undefined),
  };
}

function createTestEngine(
  overrides: Partial<InferenceEngineDependencies> = {},
) {
  const dependencies: InferenceEngineDependencies = {
    chain: "ethereum",
    catalog: catalog(),
    model: runtime(),
    session: session(),
    ...overrides,
  };
  return {
    dependencies,
    engine: createInferenceEngine(dependencies),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((next, fail) => {
    resolve = next;
    reject = fail;
  });
  return { promise, resolve, reject };
}

describe("InferenceEngine", () => {
  it("final-syncs fresh context, builds input, and decodes the run", async () => {
    const chainSession = session();
    vi.mocked(chainSession.sync)
      .mockResolvedValueOnce(context(10n, 20n))
      .mockResolvedValueOnce(context(11n, 40n));
    const model = runtime({
      actionLogits: new Float32Array([-1, 4, 1, 0]),
      minimumFeeZ: 2,
    });
    const { engine } = createTestEngine({
      session: chainSession,
      model,
    });

    await engine.prepare(4);
    const result = await engine.run(4);

    expect(chainSession.sync).toHaveBeenCalledTimes(2);
    expect(model.prepare).toHaveBeenCalledTimes(2);
    expect(model.execute).toHaveBeenCalledWith(
      selection(4),
      new Float32Array([
        Math.fround(Math.log(39)),
        Math.fround(Math.log(40)),
      ]),
    );
    expect(result).toEqual({
      chain: "ethereum",
      K: 4,
      artifact_id: chainManifest.models[4].artifact_id,
      head_block: 11,
      head_hash: hashOf(11n),
      selected_action_k: 1,
      target_block: 13,
      predicted_minimum_base_fee_per_gas: expect.closeTo(
        Math.exp(Math.log(100) + 1),
      ),
    });
    await engine.dispose();
  });

  it("retries through run after preparation fails", async () => {
    const synchronized = deferred<PreparedChainContext>();
    const chainSession = session();
    vi.mocked(chainSession.sync)
      .mockImplementationOnce(() => synchronized.promise)
      .mockResolvedValue(context(10n));
    const model = runtime();
    vi.mocked(model.prepare)
      .mockRejectedValueOnce(new Error("load failed"))
      .mockResolvedValue(undefined);
    const { engine } = createTestEngine({
      model,
      session: chainSession,
    });

    let rejected = false;
    const preparing = engine.prepare(2).catch((error: unknown) => {
      rejected = true;
      throw error;
    });
    await vi.waitFor(() => expect(model.prepare).toHaveBeenCalledOnce());
    await Promise.resolve();
    await Promise.resolve();
    expect(rejected).toBe(false);

    synchronized.resolve(context(10n));
    await expect(preparing).rejects.toThrow("load failed");
    await expect(engine.run(2)).resolves.toMatchObject({
      chain: "ethereum",
      K: 2,
    });
    expect(model.prepare).toHaveBeenCalledTimes(2);
    expect(model.execute).toHaveBeenCalledOnce();
    await engine.dispose();
  });

  it("returns short chain, model-load, and run failures with their causes", async () => {
    const unavailable = session(async () => {
      throw new Error("HTTP transport details");
    });
    const chainFailure = createTestEngine({ session: unavailable });
    await expect(chainFailure.engine.run(2)).rejects.toMatchObject({
      message: "Could not read the selected chain.",
      cause: expect.objectContaining({ message: "HTTP transport details" }),
    });
    await chainFailure.engine.dispose();

    const model = runtime();
    vi.mocked(model.prepare).mockRejectedValue(new Error("native load details"));
    const modelFailure = createTestEngine({ model });
    await expect(modelFailure.engine.run(2)).rejects.toMatchObject({
      message: "Could not load the selected model.",
      cause: expect.objectContaining({ message: "native load details" }),
    });
    await modelFailure.engine.dispose();

    const execution = runtime();
    vi.mocked(execution.execute).mockRejectedValue(
      new Error("native execution details"),
    );
    const executionFailure = createTestEngine({ model: execution });
    await expect(executionFailure.engine.run(2)).rejects.toMatchObject({
      message: "Could not run the selected model.",
      cause: expect.objectContaining({ message: "native execution details" }),
    });
    await executionFailure.engine.dispose();
  });

  it("suppresses native work completed after a selection change", async () => {
    const completed = deferred<ModelOutput>();
    const model = runtime();
    vi.mocked(model.execute).mockImplementation(() => completed.promise);
    const { engine } = createTestEngine({ model });

    const running = engine.run(5);
    await vi.waitFor(() => expect(model.execute).toHaveBeenCalledOnce());
    await engine.prepare(2);
    completed.resolve({
      actionLogits: new Float32Array([0, 0, 0, 0, 1]),
      minimumFeeZ: 0,
    });

    await expect(running).rejects.toMatchObject({ name: "AbortError" });
    await engine.dispose();
  });

  it("suppresses native work completed after disposal", async () => {
    const completed = deferred<ModelOutput>();
    const model = runtime();
    vi.mocked(model.execute).mockImplementation(() => completed.promise);
    const { engine, dependencies } = createTestEngine({ model });

    const running = engine.run(2);
    await vi.waitFor(() => expect(model.execute).toHaveBeenCalledOnce());
    await engine.dispose();
    completed.resolve({
      actionLogits: new Float32Array([0, 1]),
      minimumFeeZ: 0,
    });

    await expect(running).rejects.toMatchObject({ name: "AbortError" });
    expect(dependencies.session.dispose).toHaveBeenCalledOnce();
    expect(model.dispose).toHaveBeenCalledOnce();
  });

  it("rejects a nonfinite decoded fee", async () => {
    const { engine } = createTestEngine({
      model: runtime({
        actionLogits: new Float32Array([0, 1]),
        minimumFeeZ: 2_000,
      }),
    });
    await expect(engine.run(2)).rejects.toMatchObject({
      message: "Could not run the selected model.",
      cause: expect.objectContaining({
        message: "Predicted fee must be positive and finite",
      }),
    });
    await engine.dispose();
  });

  it("rejects an unsafe external head block without losing raw precision", async () => {
    const unsafe = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    const unsafeHead = session(async () => context(unsafe, 20n));
    const { engine } = createTestEngine({ session: unsafeHead });

    await expect(engine.run(2)).rejects.toMatchObject({
      message: "Chain data is incomplete or invalid.",
      cause: expect.objectContaining({
        message: "head block exceeds the safe integer range",
      }),
    });
    await engine.dispose();
  });

  it("passes exact outcome blocks and converts RPC fees through safe integers", async () => {
    const chainSession = session();
    const { engine } = createTestEngine({ session: chainSession });

    await expect(engine.resolveOutcome(11, 12)).resolves.toEqual({
      immediate_base_fee_per_gas: 20,
      selected_base_fee_per_gas: 18,
    });
    expect(chainSession.readOutcome).toHaveBeenCalledWith(11n, 12n);

    vi.mocked(chainSession.readOutcome).mockResolvedValueOnce({
      immediateBaseFeePerGas: BigInt(Number.MAX_SAFE_INTEGER) + 1n,
      selectedBaseFeePerGas: 18n,
    });
    await expect(engine.resolveOutcome(11, 12)).rejects.toThrow(
      "immediate base fee exceeds the safe integer range",
    );
    await engine.dispose();
  });
});
