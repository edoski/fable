import { describe, expect, it, vi } from "vitest";
import type { Hash } from "viem";

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
  type Horizon,
  type InferenceEngineDependencies,
} from "../src/inference";
import { ModelOutputError } from "../src/model";
import type {
  MobileChainManifest,
  ModelCatalog,
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

function hashOf(value: bigint): Hash {
  return `0x${value.toString(16).padStart(64, "0")}`;
}

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
    head,
    blocks: [
      block(head - 1n, headBaseFee - 1n),
      block(head, headBaseFee),
    ],
    feeHistory: null,
  };
}

const chainManifest: MobileChainManifest = {
  chain_id: 1,
  context_blocks: 2,
  features: [
    {
      name: "log_base_fee_per_gas",
      mean: 0,
      standard_deviation: 1,
    },
  ],
  models: {
    2: {
      artifact_id: "00000000-0000-4000-8000-000000000002",
      target: { mean: Math.log(100), standard_deviation: 0.5 },
    },
    3: {
      artifact_id: "00000000-0000-4000-8000-000000000003",
      target: { mean: Math.log(100), standard_deviation: 0.5 },
    },
    4: {
      artifact_id: "00000000-0000-4000-8000-000000000004",
      target: { mean: Math.log(100), standard_deviation: 0.5 },
    },
    5: {
      artifact_id: "00000000-0000-4000-8000-000000000005",
      target: { mean: Math.log(100), standard_deviation: 0.5 },
    },
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
    readSnapshot: vi.fn(async () => block(10n)),
    readOutcome: vi.fn(
      async (
        immediateBlock: bigint,
        selectedBlock: bigint,
      ): Promise<ChainOutcome> => ({
        immediateBlock,
        selectedBlock,
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
  it("passes live polling through as safe app snapshots", async () => {
    const stop = vi.fn();
    let publishBlock: ((block: BlockRow) => void) | undefined;
    let publishError: ((error: unknown) => void) | undefined;
    const chainSession = session();
    vi.mocked(chainSession.startPolling).mockImplementation(
      (onBlock, onError) => {
        publishBlock = onBlock;
        publishError = onError;
        return stop;
      },
    );
    const { engine } = createTestEngine({ session: chainSession });
    const onSnapshot = vi.fn();
    const onError = vi.fn();

    const stopPolling = engine.startPolling(onSnapshot, onError);
    publishBlock?.(block(12n, 30n));
    const failure = new Error("RPC unavailable");
    publishError?.(failure);

    expect(onSnapshot).toHaveBeenCalledWith({
      chain: "ethereum",
      head_block: 12,
      current_base_fee_per_gas: 30,
    });
    expect(onError).toHaveBeenCalledWith(failure);
    stopPolling();
    expect(stop).toHaveBeenCalledOnce();
    await engine.dispose();
  });

  it("prepares chain context and model concurrently", async () => {
    const synchronized = deferred<PreparedChainContext>();
    const loaded = deferred<void>();
    const chainSession = session(() => synchronized.promise);
    const model = runtime();
    vi.mocked(model.prepare).mockImplementation(() => loaded.promise);
    const { engine } = createTestEngine({
      session: chainSession,
      model,
    });

    const preparing = engine.prepare(5);
    await vi.waitFor(() => {
      expect(chainSession.sync).toHaveBeenCalledOnce();
      expect(model.prepare).toHaveBeenCalledWith(selection(5));
    });

    synchronized.resolve(context(10n));
    loaded.resolve();
    await preparing;
    await engine.dispose();
  });

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
      head_base_fee_per_gas: 40,
      selected_action_k: 1,
      immediate_block: 12,
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

  it("waits for cold model readiness before its one final sync", async () => {
    const loaded = deferred<void>();
    const chainSession = session();
    const model = runtime();
    vi.mocked(model.prepare)
      .mockRejectedValueOnce(new Error("load failed"))
      .mockImplementationOnce(() => loaded.promise);
    const { engine } = createTestEngine({
      model,
      session: chainSession,
    });

    await expect(engine.prepare(2)).rejects.toThrow("load failed");
    vi.mocked(chainSession.sync).mockClear();

    const running = engine.run(2);
    await vi.waitFor(() =>
      expect(model.prepare).toHaveBeenCalledTimes(2),
    );
    await Promise.resolve();
    expect(chainSession.sync).not.toHaveBeenCalled();

    loaded.resolve();
    await expect(running).resolves.toMatchObject({
      chain: "ethereum",
      K: 2,
    });
    expect(chainSession.sync).toHaveBeenCalledOnce();
    expect(model.execute).toHaveBeenCalledOnce();
    await engine.dispose();
  });

  it("returns short chain and model failures from the run interface", async () => {
    const unavailable = session(async () => {
      throw new Error("HTTP transport details");
    });
    const chainFailure = createTestEngine({ session: unavailable });
    await expect(chainFailure.engine.run(2)).rejects.toThrow(
      "Could not read the selected chain.",
    );
    await chainFailure.engine.dispose();

    const model = runtime();
    vi.mocked(model.prepare).mockRejectedValue(new Error("native load details"));
    const modelFailure = createTestEngine({ model });
    await expect(modelFailure.engine.run(2)).rejects.toThrow(
      "Could not load the selected model.",
    );
    await modelFailure.engine.dispose();

    const execution = runtime();
    vi.mocked(execution.execute).mockRejectedValue(
      new Error("native execution details"),
    );
    const executionFailure = createTestEngine({ model: execution });
    await expect(executionFailure.engine.run(2)).rejects.toThrow(
      "Could not run the selected model.",
    );
    await executionFailure.engine.dispose();

    const invalidOutput = runtime();
    vi.mocked(invalidOutput.execute).mockRejectedValue(
      new ModelOutputError(new Error("tensor details")),
    );
    const outputFailure = createTestEngine({ model: invalidOutput });
    await expect(outputFailure.engine.run(2)).rejects.toThrow(
      "The selected model returned invalid output.",
    );
    await outputFailure.engine.dispose();
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

  it.each([
    {
      name: "logit count",
      output: {
        actionLogits: new Float32Array([1]),
        minimumFeeZ: 0,
      },
      message: "returned invalid output",
    },
    {
      name: "finite logits",
      output: {
        actionLogits: new Float32Array([0, Number.NaN]),
        minimumFeeZ: 0,
      },
      message: "returned invalid output",
    },
    {
      name: "finite positive fee",
      output: {
        actionLogits: new Float32Array([0, 1]),
        minimumFeeZ: 2_000,
      },
      message: "returned invalid output",
    },
  ])("rejects invalid $name", async ({ output, message }) => {
    const { engine } = createTestEngine({ model: runtime(output) });
    await expect(engine.run(2)).rejects.toThrow(message);
    await engine.dispose();
  });

  it("checks every persisted bigint conversion for safe range", async () => {
    const unsafe = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    const unsafeHead = session(async () => context(unsafe, 20n));
    const first = createTestEngine({ session: unsafeHead });
    await expect(first.engine.run(2)).rejects.toMatchObject({
      message: "Chain data is incomplete or invalid.",
      cause: expect.objectContaining({
        message: "head block exceeds the safe integer range",
      }),
    });
    await first.engine.dispose();

    const unsafeBaseFee = session(async () =>
      context(10n, unsafe),
    );
    const nonFeeManifest: MobileChainManifest = {
      ...chainManifest,
      features: [
        {
          name: "gas_utilization",
          mean: 0,
          standard_deviation: 1,
        },
      ],
    };
    const second = createTestEngine({
      catalog: catalog(nonFeeManifest),
      session: unsafeBaseFee,
    });
    await expect(second.engine.run(2)).rejects.toMatchObject({
      message: "Chain data is incomplete or invalid.",
      cause: expect.objectContaining({
        message: "head base fee exceeds the safe integer range",
      }),
    });
    await second.engine.dispose();

    const unsafeImmediate = session(async () =>
      context(BigInt(Number.MAX_SAFE_INTEGER), 20n),
    );
    const third = createTestEngine({ session: unsafeImmediate });
    await expect(third.engine.run(2)).rejects.toMatchObject({
      message: "Chain data is incomplete or invalid.",
      cause: expect.objectContaining({
        message: "immediate block exceeds the safe integer range",
      }),
    });
    await third.engine.dispose();
  });

  it("converts snapshots and outcomes only through safe integers", async () => {
    const chainSession = session();
    const { engine } = createTestEngine({ session: chainSession });

    await expect(engine.snapshot()).resolves.toEqual({
      chain: "ethereum",
      head_block: 10,
      current_base_fee_per_gas: 20,
    });
    await expect(engine.resolveOutcome(11, 12)).resolves.toEqual({
      chain: "ethereum",
      immediate_block: 11,
      selected_block: 12,
      immediate_base_fee_per_gas: 20,
      selected_base_fee_per_gas: 18,
    });
    await expect(
      engine.resolveOutcome(Number.MAX_SAFE_INTEGER + 1, 12),
    ).rejects.toThrow("immediate block must be a nonnegative safe integer");

    vi.mocked(chainSession.readSnapshot).mockResolvedValueOnce(
      block(10n, BigInt(Number.MAX_SAFE_INTEGER) + 1n),
    );
    await expect(engine.snapshot()).rejects.toThrow(
      "current base fee exceeds the safe integer range",
    );
    await engine.dispose();
  });
});
