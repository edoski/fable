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
import type { BlockRow } from "../src/domain";
import type {
  ModelOutput,
  ModelRuntime,
} from "../src/model";
import type {
  ChainOutcome,
  ChainSession,
  PreparedChainContext,
} from "../src/rpc";
import {
  chainManifest,
  hashOf,
  modelSelection,
} from "./helpers";

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
    priorityFeeRewards: null,
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
    watchBlocks: vi.fn(),
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
    execute: vi.fn(async () => output),
    dispose: vi.fn(async () => undefined),
  };
}

function createTestEngine(
  overrides: Partial<InferenceEngineDependencies> = {},
) {
  const dependencies: InferenceEngineDependencies = {
    model: runtime(),
    selectModel: vi.fn(modelSelection),
    session: session(),
    ...overrides,
  };
  return createInferenceEngine("ethereum", dependencies);
}

describe("InferenceEngine", () => {
  it("synchronizes, builds input, and executes the selected model on Run", async () => {
    const events: string[] = [];
    const chainSession = session(async () => {
      events.push("sync");
      return context(11n, 40n);
    });
    const model = runtime({
      actionLogits: new Float32Array([-1, 4, 1, 0]),
      minimumFeeZ: 2,
    });
    vi.mocked(model.execute).mockImplementation(async () => {
      events.push("execute");
      return {
        actionLogits: new Float32Array([-1, 4, 1, 0]),
        minimumFeeZ: 2,
      };
    });
    const engine = createTestEngine({
      session: chainSession,
      model,
    });

    const result = await engine.run(4);

    expect(events).toEqual(["sync", "execute"]);
    expect(chainSession.sync).toHaveBeenCalledOnce();
    expect(model.execute).toHaveBeenCalledWith(
      modelSelection(4),
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

  it("returns short chain and model failures with their causes", async () => {
    const unavailable = session(async () => {
      throw new Error("HTTP transport details");
    });
    const chainFailure = createTestEngine({ session: unavailable });
    await expect(chainFailure.run(2)).rejects.toMatchObject({
      message: "Could not read the selected chain.",
      cause: expect.objectContaining({ message: "HTTP transport details" }),
    });
    await chainFailure.dispose();

    const model = runtime();
    vi.mocked(model.execute).mockRejectedValue(
      new Error("native load details"),
    );
    const modelFailure = createTestEngine({ model });
    await expect(modelFailure.run(2)).rejects.toMatchObject({
      message: "Could not run the selected model.",
      cause: expect.objectContaining({ message: "native load details" }),
    });
    await modelFailure.dispose();
  });

  it("rejects a nonfinite decoded fee", async () => {
    const engine = createTestEngine({
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

  it("selects the first action when maximum logits tie", async () => {
    const engine = createTestEngine({
      model: runtime({
        actionLogits: new Float32Array([1, 4, 4, 0]),
        minimumFeeZ: 0,
      }),
    });

    await expect(engine.run(4)).resolves.toMatchObject({
      selected_action_k: 1,
      target_block: 12,
    });
    await engine.dispose();
  });

  it("rejects an unsafe external head block without losing raw precision", async () => {
    const unsafe = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    const unsafeHead = session(async () => context(unsafe, 20n));
    const engine = createTestEngine({ session: unsafeHead });

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
    const engine = createTestEngine({ session: chainSession });

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
