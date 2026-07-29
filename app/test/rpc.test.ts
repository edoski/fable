import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("viem", async (importOriginal) => {
  const actual = await importOriginal<typeof import("viem")>();
  return { ...actual, http: vi.fn(actual.http) };
});

import { http } from "viem";
import type { Hash } from "viem";

import { createChainSession } from "../src/rpc";
import { flushMicrotasks, hashOf } from "./helpers";

type JsonRpcRequest = {
  id: number;
  method: string;
  params?: readonly unknown[];
};

type RpcFixture = {
  head: bigint;
  block(number: bigint): unknown | Promise<unknown>;
  history(oldestBlock: bigint, count: number): unknown | Promise<unknown>;
  batches: JsonRpcRequest[][];
};

function quantity(value: bigint): `0x${string}` {
  return `0x${value.toString(16)}`;
}

function rpcBlock(
  number: bigint,
  overrides: {
    baseFeePerGas?: `0x${string}` | null;
    hash?: Hash;
    parentHash?: Hash;
  } = {},
) {
  return {
    number: quantity(number),
    hash: overrides.hash ?? hashOf(number),
    parentHash: overrides.parentHash ?? hashOf(number - 1n),
    timestamp: quantity(1_700_000_000n + number),
    baseFeePerGas:
      overrides.baseFeePerGas === undefined
        ? quantity(1_000_000_000n + number)
        : overrides.baseFeePerGas,
    gasUsed: quantity(100n),
    gasLimit: quantity(200n),
    transactions: [],
  };
}

function feeHistory(oldestBlock: bigint, count: number) {
  return {
    oldestBlock: quantity(oldestBlock),
    baseFeePerGas: Array.from({ length: count + 1 }, (_, index) =>
      quantity(1_000_000_000n + BigInt(index)),
    ),
    gasUsedRatio: Array.from({ length: count }, () => 0.5),
    reward: Array.from({ length: count }, (_, index) => [
      quantity(2_000_000_000n + BigInt(index)),
      quantity(3_000_000_000n + BigInt(index)),
    ]),
  };
}

function installRpc(
  overrides: Partial<Pick<RpcFixture, "head" | "block" | "history">> = {},
): RpcFixture {
  const fixture: RpcFixture = {
    head: 12n,
    block: rpcBlock,
    history: feeHistory,
    batches: [],
    ...overrides,
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const parsed = JSON.parse(String(init?.body)) as
        | JsonRpcRequest
        | JsonRpcRequest[];
      const requests = Array.isArray(parsed) ? parsed : [parsed];
      fixture.batches.push(requests);
      const responses = await Promise.all(
        requests.map(async (request) => ({
          jsonrpc: "2.0",
          id: request.id,
          result: await rpcResult(fixture, request),
        })),
      );
      return new Response(
        JSON.stringify(Array.isArray(parsed) ? responses : responses[0]),
        { headers: { "Content-Type": "application/json" } },
      );
    }),
  );
  return fixture;
}

async function rpcResult(
  fixture: RpcFixture,
  request: JsonRpcRequest,
): Promise<unknown> {
  if (request.method === "eth_blockNumber") {
    return quantity(fixture.head);
  }
  if (request.method === "eth_getBlockByNumber") {
    const tag = (request.params as readonly [string])[0];
    return fixture.block(tag === "latest" ? fixture.head : BigInt(tag));
  }
  if (request.method === "eth_feeHistory") {
    const [countValue, headValue] = request.params as readonly [string, string];
    const count = Number(BigInt(countValue));
    const head = BigInt(headValue);
    return fixture.history(head - BigInt(count) + 1n, count);
  }
  throw new Error(`Unexpected RPC method: ${request.method}`);
}

function blockBatches(fixture: RpcFixture): bigint[][] {
  return fixture.batches
    .filter((batch) =>
      batch.some((request) => request.method === "eth_getBlockByNumber"),
    )
    .map((batch) =>
      batch
        .filter((request) => request.method === "eth_getBlockByNumber")
        .map((request) =>
          BigInt((request.params as readonly [string])[0]),
        ),
    );
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("createChainSession", () => {
  it("configures Viem HTTP batching, timeout, and zero retries", () => {
    createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [],
    });

    expect(vi.mocked(http)).toHaveBeenCalledWith(undefined, {
      batch: { batchSize: 40, wait: 0 },
      retryCount: 0,
      timeout: 10_000,
    });
  });

  it("reads every fresh exact range in one batch with only the required predecessor", async () => {
    const rpc = installRpc();
    const intervalSession = createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [
        "block_interval_seconds",
        "log1p_effective_priority_fee_per_gas_p50",
        "log1p_effective_priority_fee_per_gas_p90",
      ],
    });

    const first = await intervalSession.sync();
    rpc.head = 14n;
    const second = await intervalSession.sync();
    const directSession = createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [],
    });
    const direct = await directSession.sync();

    expect(first.blocks.map((block) => block.number)).toEqual([
      9n,
      10n,
      11n,
      12n,
    ]);
    expect(first.priorityFeeRewards).toEqual([
      [2_000_000_000n, 3_000_000_000n],
      [2_000_000_001n, 3_000_000_001n],
      [2_000_000_002n, 3_000_000_002n],
    ]);
    expect(second.blocks.map((block) => block.number)).toEqual([
      11n,
      12n,
      13n,
      14n,
    ]);
    expect(direct.blocks.map((block) => block.number)).toEqual([
      12n,
      13n,
      14n,
    ]);
    expect(blockBatches(rpc)).toEqual([
      [9n, 10n, 11n, 12n],
      [11n, 12n, 13n, 14n],
      [12n, 13n, 14n],
    ]);
    expect(
      rpc.batches
        .flat()
        .filter((request) => request.method === "eth_feeHistory")
        .map((request) => request.params),
    ).toEqual([
      ["0x3", "0xc", [50, 90]],
      ["0x3", "0xe", [50, 90]],
    ]);

    intervalSession.dispose();
    directSession.dispose();
  });

  it("rejects a broken parent link in the fetched range", async () => {
    const rpc = installRpc({
      block: (number) =>
        rpcBlock(
          number,
          number === 11n ? { parentHash: hashOf(99n) } : {},
        ),
    });
    const session = createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [],
    });

    await expect(session.sync()).rejects.toThrow(
      "Broken parent link between blocks 10 and 11",
    );
    expect(blockBatches(rpc)).toEqual([[10n, 11n, 12n]]);
    session.dispose();
  });

  it("requires fee history to start at the first context block", async () => {
    const session = createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [
        "log1p_effective_priority_fee_per_gas_p50",
      ],
    });
    installRpc({
      history: (oldestBlock, count) =>
        feeHistory(oldestBlock + 1n, count),
    });

    await expect(session.sync()).rejects.toThrow(
      "Fee history must start at block 10, got 11",
    );
    session.dispose();
  });

  it("requires an EIP-1559 base fee", async () => {
    installRpc({
      block: (number) =>
        rpcBlock(number, {
          baseFeePerGas: number === 12n ? null : undefined,
        }),
    });
    const session = createChainSession({
      chain: "ethereum",
      contextBlocks: 1,
      orderedFeatures: [],
    });

    await expect(session.sync()).rejects.toThrow(
      "RPC returned block 12 without a base fee",
    );
    session.dispose();
  });

  it("forwards Viem watched blocks and unwatches on disposal", async () => {
    vi.useFakeTimers();
    const rpc = installRpc();
    const session = createChainSession({
      chain: "ethereum",
      contextBlocks: 1,
      orderedFeatures: [],
    });
    const publish = vi.fn();

    session.watchBlocks(publish);
    await vi.advanceTimersByTimeAsync(0);
    await flushMicrotasks();

    expect(publish).toHaveBeenCalledWith(
      expect.objectContaining({
        number: 12n,
        baseFeePerGas: 1_000_000_012n,
      }),
    );
    const readsBeforeDisposal = rpc.batches.flat().length;

    session.dispose();
    await vi.advanceTimersByTimeAsync(10_000);
    await flushMicrotasks();
    expect(rpc.batches.flat()).toHaveLength(readsBeforeDisposal);
  });

  it("reads exact outcome blocks directly", async () => {
    const rpc = installRpc();
    const session = createChainSession({
      chain: "ethereum",
      contextBlocks: 3,
      orderedFeatures: [],
    });

    await expect(session.readOutcome(20n, 22n)).resolves.toEqual({
      immediateBaseFeePerGas: 1_000_000_020n,
      selectedBaseFeePerGas: 1_000_000_022n,
    });
    expect(blockBatches(rpc)).toEqual([[20n, 22n]]);
    session.dispose();
  });
});
