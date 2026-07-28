import { afterEach, describe, expect, it, vi } from "vitest";
import { custom } from "viem";
import type { Hash, Transport } from "viem";

import type { Chain } from "../src/domain";
import type { FeatureName } from "../src/features";
import { createChainSession } from "../src/rpc";
import { flushMicrotasks, hashOf } from "./helpers";

type RequestArguments = {
  method: string;
  params?: readonly unknown[];
};

type RpcProvider = {
  request(args: RequestArguments): Promise<unknown>;
};

const CHAIN_IDS: Record<Chain, number> = {
  ethereum: 1,
  polygon: 137,
  avalanche: 43_114,
};

function quantity(value: bigint): `0x${string}` {
  return `0x${value.toString(16)}`;
}

function rpcBlock(
  number: bigint,
  overrides: {
    hash?: Hash | null;
    parentHash?: Hash;
    returnedNumber?: bigint;
  } = {},
) {
  return {
    number: quantity(overrides.returnedNumber ?? number),
    hash: overrides.hash === undefined ? hashOf(number) : overrides.hash,
    parentHash: overrides.parentHash ?? hashOf(number - 1n),
    timestamp: quantity(1_700_000_000n + number),
    baseFeePerGas: quantity(1_000_000_000n + number),
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

function transport(provider: RpcProvider): Transport {
  return custom(provider, { retryCount: 0 });
}

type FakeChain = RpcProvider & {
  chainId(): number | Promise<number>;
  head: bigint;
  block(number: bigint): unknown | Promise<unknown>;
  history(oldestBlock: bigint, count: number): unknown | Promise<unknown>;
  requests: RequestArguments[];
  reads: bigint[];
};

function fakeChain(
  overrides: Partial<
    Pick<FakeChain, "chainId" | "head" | "block" | "history">
  > = {},
): FakeChain {
  const chain: FakeChain = {
    chainId: () => 1,
    head: 12n,
    block: rpcBlock,
    history: feeHistory,
    requests: [],
    reads: [],
    async request(args) {
      chain.requests.push(args);
      if (args.method === "eth_chainId") {
        return quantity(BigInt(await chain.chainId()));
      }
      if (args.method === "eth_blockNumber") {
        return quantity(chain.head);
      }
      if (args.method === "eth_getBlockByNumber") {
        const number = blockNumberFrom(args);
        chain.reads.push(number);
        return chain.block(number);
      }
      if (args.method === "eth_feeHistory") {
        const [countValue, headValue] = args.params as readonly [string, string];
        const count = Number(BigInt(countValue));
        const head = BigInt(headValue);
        return chain.history(head - BigInt(count) + 1n, count);
      }
      throw new Error(`Unexpected RPC method: ${args.method}`);
    },
    ...overrides,
  };
  return chain;
}

function session(
  chain: Chain,
  rpcTransport: Transport,
  {
    contextBlocks = 3,
    orderedFeatures = [],
  }: {
    contextBlocks?: number;
    orderedFeatures?: readonly FeatureName[];
  } = {},
) {
  return createChainSession(
    {
      chain,
      contextBlocks,
      orderedFeatures,
    },
    rpcTransport,
  );
}

function blockNumberFrom(args: RequestArguments): bigint {
  return BigInt((args.params as readonly [string])[0]);
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("createChainSession", () => {
  it.each(Object.entries(CHAIN_IDS) as [Chain, number][])(
    "verifies the %s chain ID",
    async (chain, chainId) => {
      const rpc = fakeChain({
        chainId: () => chainId,
        head: 5n,
      });
      const chainSession = session(chain, transport(rpc), {
        contextBlocks: 1,
      });

      const context = await chainSession.sync();
      await chainSession.sync();

      expect(context.blocks.at(-1)?.number).toBe(5n);
      expect(context.blocks.map((block) => block.number)).toEqual([5n]);
      expect(
        rpc.requests.filter((request) => request.method === "eth_chainId"),
      ).toHaveLength(1);
      chainSession.dispose();
    },
  );

  it("rejects a provider connected to the wrong chain", async () => {
    const rpc = fakeChain({ chainId: () => 137 });
    const chainSession = session("ethereum", transport(rpc));

    await expect(chainSession.sync()).rejects.toThrow(
      "RPC chain ID 137 does not match expected chain ID 1",
    );
    chainSession.dispose();
  });

  it("retries chain verification after a transient failure", async () => {
    let chainIdReads = 0;
    const rpc = fakeChain({
      head: 5n,
      chainId: () => {
        chainIdReads += 1;
        if (chainIdReads === 1) {
          throw new Error("temporary chain ID failure");
        }
        return 1;
      },
    });
    const chainSession = session("ethereum", transport(rpc), {
      contextBlocks: 1,
    });

    await expect(chainSession.sync()).rejects.toThrow();
    const context = await chainSession.sync();

    expect(context.blocks.at(-1)?.number).toBe(5n);
    expect(chainIdReads).toBe(2);
    chainSession.dispose();
  });

  it("performs a cold fetch, then appends and trims a warm context", async () => {
    const rpc = fakeChain();
    const chainSession = session("ethereum", transport(rpc));

    const cold = await chainSession.sync();
    rpc.head = 14n;
    const warm = await chainSession.sync();

    expect(cold.blocks.map((block) => block.number)).toEqual([10n, 11n, 12n]);
    expect(warm.blocks.map((block) => block.number)).toEqual([12n, 13n, 14n]);
    expect(rpc.reads).toEqual([10n, 11n, 12n, 13n, 14n]);
    chainSession.dispose();
  });

  it("bounds catch-up after a large head jump", async () => {
    const rpc = fakeChain();
    const chainSession = session("ethereum", transport(rpc));

    await chainSession.sync();
    rpc.head = 20n;
    const context = await chainSession.sync();

    expect(context.blocks.map((block) => block.number)).toEqual([
      18n,
      19n,
      20n,
    ]);
    expect(rpc.reads).toEqual([10n, 11n, 12n, 18n, 19n, 20n]);
    chainSession.dispose();
  });

  it("serializes concurrent different-head synchronizations", async () => {
    const heads = [10n, 11n, 12n];
    let headReads = 0;
    let blockElevenReads = 0;
    let releaseBlockEleven:
      | ((value: ReturnType<typeof rpcBlock>) => void)
      | undefined;
    const delayedBlockEleven = new Promise<ReturnType<typeof rpcBlock>>(
      (resolve) => {
        releaseBlockEleven = resolve;
      },
    );
    const provider: RpcProvider = {
      async request(args) {
        if (args.method === "eth_chainId") return quantity(1n);
        if (args.method === "eth_blockNumber") {
          const head = heads[Math.min(headReads, heads.length - 1)];
          headReads += 1;
          return quantity(head);
        }
        if (args.method === "eth_getBlockByNumber") {
          const number = blockNumberFrom(args);
          if (number === 11n) {
            blockElevenReads += 1;
            if (blockElevenReads === 1) return delayedBlockEleven;
          }
          return rpcBlock(number);
        }
        throw new Error(`Unexpected RPC method: ${args.method}`);
      },
    };
    const chainSession = session("ethereum", transport(provider));
    await chainSession.sync();

    const first = chainSession.sync();
    await vi.waitFor(() => expect(blockElevenReads).toBe(1));
    const second = chainSession.sync();
    await flushMicrotasks();

    expect(headReads).toBe(2);
    releaseBlockEleven?.(rpcBlock(11n));
    const [firstContext, secondContext] = await Promise.all([first, second]);

    expect(firstContext.blocks.map((block) => block.number)).toEqual([
      9n,
      10n,
      11n,
    ]);
    expect(secondContext.blocks.map((block) => block.number)).toEqual([
      10n,
      11n,
      12n,
    ]);
    expect(blockElevenReads).toBe(1);
    chainSession.dispose();
  });

  it.each([
    {
      name: "oldest block",
      mutate: (history: ReturnType<typeof feeHistory>) => ({
        ...history,
        oldestBlock: quantity(11n),
      }),
    },
    {
      name: "row count",
      mutate: (history: ReturnType<typeof feeHistory>) => ({
        ...history,
        reward: history.reward.slice(1),
      }),
    },
    {
      name: "row width",
      mutate: (history: ReturnType<typeof feeHistory>) => ({
        ...history,
        reward: history.reward.map((row, index) =>
          index === 0 ? [...row, quantity(3n)] : row,
        ),
      }),
    },
  ])("rejects incomplete fee-history $name coverage", async ({ mutate }) => {
    const rpc = fakeChain({
      history: (oldestBlock, count) =>
        mutate(feeHistory(oldestBlock, count)),
    });
    const chainSession = session("ethereum", transport(rpc), {
      orderedFeatures: ["log1p_effective_priority_fee_per_gas_p50"],
    });

    await expect(chainSession.sync()).rejects.toThrow(
      "Fee history must exactly cover blocks 10 through 12",
    );
    chainSession.dispose();
  });

  it("returns exact P50 and P90 coverage ending at the synchronized head", async () => {
    const rpc = fakeChain();
    const chainSession = session("ethereum", transport(rpc), {
      orderedFeatures: [
        "log1p_effective_priority_fee_per_gas_p50",
        "log1p_effective_priority_fee_per_gas_p90",
      ],
    });

    const context = await chainSession.sync();

    expect(context.priorityFeeRewards).toEqual([
      [2_000_000_000n, 3_000_000_000n],
      [2_000_000_001n, 3_000_000_001n],
      [2_000_000_002n, 3_000_000_002n],
    ]);
    expect(
      rpc.requests.find((request) => request.method === "eth_feeHistory")
        ?.params,
    ).toEqual(["0x3", "0xc", [50, 90]]);
    chainSession.dispose();
  });

  it("recovers a broken incremental parent link with one full refetch", async () => {
    let block13Reads = 0;
    const rpc = fakeChain({
      block: (number) => {
        if (number === 13n) {
          block13Reads += 1;
          return rpcBlock(number, {
            parentHash: block13Reads === 1 ? hashOf(99n) : hashOf(12n),
          });
        }
        return rpcBlock(number);
      },
    });
    const chainSession = session("ethereum", transport(rpc));

    await chainSession.sync();
    rpc.head = 13n;
    const recovered = await chainSession.sync();

    expect(recovered.blocks.map((block) => block.number)).toEqual([11n, 12n, 13n]);
    expect(rpc.reads.slice(3)).toEqual([13n, 11n, 12n, 13n]);
    chainSession.dispose();
  });

  it("fully refetches a regressed head", async () => {
    const rpc = fakeChain();
    const chainSession = session("ethereum", transport(rpc));

    await chainSession.sync();
    rpc.head = 11n;
    const recovered = await chainSession.sync();

    expect(recovered.blocks.map((block) => block.number)).toEqual([9n, 10n, 11n]);
    expect(rpc.reads.slice(3)).toEqual([9n, 10n, 11n]);
    chainSession.dispose();
  });

  it("fully refetches a changed same-height hash", async () => {
    let headReads = 0;
    const replacementHash = hashOf(12_000n);
    const rpc = fakeChain({
      block: (number) => {
        if (number === 12n) {
          headReads += 1;
          return rpcBlock(number, {
            hash: headReads === 1 ? hashOf(12n) : replacementHash,
          });
        }
        return rpcBlock(number);
      },
    });
    const chainSession = session("ethereum", transport(rpc));

    await chainSession.sync();
    const recovered = await chainSession.sync();

    expect(recovered.blocks.at(-1)?.hash).toBe(replacementHash);
    expect(headReads).toBe(3);
    chainSession.dispose();
  });

  it("rejects a block whose exact number or hash is invalid", async () => {
    let invalid: "number" | "hash" = "number";
    const rpc = fakeChain({
      head: 10n,
      block: () =>
        invalid === "number"
          ? rpcBlock(10n, { returnedNumber: 9n })
          : rpcBlock(10n, { hash: null }),
    });

    const wrongNumber = session("ethereum", transport(rpc), {
      contextBlocks: 1,
    });
    await expect(wrongNumber.sync()).rejects.toThrow(
      "RPC returned block 9 for requested block 10",
    );
    wrongNumber.dispose();

    invalid = "hash";
    const missingHash = session("ethereum", transport(rpc), {
      contextBlocks: 1,
    });
    await expect(missingHash.sync()).rejects.toThrow(
      "RPC returned block 10 without a hash",
    );
    missingHash.dispose();
  });

  it("returns both action-zero outcome values", async () => {
    const chainSession = session("ethereum", transport(fakeChain()));

    const outcome = await chainSession.readOutcome(20n, 20n);

    expect(outcome.immediateBaseFeePerGas).toBe(1_000_000_020n);
    expect(outcome.selectedBaseFeePerGas).toBe(1_000_000_020n);
    chainSession.dispose();
  });

  it("polls after settlement and never overlaps a slow head read", async () => {
    vi.useFakeTimers();
    let headReads = 0;
    let releaseHead: ((value: `0x${string}`) => void) | undefined;
    const firstHead = new Promise<`0x${string}`>((resolve) => {
      releaseHead = resolve;
    });
    const provider: RpcProvider = {
      async request(args) {
        if (args.method === "eth_chainId") return quantity(1n);
        if (args.method === "eth_blockNumber") {
          headReads += 1;
          return headReads === 1 ? firstHead : quantity(12n);
        }
        if (args.method === "eth_getBlockByNumber") {
          return rpcBlock(blockNumberFrom(args));
        }
        throw new Error(`Unexpected RPC method: ${args.method}`);
      },
    };
    const chainSession = session("ethereum", transport(provider));
    const publish = vi.fn();
    const stop = chainSession.startPolling(publish);

    await flushMicrotasks();
    expect(headReads).toBe(1);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(headReads).toBe(1);

    releaseHead?.(quantity(12n));
    await flushMicrotasks();
    await vi.advanceTimersByTimeAsync(999);
    expect(headReads).toBe(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(headReads).toBe(2);
    expect(publish).toHaveBeenCalledTimes(2);

    stop();
    chainSession.dispose();
  });

  it("keeps visible polling isolated from the inference context", async () => {
    const rpc = fakeChain();
    const chainSession = session("ethereum", transport(rpc));
    await chainSession.sync();
    rpc.head = 13n;

    await new Promise<void>((resolve, reject) => {
      const stop = chainSession.startPolling(
        () => {
          stop();
          resolve();
        },
        reject,
      );
    });

    rpc.head = 14n;
    const context = await chainSession.sync();

    expect(context.blocks.map((block) => block.number)).toEqual([12n, 13n, 14n]);
    expect(rpc.reads.filter((number) => number === 13n)).toHaveLength(2);
    chainSession.dispose();
  });

  it("suppresses a stale synchronization result after disposal", async () => {
    let releaseBlock: ((value: ReturnType<typeof rpcBlock>) => void) | undefined;
    const pendingBlock = new Promise<ReturnType<typeof rpcBlock>>((resolve) => {
      releaseBlock = resolve;
    });
    const provider: RpcProvider = {
      async request(args) {
        if (args.method === "eth_chainId") return quantity(1n);
        if (args.method === "eth_blockNumber") return quantity(10n);
        if (args.method === "eth_getBlockByNumber") return pendingBlock;
        throw new Error(`Unexpected RPC method: ${args.method}`);
      },
    };
    const chainSession = session("ethereum", transport(provider), {
      contextBlocks: 1,
    });

    const synchronization = chainSession.sync();
    await flushMicrotasks();
    chainSession.dispose();
    releaseBlock?.(rpcBlock(10n));

    await expect(synchronization).rejects.toMatchObject({ name: "AbortError" });
  });

  it("isolates production HTTP batches by session abort signal", async () => {
    type PendingChainId = {
      signal: AbortSignal;
      respond: () => void;
    };

    const pendingChainIds: PendingChainId[] = [];
    const responseFor = (
      body: readonly Record<string, unknown>[],
    ): Response =>
      new Response(
        JSON.stringify(
          body.map((request) => ({
            jsonrpc: "2.0",
            id: request.id,
            result:
              request.method === "eth_chainId"
                ? quantity(1n)
                : request.method === "eth_blockNumber"
                  ? quantity(10n)
                  : rpcBlock(
                      BigInt((request.params as readonly [string])[0]),
                    ),
          })),
        ),
        { headers: { "Content-Type": "application/json" } },
      );

    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const parsed = JSON.parse(String(init?.body)) as
          | Record<string, unknown>
          | Record<string, unknown>[];
        const body = Array.isArray(parsed) ? parsed : [parsed];
        const signal = init?.signal;
        if (!(signal instanceof AbortSignal)) {
          throw new Error("Production HTTP request is missing its session signal");
        }
        if (body.every((request) => request.method === "eth_chainId")) {
          return new Promise<Response>((resolve, reject) => {
            pendingChainIds.push({
              signal,
              respond: () => resolve(responseFor(body)),
            });
            signal.addEventListener(
              "abort",
              () => reject(signal.reason),
              { once: true },
            );
          });
        }
        return responseFor(body);
      }),
    );

    const first = createChainSession({
      chain: "ethereum",
      contextBlocks: 1,
      orderedFeatures: [],
    });
    const replacement = createChainSession({
      chain: "ethereum",
      contextBlocks: 1,
      orderedFeatures: [],
    });

    const discardedSynchronization = first.sync();
    const replacementSynchronization = replacement.sync();
    await vi.waitFor(() => expect(pendingChainIds).toHaveLength(2));

    expect(pendingChainIds[0].signal).not.toBe(pendingChainIds[1].signal);
    first.dispose();
    expect(pendingChainIds[0].signal.aborted).toBe(true);
    expect(pendingChainIds[1].signal.aborted).toBe(false);
    pendingChainIds[1].respond();

    await expect(discardedSynchronization).rejects.toMatchObject({
      name: "AbortError",
    });
    await expect(replacementSynchronization).resolves.toMatchObject({
      blocks: [expect.objectContaining({ number: 10n })],
    });
    replacement.dispose();
  });

  it("times out a hanging production HTTP fetch without disposing the session", async () => {
    vi.useFakeTimers();
    let abortedFetches = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const signal = init?.signal;
        if (!(signal instanceof AbortSignal)) {
          throw new Error("Production HTTP request is missing an abort signal");
        }
        return new Promise<Response>((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => {
              abortedFetches += 1;
              reject(signal.reason);
            },
            { once: true },
          );
        });
      }),
    );
    const chainSession = createChainSession({
      chain: "ethereum",
      contextBlocks: 1,
      orderedFeatures: [],
    });
    let result:
      | { status: "fulfilled" }
      | { status: "rejected"; error: unknown }
      | undefined;
    const completion = chainSession.sync().then(
      () => {
        result = { status: "fulfilled" };
      },
      (error: unknown) => {
        result = { status: "rejected", error };
      },
    );

    try {
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(9_999);
      expect(result).toBeUndefined();

      await vi.advanceTimersByTimeAsync(1);
      await flushMicrotasks();

      expect(result).toMatchObject({
        status: "rejected",
        error: { name: "TimeoutError" },
      });
      expect(abortedFetches).toBe(1);
    } finally {
      chainSession.dispose();
      await completion;
    }
  });
});
