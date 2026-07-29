import { createPublicClient, http } from "viem";
import type { Hash, Transport } from "viem";
import { avalanche, mainnet, polygon } from "viem/chains";

import { abortError } from "./abort";
import type { Chain } from "./domain";
import {
  INTERVAL_FEATURE,
  PRIORITY_FEE_FEATURES,
  type FeatureName,
  type PriorityFeeRewards,
} from "./features";
import { createSerialQueue } from "./serialQueue";

export type BlockRow = {
  number: bigint;
  hash: Hash;
  parentHash: Hash;
  timestamp: bigint;
  baseFeePerGas: bigint;
  gasUsed: bigint;
  gasLimit: bigint;
  transactionCount: number;
};

export type PreparedChainContext = {
  blocks: readonly BlockRow[];
  priorityFeeRewards: readonly PriorityFeeRewards[] | null;
};

export type ChainOutcome = {
  immediateBaseFeePerGas: bigint;
  selectedBaseFeePerGas: bigint;
};

export type ChainSessionConfig = {
  chain: Chain;
  contextBlocks: number;
  orderedFeatures: readonly FeatureName[];
};

export type ChainSession = {
  sync(): Promise<PreparedChainContext>;
  readOutcome(
    immediateBlock: bigint,
    selectedBlock: bigint,
  ): Promise<ChainOutcome>;
  startPolling(
    onBlock: (block: BlockRow) => void,
    onError?: (error: unknown) => void,
  ): () => void;
  dispose(): void;
};

const CHAIN_DEFINITIONS = {
  ethereum: mainnet,
  polygon,
  avalanche,
} as const;

const BLOCK_BATCH_SIZE = 40;
const POLL_INTERVAL_MS = 1_000;
const RPC_TIMEOUT_MS = 10_000;
const HASH_PATTERN = /^0x[0-9a-fA-F]{64}$/;

export function createChainSession(
  config: ChainSessionConfig,
  transportOverride?: Transport,
): ChainSession {
  const controller = new AbortController();
  const definition = CHAIN_DEFINITIONS[config.chain];
  const baseTransport =
    transportOverride ??
    http(undefined, {
      batch: { batchSize: BLOCK_BATCH_SIZE, wait: 0 },
      fetchFn: fetchWithTimeout,
      retryCount: 0,
      timeout: RPC_TIMEOUT_MS,
    });
  const transport = withSessionSignal(baseTransport, controller.signal);
  const client = createPublicClient({
    chain: definition,
    cacheTime: 0,
    transport,
  });
  const rawBlockCount =
    config.contextBlocks +
    Number(config.orderedFeatures.includes(INTERVAL_FEATURE));
  const needsFeeHistory = config.orderedFeatures.some((feature) =>
    PRIORITY_FEE_FEATURES.includes(feature),
  );

  let blocks: BlockRow[] = [];
  let disposed = false;
  let verified = false;
  let activePollStop: (() => void) | null = null;
  const serializeSync = createSerialQueue();

  function requireActive(): void {
    if (disposed) throw abortError("Chain session is disposed");
  }

  async function verifyChain(): Promise<void> {
    requireActive();
    if (verified) return;

    const chainId = await client.getChainId();
    requireActive();
    if (chainId !== definition.id) {
      throw new Error(
        `RPC chain ID ${chainId} does not match expected chain ID ${definition.id}`,
      );
    }
    verified = true;
  }

  async function readBlock(number: bigint): Promise<BlockRow> {
    const block = await client.getBlock({ blockNumber: number });
    requireActive();
    if (block.number !== number) {
      throw new Error(
        `RPC returned block ${String(block.number)} for requested block ${number}`,
      );
    }
    if (block.hash === null) {
      throw new Error(`RPC returned block ${number} without a hash`);
    }
    if (!HASH_PATTERN.test(block.hash)) {
      throw new Error(`RPC returned an invalid hash for block ${number}`);
    }
    if (!HASH_PATTERN.test(block.parentHash)) {
      throw new Error(`RPC returned an invalid parent hash for block ${number}`);
    }
    if (block.baseFeePerGas === null) {
      throw new Error(`RPC returned block ${number} without a base fee`);
    }
    return {
      number,
      hash: block.hash,
      parentHash: block.parentHash,
      timestamp: requireBigInt(block.timestamp, "timestamp", number),
      baseFeePerGas: block.baseFeePerGas,
      gasUsed: requireBigInt(block.gasUsed, "gas used", number),
      gasLimit: requireBigInt(block.gasLimit, "gas limit", number),
      transactionCount: block.transactions.length,
    };
  }

  async function readBlockRange(
    firstBlock: bigint,
    lastBlock: bigint,
  ): Promise<BlockRow[]> {
    return Promise.all(
      Array.from(
        { length: Number(lastBlock - firstBlock + 1n) },
        (_, offset) => readBlock(firstBlock + BigInt(offset)),
      ),
    );
  }

  async function fullFetch(head: bigint): Promise<readonly BlockRow[]> {
    const firstBlock = head - BigInt(rawBlockCount) + 1n;
    if (firstBlock < 0n) {
      throw new Error(
        `Head block ${head} cannot provide ${rawBlockCount} context blocks`,
      );
    }
    const fetched = await readBlockRange(firstBlock, head);
    const broken = findBrokenLink(fetched);
    if (broken !== null) {
      throw new Error(
        `Broken parent link between blocks ${broken[0].number} and ${broken[1].number}`,
      );
    }
    blocks = fetched;
    return blocks;
  }

  async function recoverWithFullFetch(
    head: bigint,
  ): Promise<readonly BlockRow[]> {
    blocks = [];
    return fullFetch(head);
  }

  async function synchronizeBlocks(
    head: bigint,
  ): Promise<readonly BlockRow[]> {
    if (blocks.length === 0) {
      return fullFetch(head);
    }

    const cachedHead = blocks[blocks.length - 1];
    if (head < cachedHead.number) {
      return recoverWithFullFetch(head);
    }
    if (head === cachedHead.number) {
      const currentHead = await readBlock(head);
      if (currentHead.hash !== cachedHead.hash) {
        return recoverWithFullFetch(head);
      }
      return blocks;
    }
    if (head - cachedHead.number >= BigInt(rawBlockCount)) {
      return recoverWithFullFetch(head);
    }

    const appended = await readBlockRange(cachedHead.number + 1n, head);
    if (
      appended[0].parentHash !== cachedHead.hash ||
      findBrokenLink(appended) !== null
    ) {
      return recoverWithFullFetch(head);
    }
    blocks = [...blocks, ...appended].slice(-rawBlockCount);
    return blocks;
  }

  async function readPriorityFeeRewards(
    head: bigint,
  ): Promise<readonly PriorityFeeRewards[] | null> {
    if (!needsFeeHistory) return null;

    const firstBlock = head - BigInt(config.contextBlocks) + 1n;
    if (firstBlock < 0n) {
      throw new Error(
        `Head block ${head} cannot provide ${config.contextBlocks} fee rows`,
      );
    }
    const history = await client.getFeeHistory({
      blockCount: config.contextBlocks,
      blockNumber: head,
      rewardPercentiles: [50, 90],
    });
    requireActive();
    if (
      history.oldestBlock !== firstBlock ||
      history.reward === undefined ||
      history.reward.length !== config.contextBlocks ||
      history.reward.some((row) => row.length !== 2)
    ) {
      throw new Error(
        `Fee history must exactly cover blocks ${firstBlock} through ${head}`,
      );
    }
    return history.reward.map((row) => [row[0], row[1]]);
  }

  async function synchronize(): Promise<PreparedChainContext> {
    await verifyChain();
    const head = await client.getBlockNumber();
    requireActive();
    const [contextBlocks, priorityFeeRewards] = await Promise.all([
      synchronizeBlocks(head),
      readPriorityFeeRewards(head),
    ]);
    requireActive();
    return { blocks: contextBlocks, priorityFeeRewards };
  }

  function sync(): Promise<PreparedChainContext> {
    requireActive();
    return serializeSync(synchronize);
  }

  async function readOutcome(
    immediateBlock: bigint,
    selectedBlock: bigint,
  ): Promise<ChainOutcome> {
    await verifyChain();

    const [immediate, selected] = await Promise.all([
      readBlock(immediateBlock),
      readBlock(selectedBlock),
    ]);
    requireActive();
    return {
      immediateBaseFeePerGas: immediate.baseFeePerGas,
      selectedBaseFeePerGas: selected.baseFeePerGas,
    };
  }

  function startPolling(
    onBlock: (block: BlockRow) => void,
    onError?: (error: unknown) => void,
  ): () => void {
    requireActive();
    activePollStop?.();

    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let visibleBlock: BlockRow | null = null;

    const stop = () => {
      stopped = true;
      if (timer !== undefined) clearTimeout(timer);
      if (activePollStop === stop) activePollStop = null;
    };

    const poll = async () => {
      try {
        await verifyChain();
        const head = await client.getBlockNumber();
        requireActive();
        if (head !== visibleBlock?.number) {
          visibleBlock = await readBlock(head);
          requireActive();
        }
        if (!stopped) onBlock(visibleBlock);
      } catch (error) {
        if (!stopped && !disposed) onError?.(error);
      } finally {
        if (!stopped && !disposed) {
          timer = setTimeout(() => {
            void poll();
          }, POLL_INTERVAL_MS);
        }
      }
    };

    activePollStop = stop;
    void poll();
    return stop;
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    activePollStop?.();
    controller.abort();
    blocks = [];
  }

  return {
    sync,
    readOutcome,
    startPolling,
    dispose,
  };
}

function withSessionSignal(
  transport: Transport,
  signal: AbortSignal,
): Transport {
  return (parameters) => {
    const configured = transport(parameters);
    const request: typeof configured.request = (args, options) =>
      configured.request(args, { ...options, signal });
    return { ...configured, request };
  };
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const sessionSignal = init?.signal;
  const abortFromSession = () => controller.abort(sessionSignal?.reason);
  if (sessionSignal?.aborted) {
    abortFromSession();
  } else {
    sessionSignal?.addEventListener("abort", abortFromSession, { once: true });
  }
  const timer = setTimeout(() => controller.abort(), RPC_TIMEOUT_MS);

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
    sessionSignal?.removeEventListener("abort", abortFromSession);
  }
}

function findBrokenLink(
  rows: readonly BlockRow[],
): readonly [BlockRow, BlockRow] | null {
  for (let index = 1; index < rows.length; index += 1) {
    const previous = rows[index - 1];
    const current = rows[index];
    if (
      current.number !== previous.number + 1n ||
      current.parentHash !== previous.hash
    ) {
      return [previous, current];
    }
  }
  return null;
}

function requireBigInt(
  value: unknown,
  name: string,
  blockNumber: bigint,
): bigint {
  if (typeof value !== "bigint") {
    throw new Error(`RPC returned block ${blockNumber} without ${name}`);
  }
  return value;
}
