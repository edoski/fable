import { createPublicClient, http } from "viem";
import type {
  FeeHistory as ViemFeeHistory,
  Hash,
  Transport,
} from "viem";
import { avalanche, mainnet, polygon } from "viem/chains";

export const SUPPORTED_CHAINS = [
  "ethereum",
  "polygon",
  "avalanche",
] as const;

export type SupportedChain = (typeof SUPPORTED_CHAINS)[number];
export type FeeHistory = ViemFeeHistory;

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
  head: bigint;
  blocks: readonly BlockRow[];
  feeHistory: FeeHistory | null;
};

export type ChainOutcome = {
  immediateBlock: bigint;
  selectedBlock: bigint;
  immediateBaseFeePerGas: bigint;
  selectedBaseFeePerGas: bigint;
};

export type ChainSessionConfig = {
  chain: SupportedChain;
  rpcUrl: string;
  contextBlocks: number;
  orderedFeatures: readonly string[];
};

export type ChainSession = {
  sync(): Promise<PreparedChainContext>;
  readSnapshot(): Promise<BlockRow>;
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
const PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p50";
const INTERVAL_FEATURE = "block_interval_seconds";
const HASH_PATTERN = /^0x[0-9a-fA-F]{64}$/;

export function defaultRpcUrl(chain: SupportedChain): string {
  return CHAIN_DEFINITIONS[chain].rpcUrls.default.http[0];
}

export function createChainSession(
  config: ChainSessionConfig,
  transportOverride?: Transport,
): ChainSession {
  validateConfig(config);

  const controller = new AbortController();
  const definition = CHAIN_DEFINITIONS[config.chain];
  const baseTransport =
    transportOverride ??
    http(config.rpcUrl, {
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
  const needsFeeHistory = config.orderedFeatures.includes(
    PRIORITY_FEE_FEATURE,
  );

  let blocks: BlockRow[] = [];
  let disposed = false;
  let verified = false;
  let verification: Promise<void> | null = null;
  let activePollStop: (() => void) | null = null;
  let synchronizations: Promise<void> = Promise.resolve();

  function requireActive(): void {
    if (disposed) throw abortError();
  }

  async function verifyChain(): Promise<void> {
    requireActive();
    if (verified) return;

    const current =
      verification ??
      (verification = (async () => {
        const chainId = await client.getChainId();
        requireActive();
        if (chainId !== definition.id) {
          throw new Error(
            `RPC chain ID ${chainId} does not match expected chain ID ${definition.id}`,
          );
        }
      })());
    try {
      await current;
    } catch (error) {
      if (verification === current) verification = null;
      throw error;
    }
    requireActive();
    verified = true;
    if (verification === current) verification = null;
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
    const rows: BlockRow[] = [];
    let nextBlock = firstBlock;
    while (nextBlock <= lastBlock) {
      const numbers: bigint[] = [];
      while (
        numbers.length < BLOCK_BATCH_SIZE &&
        nextBlock <= lastBlock
      ) {
        numbers.push(nextBlock);
        nextBlock += 1n;
      }
      rows.push(...(await Promise.all(numbers.map(readBlock))));
      requireActive();
    }
    return rows;
  }

  async function fullFetch(head: bigint): Promise<readonly BlockRow[]> {
    const firstBlock = head - BigInt(rawBlockCount) + 1n;
    if (firstBlock < 0n) {
      throw new Error(
        `Head block ${head} cannot provide ${rawBlockCount} context blocks`,
      );
    }
    const fetched = await readBlockRange(firstBlock, head);
    validateLinks(fetched);
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

  async function readFeeHistory(
    head: bigint,
  ): Promise<FeeHistory | null> {
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
      rewardPercentiles: [50],
    });
    requireActive();
    if (
      history.oldestBlock !== firstBlock ||
      history.reward === undefined ||
      history.reward.length !== config.contextBlocks ||
      history.reward.some((row) => row.length !== 1)
    ) {
      throw new Error(
        `Fee history must exactly cover blocks ${firstBlock} through ${head}`,
      );
    }
    return history;
  }

  async function synchronize(): Promise<PreparedChainContext> {
    await verifyChain();
    const head = await client.getBlockNumber();
    requireActive();
    const [contextBlocks, feeHistory] = await Promise.all([
      synchronizeBlocks(head),
      readFeeHistory(head),
    ]);
    requireActive();
    return { head, blocks: contextBlocks, feeHistory };
  }

  function sync(): Promise<PreparedChainContext> {
    requireActive();
    const result = synchronizations.then(synchronize, synchronize);
    synchronizations = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  async function readSnapshot(): Promise<BlockRow> {
    await verifyChain();
    const head = await client.getBlockNumber();
    requireActive();
    return readBlock(head);
  }

  async function readOutcome(
    immediateBlock: bigint,
    selectedBlock: bigint,
  ): Promise<ChainOutcome> {
    requireBlockNumber(immediateBlock, "immediateBlock");
    requireBlockNumber(selectedBlock, "selectedBlock");
    await verifyChain();

    let immediate: BlockRow;
    let selected: BlockRow;
    if (immediateBlock === selectedBlock) {
      immediate = await readBlock(immediateBlock);
      selected = immediate;
    } else {
      [immediate, selected] = await Promise.all([
        readBlock(immediateBlock),
        readBlock(selectedBlock),
      ]);
    }
    requireActive();
    return {
      immediateBlock,
      selectedBlock,
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
    readSnapshot,
    readOutcome,
    startPolling,
    dispose,
  };
}

function validateConfig(config: ChainSessionConfig): void {
  if (config.rpcUrl.trim() === "") {
    throw new Error("rpcUrl must be nonempty");
  }
  if (
    !Number.isSafeInteger(config.contextBlocks) ||
    config.contextBlocks <= 0
  ) {
    throw new Error("contextBlocks must be a positive safe integer");
  }
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

function validateLinks(rows: readonly BlockRow[]): void {
  const broken = findBrokenLink(rows);
  if (broken !== null) {
    throw new Error(
      `Broken parent link between blocks ${broken[0].number} and ${broken[1].number}`,
    );
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

function requireBlockNumber(value: bigint, name: string): void {
  if (value < 0n) {
    throw new Error(`${name} must be nonnegative`);
  }
}

function abortError(): Error {
  const error = new Error("Chain session is disposed");
  error.name = "AbortError";
  return error;
}
