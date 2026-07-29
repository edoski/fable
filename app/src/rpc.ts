import { createPublicClient, http } from "viem";
import type { GetBlockReturnType } from "viem";
import { avalanche, mainnet, polygon } from "viem/chains";

import type { BlockRow, Chain } from "./domain";
import {
  INTERVAL_FEATURE,
  PRIORITY_FEE_FEATURES,
  type FeatureName,
  type PriorityFeeRewards,
} from "./features";

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
  watchBlocks(
    onBlock: (block: BlockRow) => void,
    onError?: (error: unknown) => void,
  ): void;
  dispose(): void;
};

const CHAIN_DEFINITIONS = {
  ethereum: mainnet,
  polygon,
  avalanche,
} as const;

const BLOCK_BATCH_SIZE = 40;
const RPC_TIMEOUT_MS = 10_000;

export function createChainSession(config: ChainSessionConfig): ChainSession {
  const definition = CHAIN_DEFINITIONS[config.chain];
  const client = createPublicClient({
    chain: definition,
    cacheTime: 0,
    transport: http(undefined, {
      batch: { batchSize: BLOCK_BATCH_SIZE, wait: 0 },
      retryCount: 0,
      timeout: RPC_TIMEOUT_MS,
    }),
  });
  const needsPredecessor = config.orderedFeatures.includes(INTERVAL_FEATURE);
  const needsFeeHistory = config.orderedFeatures.some((feature) =>
    PRIORITY_FEE_FEATURES.includes(feature),
  );
  let unwatch: (() => void) | undefined;

  function blockRow(
    block: GetBlockReturnType<typeof definition, false, "latest">,
  ): BlockRow {
    if (block.baseFeePerGas === null) {
      throw new Error(`RPC returned block ${block.number} without a base fee`);
    }
    return {
      number: block.number,
      hash: block.hash,
      parentHash: block.parentHash,
      timestamp: block.timestamp,
      baseFeePerGas: block.baseFeePerGas,
      gasUsed: block.gasUsed,
      gasLimit: block.gasLimit,
      transactionCount: block.transactions.length,
    };
  }

  async function readBlock(number: bigint): Promise<BlockRow> {
    return blockRow(await client.getBlock({ blockNumber: number }));
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

  async function readPriorityFeeRewards(
    head: bigint,
    firstBlock: bigint,
  ): Promise<readonly PriorityFeeRewards[] | null> {
    if (!needsFeeHistory) return null;

    const history = await client.getFeeHistory({
      blockCount: config.contextBlocks,
      blockNumber: head,
      rewardPercentiles: [50, 90],
    });
    if (history.oldestBlock !== firstBlock) {
      throw new Error(
        `Fee history must start at block ${firstBlock}, got ${history.oldestBlock}`,
      );
    }
    return (
      history.reward?.map(([p50, p90]) => [p50, p90] as const) ?? []
    );
  }

  async function sync(): Promise<PreparedChainContext> {
    const head = await client.getBlockNumber();
    const firstContextBlock =
      head - BigInt(config.contextBlocks) + 1n;
    const firstRawBlock =
      firstContextBlock - BigInt(needsPredecessor);
    const [blocks, priorityFeeRewards] = await Promise.all([
      readBlockRange(firstRawBlock, head),
      readPriorityFeeRewards(head, firstContextBlock),
    ]);
    const broken = findBrokenLink(blocks);
    if (broken !== null) {
      throw new Error(
        `Broken parent link between blocks ${broken[0].number} and ${broken[1].number}`,
      );
    }
    return { blocks, priorityFeeRewards };
  }

  async function readOutcome(
    immediateBlock: bigint,
    selectedBlock: bigint,
  ): Promise<ChainOutcome> {
    const [immediate, selected] = await Promise.all([
      readBlock(immediateBlock),
      readBlock(selectedBlock),
    ]);
    return {
      immediateBaseFeePerGas: immediate.baseFeePerGas,
      selectedBaseFeePerGas: selected.baseFeePerGas,
    };
  }

  function watchBlocks(
    onBlock: (block: BlockRow) => void,
    onError?: (error: unknown) => void,
  ): void {
    unwatch?.();
    unwatch = client.watchBlocks({
      emitOnBegin: true,
      onBlock: (block) => onBlock(blockRow(block)),
      onError,
    });
  }

  function dispose(): void {
    unwatch?.();
    unwatch = undefined;
  }

  return {
    sync,
    readOutcome,
    watchBlocks,
    dispose,
  };
}

function findBrokenLink(
  rows: readonly BlockRow[],
): readonly [BlockRow, BlockRow] | null {
  for (let index = 1; index < rows.length; index += 1) {
    const previous = rows[index - 1];
    const current = rows[index];
    if (current.parentHash !== previous.hash) {
      return [previous, current];
    }
  }
  return null;
}
