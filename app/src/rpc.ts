import { createPublicClient, http } from "viem";
import type { GetBlockReturnType } from "viem";
import { avalanche, mainnet, polygon } from "viem/chains";

import type { BlockRow, Chain } from "./domain";
import {
  needsPredecessor,
  usesPriorityFees,
  type ChainManifest,
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

export function createChainSession(
  chain: Chain,
  manifest: ChainManifest,
): ChainSession {
  const definition = CHAIN_DEFINITIONS[chain];
  const client = createPublicClient({
    chain: definition,
    cacheTime: 0,
    transport: http(undefined, {
      batch: { batchSize: BLOCK_BATCH_SIZE, wait: 0 },
      retryCount: 0,
      timeout: RPC_TIMEOUT_MS,
    }),
  });
  const predecessorOffset = Number(needsPredecessor(manifest));
  const needsFeeHistory = usesPriorityFees(manifest);
  let unwatch: (() => void) | undefined;

  function blockRow(
    block: GetBlockReturnType<typeof definition, false, "latest">,
  ): BlockRow {
    if (block.baseFeePerGas === null || block.baseFeePerGas <= 0n) {
      throw new Error(
        `RPC returned block ${block.number} without a positive base fee`,
      );
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
      blockCount: manifest.context_blocks,
      blockNumber: head,
      rewardPercentiles: [50, 90],
    });
    if (history.oldestBlock !== firstBlock) {
      throw new Error(
        `Fee history must start at block ${firstBlock}, got ${history.oldestBlock}`,
      );
    }
    if (history.reward === undefined) {
      throw new Error("Fee history must include priority-fee rewards");
    }
    if (history.reward.length !== manifest.context_blocks) {
      throw new Error(
        `Fee history must contain exactly ${manifest.context_blocks} reward rows, got ${history.reward.length}`,
      );
    }
    return history.reward.map((rewards) => {
      const [p50, p90] = rewards;
      if (
        rewards.length !== 2 ||
        typeof p50 !== "bigint" ||
        typeof p90 !== "bigint" ||
        p50 < 0n ||
        p90 < 0n
      ) {
        throw new Error(
          "Fee history reward rows must contain nonnegative P50 and P90 values",
        );
      }
      return [p50, p90] as const;
    });
  }

  async function sync(): Promise<PreparedChainContext> {
    const head = await client.getBlockNumber();
    const firstContextBlock =
      head - BigInt(manifest.context_blocks) + 1n;
    const firstRawBlock =
      firstContextBlock - BigInt(predecessorOffset);
    const [blocks, priorityFeeRewards] = await Promise.all([
      readBlockRange(firstRawBlock, head),
      readPriorityFeeRewards(head, firstContextBlock),
    ]);
    for (let index = 1; index < blocks.length; index += 1) {
      const previous = blocks[index - 1];
      const current = blocks[index];
      if (current.parentHash !== previous.hash) {
        throw new Error(
          `Broken parent link between blocks ${previous.number} and ${current.number}`,
        );
      }
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
