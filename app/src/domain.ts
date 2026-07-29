import type { Hash } from "viem";

export const CHAINS = ["ethereum", "polygon", "avalanche"] as const;
export type Chain = (typeof CHAINS)[number];

export const HORIZONS = [2, 3, 4, 5] as const;
export type Horizon = (typeof HORIZONS)[number];

export const CHAIN_LABELS: Record<Chain, string> = {
  ethereum: "Ethereum",
  polygon: "Polygon",
  avalanche: "Avalanche",
};

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
