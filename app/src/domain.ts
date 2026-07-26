export const CHAINS = ["ethereum", "polygon", "avalanche"] as const;
export type Chain = (typeof CHAINS)[number];

export const HORIZONS = [2, 3, 4, 5] as const;
export type Horizon = (typeof HORIZONS)[number];

export const CHAIN_DETAILS: Record<Chain, { label: string }> = {
  ethereum: { label: "Ethereum" },
  polygon: { label: "Polygon" },
  avalanche: { label: "Avalanche" },
};
