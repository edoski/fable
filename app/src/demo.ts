import type { InferenceRun } from "./history";
import type { Chain, Horizon } from "./inference";

type DemoChain = {
  blockTimeSeconds: number;
  feeScale: number;
  headBlock: number;
  savingsShift: number;
};

type DemoRun = {
  horizon: Horizon;
  immediateGwei: number;
  minutesAgo: number;
  offset: number;
  predictedGwei: number;
  selectedGwei: number;
};

const CHAINS: Record<Chain, DemoChain> = {
  ethereum: {
    blockTimeSeconds: 12,
    feeScale: 1,
    headBlock: 25_400_000,
    savingsShift: 0,
  },
  polygon: {
    blockTimeSeconds: 2,
    feeScale: 3.1,
    headBlock: 78_200_000,
    savingsShift: 0.018,
  },
  avalanche: {
    blockTimeSeconds: 2,
    feeScale: 1.8,
    headBlock: 72_600_000,
    savingsShift: -0.012,
  },
};

const RUNS: readonly DemoRun[] = [
  {
    horizon: 5,
    offset: 2,
    minutesAgo: 9,
    immediateGwei: 14.2,
    selectedGwei: 12.4,
    predictedGwei: 12.1,
  },
  {
    horizon: 5,
    offset: 0,
    minutesAgo: 24,
    immediateGwei: 11.8,
    selectedGwei: 11.8,
    predictedGwei: 11.5,
  },
  {
    horizon: 5,
    offset: 4,
    minutesAgo: 51,
    immediateGwei: 18.6,
    selectedGwei: 14.9,
    predictedGwei: 14.4,
  },
  {
    horizon: 5,
    offset: 1,
    minutesAgo: 95,
    immediateGwei: 9.7,
    selectedGwei: 9.2,
    predictedGwei: 9.0,
  },
  {
    horizon: 5,
    offset: 3,
    minutesAgo: 180,
    immediateGwei: 16.3,
    selectedGwei: 15.7,
    predictedGwei: 15.4,
  },
  {
    horizon: 4,
    offset: 2,
    minutesAgo: 320,
    immediateGwei: 13.1,
    selectedGwei: 11.7,
    predictedGwei: 11.4,
  },
  {
    horizon: 4,
    offset: 1,
    minutesAgo: 520,
    immediateGwei: 15.6,
    selectedGwei: 14.1,
    predictedGwei: 13.8,
  },
  {
    horizon: 4,
    offset: 0,
    minutesAgo: 780,
    immediateGwei: 10.4,
    selectedGwei: 10.4,
    predictedGwei: 10.1,
  },
  {
    horizon: 4,
    offset: 3,
    minutesAgo: 980,
    immediateGwei: 19.4,
    selectedGwei: 16.1,
    predictedGwei: 15.7,
  },
  {
    horizon: 3,
    offset: 0,
    minutesAgo: 1_200,
    immediateGwei: 8.9,
    selectedGwei: 8.9,
    predictedGwei: 8.7,
  },
  {
    horizon: 3,
    offset: 1,
    minutesAgo: 1_450,
    immediateGwei: 17.2,
    selectedGwei: 17.5,
    predictedGwei: 16.9,
  },
  {
    horizon: 3,
    offset: 2,
    minutesAgo: 2_200,
    immediateGwei: 12.9,
    selectedGwei: 11.1,
    predictedGwei: 10.8,
  },
  {
    horizon: 2,
    offset: 0,
    minutesAgo: 3_000,
    immediateGwei: 10.7,
    selectedGwei: 10.7,
    predictedGwei: 10.5,
  },
  {
    horizon: 2,
    offset: 1,
    minutesAgo: 3_600,
    immediateGwei: 15.0,
    selectedGwei: 13.8,
    predictedGwei: 13.5,
  },
];

function wei(gwei: number, scale: number): number {
  return Math.round(gwei * scale * 1_000_000_000);
}

export function createDemoRuns(now = Date.now()): InferenceRun[] {
  return Object.entries(CHAINS).flatMap(([chainValue, network]) => {
    const chain = chainValue as Chain;
    return RUNS.map((spec, index) => {
      const ranAt = now - spec.minutesAgo * 60_000;
      const elapsedBlocks = Math.round(
        (spec.minutesAgo * 60) / network.blockTimeSeconds,
      );
      const headBlock = network.headBlock - elapsedBlocks;
      const savings =
        spec.offset === 0
          ? 0
          : (spec.immediateGwei - spec.selectedGwei) / spec.immediateGwei +
            network.savingsShift;
      const selectedGwei = spec.immediateGwei * (1 - savings);
      return {
        id: `demo:${chain}:${index}`,
        ran_at: new Date(ranAt).toISOString(),
        chain,
        K: spec.horizon,
        head_block: headBlock,
        selected_action_k: spec.offset,
        target_block: headBlock + 1 + spec.offset,
        predicted_minimum_base_fee_per_gas: wei(
          spec.predictedGwei,
          network.feeScale,
        ),
        outcome: {
          resolved_at: new Date(
            ranAt + (spec.offset + 1) * network.blockTimeSeconds * 1_000,
          ).toISOString(),
          immediate_base_fee_per_gas: wei(spec.immediateGwei, network.feeScale),
          selected_base_fee_per_gas: wei(selectedGwei, network.feeScale),
        },
      };
    });
  });
}
