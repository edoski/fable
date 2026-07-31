import type { BlockRow } from "./domain";

export const FEATURE_NAMES = [
  "log_base_fee_per_gas",
  "gas_utilization",
  "log_exact_forming_base_fee_per_gas",
  "log_gas_limit",
  "log1p_tx_count",
  "log1p_effective_priority_fee_per_gas_p50",
  "log1p_effective_priority_fee_per_gas_p90",
  "block_interval_seconds",
  "hour_sin",
  "hour_cos",
  "dow_sin",
  "dow_cos",
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];

export type FeatureManifest = {
  name: FeatureName;
  mean: number;
  standard_deviation: number;
};

export type ChainManifest = {
  context_blocks: number;
  features: readonly FeatureManifest[];
};

const P50_PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p50" satisfies FeatureName;
const P90_PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p90" satisfies FeatureName;
const INTERVAL_FEATURE =
  "block_interval_seconds" satisfies FeatureName;

export type PriorityFeeRewards = readonly [p50: bigint, p90: bigint];

export function needsPredecessor(manifest: ChainManifest): boolean {
  return manifest.features.some(
    (feature) => feature.name === INTERVAL_FEATURE,
  );
}

export function usesPriorityFees(manifest: ChainManifest): boolean {
  return manifest.features.some(
    (feature) =>
      feature.name === P50_PRIORITY_FEE_FEATURE ||
      feature.name === P90_PRIORITY_FEE_FEATURE,
  );
}

export function buildModelInput(
  blocks: readonly BlockRow[],
  priorityFeeRewards: readonly PriorityFeeRewards[] | null,
  manifest: ChainManifest,
): Float32Array {
  if (
    usesPriorityFees(manifest) &&
    (priorityFeeRewards === null ||
      priorityFeeRewards.length !== manifest.context_blocks ||
      priorityFeeRewards.some(
        (rewards) =>
          rewards.length !== 2 ||
          Array.from(rewards).some(
            (reward) =>
              typeof reward !== "bigint" || reward < 0n,
          ),
      ))
  ) {
    throw new Error(
      "Priority-fee rewards must provide nonnegative P50 and P90 values for every context block",
    );
  }
  const predecessorOffset = Number(needsPredecessor(manifest));
  const featureCount = manifest.features.length;
  const output = new Float32Array(manifest.context_blocks * featureCount);

  for (let row = 0; row < manifest.context_blocks; row += 1) {
    const blockIndex = row + predecessorOffset;
    const block = blocks[blockIndex];
    for (let column = 0; column < featureCount; column += 1) {
      const feature = manifest.features[column];
      const raw = rawFeature(
        feature.name,
        block,
        blocks[blockIndex - 1],
        priorityFeeRewards?.[row],
      );
      const index = row * featureCount + column;
      output[index] =
        (raw - feature.mean) / feature.standard_deviation;
      if (!Number.isFinite(output[index])) {
        throw new Error("Model input must contain finite float32 values");
      }
    }
  }

  return output;
}

function rawFeature(
  feature: FeatureName,
  block: BlockRow,
  predecessor: BlockRow,
  priorityFeeRewards: PriorityFeeRewards | undefined,
): number {
  switch (feature) {
    case "log_base_fee_per_gas":
      return positiveLog(block.baseFeePerGas);
    case "gas_utilization":
      return gasUtilization(block);
    case "log_exact_forming_base_fee_per_gas":
      return positiveLog(formingChildBaseFee(block));
    case "log_gas_limit":
      return positiveLog(block.gasLimit);
    case "log1p_tx_count":
      return Math.log1p(block.transactionCount);
    case P50_PRIORITY_FEE_FEATURE:
      return Math.log1p(Number(priorityFeeRewards![0]));
    case P90_PRIORITY_FEE_FEATURE:
      return Math.log1p(Number(priorityFeeRewards![1]));
    case INTERVAL_FEATURE: {
      return Number(block.timestamp - predecessor.timestamp);
    }
    case "hour_sin":
      return Math.sin(hourAngle(block.timestamp));
    case "hour_cos":
      return Math.cos(hourAngle(block.timestamp));
    case "dow_sin":
      return Math.sin(dayOfWeekAngle(block.timestamp));
    case "dow_cos":
      return Math.cos(dayOfWeekAngle(block.timestamp));
  }
}

function positiveLog(value: bigint): number {
  return Math.log(Number(value));
}

function gasUtilization(block: BlockRow): number {
  return Number(block.gasUsed) / Number(block.gasLimit);
}

function formingChildBaseFee(block: BlockRow): bigint {
  const gasTarget = block.gasLimit / 2n;
  if (block.gasUsed === gasTarget) {
    return block.baseFeePerGas;
  }
  if (block.gasUsed > gasTarget) {
    const increase =
      (block.baseFeePerGas * (block.gasUsed - gasTarget)) /
      gasTarget /
      8n;
    return block.baseFeePerGas + (increase > 0n ? increase : 1n);
  }
  const decrease =
    (block.baseFeePerGas * (gasTarget - block.gasUsed)) /
    gasTarget /
    8n;
  return block.baseFeePerGas - decrease;
}

function hourAngle(timestamp: bigint): number {
  const hour = Number((timestamp / 3_600n) % 24n);
  return (2 * Math.PI * hour) / 24;
}

function dayOfWeekAngle(timestamp: bigint): number {
  const day = Number((timestamp / 86_400n + 4n) % 7n);
  return (2 * Math.PI * day) / 7;
}
