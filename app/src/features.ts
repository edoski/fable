import type { BlockRow } from "./rpc";

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

export const P50_PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p50" satisfies FeatureName;
export const P90_PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p90" satisfies FeatureName;
export const PRIORITY_FEE_FEATURES: readonly FeatureName[] = [
  P50_PRIORITY_FEE_FEATURE,
  P90_PRIORITY_FEE_FEATURE,
];
export const INTERVAL_FEATURE =
  "block_interval_seconds" satisfies FeatureName;

export type PriorityFeeRewards = readonly [p50: bigint, p90: bigint];

export function buildModelInput(
  blocks: readonly BlockRow[],
  priorityFeeRewards: readonly PriorityFeeRewards[] | null,
  manifest: ChainManifest,
): Float32Array {
  const needsPredecessor = manifest.features.some(
    (feature) => feature.name === INTERVAL_FEATURE,
  );
  const featureCount = manifest.features.length;
  const output = new Float32Array(manifest.context_blocks * featureCount);

  for (let row = 0; row < manifest.context_blocks; row += 1) {
    const blockIndex = row + Number(needsPredecessor);
    const block = blocks[blockIndex];
    for (let column = 0; column < featureCount; column += 1) {
      const feature = manifest.features[column];
      const raw = rawFeature(
        feature.name,
        block,
        needsPredecessor ? blocks[blockIndex - 1] : null,
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
  predecessor: BlockRow | null,
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
    case P50_PRIORITY_FEE_FEATURE: {
      const reward = priorityFeeRewards?.[0];
      if (reward === undefined || reward < 0n) {
        throw new Error("P50 priority fee must be present and nonnegative");
      }
      return Math.log1p(Number(reward));
    }
    case P90_PRIORITY_FEE_FEATURE: {
      const reward = priorityFeeRewards?.[1];
      if (reward === undefined || reward < 0n) {
        throw new Error("P90 priority fee must be present and nonnegative");
      }
      return Math.log1p(Number(reward));
    }
    case INTERVAL_FEATURE: {
      if (predecessor === null) {
        throw new Error("block_interval_seconds requires a predecessor block");
      }
      const interval = block.timestamp - predecessor.timestamp;
      if (interval < 0n) {
        throw new Error("block_interval_seconds values must be nonnegative");
      }
      return Number(interval);
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
  if (block.gasLimit <= 0n) {
    throw new Error("gasLimit must be positive");
  }
  if (block.gasUsed < 0n || block.gasUsed > block.gasLimit) {
    throw new Error("gasUsed must be between zero and gasLimit");
  }
  return (
    Number(block.gasUsed) /
    Number(block.gasLimit)
  );
}

function formingChildBaseFee(block: BlockRow): bigint {
  const gasTarget = block.gasLimit / 2n;
  if (gasTarget <= 0n) {
    throw new Error("gas target must be positive");
  }
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
