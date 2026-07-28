import type { BlockRow } from "./rpc";

export const FEATURE_NAMES = [
  "log_base_fee_per_gas",
  "gas_utilization",
  "log_exact_forming_base_fee_per_gas",
  "log_gas_limit",
  "log1p_tx_count",
  "log1p_effective_priority_fee_per_gas_p50",
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

export const PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p50" satisfies FeatureName;
export const INTERVAL_FEATURE =
  "block_interval_seconds" satisfies FeatureName;

export function buildModelInput(
  blocks: readonly BlockRow[],
  p50Rewards: readonly bigint[] | null,
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
        p50Rewards?.[row],
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
  reward: bigint | undefined,
): number {
  switch (feature) {
    case "log_base_fee_per_gas":
      return positiveLog(block.baseFeePerGas, "baseFeePerGas");
    case "gas_utilization":
      return gasUtilization(block);
    case "log_exact_forming_base_fee_per_gas":
      return positiveLog(formingChildBaseFee(block), "forming base fee");
    case "log_gas_limit":
      return positiveLog(block.gasLimit, "gasLimit");
    case "log1p_tx_count":
      return Math.log1p(block.transactionCount);
    case PRIORITY_FEE_FEATURE: {
      if (reward === undefined || reward < 0n) {
        throw new Error("P50 priority fee must be present and nonnegative");
      }
      return Math.log1p(safeNumber(reward, "P50 priority fee"));
    }
    case INTERVAL_FEATURE: {
      if (predecessor === null) {
        throw new Error("block_interval_seconds requires a predecessor block");
      }
      const interval = block.timestamp - predecessor.timestamp;
      if (interval < 0n) {
        throw new Error("block_interval_seconds values must be nonnegative");
      }
      return safeNumber(interval, "block interval");
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

function positiveLog(value: bigint, name: string): number {
  return Math.log(safeNumber(value, name));
}

function gasUtilization(block: BlockRow): number {
  if (block.gasLimit <= 0n) {
    throw new Error("gasLimit must be positive");
  }
  if (block.gasUsed < 0n || block.gasUsed > block.gasLimit) {
    throw new Error("gasUsed must be between zero and gasLimit");
  }
  return (
    safeNumber(block.gasUsed, "gasUsed") /
    safeNumber(block.gasLimit, "gasLimit")
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

function safeNumber(value: bigint, name: string): number {
  const converted = Number(value);
  if (!Number.isSafeInteger(converted)) {
    throw new Error(`${name} exceeds the safe integer range`);
  }
  return converted;
}
