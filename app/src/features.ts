import type { BlockRow, FeeHistory } from "./rpc";

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
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];

export type FeatureManifest = {
  name: FeatureName;
  mean: number;
  standard_deviation: number;
};

export type ChainManifest = {
  chain_id: number;
  context_blocks: number;
  features: readonly FeatureManifest[];
};

const PRIORITY_FEE_FEATURE =
  "log1p_effective_priority_fee_per_gas_p50";
const INTERVAL_FEATURE = "block_interval_seconds";
const FORMING_FEE_FEATURE = "log_exact_forming_base_fee_per_gas";

export function buildModelInput(
  blocks: readonly BlockRow[],
  feeHistory: FeeHistory | null,
  manifest: ChainManifest,
): Float32Array {
  validateManifest(manifest);

  const needsPredecessor = manifest.features.some(
    (feature) => feature.name === INTERVAL_FEATURE,
  );
  const expectedBlocks = manifest.context_blocks + Number(needsPredecessor);
  if (blocks.length !== expectedBlocks) {
    throw new Error(
      `Model input requires exactly ${expectedBlocks} blocks, got ${blocks.length}`,
    );
  }

  const rewards = feeRewards(blocks, feeHistory, manifest, needsPredecessor);
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
        rewards?.[row],
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

function validateManifest(manifest: ChainManifest): void {
  if (
    !Number.isSafeInteger(manifest.context_blocks) ||
    manifest.context_blocks <= 0
  ) {
    throw new Error("context_blocks must be a positive safe integer");
  }
  if (manifest.features.length === 0) {
    throw new Error("Manifest must select at least one feature");
  }

  const supported = new Set<string>(FEATURE_NAMES);
  const selected = new Set<string>();
  for (const feature of manifest.features) {
    if (!supported.has(feature.name)) {
      throw new Error(`Unsupported feature: ${feature.name}`);
    }
    if (selected.has(feature.name)) {
      throw new Error(`Duplicate feature: ${feature.name}`);
    }
    selected.add(feature.name);
    if (
      !Number.isFinite(feature.mean) ||
      !Number.isFinite(feature.standard_deviation) ||
      feature.standard_deviation <= 0
    ) {
      throw new Error(`Invalid normalization for feature: ${feature.name}`);
    }
  }

  if (selected.has(FORMING_FEE_FEATURE) && manifest.chain_id !== 1) {
    throw new Error(`${FORMING_FEE_FEATURE} is Ethereum-only`);
  }
}

function feeRewards(
  blocks: readonly BlockRow[],
  feeHistory: FeeHistory | null,
  manifest: ChainManifest,
  needsPredecessor: boolean,
): readonly (readonly bigint[])[] | null {
  const needsPriorityFee = manifest.features.some(
    (feature) => feature.name === PRIORITY_FEE_FEATURE,
  );
  if (!needsPriorityFee) return null;

  const rewards = feeHistory?.reward;
  const firstModelBlock = blocks[Number(needsPredecessor)].number;
  if (
    feeHistory === null ||
    feeHistory.oldestBlock !== firstModelBlock ||
    rewards === undefined ||
    rewards.length !== manifest.context_blocks ||
    rewards.some((row) => row.length !== 1)
  ) {
    throw new Error("Fee history must exactly cover model rows");
  }
  return rewards;
}

function rawFeature(
  feature: FeatureName,
  block: BlockRow,
  predecessor: BlockRow | null,
  reward: readonly bigint[] | undefined,
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
      if (
        !Number.isSafeInteger(block.transactionCount) ||
        block.transactionCount < 0
      ) {
        throw new Error("transactionCount must be a nonnegative safe integer");
      }
      return Math.log1p(block.transactionCount);
    case PRIORITY_FEE_FEATURE: {
      const value = reward?.[0];
      if (value === undefined || value < 0n) {
        throw new Error("P50 priority fee must be present and nonnegative");
      }
      return Math.log1p(safeNumber(value, "P50 priority fee"));
    }
    case INTERVAL_FEATURE: {
      if (predecessor === null) {
        throw new Error("block_interval_seconds requires a predecessor block");
      }
      const interval = block.timestamp - predecessor.timestamp;
      if (interval <= 0n) {
        throw new Error("block_interval_seconds values must be positive");
      }
      return safeNumber(interval, "block interval");
    }
    case "hour_sin":
      return Math.sin(hourAngle(block.timestamp));
    case "hour_cos":
      return Math.cos(hourAngle(block.timestamp));
  }
}

function positiveLog(value: bigint, name: string): number {
  if (value <= 0n) {
    throw new Error(`${name} must be positive`);
  }
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
  if (timestamp < 0n) {
    throw new Error("timestamp must be nonnegative");
  }
  const hour = Number((timestamp / 3_600n) % 24n);
  return (2 * Math.PI * hour) / 24;
}

function safeNumber(value: bigint, name: string): number {
  const converted = Number(value);
  if (!Number.isSafeInteger(converted)) {
    throw new Error(`${name} exceeds the safe integer range`);
  }
  return converted;
}
