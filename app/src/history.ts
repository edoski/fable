import AsyncStorage from "@react-native-async-storage/async-storage";

import type {
  Chain,
  Horizon,
  InferenceOutcome,
  InferenceRequest,
  InferenceResponse,
} from "./inference";

const STORAGE_KEY = "fable.inference-runs-v3";
const MAX_RUNS = 100;

export type InferenceRun = InferenceRequest &
  InferenceResponse & {
    id: string;
    ran_at: string;
    outcome?: RunOutcome;
  };

export type RunOutcome = {
  resolved_at: string;
  immediate_base_fee_per_gas: number;
  selected_base_fee_per_gas: number;
};

function isChain(value: unknown): value is Chain {
  return value === "ethereum" || value === "polygon" || value === "avalanche";
}

function isHorizon(value: unknown): value is Horizon {
  return value === 2 || value === 3 || value === 4 || value === 5;
}

function requireInteger(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error(`Stored ${name} must be a nonnegative integer`);
  }
  return value;
}

function parseOutcome(value: unknown): RunOutcome | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Stored inference outcome must be an object");
  }
  const outcome = value as Record<string, unknown>;
  if (typeof outcome.resolved_at !== "string") {
    throw new Error("Stored inference outcome identity is invalid");
  }
  const immediateBaseFee = requireInteger(
    outcome.immediate_base_fee_per_gas,
    "immediate base fee",
  );
  const selectedBaseFee = requireInteger(
    outcome.selected_base_fee_per_gas,
    "selected base fee",
  );
  if (immediateBaseFee === 0 || selectedBaseFee === 0) {
    throw new Error("Stored inference outcome base fees must be positive");
  }
  return {
    resolved_at: outcome.resolved_at,
    immediate_base_fee_per_gas: immediateBaseFee,
    selected_base_fee_per_gas: selectedBaseFee,
  };
}

function parseRun(value: unknown): InferenceRun {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Stored inference run must be an object");
  }
  const run = value as Record<string, unknown>;
  if (typeof run.id !== "string" || typeof run.ran_at !== "string") {
    throw new Error("Stored inference run identity is invalid");
  }
  if (!isChain(run.chain) || !isHorizon(run.K)) {
    throw new Error("Stored inference run selection is invalid");
  }
  const headBlock = requireInteger(run.head_block, "head block");
  const selectedAction = requireInteger(
    run.selected_action_k,
    "selected action",
  );
  const targetBlock = requireInteger(run.target_block, "target block");
  const predictedMinimum = run.predicted_minimum_base_fee_per_gas;
  if (
    typeof predictedMinimum !== "number" ||
    !Number.isFinite(predictedMinimum) ||
    predictedMinimum <= 0
  ) {
    throw new Error(
      "Stored predicted minimum base fee must be positive and finite",
    );
  }
  if (
    selectedAction >= run.K ||
    targetBlock !== headBlock + 1 + selectedAction
  ) {
    throw new Error("Stored inference run geometry is invalid");
  }
  return {
    id: run.id,
    ran_at: run.ran_at,
    chain: run.chain,
    K: run.K,
    head_block: headBlock,
    selected_action_k: selectedAction,
    target_block: targetBlock,
    predicted_minimum_base_fee_per_gas: predictedMinimum,
    outcome: parseOutcome(run.outcome),
  };
}

export function createRun(
  request: InferenceRequest,
  response: InferenceResponse,
): InferenceRun {
  const ranAt = new Date().toISOString();
  return {
    id: `${ranAt}:${request.chain}:${request.K}:${response.head_block}`,
    ran_at: ranAt,
    ...request,
    ...response,
  };
}

export function recordOutcome(
  run: InferenceRun,
  outcome: InferenceOutcome,
): InferenceRun {
  if (
    outcome.chain !== run.chain ||
    outcome.immediate_block !== run.head_block + 1 ||
    outcome.selected_block !== run.target_block
  ) {
    throw new Error("Outcome does not match the inference run");
  }
  return {
    ...run,
    outcome: {
      resolved_at: new Date().toISOString(),
      immediate_base_fee_per_gas: outcome.immediate_base_fee_per_gas,
      selected_base_fee_per_gas: outcome.selected_base_fee_per_gas,
    },
  };
}

export async function loadRuns(): Promise<InferenceRun[]> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  if (stored === null) {
    return [];
  }
  const value: unknown = JSON.parse(stored);
  if (!Array.isArray(value)) {
    throw new Error("Stored inference history must be an array");
  }
  return value.map(parseRun).slice(0, MAX_RUNS);
}

export async function saveRuns(runs: readonly InferenceRun[]): Promise<void> {
  await AsyncStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(runs.slice(0, MAX_RUNS)),
  );
}
