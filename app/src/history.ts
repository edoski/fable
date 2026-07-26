import AsyncStorage from "@react-native-async-storage/async-storage";

import type {
  Chain,
  Horizon,
  InferenceEngine,
  InferenceOutcome,
  InferenceResult,
} from "./inference";

const STORAGE_KEY = "fable.runs";
const MAX_RUNS = 100;

export type InferenceRun = {
  id: string;
  ran_at: string;
  chain: Chain;
  K: Horizon;
  artifact_id: string;
  head_block: number;
  head_hash: string;
  head_base_fee_per_gas: number;
  selected_action_k: number;
  target_block: number;
  predicted_minimum_base_fee_per_gas: number;
  outcome?: RunOutcome;
};

export type RunOutcome = {
  resolved_at: string;
  immediate_base_fee_per_gas: number;
  selected_base_fee_per_gas: number;
};

export type OutcomeResolver = InferenceEngine["resolveOutcome"];

let runSequence = 0;

export function createRun(result: InferenceResult): InferenceRun {
  const ranAt = new Date().toISOString();
  runSequence += 1;
  return {
    id: `${ranAt}:${runSequence}:${result.chain}:${result.K}:${result.head_hash}`,
    ran_at: ranAt,
    chain: result.chain,
    K: result.K,
    artifact_id: result.artifact_id,
    head_block: result.head_block,
    head_hash: result.head_hash,
    head_base_fee_per_gas: result.head_base_fee_per_gas,
    selected_action_k: result.selected_action_k,
    target_block: result.target_block,
    predicted_minimum_base_fee_per_gas:
      result.predicted_minimum_base_fee_per_gas,
  };
}

function recordOutcome(
  run: InferenceRun,
  outcome: InferenceOutcome,
): InferenceRun {
  if (run.outcome !== undefined) {
    return run;
  }
  if (outcome.chain !== run.chain) {
    throw new Error("Outcome chain does not match the run");
  }
  if (outcome.immediate_block !== run.head_block + 1) {
    throw new Error("Outcome immediate block does not match the run");
  }
  if (outcome.selected_block !== run.target_block) {
    throw new Error("Outcome selected block does not match the run");
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

export async function resolvePendingRuns(
  runs: readonly InferenceRun[],
  chain: Chain,
  headBlock: number,
  resolveOutcome: OutcomeResolver,
): Promise<InferenceRun[]> {
  return Promise.all(
    runs.map(async (run) => {
      if (
        run.chain !== chain ||
        run.outcome !== undefined ||
        run.target_block > headBlock
      ) {
        return run;
      }
      const outcome = await resolveOutcome(
        run.head_block + 1,
        run.target_block,
      );
      return recordOutcome(run, outcome);
    }),
  );
}

export async function loadRuns(): Promise<InferenceRun[]> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  if (stored === null) {
    return [];
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(stored);
  } catch {
    throw new Error("Stored inference runs are not valid JSON");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("Stored inference runs must be a JSON array");
  }
  return (parsed as InferenceRun[]).slice(0, MAX_RUNS);
}

export async function saveRuns(runs: readonly InferenceRun[]): Promise<void> {
  await AsyncStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(runs.slice(0, MAX_RUNS)),
  );
}
