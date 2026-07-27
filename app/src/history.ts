import AsyncStorage from "@react-native-async-storage/async-storage";

import type { Chain, Horizon } from "./domain";
import type {
  InferenceEngine,
  InferenceOutcome,
  InferenceResult,
} from "./inference";

const STORAGE_KEY = "fable.runs";

export type InferenceRun = {
  id: string;
  ran_at: string;
  chain: Chain;
  K: Horizon;
  artifact_id: string;
  head_block: number;
  head_hash: string;
  selected_action_k: number;
  target_block: number;
  predicted_minimum_base_fee_per_gas: number;
  outcome?: RunOutcome;
};

export type RunOutcome = {
  immediate_base_fee_per_gas: number;
  selected_base_fee_per_gas: number;
};

export type OutcomeResolver = InferenceEngine["resolveOutcome"];

let runSequence = 0;

export function addRun(
  runs: readonly InferenceRun[],
  result: InferenceResult,
): InferenceRun[] {
  return [createRun(result), ...runs];
}

function createRun(result: InferenceResult): InferenceRun {
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
    selected_action_k: result.selected_action_k,
    target_block: result.target_block,
    predicted_minimum_base_fee_per_gas:
      result.predicted_minimum_base_fee_per_gas,
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
      return {
        ...run,
        outcome: {
          immediate_base_fee_per_gas: outcome.immediate_base_fee_per_gas,
          selected_base_fee_per_gas: outcome.selected_base_fee_per_gas,
        },
      };
    }),
  );
}

export async function loadRuns(): Promise<InferenceRun[]> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  if (stored === null) {
    return [];
  }

  try {
    return JSON.parse(stored) as InferenceRun[];
  } catch {
    throw new Error("Stored inference runs are not valid JSON");
  }
}

export async function saveRuns(runs: readonly InferenceRun[]): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
}
