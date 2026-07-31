import AsyncStorage from "@react-native-async-storage/async-storage";

import type { Chain } from "./domain";
import type {
  InferenceEngine,
  InferenceOutcome,
  InferenceResult,
} from "./inference";

const STORAGE_KEY = "fable.runs";

export type InferenceRun = InferenceResult & {
  id: string;
  ran_at: string;
  outcome?: InferenceOutcome;
};

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
    ...result,
  };
}

export async function resolvePendingRuns(
  runs: readonly InferenceRun[],
  chain: Chain,
  headBlock: number,
  resolveOutcome: InferenceEngine["resolveOutcome"],
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
      try {
        const outcome = await resolveOutcome(
          run.head_block + 1,
          run.target_block,
        );
        return { ...run, outcome };
      } catch {
        return run;
      }
    }),
  );
}

export async function loadRuns(): Promise<InferenceRun[]> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  if (stored === null) {
    return [];
  }

  return JSON.parse(stored) as InferenceRun[];
}

export async function saveRuns(runs: readonly InferenceRun[]): Promise<void> {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
}
