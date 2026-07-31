import type { Hash } from "viem";

import type { Horizon } from "../src/domain";
import type { InferenceRun } from "../src/history";
import type { InferenceResult } from "../src/inference";
import type {
  MobileChainManifest,
  ModelManifest,
  ModelSelection,
} from "../src/model";

function modelEntry(K: Horizon): ModelManifest {
  return {
    artifact_id: `00000000-0000-4000-8000-${K.toString().padStart(12, "0")}`,
    target: { mean: Math.log(100), standard_deviation: 0.5 },
  };
}

export const chainManifest: MobileChainManifest = {
  context_blocks: 2,
  features: [
    {
      name: "log_base_fee_per_gas",
      mean: 0,
      standard_deviation: 1,
    },
  ],
  models: {
    2: modelEntry(2),
    3: modelEntry(3),
    4: modelEntry(4),
    5: modelEntry(5),
  },
};

export function modelSelection(K: Horizon): ModelSelection {
  return {
    K,
    source: 10 + K,
    chainManifest,
    modelManifest: chainManifest.models[K],
  };
}

export function inferenceResult(
  overrides: Partial<InferenceResult> = {},
): InferenceResult {
  return {
    chain: "ethereum",
    K: 5,
    artifact_id: "artifact-5",
    head_block: 10,
    head_hash: "0xhead",
    selected_action_k: 1,
    target_block: 12,
    predicted_minimum_base_fee_per_gas: 9_000_000_000,
    ...overrides,
  };
}

export function inferenceRun(
  overrides: Partial<InferenceRun> = {},
): InferenceRun {
  return {
    id: "run",
    ran_at: "2026-07-26T10:00:00.000Z",
    ...inferenceResult(),
    ...overrides,
  };
}

export function hashOf(value: bigint): Hash {
  return `0x${value.toString(16).padStart(64, "0")}`;
}

export async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 10; index += 1) {
    await Promise.resolve();
  }
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}
