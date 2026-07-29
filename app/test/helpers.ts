import type { Hash } from "viem";

import type { InferenceRun } from "../src/history";
import type { InferenceResult } from "../src/inference";

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
