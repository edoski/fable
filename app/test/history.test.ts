import { describe, expect, it, vi } from "vitest";

import type { InferenceOutcome } from "../src/inference";
import { inferenceResult, inferenceRun } from "./helpers";

vi.mock("@react-native-async-storage/async-storage", () => ({
  default: {
    getItem: vi.fn(),
    setItem: vi.fn(),
  },
}));

import { addRun, resolvePendingRuns } from "../src/history";

function outcome(
  overrides: Partial<InferenceOutcome> = {},
): InferenceOutcome {
  return {
    immediate_base_fee_per_gas: 12_000_000_000,
    selected_base_fee_per_gas: 10_000_000_000,
    ...overrides,
  };
}

describe("history", () => {
  it("adds a unique canonical run before every existing run", () => {
    const existing = Array.from({ length: 3 }, (_, index) =>
      inferenceRun({ id: `existing-${index}` }),
    );
    const result = inferenceResult({
      head_block: 100,
      selected_action_k: 2,
      target_block: 103,
      predicted_minimum_base_fee_per_gas: 10_000_000_000,
    });
    const [first, ...retained] = addRun(existing, result);
    const [second] = addRun(existing, result);

    expect(first).toEqual({
      id: expect.any(String),
      ran_at: expect.any(String),
      chain: "ethereum",
      K: 5,
      artifact_id: "artifact-5",
      head_block: 100,
      head_hash: "0xhead",
      selected_action_k: 2,
      target_block: 103,
      predicted_minimum_base_fee_per_gas: 10_000_000_000,
    });
    expect(first.id).not.toBe(second.id);
    expect(retained).toEqual(existing);
  });

  it("commits successful outcomes while failed siblings remain retryable", async () => {
    const failed = inferenceRun({
      id: "failed",
      head_block: 10,
      target_block: 12,
    });
    const successful = inferenceRun({
      id: "successful",
      head_block: 20,
      target_block: 22,
    });
    const resolve = vi
      .fn()
      .mockRejectedValueOnce(new Error("RPC unavailable"))
      .mockResolvedValue(outcome());

    const resolved = await resolvePendingRuns(
      [failed, successful],
      "ethereum",
      successful.target_block,
      resolve,
    );
    expect(resolved[0]).toBe(failed);
    expect(resolved[0].outcome).toBeUndefined();
    expect(resolved[1]).toEqual({
      ...successful,
      outcome: outcome(),
    });

    const retried = await resolvePendingRuns(
      resolved,
      "ethereum",
      successful.target_block,
      resolve,
    );
    expect(resolve).toHaveBeenCalledTimes(3);
    expect(retried[0].outcome).toBeDefined();
    expect(retried[1]).toBe(resolved[1]);
  });
});
