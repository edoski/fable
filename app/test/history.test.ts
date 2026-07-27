import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  InferenceOutcome,
  InferenceResult,
} from "../src/inference";
import type { InferenceRun } from "../src/history";

const storage = vi.hoisted(() => {
  const values = new Map<string, string>();
  return {
    values,
    getItem: vi.fn(async (key: string) => values.get(key) ?? null),
    setItem: vi.fn(async (key: string, value: string) => {
      values.set(key, value);
    }),
  };
});

vi.mock("@react-native-async-storage/async-storage", () => ({
  default: {
    getItem: storage.getItem,
    setItem: storage.setItem,
  },
}));

import {
  addRun,
  loadRuns,
  resolvePendingRuns,
  saveRuns,
} from "../src/history";

function inferenceResult(
  overrides: Partial<InferenceResult> = {},
): InferenceResult {
  return {
    chain: "ethereum",
    K: 5,
    artifact_id: "artifact-5",
    head_block: 100,
    head_hash: "0xhead",
    selected_action_k: 2,
    target_block: 103,
    predicted_minimum_base_fee_per_gas: 10_000_000_000,
    ...overrides,
  };
}

function storedRun(
  overrides: Partial<InferenceRun> = {},
): InferenceRun {
  return {
    id: "run",
    ran_at: "2026-07-26T10:00:00.000Z",
    chain: "ethereum",
    K: 5,
    artifact_id: "artifact-5",
    head_block: 10,
    head_hash: "0xhead",
    selected_action_k: 2,
    target_block: 13,
    predicted_minimum_base_fee_per_gas: 10_000_000_000,
    ...overrides,
  };
}

function outcome(
  overrides: Partial<InferenceOutcome> = {},
): InferenceOutcome {
  return {
    immediate_base_fee_per_gas: 12_000_000_000,
    selected_base_fee_per_gas: 10_000_000_000,
    ...overrides,
  };
}

beforeEach(() => {
  storage.values.clear();
  vi.clearAllMocks();
});

describe("history", () => {
  it("adds a unique canonical run before every existing run", () => {
    const existing = Array.from({ length: 3 }, (_, index) =>
      storedRun({ id: `existing-${index}` }),
    );
    const [first, ...retained] = addRun(existing, inferenceResult());
    const [second] = addRun(existing, inferenceResult());

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

  it("round-trips the complete ordered array under fable.runs", async () => {
    const runs = Array.from({ length: 105 }, (_, index) =>
      storedRun({ id: `run-${index}`, head_block: index }),
    );

    await saveRuns(runs);
    const saved = JSON.parse(storage.values.get("fable.runs") ?? "null");

    expect([...storage.values.keys()]).toEqual(["fable.runs"]);
    expect(saved).toEqual(runs);
    await expect(loadRuns()).resolves.toEqual(runs);
  });

  it("rejects malformed stored JSON", async () => {
    storage.values.set("fable.runs", "{");
    await expect(loadRuns()).rejects.toThrow(
      "Stored inference runs are not valid JSON",
    );
  });

  it("leaves the original pending run retryable after resolver failure", async () => {
    const run = storedRun();
    const resolve = vi
      .fn()
      .mockRejectedValueOnce(new Error("RPC unavailable"))
      .mockResolvedValueOnce(outcome());

    await expect(
      resolvePendingRuns([run], "ethereum", run.target_block, resolve),
    ).rejects.toThrow("RPC unavailable");
    expect(run.outcome).toBeUndefined();

    const retried = await resolvePendingRuns(
      [run],
      "ethereum",
      run.target_block,
      resolve,
    );
    expect(resolve).toHaveBeenCalledTimes(2);
    expect(retried[0].outcome).toBeDefined();
  });
});
