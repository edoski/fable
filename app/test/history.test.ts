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
  createRun,
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
    head_base_fee_per_gas: 12_000_000_000,
    selected_action_k: 2,
    immediate_block: 101,
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
    head_base_fee_per_gas: 12_000_000_000,
    selected_action_k: 2,
    target_block: 13,
    predicted_minimum_base_fee_per_gas: 10_000_000_000,
    ...overrides,
  };
}

function outcome(
  run: InferenceRun,
  overrides: Partial<InferenceOutcome> = {},
): InferenceOutcome {
  return {
    chain: run.chain,
    immediate_block: run.head_block + 1,
    selected_block: run.target_block,
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
  it("selects only canonical fields from a local inference result", () => {
    const source = {
      ...inferenceResult(),
      action_logits: [0, 1, 0, 0, 0],
      feature_rows: [[1, 2]],
      cached_blocks: [99, 100],
    };

    const first = createRun(source);
    const second = createRun(source);

    expect(first).toEqual({
      id: expect.any(String),
      ran_at: expect.any(String),
      chain: "ethereum",
      K: 5,
      artifact_id: "artifact-5",
      head_block: 100,
      head_hash: "0xhead",
      head_base_fee_per_gas: 12_000_000_000,
      selected_action_k: 2,
      target_block: 103,
      predicted_minimum_base_fee_per_gas: 10_000_000_000,
    });
    expect(first.id).not.toBe(second.id);
    expect(first).not.toHaveProperty("immediate_block");
    expect(first).not.toHaveProperty("action_logits");
    expect(first).not.toHaveProperty("feature_rows");
    expect(first).not.toHaveProperty("cached_blocks");
  });

  it("round-trips one capped ordered array under fable.runs", async () => {
    const runs = Array.from({ length: 105 }, (_, index) =>
      storedRun({ id: `run-${index}`, head_block: index }),
    );

    await saveRuns(runs);
    const saved = JSON.parse(storage.values.get("fable.runs") ?? "null");

    expect([...storage.values.keys()]).toEqual(["fable.runs"]);
    expect(saved).toHaveLength(100);
    expect(saved.map((run: InferenceRun) => run.id)).toEqual(
      runs.slice(0, 100).map((run) => run.id),
    );
    await expect(loadRuns()).resolves.toEqual(runs.slice(0, 100));
  });

  it("rejects malformed top-level storage", async () => {
    storage.values.set("fable.runs", "{");
    await expect(loadRuns()).rejects.toThrow(
      "Stored inference runs are not valid JSON",
    );

    storage.values.set("fable.runs", JSON.stringify({ runs: [] }));
    await expect(loadRuns()).rejects.toThrow(
      "Stored inference runs must be a JSON array",
    );
  });

  it("records only an outcome with matching chain and exact blocks", async () => {
    const run = storedRun();
    const [resolved] = await resolvePendingRuns(
      [run],
      "ethereum",
      run.target_block,
      async () => outcome(run),
    );

    expect(resolved.id).toBe(run.id);
    expect(resolved.outcome).toEqual({
      resolved_at: expect.any(String),
      immediate_base_fee_per_gas: 12_000_000_000,
      selected_base_fee_per_gas: 10_000_000_000,
    });
    expect(resolved.outcome).not.toHaveProperty("immediate_block");
    expect(resolved.outcome).not.toHaveProperty("selected_block");

    await expect(
      resolvePendingRuns(
        [run],
        "ethereum",
        run.target_block,
        async () => outcome(run, { chain: "polygon" }),
      ),
    ).rejects.toThrow("Outcome chain does not match the run");
    await expect(
      resolvePendingRuns(
        [run],
        "ethereum",
        run.target_block,
        async () => outcome(run, { immediate_block: 12 }),
      ),
    ).rejects.toThrow("Outcome immediate block does not match the run");
    await expect(
      resolvePendingRuns(
        [run],
        "ethereum",
        run.target_block,
        async () => outcome(run, { selected_block: 14 }),
      ),
    ).rejects.toThrow("Outcome selected block does not match the run");
  });

  it("resolves only eligible runs on the selected chain and preserves order", async () => {
    const alreadyResolved = storedRun({
      id: "resolved",
      outcome: {
        resolved_at: "2026-07-26T10:01:00.000Z",
        immediate_base_fee_per_gas: 12,
        selected_base_fee_per_gas: 10,
      },
    });
    const waited = storedRun({ id: "waited" });
    const actionZero = storedRun({
      id: "action-zero",
      head_block: 20,
      selected_action_k: 0,
      target_block: 21,
    });
    const future = storedRun({
      id: "future",
      head_block: 30,
      target_block: 33,
    });
    const otherChain = storedRun({
      id: "other-chain",
      chain: "polygon",
    });
    const resolve = vi.fn(
      async (
        immediateBlock: number,
        selectedBlock: number,
      ): Promise<InferenceOutcome> => ({
        chain: "ethereum",
        immediate_block: immediateBlock,
        selected_block: selectedBlock,
        immediate_base_fee_per_gas: 12,
        selected_base_fee_per_gas: 10,
      }),
    );
    const runs = [
      alreadyResolved,
      waited,
      actionZero,
      future,
      otherChain,
    ];

    const resolved = await resolvePendingRuns(
      runs,
      "ethereum",
      25,
      resolve,
    );

    expect(resolved.map((run) => run.id)).toEqual(runs.map((run) => run.id));
    expect(resolve.mock.calls).toEqual([
      [11, 13],
      [21, 21],
    ]);
    expect(resolved[0]).toBe(alreadyResolved);
    expect(resolved[1].outcome).toBeDefined();
    expect(resolved[2].outcome).toBeDefined();
    expect(resolved[3]).toBe(future);
    expect(resolved[4]).toBe(otherChain);
  });

  it("leaves the original pending run retryable after resolver failure", async () => {
    const run = storedRun();
    const resolve = vi
      .fn()
      .mockRejectedValueOnce(new Error("RPC unavailable"))
      .mockResolvedValueOnce(outcome(run));

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
