import { describe, expect, it } from "vitest";

import {
  feeComparisonData,
  recommendedWaitData,
  runsForSelection,
  savingsByWaitData,
  summarizeRuns,
} from "../src/analytics";
import type { InferenceRun } from "../src/history";

const GWEI = 1_000_000_000;

function run(overrides: Partial<InferenceRun> = {}): InferenceRun {
  return {
    id: "run",
    ran_at: "2026-07-26T10:00:00.000Z",
    chain: "ethereum",
    K: 5,
    artifact_id: "artifact-5",
    head_block: 10,
    head_hash: "0xhead",
    selected_action_k: 1,
    target_block: 12,
    predicted_minimum_base_fee_per_gas: 9 * GWEI,
    ...overrides,
  };
}

function resolved(
  id: string,
  wait: number,
  immediateGwei: number,
  selectedGwei: number,
): InferenceRun {
  return run({
    id,
    selected_action_k: wait,
    target_block: 11 + wait,
    outcome: {
      immediate_base_fee_per_gas: immediateGwei * GWEI,
      selected_base_fee_per_gas: selectedGwei * GWEI,
    },
  });
}

describe("analytics", () => {
  it("computes realized metrics from supported outcomes and wait from every run", () => {
    const runs = [
      resolved("saved", 1, 100, 80),
      resolved("lost", 2, 100, 120),
      resolved("act-now", 0, 100, 100),
      run({ id: "pending", selected_action_k: 3, target_block: 14 }),
      resolved("zero-baseline", 4, 0, 0),
      resolved("zero-selected", 2, 100, 0),
    ];

    const summary = summarizeRuns(runs);

    expect(summary.averageWait).toBe(2);
    expect(summary.averageSavingsPercent).toBeCloseTo(25);
    expect(summary.winPercent).toBeCloseTo((2 / 3) * 100);
  });

  it("builds all charts from resolved and pending cases consistently", () => {
    const selectedRuns = [
      resolved("act-now", 0, 10, 10),
      resolved("saved", 1, 10, 8),
      run({ id: "pending", selected_action_k: 1, target_block: 12 }),
      resolved("lost", 2, 10, 12),
      resolved("zero-selected", 3, 10, 0),
      resolved("invalid", 4, 0, 0),
    ];
    const collection = runsForSelection(
      [
        ...selectedRuns,
        resolved("other-chain", 1, 100, 1),
        run({ id: "other-horizon", K: 4 }),
      ].map((item, index) =>
        index === selectedRuns.length
          ? { ...item, chain: "polygon" as const }
          : item,
      ),
      "ethereum",
      5,
    );

    expect(recommendedWaitData(collection, 5)).toEqual([
      { label: "0", value: 1 },
      { label: "1", value: 2 },
      { label: "2", value: 1 },
      { label: "3", value: 1 },
      { label: "4", value: 1 },
    ]);
    expect(savingsByWaitData(collection, 5)).toEqual([
      { label: "0", value: 0 },
      { label: "1", value: 20 },
      { label: "2", value: -20 },
      { label: "3", value: 100 },
      { label: "4", value: null },
    ]);
    expect(feeComparisonData(collection, 5)).toEqual([
      { label: "0", immediate: 10, fable: 10 },
      { label: "1", immediate: 10, fable: 8 },
      { label: "2", immediate: 10, fable: 12 },
      { label: "3", immediate: 10, fable: 0 },
    ]);
  });

  it("returns empty analytics for an empty selection", () => {
    expect(summarizeRuns([])).toEqual({
      averageWait: null,
      averageSavingsPercent: null,
      winPercent: null,
    });
    expect(recommendedWaitData([], 5)).toEqual([]);
    expect(savingsByWaitData([], 5)).toEqual([]);
    expect(feeComparisonData([], 5)).toEqual([]);
  });
});
