import { describe, expect, it } from "vitest";

import {
  runsForSelection,
  summarizeRuns,
  waitBuckets,
} from "../src/analytics";
import type { InferenceRun } from "../src/history";
import { inferenceRun } from "./helpers";

const GWEI = 1_000_000_000;

function resolved(
  id: string,
  wait: number,
  immediateGwei: number,
  selectedGwei: number,
): InferenceRun {
  return inferenceRun({
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
      inferenceRun({
        id: "pending",
        selected_action_k: 3,
        target_block: 14,
      }),
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
      resolved("saved-more", 1, 20, 12),
      inferenceRun({
        id: "pending",
        selected_action_k: 1,
        target_block: 12,
      }),
      resolved("lost", 2, 10, 12),
      resolved("zero-selected", 3, 10, 0),
      resolved("invalid", 4, 0, 0),
    ];
    const collection = runsForSelection(
      [
        ...selectedRuns,
        resolved("other-chain", 1, 100, 1),
        inferenceRun({ id: "other-horizon", K: 4 }),
      ].map((item, index) =>
        index === selectedRuns.length
          ? { ...item, chain: "polygon" as const }
          : item,
      ),
      "ethereum",
      5,
    );

    expect(waitBuckets(collection, 5)).toEqual([
      {
        fableGwei: 10,
        immediateGwei: 10,
        label: "0",
        runCount: 1,
        savingsPercent: 0,
      },
      {
        fableGwei: 10,
        immediateGwei: 15,
        label: "1",
        runCount: 3,
        savingsPercent: 30,
      },
      {
        fableGwei: 12,
        immediateGwei: 10,
        label: "2",
        runCount: 1,
        savingsPercent: -20,
      },
      {
        fableGwei: 0,
        immediateGwei: 10,
        label: "3",
        runCount: 1,
        savingsPercent: 100,
      },
      {
        fableGwei: null,
        immediateGwei: null,
        label: "4",
        runCount: 1,
        savingsPercent: null,
      },
    ]);
  });

  it("returns empty analytics for an empty selection", () => {
    expect(summarizeRuns([])).toEqual({
      averageWait: null,
      averageSavingsPercent: null,
      winPercent: null,
    });
    expect(waitBuckets([], 5)).toEqual([]);
  });
});
