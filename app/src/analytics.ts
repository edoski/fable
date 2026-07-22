import type { Horizon } from "./inference";
import type { InferenceRun } from "./history";

export const GRAPH_OPTIONS = [
  { value: "waits", label: "Recommended wait distribution" },
  { value: "savings", label: "Average savings by wait" },
  { value: "fees", label: "Base fee (lower is better)" },
] as const;

export type GraphKind = (typeof GRAPH_OPTIONS)[number]["value"];

type ChartDatum = {
  label: string;
  value: number | null;
};

type FeeComparisonDatum = {
  label: string;
  immediate: number;
  fable: number;
};

type RunSummary = {
  averageOffset: number | null;
  averageSavingsPercent: number | null;
  winPercent: number | null;
};

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export function summarizeRuns(runs: readonly InferenceRun[]): RunSummary {
  if (runs.length === 0) {
    return {
      averageOffset: null,
      averageSavingsPercent: null,
      winPercent: null,
    };
  }
  const offsets = runs.reduce((total, run) => total + run.selected_action_k, 0);
  const savings = runs.flatMap((run) => {
    const value = realizedSavingsPercent(run);
    return value === null ? [] : [value];
  });
  const waitedSavings = runs.flatMap((run) => {
    const value = realizedSavingsPercent(run);
    return run.selected_action_k === 0 || value === null ? [] : [value];
  });
  return {
    averageOffset: offsets / runs.length,
    averageSavingsPercent:
      savings.length === 0
        ? null
        : savings.reduce((total, value) => total + value, 0) / savings.length,
    winPercent:
      waitedSavings.length === 0
        ? null
        : (waitedSavings.filter((value) => value > 0).length /
            waitedSavings.length) *
          100,
  };
}

export function realizedSavingsPercent(run: InferenceRun): number | null {
  if (run.outcome === undefined) {
    return null;
  }
  return (
    ((run.outcome.immediate_base_fee_per_gas -
      run.outcome.selected_base_fee_per_gas) /
      run.outcome.immediate_base_fee_per_gas) *
    100
  );
}

function shortTime(value: string): string {
  const date = new Date(value);
  return `${date.getHours().toString().padStart(2, "0")}:${date
    .getMinutes()
    .toString()
    .padStart(2, "0")}`;
}

function shortDate(value: string): string {
  const date = new Date(value);
  return `${date.getDate()} ${MONTHS[date.getMonth()]}`;
}

export function formatRunDate(value: string): string {
  return `${shortDate(value)}, ${shortTime(value)}`;
}

export function formatGwei(value: number): string {
  const gwei = value / 1_000_000_000;
  if (gwei >= 100) {
    return `${gwei.toFixed(0)} Gwei`;
  }
  if (gwei >= 10) {
    return `${gwei.toFixed(1)} Gwei`;
  }
  return `${gwei.toFixed(2)} Gwei`;
}

export function recommendedWaitData(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): ChartDatum[] {
  if (runs.length === 0) {
    return [];
  }
  return Array.from({ length: horizon }, (_, offset) => ({
    label: String(offset),
    value: runs.filter((run) => run.selected_action_k === offset).length,
  }));
}

export function savingsByWaitData(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): ChartDatum[] {
  if (runs.length === 0) {
    return [];
  }
  return Array.from({ length: horizon }, (_, offset) => {
    const savings = runs.flatMap((run) => {
      if (run.selected_action_k !== offset) {
        return [];
      }
      const value = realizedSavingsPercent(run);
      return value === null ? [] : [value];
    });
    return {
      label: String(offset),
      value:
        savings.length === 0
          ? null
          : savings.reduce((total, value) => total + value, 0) /
            savings.length,
    };
  });
}

export function feeComparisonData(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): FeeComparisonDatum[] {
  return Array.from({ length: horizon }, (_, offset) => {
    const outcomes = runs.flatMap((run) =>
      run.selected_action_k === offset && run.outcome !== undefined
        ? [run.outcome]
        : [],
    );
    if (outcomes.length === 0) {
      return [];
    }
    return [
      {
        label: String(offset),
        immediate:
          outcomes.reduce(
            (total, outcome) => total + outcome.immediate_base_fee_per_gas,
            0,
          ) /
          outcomes.length /
          1_000_000_000,
        fable:
          outcomes.reduce(
            (total, outcome) => total + outcome.selected_base_fee_per_gas,
            0,
          ) /
          outcomes.length /
          1_000_000_000,
      },
    ];
  }).flat();
}
