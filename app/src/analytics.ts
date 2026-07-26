import type { Chain, Horizon } from "./domain";
import type { InferenceRun, RunOutcome } from "./history";

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

export type RunSummary = {
  averageWait: number | null;
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

export function runsForSelection(
  runs: readonly InferenceRun[],
  chain: Chain,
  horizon: Horizon,
): InferenceRun[] {
  return runs.filter((run) => run.chain === chain && run.K === horizon);
}

export function summarizeRuns(runs: readonly InferenceRun[]): RunSummary {
  if (runs.length === 0) {
    return {
      averageWait: null,
      averageSavingsPercent: null,
      winPercent: null,
    };
  }
  const savings = runs.flatMap((run) => {
    const value = realizedSavingsPercent(run);
    return value === null ? [] : [value];
  });
  const waitedSavings = runs.flatMap((run) => {
    const value = realizedSavingsPercent(run);
    return run.selected_action_k === 0 || value === null ? [] : [value];
  });
  return {
    averageWait: mean(runs.map((run) => run.selected_action_k)),
    averageSavingsPercent: mean(savings),
    winPercent:
      waitedSavings.length === 0
        ? null
        : (waitedSavings.filter((value) => value > 0).length /
            waitedSavings.length) *
          100,
  };
}

export function realizedSavingsPercent(run: InferenceRun): number | null {
  const outcome = validOutcome(run);
  if (outcome === null) {
    return null;
  }
  return (
    ((outcome.immediate_base_fee_per_gas -
      outcome.selected_base_fee_per_gas) /
      outcome.immediate_base_fee_per_gas) *
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
  if (!Number.isFinite(value) || value < 0) {
    return "—";
  }
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
      value: mean(savings),
    };
  });
}

export function feeComparisonData(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): FeeComparisonDatum[] {
  return Array.from({ length: horizon }, (_, offset) => {
    const outcomes = runs.flatMap((run) => {
      const outcome = validOutcome(run);
      return run.selected_action_k === offset && outcome !== null
        ? [outcome]
        : [];
    });
    if (outcomes.length === 0) {
      return [];
    }
    return [
      {
        label: String(offset),
        immediate:
          (mean(
            outcomes.map(
              (outcome) => outcome.immediate_base_fee_per_gas,
            ),
          ) ?? 0) / 1_000_000_000,
        fable:
          (mean(
            outcomes.map((outcome) => outcome.selected_base_fee_per_gas),
          ) ?? 0) / 1_000_000_000,
      },
    ];
  }).flat();
}

function validOutcome(run: InferenceRun): RunOutcome | null {
  const outcome = run.outcome;
  if (
    outcome === undefined ||
    !Number.isFinite(outcome.immediate_base_fee_per_gas) ||
    outcome.immediate_base_fee_per_gas <= 0 ||
    !Number.isFinite(outcome.selected_base_fee_per_gas) ||
    outcome.selected_base_fee_per_gas < 0
  ) {
    return null;
  }
  return outcome;
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce(
    (average, value, index) => average + (value - average) / (index + 1),
    0,
  );
}
