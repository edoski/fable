import type { Chain, Horizon } from "./domain";
import type { InferenceRun, RunOutcome } from "./history";

export const GRAPH_OPTIONS = [
  { value: "waits", label: "Recommended wait distribution" },
  { value: "savings", label: "Savings by wait (%)" },
  { value: "fees", label: "Base fee by wait (Gwei)" },
] as const;

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

const RUN_DATE_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  hour: "2-digit",
  hourCycle: "h23",
  minute: "2-digit",
  month: "short",
});

export function runsForSelection(
  runs: readonly InferenceRun[],
  chain: Chain,
  horizon: Horizon,
): InferenceRun[] {
  return runs.filter((run) => run.chain === chain && run.K === horizon);
}

export function summarizeRuns(runs: readonly InferenceRun[]): RunSummary {
  let averageWait = 0;
  let averageSavings = 0;
  let realizedCount = 0;
  let waitedCount = 0;
  let winCount = 0;

  for (const [index, run] of runs.entries()) {
    averageWait += (run.selected_action_k - averageWait) / (index + 1);
    const savings = realizedSavingsPercent(run);
    if (savings === null) {
      continue;
    }
    realizedCount += 1;
    averageSavings += (savings - averageSavings) / realizedCount;
    if (run.selected_action_k !== 0) {
      waitedCount += 1;
      winCount += Number(savings > 0);
    }
  }

  return {
    averageWait: runs.length === 0 ? null : averageWait,
    averageSavingsPercent:
      realizedCount === 0 ? null : averageSavings,
    winPercent: waitedCount === 0 ? null : (winCount / waitedCount) * 100,
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

export function formatRunDate(value: string): string {
  return RUN_DATE_FORMATTER.format(new Date(value));
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
    outcome.immediate_base_fee_per_gas <= 0
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
