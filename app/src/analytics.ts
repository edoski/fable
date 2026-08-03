import type { Chain, Horizon } from "./domain";
import type { InferenceRun } from "./history";

export type WaitBucket = {
  kairosGwei: number | null;
  immediateGwei: number | null;
  label: string;
  runCount: number;
  savingsPercent: number | null;
};

const RUN_DATE_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  hour: "2-digit",
  hourCycle: "h23",
  minute: "2-digit",
  month: "short",
});
const GWEI = 1_000_000_000;

export function runsForSelection(
  runs: readonly InferenceRun[],
  chain: Chain,
  horizon: Horizon,
): InferenceRun[] {
  return runs.filter((run) => run.chain === chain && run.K === horizon);
}

export function summarizeRuns(runs: readonly InferenceRun[]) {
  const realized = runs.flatMap((run) => {
    const savings = realizedSavingsPercent(run);
    return savings === null ? [] : [[run.selected_action_k, savings] as const];
  });
  const waited = realized.filter(([action]) => action !== 0);
  const winFraction = mean(waited.map(([, savings]) => Number(savings > 0)));

  return {
    averageWait: mean(runs.map((run) => run.selected_action_k)),
    averageSavingsPercent: mean(realized.map(([, savings]) => savings)),
    winPercent: winFraction === null ? null : winFraction * 100,
  };
}

export function realizedSavingsPercent(run: InferenceRun): number | null {
  return run.outcome === undefined ? null : savingsPercent(run.outcome);
}

function savingsPercent(outcome: NonNullable<InferenceRun["outcome"]>): number {
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
  const gwei = value / GWEI;
  if (gwei >= 100) {
    return `${gwei.toFixed(0)} Gwei`;
  }
  if (gwei >= 10) {
    return `${gwei.toFixed(1)} Gwei`;
  }
  return `${gwei.toFixed(2)} Gwei`;
}

export function waitBuckets(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): WaitBucket[] {
  if (runs.length === 0) {
    return [];
  }

  return Array.from({ length: horizon }, (_, offset) => {
    const matchingRuns = runs.filter(
      (run) => run.selected_action_k === offset,
    );
    const outcomes = matchingRuns.flatMap((run) =>
      run.outcome === undefined ? [] : [run.outcome],
    );
    const kairosFeeMean = mean(
      outcomes.map((outcome) => outcome.selected_base_fee_per_gas),
    );
    const immediateFeeMean = mean(
      outcomes.map((outcome) => outcome.immediate_base_fee_per_gas),
    );

    return {
      kairosGwei: kairosFeeMean === null ? null : kairosFeeMean / GWEI,
      immediateGwei:
        immediateFeeMean === null ? null : immediateFeeMean / GWEI,
      label: String(offset),
      runCount: matchingRuns.length,
      savingsPercent: mean(outcomes.map(savingsPercent)),
    };
  });
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
