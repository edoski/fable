import type { Chain, Horizon } from "./domain";
import type { InferenceOutcome } from "./inference";
import type { InferenceRun } from "./history";

export type WaitBucket = {
  fableGwei: number | null;
  immediateGwei: number | null;
  label: string;
  runCount: number;
  savingsPercent: number | null;
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
const GWEI = 1_000_000_000;

export function runsForSelection(
  runs: readonly InferenceRun[],
  chain: Chain,
  horizon: Horizon,
): InferenceRun[] {
  return runs.filter((run) => run.chain === chain && run.K === horizon);
}

export function summarizeRuns(runs: readonly InferenceRun[]): RunSummary {
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
  const outcome = validOutcome(run);
  if (outcome === null) {
    return null;
  }
  return savingsPercent(outcome);
}

function savingsPercent(outcome: InferenceOutcome): number {
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

export function waitBuckets(
  runs: readonly InferenceRun[],
  horizon: Horizon,
): WaitBucket[] {
  if (runs.length === 0) {
    return [];
  }

  const buckets = Array.from({ length: horizon }, (_, offset) => ({
    fableFeeMean: null as number | null,
    immediateFeeMean: null as number | null,
    label: String(offset),
    outcomeCount: 0,
    runCount: 0,
    savingsPercent: null as number | null,
  }));

  for (const run of runs) {
    const bucket = buckets[run.selected_action_k];
    if (bucket === undefined) {
      continue;
    }
    bucket.runCount += 1;
    const outcome = validOutcome(run);
    if (outcome === null) {
      continue;
    }
    bucket.outcomeCount += 1;
    bucket.savingsPercent = nextMean(
      bucket.savingsPercent,
      savingsPercent(outcome),
      bucket.outcomeCount,
    );
    bucket.immediateFeeMean = nextMean(
      bucket.immediateFeeMean,
      outcome.immediate_base_fee_per_gas,
      bucket.outcomeCount,
    );
    bucket.fableFeeMean = nextMean(
      bucket.fableFeeMean,
      outcome.selected_base_fee_per_gas,
      bucket.outcomeCount,
    );
  }

  return buckets.map(
    ({
      fableFeeMean,
      immediateFeeMean,
      label,
      runCount,
      savingsPercent,
    }) => ({
      fableGwei: fableFeeMean === null ? null : fableFeeMean / GWEI,
      immediateGwei:
        immediateFeeMean === null ? null : immediateFeeMean / GWEI,
      label,
      runCount,
      savingsPercent,
    }),
  );
}

function validOutcome(run: InferenceRun): InferenceOutcome | null {
  const outcome = run.outcome;
  if (
    outcome === undefined ||
    outcome.immediate_base_fee_per_gas <= 0
  ) {
    return null;
  }
  return outcome;
}

function nextMean(
  average: number | null,
  value: number,
  count: number,
): number {
  const current = average ?? 0;
  return current + (value - current) / count;
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
