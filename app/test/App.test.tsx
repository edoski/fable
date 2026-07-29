import type { ReactNode } from "react";
import {
  act,
  create,
  type ReactTestRenderer,
} from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Chain, Horizon } from "../src/domain";
import type {
  ChainSnapshot,
  InferenceEngine,
  InferenceResult,
} from "../src/inference";
import type { InferenceRun } from "../src/history";
import { deferred, flushMicrotasks } from "./helpers";

const mocks = vi.hoisted(() => ({
  addRun: vi.fn(),
  analyticsProps: null as Record<string, unknown> | null,
  bottomTabsProps: null as Record<string, unknown> | null,
  createInferenceEngine: vi.fn(),
  inferenceProps: null as Record<string, unknown> | null,
  headerProps: null as Record<string, unknown> | null,
  loadRuns: vi.fn(),
  resolvePendingRuns: vi.fn(),
  saveRuns: vi.fn(),
}));

vi.mock("react-native", () => {
  const View = ({ children }: { children?: ReactNode }) => children ?? null;
  return {
    StatusBar: () => null,
    StyleSheet: { create: <T,>(styles: T) => styles },
    View,
  };
});

vi.mock("react-native-safe-area-context", () => {
  const View = ({ children }: { children?: ReactNode }) => children ?? null;
  return {
    SafeAreaProvider: View,
    SafeAreaView: View,
  };
});

vi.mock("../src/components/AppHeader", () => ({
  AppHeader: (props: Record<string, unknown>) => {
    mocks.headerProps = props;
    return null;
  },
}));

vi.mock("../src/components/BottomTabs", () => ({
  BottomTabs: (props: Record<string, unknown>) => {
    mocks.bottomTabsProps = props;
    return null;
  },
}));

vi.mock("../src/screens/AnalyticsScreen", () => ({
  AnalyticsScreen: (props: Record<string, unknown>) => {
    mocks.analyticsProps = props;
    return null;
  },
}));

vi.mock("../src/screens/InferenceScreen", () => ({
  InferenceScreen: (props: Record<string, unknown>) => {
    mocks.inferenceProps = props;
    return null;
  },
}));

vi.mock("../src/history", () => ({
  addRun: mocks.addRun,
  loadRuns: mocks.loadRuns,
  resolvePendingRuns: mocks.resolvePendingRuns,
  saveRuns: mocks.saveRuns,
}));

vi.mock("../src/inference", () => ({
  createInferenceEngine: mocks.createInferenceEngine,
}));

import App from "../App";

type EngineHarness = {
  engine: InferenceEngine;
  publish(snapshot: ChainSnapshot): void;
  resolveRun(result: InferenceResult): void;
};

const engines: EngineHarness[] = [];
let root: ReactTestRenderer | null = null;

function engine(): EngineHarness {
  let publish = (_snapshot: ChainSnapshot) => undefined;
  const run = deferred<InferenceResult>();
  const value: InferenceEngine = {
    startPolling: vi.fn((onSnapshot) => {
      publish = onSnapshot;
    }),
    run: vi.fn(() => run.promise),
    resolveOutcome: vi.fn(async () => {
      throw new Error("unused");
    }),
    dispose: vi.fn(async () => undefined),
  };
  return {
    engine: value,
    publish(snapshot) {
      publish(snapshot);
    },
    resolveRun(result) {
      run.resolve(result);
    },
  };
}

function inferenceProps(): {
  chain: Chain;
  horizon: Horizon;
  onChainChange(chain: Chain): void;
  onHorizonChange(horizon: Horizon): void;
  onRun(): void;
  snapshot: ChainSnapshot | null;
  state: Record<string, unknown>;
} {
  return mocks.inferenceProps as ReturnType<typeof inferenceProps>;
}

function bottomTabsProps(): {
  onSelect(tab: "inference" | "analytics"): void;
} {
  return mocks.bottomTabsProps as ReturnType<typeof bottomTabsProps>;
}

function analyticsProps(): {
  runs: readonly InferenceRun[];
} {
  return mocks.analyticsProps as ReturnType<typeof analyticsProps>;
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  engines.length = 0;
  mocks.analyticsProps = null;
  mocks.bottomTabsProps = null;
  mocks.inferenceProps = null;
  mocks.headerProps = null;
  mocks.addRun.mockReset();
  mocks.loadRuns.mockReset().mockResolvedValue([]);
  mocks.resolvePendingRuns.mockReset().mockResolvedValue([]);
  mocks.saveRuns.mockReset().mockResolvedValue(undefined);
  mocks.createInferenceEngine.mockReset().mockImplementation(() => {
    const created = engine();
    engines.push(created);
    return created.engine;
  });
});

afterEach(async () => {
  if (root !== null) {
    await act(async () => root?.unmount());
    root = null;
  }
});

async function renderApp(): Promise<void> {
  await act(async () => {
    root = create(<App />);
  });
}

describe("App engine selection", () => {
  it("removes a persisted result made stale during save", async () => {
    const result: InferenceResult = {
      chain: "ethereum",
      K: 5,
      artifact_id: "artifact-5",
      head_block: 10,
      head_hash: "0xhead",
      selected_action_k: 1,
      target_block: 12,
      predicted_minimum_base_fee_per_gas: 20,
    };
    const staleRun: InferenceRun = {
      id: "stale-run",
      ran_at: "2026-07-29T12:00:00.000Z",
      ...result,
    };
    const firstSave = deferred<void>();
    mocks.addRun.mockReturnValue([staleRun]);
    mocks.saveRuns
      .mockImplementationOnce(() => firstSave.promise)
      .mockResolvedValue(undefined);

    await renderApp();
    expect(inferenceProps().state).toEqual({ status: "idle" });

    act(() => inferenceProps().onRun());
    expect(engines[0].engine.run).toHaveBeenCalledWith(5);
    expect(inferenceProps().state).toEqual({ status: "loading" });

    act(() => engines[0].resolveRun(result));
    await vi.waitFor(() => expect(mocks.saveRuns).toHaveBeenCalledOnce());
    expect(mocks.saveRuns).toHaveBeenLastCalledWith([staleRun]);

    act(() => {
      inferenceProps().onHorizonChange(4);
      inferenceProps().onChainChange("polygon");
    });
    await act(async () => {
      firstSave.resolve();
      await flushMicrotasks();
    });

    expect(mocks.saveRuns).toHaveBeenCalledTimes(2);
    expect(mocks.saveRuns).toHaveBeenLastCalledWith([]);
    expect(inferenceProps()).toMatchObject({
      chain: "polygon",
      horizon: 4,
      snapshot: null,
      state: { status: "idle" },
    });
    expect(mocks.headerProps).toMatchObject({ status: "checking" });
    expect(engines).toHaveLength(2);
    act(() => bottomTabsProps().onSelect("analytics"));
    expect(analyticsProps().runs).toEqual([]);
  });
});
