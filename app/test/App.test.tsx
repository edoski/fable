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
import {
  deferred,
  flushMicrotasks,
  inferenceResult,
  inferenceRun,
} from "./helpers";

const mocks = vi.hoisted(() => ({
  addRun: vi.fn(),
  analyticsProps: null as Record<string, unknown> | null,
  bottomTabsProps: null as Record<string, unknown> | null,
  createInferenceEngine: vi.fn(),
  inferenceProps: null as Record<string, unknown> | null,
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
  AppHeader: () => null,
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
  resolveRun(result: InferenceResult): void;
};

const engines: EngineHarness[] = [];
let root: ReactTestRenderer | null = null;

function engine(): EngineHarness {
  const run = deferred<InferenceResult>();
  const value: InferenceEngine = {
    watchBlocks: vi.fn(),
    run: vi.fn(() => run.promise),
    resolveOutcome: vi.fn(async () => {
      throw new Error("unused");
    }),
    dispose: vi.fn(async () => undefined),
  };
  return {
    engine: value,
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
  onSelect(tab: "analytics" | "inference"): void;
} {
  return mocks.bottomTabsProps as ReturnType<typeof bottomTabsProps>;
}

function analyticsProps(): {
  loadError: string | null;
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
  it("does not publish a result after the horizon changes and returns", async () => {
    const result = inferenceResult();
    await renderApp();

    act(() => inferenceProps().onRun());
    expect(engines[0].engine.run).toHaveBeenCalledWith(5);

    await act(async () => {
      inferenceProps().onHorizonChange(4);
      inferenceProps().onHorizonChange(5);
      await flushMicrotasks();
    });
    expect(inferenceProps()).toMatchObject({
      chain: "ethereum",
      horizon: 5,
      state: { status: "idle" },
    });

    await act(async () => {
      engines[0].resolveRun(result);
      await flushMicrotasks();
    });

    expect(inferenceProps().state).toEqual({ status: "idle" });
    expect(mocks.addRun).not.toHaveBeenCalled();
    expect(mocks.saveRuns).not.toHaveBeenCalled();
  });

  it("does not publish a result from a replaced engine", async () => {
    const result = inferenceResult();
    await renderApp();
    act(() => inferenceProps().onRun());

    await act(async () => {
      inferenceProps().onChainChange("polygon");
      await flushMicrotasks();
    });
    expect(inferenceProps().chain).toBe("polygon");
    expect(engines).toHaveLength(2);

    await act(async () => {
      engines[0].resolveRun(result);
      await flushMicrotasks();
    });

    expect(inferenceProps().state).toEqual({ status: "idle" });
    expect(mocks.addRun).not.toHaveBeenCalled();
    expect(mocks.saveRuns).not.toHaveBeenCalled();
  });
});

describe("App history persistence", () => {
  it("reports initial history load failures in Analytics", async () => {
    mocks.loadRuns.mockRejectedValueOnce(new Error("Corrupt history"));
    await renderApp();

    act(() => bottomTabsProps().onSelect("analytics"));

    expect(analyticsProps().loadError).toBe("Corrupt history");
  });

  it("reports inference save failures only through the inference state", async () => {
    const result = inferenceResult();
    mocks.addRun.mockReturnValue([inferenceRun()]);
    mocks.saveRuns.mockRejectedValueOnce(new Error("Storage unavailable"));
    await renderApp();

    act(() => inferenceProps().onRun());
    await act(async () => {
      engines[0].resolveRun(result);
      await flushMicrotasks();
    });

    expect(inferenceProps().state).toEqual({
      message: "Could not save this run.",
      status: "error",
    });
    act(() => bottomTabsProps().onSelect("analytics"));
    expect(analyticsProps().loadError).toBeNull();
  });
});
