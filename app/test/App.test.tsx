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
} from "../src/inference";

const mocks = vi.hoisted(() => ({
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
  BottomTabs: () => null,
}));

vi.mock("../src/screens/AnalyticsScreen", () => ({
  AnalyticsScreen: () => null,
}));

vi.mock("../src/screens/InferenceScreen", () => ({
  InferenceScreen: (props: Record<string, unknown>) => {
    mocks.inferenceProps = props;
    return null;
  },
}));

vi.mock("../src/history", () => ({
  addRun: vi.fn(),
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
};

const engines: EngineHarness[] = [];
let root: ReactTestRenderer | null = null;

function engine(): EngineHarness {
  let publish = (_snapshot: ChainSnapshot) => undefined;
  const value: InferenceEngine = {
    prepare: vi.fn(async () => undefined),
    startPolling: vi.fn((onSnapshot) => {
      publish = onSnapshot;
    }),
    run: vi.fn(async () => {
      throw new Error("unused");
    }),
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
  };
}

function inferenceProps(): {
  chain: Chain;
  onChainChange(chain: Chain): void;
  onHorizonChange(horizon: Horizon): void;
  snapshot: ChainSnapshot | null;
} {
  return mocks.inferenceProps as ReturnType<typeof inferenceProps>;
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  engines.length = 0;
  mocks.inferenceProps = null;
  mocks.headerProps = null;
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
  it("rejects an old-chain poll published during chain selection", async () => {
    await renderApp();
    act(() => {
      engines[0].publish({
        head_block: 10,
        current_base_fee_per_gas: 20,
      });
    });
    expect(inferenceProps().snapshot?.head_block).toBe(10);

    await act(async () => {
      inferenceProps().onChainChange("polygon");
      engines[0].publish({
        head_block: 11,
        current_base_fee_per_gas: 21,
      });
    });

    expect(inferenceProps()).toMatchObject({
      chain: "polygon",
      snapshot: null,
    });
    expect(mocks.headerProps).toMatchObject({ status: "checking" });
  });

  it("prepares each committed engine revision once", async () => {
    await renderApp();

    expect(engines).toHaveLength(1);
    expect(engines[0].engine.prepare).toHaveBeenCalledOnce();
    expect(engines[0].engine.prepare).toHaveBeenCalledWith(5);

    await act(async () => inferenceProps().onChainChange("polygon"));
    expect(engines).toHaveLength(2);
    expect(engines[1].engine.prepare).toHaveBeenCalledOnce();
    expect(engines[1].engine.prepare).toHaveBeenCalledWith(5);

    await act(async () => inferenceProps().onHorizonChange(4));
    expect(engines[1].engine.prepare).toHaveBeenCalledTimes(2);
    expect(engines[1].engine.prepare).toHaveBeenLastCalledWith(4);
  });
});
