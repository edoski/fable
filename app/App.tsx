import { useEffect, useRef, useState } from "react";
import { StatusBar, StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { AppHeader } from "./src/components/AppHeader";
import { BottomTabs, type AppTab } from "./src/components/BottomTabs";
import {
  createEngineLifecycle,
  type EngineLifecycle,
  type RpcStatus,
} from "./src/engineLifecycle";
import {
  addRun,
  loadRuns,
  resolvePendingRuns,
  saveRuns,
  type InferenceRun,
} from "./src/history";
import {
  createInferenceEngine,
  type Chain,
  type ChainSnapshot,
  type Horizon,
  type InferenceEngine,
} from "./src/inference";
import { AnalyticsScreen } from "./src/screens/AnalyticsScreen";
import {
  InferenceScreen,
  type InferenceState,
} from "./src/screens/InferenceScreen";
import { colors } from "./src/theme";

type ActiveEngine = {
  chain: Chain;
  engine: InferenceEngine;
  outcomeHead: number | null;
  outcomesRunning: boolean;
};

export default function App() {
  const [tab, setTab] = useState<AppTab>("inference");
  const [chain, setChain] = useState<Chain>("ethereum");
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [inference, setInference] = useState<InferenceState>({
    status: "preparing",
  });
  const [rpcStatus, setRpcStatus] = useState<RpcStatus>("checking");
  const [snapshot, setSnapshot] = useState<ChainSnapshot | null>(null);
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [engineRevision, setEngineRevision] = useState(0);
  const activeEngine = useRef<ActiveEngine | null>(null);
  const selectionRevision = useRef(0);
  const runsRef = useRef<InferenceRun[]>([]);
  const historyWrites = useRef<Promise<void>>(Promise.resolve());
  const mounted = useRef(true);
  const engineLifecycle = useRef<EngineLifecycle<InferenceEngine> | null>(null);
  if (engineLifecycle.current === null) {
    engineLifecycle.current = createEngineLifecycle<
      InferenceEngine,
      ChainSnapshot
    >({
      onConstructionError() {
        if (!mounted.current) return;
        setInference({
          status: "error",
          message: "Could not load the bundled models.",
        });
      },
      onDisposalError() {
        if (!mounted.current) return;
        setInference({
          status: "error",
          message: "Could not release the previous model.",
        });
      },
      onRpcUnavailable() {
        if (mounted.current) setSnapshot(null);
      },
      onSnapshot(engine, nextSnapshot) {
        if (!mounted.current) return;
        const current = activeEngine.current;
        if (current?.engine !== engine) return;
        setSnapshot(nextSnapshot);
        resolveOutcomes(current, nextSnapshot.head_block);
      },
      onStatus(status) {
        if (mounted.current) setRpcStatus(status);
      },
    });
  }
  const lifecycle = engineLifecycle.current;

  function commitRuns(
    update: (
      current: readonly InferenceRun[],
    ) => InferenceRun[] | Promise<InferenceRun[]>,
    isCurrent: () => boolean,
  ): Promise<InferenceRun[]> {
    const committed = historyWrites.current.then(async () => {
      const current = runsRef.current;
      if (!isCurrent()) return current;
      const next = await update(current);
      if (
        !isCurrent() ||
        (next.length === current.length &&
          next.every((run, index) => run === current[index]))
      ) {
        return current;
      }
      try {
        await saveRuns(next);
      } catch (error) {
        setStorageError(
          error instanceof Error ? error.message : String(error),
        );
        throw error;
      }
      runsRef.current = next;
      setRuns(next);
      setStorageError(null);
      return next;
    });
    historyWrites.current = committed.then(
      () => undefined,
      () => undefined,
    );
    return committed;
  }

  function resolveOutcomes(engine: ActiveEngine, headBlock: number): void {
    if (engine.outcomesRunning) {
      engine.outcomeHead = Math.max(engine.outcomeHead ?? 0, headBlock);
      return;
    }

    engine.outcomesRunning = true;
    void (async () => {
      let head = headBlock;
      try {
        while (activeEngine.current === engine) {
          engine.outcomeHead = null;
          await commitRuns(
            (current) =>
              resolvePendingRuns(
                current,
                engine.chain,
                head,
                engine.engine.resolveOutcome,
              ),
            () => activeEngine.current === engine,
          );
          if (engine.outcomeHead === null) break;
          head = engine.outcomeHead;
        }
      } catch {
        // Pending chain outcomes remain retryable on the next successful poll.
      } finally {
        engine.outcomesRunning = false;
        const queuedHead = engine.outcomeHead;
        if (activeEngine.current === engine && queuedHead !== null) {
          engine.outcomeHead = null;
          resolveOutcomes(engine, queuedHead);
        }
      }
    })();
  }

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const loaded = historyWrites.current.then(async () => {
      try {
        const storedRuns = await loadRuns();
        if (active) {
          runsRef.current = storedRuns;
          setRuns(storedRuns);
          setStorageError(null);
        }
      } catch (error) {
        if (active) {
          setStorageError(
            error instanceof Error ? error.message : String(error),
          );
        }
      }
    });
    historyWrites.current = loaded;
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    setSnapshot(null);
    let active = true;
    let activated: InferenceEngine | null = null;
    const lease = lifecycle.replace(() => createInferenceEngine(chain));
    void lease
      .then((engine) => {
        if (!active || engine === null) return;
        activated = engine;
        activeEngine.current = {
          chain,
          engine,
          outcomeHead: null,
          outcomesRunning: false,
        };
        setEngineRevision((revision) => revision + 1);
      })
      .catch(() => {
        if (!mounted.current) return;
        setInference({
          status: "error",
          message: "Could not start local inference.",
        });
      });

    return () => {
      active = false;
      selectionRevision.current += 1;
      if (activeEngine.current?.engine === activated) {
        activeEngine.current = null;
      }
      void lifecycle.release(lease).catch(() => {
        if (!mounted.current) return;
        setInference({
          status: "error",
          message: "Could not release the previous model.",
        });
      });
    };
  }, [chain, lifecycle]);

  useEffect(() => {
    const current = activeEngine.current;
    if (current === null || current.chain !== chain) return;

    const revision = selectionRevision.current + 1;
    selectionRevision.current = revision;
    let active = true;
    setInference({ status: "preparing" });
    void current.engine.prepare(horizon).then(
      () => {
        if (
          active &&
          activeEngine.current === current &&
          selectionRevision.current === revision
        ) {
          setInference({ status: "idle" });
        }
      },
      () => {
        if (
          active &&
          activeEngine.current === current &&
          selectionRevision.current === revision
        ) {
          setInference({ status: "idle" });
        }
      },
    );
    return () => {
      active = false;
    };
  }, [chain, engineRevision, horizon]);

  function selectChain(nextChain: Chain) {
    if (nextChain === chain) return;
    selectionRevision.current += 1;
    setInference({ status: "preparing" });
    setRpcStatus("checking");
    setSnapshot(null);
    setChain(nextChain);
  }

  function selectHorizon(nextHorizon: Horizon) {
    if (nextHorizon === horizon) return;
    selectionRevision.current += 1;
    setInference({ status: "preparing" });
    setHorizon(nextHorizon);
  }

  async function runInference() {
    const current = activeEngine.current;
    if (current === null || current.chain !== chain) {
      setInference({
        status: "error",
        message: "Could not connect to the selected chain.",
      });
      return;
    }
    const revision = selectionRevision.current;
    const isCurrent = () =>
      activeEngine.current === current &&
      selectionRevision.current === revision;

    setInference({ status: "loading" });
    let result;
    try {
      result = await current.engine.run(horizon);
    } catch (error) {
      if (
        isCurrent() &&
        !(error instanceof Error && error.name === "AbortError")
      ) {
        setInference({
          status: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
      return;
    }
    if (!isCurrent()) return;

    try {
      await commitRuns(
        (storedRuns) => addRun(storedRuns, result),
        isCurrent,
      );
    } catch {
      if (isCurrent()) {
        setInference({
          status: "error",
          message: "Could not save this run.",
        });
      }
      return;
    }
    if (isCurrent()) {
      setInference({ status: "success", result });
    }
  }

  return (
    <SafeAreaProvider>
      <StatusBar backgroundColor={colors.navy} barStyle="light-content" />
      <View style={styles.app}>
        <SafeAreaView edges={["top"]} style={styles.headerSafeArea}>
          <AppHeader chain={chain} status={rpcStatus} />
        </SafeAreaView>
        <View style={styles.content}>
          {tab === "inference" ? (
            <InferenceScreen
              chain={chain}
              horizon={horizon}
              onChainChange={selectChain}
              onHorizonChange={selectHorizon}
              onRun={() => void runInference()}
              onRunAgain={() => setInference({ status: "idle" })}
              snapshot={snapshot}
              state={inference}
            />
          ) : (
            <AnalyticsScreen
              chain={chain}
              horizon={horizon}
              onChainChange={selectChain}
              runs={runs}
              storageError={storageError}
            />
          )}
        </View>
        <SafeAreaView edges={["bottom"]} style={styles.tabSafeArea}>
          <BottomTabs onSelect={setTab} selected={tab} />
        </SafeAreaView>
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  app: { backgroundColor: colors.background, flex: 1 },
  headerSafeArea: { backgroundColor: colors.navy },
  content: { flex: 1 },
  tabSafeArea: { backgroundColor: colors.surface },
});
