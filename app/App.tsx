import { useEffect, useRef, useState } from "react";
import { StatusBar, StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import {
  AppHeader,
  type RpcStatus,
} from "./src/components/AppHeader";
import { BottomTabs, type AppTab } from "./src/components/BottomTabs";
import type { Chain, Horizon } from "./src/domain";
import {
  addRun,
  loadRuns,
  resolvePendingRuns,
  saveRuns,
  type InferenceRun,
} from "./src/history";
import {
  createInferenceEngine,
  type ChainSnapshot,
  type InferenceEngine,
} from "./src/inference";
import { AnalyticsScreen } from "./src/screens/AnalyticsScreen";
import {
  InferenceScreen,
  type InferenceState,
} from "./src/screens/InferenceScreen";
import { createSerialQueue } from "./src/serialQueue";
import { colors } from "./src/theme";

type ActiveEngine = {
  chain: Chain;
  engine: InferenceEngine;
  outcomesRunning: boolean;
  revision: number;
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
  const engineRevisionSequence = useRef(0);
  const selectionRevision = useRef(0);
  const runsRef = useRef<InferenceRun[]>([]);
  const serializeHistory = useRef(createSerialQueue()).current;

  function fail(message: string): void {
    setInference({ status: "error", message });
  }

  function commitRuns(
    update: (
      current: readonly InferenceRun[],
    ) => InferenceRun[] | Promise<InferenceRun[]>,
    isCurrent: () => boolean,
  ): Promise<void> {
    return serializeHistory(async () => {
      const current = runsRef.current;
      if (!isCurrent()) return;
      const next = await update(current);
      if (
        !isCurrent() ||
        (next.length === current.length &&
          next.every((run, index) => run === current[index]))
      ) {
        return;
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
    });
  }

  function resolveOutcomes(engine: ActiveEngine, headBlock: number): void {
    if (engine.outcomesRunning) return;

    engine.outcomesRunning = true;
    void (async () => {
      try {
        await commitRuns(
          (current) =>
            resolvePendingRuns(
              current,
              engine.chain,
              headBlock,
              engine.engine.resolveOutcome,
            ),
          () => activeEngine.current === engine,
        );
      } catch {
        // Pending chain outcomes remain retryable on the next successful poll.
      } finally {
        engine.outcomesRunning = false;
      }
    })();
  }

  useEffect(() => {
    let active = true;
    void serializeHistory(async () => {
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
    return () => {
      active = false;
    };
  }, [serializeHistory]);

  useEffect(() => {
    const engine = createInferenceEngine(chain);
    const revision = engineRevisionSequence.current + 1;
    engineRevisionSequence.current = revision;
    const current: ActiveEngine = {
      chain,
      engine,
      outcomesRunning: false,
      revision,
    };
    activeEngine.current = current;
    setRpcStatus("checking");
    const onStatus = (status: RpcStatus) => {
      if (activeEngine.current !== current) return;
      setRpcStatus(status);
      if (status === "offline") setSnapshot(null);
    };
    engine.startPolling(
      (nextSnapshot) => {
        if (activeEngine.current !== current) return;
        onStatus("live");
        setSnapshot(nextSnapshot);
        resolveOutcomes(current, nextSnapshot.head_block);
      },
      () => onStatus("offline"),
    );
    setEngineRevision(revision);

    return () => {
      selectionRevision.current += 1;
      if (activeEngine.current === current) {
        activeEngine.current = null;
      }
      void engine.dispose();
    };
  }, [chain]);

  useEffect(() => {
    const current = activeEngine.current;
    if (
      current === null ||
      current.chain !== chain ||
      current.revision !== engineRevision
    ) {
      return;
    }

    const revision = selectionRevision.current + 1;
    selectionRevision.current = revision;
    let active = true;
    setInference({ status: "preparing" });
    const settlePreparation = () => {
      if (
        active &&
        activeEngine.current === current &&
        selectionRevision.current === revision
      ) {
        setInference({ status: "idle" });
      }
    };
    void current.engine.prepare(horizon).then(
      settlePreparation,
      settlePreparation,
    );
    return () => {
      active = false;
    };
  }, [chain, engineRevision, horizon]);

  function selectChain(nextChain: Chain) {
    if (nextChain === chain) return;
    activeEngine.current = null;
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
      fail("Could not connect to the selected chain.");
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
        fail(error instanceof Error ? error.message : String(error));
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
        fail("Could not save this run.");
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
