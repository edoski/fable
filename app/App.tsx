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
};

type Selection = {
  chain: Chain;
  horizon: Horizon;
};

const INITIAL_SELECTION: Selection = {
  chain: "ethereum",
  horizon: 5,
};

export default function App() {
  const [tab, setTab] = useState<AppTab>("inference");
  const [chain, setChain] = useState<Chain>(INITIAL_SELECTION.chain);
  const [horizon, setHorizon] = useState<Horizon>(
    INITIAL_SELECTION.horizon,
  );
  const [inference, setInference] = useState<InferenceState>({
    status: "idle",
  });
  const [rpcStatus, setRpcStatus] = useState<RpcStatus>("checking");
  const [snapshot, setSnapshot] = useState<ChainSnapshot | null>(null);
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [storageError, setStorageError] = useState<string | null>(null);
  const activeEngine = useRef<ActiveEngine | null>(null);
  const selectionRevision = useRef(0);
  const selection = useRef(INITIAL_SELECTION);
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
    void commitRuns(
      (current) =>
        resolvePendingRuns(
          current,
          engine.chain,
          headBlock,
          engine.engine.resolveOutcome,
        ),
      () => activeEngine.current === engine,
    ).catch(() => {
      // Pending chain outcomes remain retryable on the next successful poll.
    });
  }

  useEffect(() => {
    void serializeHistory(async () => {
      try {
        const storedRuns = await loadRuns();
        runsRef.current = storedRuns;
        setRuns(storedRuns);
        setStorageError(null);
      } catch (error) {
        setStorageError(
          error instanceof Error ? error.message : String(error),
        );
      }
    });
  }, [serializeHistory]);

  useEffect(() => {
    const engine = createInferenceEngine(chain);
    const current: ActiveEngine = {
      chain,
      engine,
    };
    activeEngine.current = current;
    setRpcStatus("checking");
    const onStatus = (status: RpcStatus) => {
      if (activeEngine.current !== current) return;
      setRpcStatus(status);
      if (status === "offline") setSnapshot(null);
    };
    engine.watchBlocks(
      (nextSnapshot) => {
        if (activeEngine.current !== current) return;
        onStatus("live");
        setSnapshot(nextSnapshot);
        resolveOutcomes(current, nextSnapshot.head_block);
      },
      () => onStatus("offline"),
    );

    return () => {
      selectionRevision.current += 1;
      if (activeEngine.current === current) {
        activeEngine.current = null;
      }
      void engine.dispose();
    };
  }, [chain]);

  function selectChain(nextChain: Chain) {
    const current = selection.current;
    if (nextChain === current.chain) return;
    selection.current = { ...current, chain: nextChain };
    selectionRevision.current += 1;
    activeEngine.current = null;
    setInference({ status: "idle" });
    setRpcStatus("checking");
    setSnapshot(null);
    setChain(nextChain);
  }

  function selectHorizon(nextHorizon: Horizon) {
    const current = selection.current;
    if (nextHorizon === current.horizon) return;
    selection.current = {
      ...current,
      horizon: nextHorizon,
    };
    selectionRevision.current += 1;
    setInference({ status: "idle" });
    setHorizon(nextHorizon);
  }

  async function runInference() {
    const selected = selection.current;
    const current = activeEngine.current;
    if (current === null || current.chain !== selected.chain) {
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
      result = await current.engine.run(selected.horizon);
    } catch (error) {
      if (isCurrent()) {
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
