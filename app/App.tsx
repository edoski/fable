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
  const [selection, setSelection] = useState(INITIAL_SELECTION);
  const [inference, setInference] = useState<InferenceState>({
    status: "idle",
  });
  const [rpcStatus, setRpcStatus] = useState<RpcStatus>("checking");
  const [snapshot, setSnapshot] = useState<ChainSnapshot | null>(null);
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const activeEngine = useRef<ActiveEngine | null>(null);
  const selectionIdentity = useRef(INITIAL_SELECTION);
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
      await saveRuns(next);
      runsRef.current = next;
      setRuns(next);
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
        setLoadError(null);
      } catch (error) {
        setLoadError(
          error instanceof Error ? error.message : String(error),
        );
      }
    });
  }, [serializeHistory]);

  useEffect(() => {
    const engine = createInferenceEngine(selection.chain);
    const current: ActiveEngine = {
      chain: selection.chain,
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
      if (activeEngine.current === current) {
        activeEngine.current = null;
      }
      void engine.dispose();
    };
  }, [selection.chain]);

  function select(next: Selection): void {
    const current = selectionIdentity.current;
    if (next.chain === current.chain && next.horizon === current.horizon) {
      return;
    }
    selectionIdentity.current = next;
    setInference({ status: "idle" });
    if (next.chain !== current.chain) {
      activeEngine.current = null;
      setRpcStatus("checking");
      setSnapshot(null);
    }
    setSelection(next);
  }

  function selectChain(chain: Chain): void {
    select({ ...selectionIdentity.current, chain });
  }

  async function runInference() {
    const selected = selectionIdentity.current;
    const current = activeEngine.current;
    if (current === null || current.chain !== selected.chain) {
      fail("Could not connect to the selected chain.");
      return;
    }
    const isCurrent = () =>
      activeEngine.current === current &&
      selectionIdentity.current === selected;

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
          <AppHeader chain={selection.chain} status={rpcStatus} />
        </SafeAreaView>
        <View style={styles.content}>
          {tab === "inference" ? (
            <InferenceScreen
              chain={selection.chain}
              horizon={selection.horizon}
              onChainChange={selectChain}
              onHorizonChange={(horizon) =>
                select({ ...selectionIdentity.current, horizon })
              }
              onRun={() => void runInference()}
              onRunAgain={() => setInference({ status: "idle" })}
              snapshot={snapshot}
              state={inference}
            />
          ) : (
            <AnalyticsScreen
              chain={selection.chain}
              horizon={selection.horizon}
              loadError={loadError}
              onChainChange={selectChain}
              runs={runs}
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
