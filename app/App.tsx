import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBar, StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { AppHeader, type ServiceStatus } from "./src/components/AppHeader";
import { BottomTabs, type AppTab } from "./src/components/BottomTabs";
import { createDemoRuns } from "./src/demo";
import {
  createRun,
  loadRuns,
  recordOutcome,
  saveRuns,
  type InferenceRun,
} from "./src/history";
import {
  requestChainSnapshot,
  requestInference,
  requestInferenceOutcome,
  type Chain,
  type ChainSnapshot,
  type Horizon,
} from "./src/inference";
import { AnalyticsScreen } from "./src/screens/AnalyticsScreen";
import {
  InferenceScreen,
  type InferenceState,
} from "./src/screens/InferenceScreen";
import { colors } from "./src/theme";

const SNAPSHOT_INTERVAL_MS = 1_000;
const MAX_RUNS = 100;

export default function App() {
  const [tab, setTab] = useState<AppTab>("inference");
  const [chain, setChain] = useState<Chain>("ethereum");
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [inference, setInference] = useState<InferenceState>({
    status: "idle",
  });
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>("checking");
  const [snapshot, setSnapshot] = useState<ChainSnapshot | null>(null);
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [storageError, setStorageError] = useState<string | null>(null);
  const inferenceController = useRef<AbortController | null>(null);
  const runsRef = useRef<InferenceRun[]>([]);

  const publishRuns = useCallback((nextRuns: InferenceRun[]) => {
    runsRef.current = nextRuns;
    setRuns(nextRuns);
  }, []);

  useEffect(() => {
    let active = true;
    loadRuns()
      .then(async (storedRuns) => {
        if (active) {
          const loadedRuns = [
            ...createDemoRuns(),
            ...storedRuns.filter((run) => !run.id.startsWith("demo:")),
          ];
          const currentRuns = runsRef.current;
          const currentIds = new Set(currentRuns.map((run) => run.id));
          const mergedRuns = [
            ...currentRuns,
            ...loadedRuns.filter((run) => !currentIds.has(run.id)),
          ].slice(0, MAX_RUNS);
          publishRuns(mergedRuns);
          await saveRuns(mergedRuns);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setStorageError(
            error instanceof Error ? error.message : String(error),
          );
        }
      });
    return () => {
      active = false;
    };
  }, [publishRuns]);

  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;

    async function probe(checking: boolean) {
      controller?.abort();
      controller = new AbortController();
      if (checking) {
        setServiceStatus("checking");
        setSnapshot(null);
      }
      try {
        const nextSnapshot = await requestChainSnapshot(
          chain,
          controller.signal,
        );
        if (active) {
          setServiceStatus("live");
          setSnapshot(nextSnapshot);
        }
      } catch (error) {
        if (
          active &&
          !(error instanceof Error && error.name === "AbortError")
        ) {
          setServiceStatus("offline");
          setSnapshot(null);
        }
      }
    }

    void probe(true);
    const interval = setInterval(() => void probe(false), SNAPSHOT_INTERVAL_MS);
    return () => {
      active = false;
      controller?.abort();
      clearInterval(interval);
    };
  }, [chain]);

  useEffect(() => {
    if (snapshot === null) {
      return;
    }
    const pendingRuns = runs.filter(
      (run) =>
        run.chain === snapshot.chain &&
        run.outcome === undefined &&
        run.target_block <= snapshot.head_block,
    );
    if (pendingRuns.length === 0) {
      return;
    }
    const controller = new AbortController();
    Promise.all(
      pendingRuns.map(async (run) => ({
        id: run.id,
        outcome: await requestInferenceOutcome(
          run.chain,
          run.head_block + 1,
          run.target_block,
          controller.signal,
        ),
      })),
    )
      .then(async (resolved) => {
        if (controller.signal.aborted) {
          return;
        }
        const outcomes = new Map(
          resolved.map(({ id, outcome }) => [id, outcome]),
        );
        const nextRuns = runsRef.current.map((run) => {
          const outcome = outcomes.get(run.id);
          return outcome === undefined || run.outcome !== undefined
            ? run
            : recordOutcome(run, outcome);
        });
        publishRuns(nextRuns);
        try {
          await saveRuns(nextRuns);
          setStorageError(null);
        } catch (error) {
          setStorageError(
            error instanceof Error ? error.message : String(error),
          );
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [publishRuns, runs, snapshot]);

  useEffect(
    () => () => {
      inferenceController.current?.abort();
    },
    [],
  );

  const selectChain = useCallback((nextChain: Chain) => {
    inferenceController.current?.abort();
    setChain(nextChain);
    setSnapshot(null);
    setInference({ status: "idle" });
  }, []);

  const selectHorizon = useCallback((nextHorizon: Horizon) => {
    inferenceController.current?.abort();
    setHorizon(nextHorizon);
    setInference({ status: "idle" });
  }, []);

  const runInference = useCallback(async () => {
    inferenceController.current?.abort();
    const controller = new AbortController();
    inferenceController.current = controller;
    setInference({ status: "loading" });
    const request = { chain, K: horizon } as const;
    try {
      const result = await requestInference(request, controller.signal);
      const run = createRun(request, result);
      const nextRuns = [run, ...runsRef.current].slice(0, MAX_RUNS);
      setInference({ status: "success", result });
      publishRuns(nextRuns);
      setStorageError(null);
      try {
        await saveRuns(nextRuns);
      } catch (error) {
        setStorageError(error instanceof Error ? error.message : String(error));
      }
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        setInference({
          status: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      if (inferenceController.current === controller) {
        inferenceController.current = null;
      }
    }
  }, [chain, horizon, publishRuns]);

  return (
    <SafeAreaProvider>
      <StatusBar backgroundColor={colors.navy} barStyle="light-content" />
      <View style={styles.app}>
        <SafeAreaView edges={["top"]} style={styles.headerSafeArea}>
          <AppHeader chain={chain} status={serviceStatus} />
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
