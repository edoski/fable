import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBar, StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { AppHeader, type ServiceStatus } from "./src/components/AppHeader";
import { BottomTabs, type AppTab } from "./src/components/BottomTabs";
import { createRun, loadRuns, saveRuns, type InferenceRun } from "./src/history";
import {
  checkHealth,
  requestInference,
  type Chain,
  type Horizon,
} from "./src/inference";
import { AnalyticsScreen } from "./src/screens/AnalyticsScreen";
import { InferenceScreen, type InferenceState } from "./src/screens/InferenceScreen";
import { colors } from "./src/theme";

const HEALTH_INTERVAL_MS = 30_000;
const MAX_RUNS = 100;

export default function App() {
  const [tab, setTab] = useState<AppTab>("inference");
  const [chain, setChain] = useState<Chain>("ethereum");
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [inference, setInference] = useState<InferenceState>({ status: "idle" });
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>("checking");
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
          const currentRuns = runsRef.current;
          const currentIds = new Set(currentRuns.map((run) => run.id));
          const mergedRuns = [
            ...currentRuns,
            ...storedRuns.filter((run) => !currentIds.has(run.id)),
          ].slice(0, MAX_RUNS);
          publishRuns(mergedRuns);
          if (currentRuns.length > 0) {
            await saveRuns(mergedRuns);
          }
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setStorageError(error instanceof Error ? error.message : String(error));
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
      }
      try {
        await checkHealth(chain, controller.signal);
        if (active) {
          setServiceStatus("live");
        }
      } catch (error) {
        if (active && !(error instanceof Error && error.name === "AbortError")) {
          setServiceStatus("offline");
        }
      }
    }

    void probe(true);
    const interval = setInterval(() => void probe(false), HEALTH_INTERVAL_MS);
    return () => {
      active = false;
      controller?.abort();
      clearInterval(interval);
    };
  }, [chain]);

  useEffect(
    () => () => {
      inferenceController.current?.abort();
    },
    [],
  );

  const selectChain = useCallback((nextChain: Chain) => {
    inferenceController.current?.abort();
    setChain(nextChain);
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
          <AppHeader status={serviceStatus} />
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
              state={inference}
            />
          ) : (
            <AnalyticsScreen
              chain={chain}
              horizon={horizon}
              onChainChange={selectChain}
              onHorizonChange={selectHorizon}
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
