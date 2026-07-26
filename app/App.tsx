import { useEffect, useRef, useState } from "react";
import { StatusBar, StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { AppHeader, type ServiceStatus } from "./src/components/AppHeader";
import { BottomTabs, type AppTab } from "./src/components/BottomTabs";
import { loadRuns, type InferenceRun } from "./src/history";
import {
  requestChainSnapshot,
  requestInference,
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

  useEffect(() => {
    let active = true;
    loadRuns()
      .then((storedRuns) => {
        if (active) {
          setRuns(storedRuns);
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
  }, []);

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

  useEffect(
    () => () => {
      inferenceController.current?.abort();
    },
    [],
  );

  function selectChain(nextChain: Chain) {
    inferenceController.current?.abort();
    setChain(nextChain);
    setSnapshot(null);
    setInference({ status: "idle" });
  }

  function selectHorizon(nextHorizon: Horizon) {
    inferenceController.current?.abort();
    setHorizon(nextHorizon);
    setInference({ status: "idle" });
  }

  async function runInference() {
    inferenceController.current?.abort();
    const controller = new AbortController();
    inferenceController.current = controller;
    setInference({ status: "loading" });
    const request = { chain, K: horizon } as const;
    try {
      const result = await requestInference(request, controller.signal);
      setInference({ status: "success", result });
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
  }

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
