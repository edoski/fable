import { useState } from "react";
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import {
  CHAINS,
  HORIZONS,
  requestInference,
  type Chain,
  type Horizon,
  type InferenceRequest,
  type InferenceResponse,
} from "./src/inference";

export default function App() {
  const [chain, setChain] = useState<Chain>("ethereum");
  const [K, setK] = useState<Horizon>(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function selectChain(nextChain: Chain) {
    setChain(nextChain);
    setResult(null);
    setError(null);
  }

  function selectHorizon(nextHorizon: Horizon) {
    setK(nextHorizon);
    setResult(null);
    setError(null);
  }

  async function runInference() {
    const request: InferenceRequest = { chain, K };
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      setResult(await requestInference(request));
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.page}>
        <Text style={styles.title}>Inference</Text>

        <View style={styles.section}>
          <Text style={styles.label}>Chain</Text>
          <View style={styles.choices}>
            {CHAINS.map((choice) => (
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ disabled: loading, selected: choice === chain }}
                disabled={loading}
                key={choice}
                onPress={() => selectChain(choice)}
                style={[styles.choice, choice === chain && styles.choiceSelected]}
              >
                <Text
                  style={[
                    styles.choiceText,
                    choice === chain && styles.choiceTextSelected,
                  ]}
                >
                  {choice}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>K (blocks)</Text>
          <View style={styles.choices}>
            {HORIZONS.map((choice) => (
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ disabled: loading, selected: choice === K }}
                disabled={loading}
                key={choice}
                onPress={() => selectHorizon(choice)}
                style={[styles.choice, choice === K && styles.choiceSelected]}
              >
                <Text
                  style={[
                    styles.choiceText,
                    choice === K && styles.choiceTextSelected,
                  ]}
                >
                  {choice} blocks
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <Pressable
          accessibilityRole="button"
          disabled={loading}
          onPress={runInference}
          style={[styles.runButton, loading && styles.runButtonDisabled]}
        >
          <Text style={styles.runButtonText}>
            {loading ? "Running inference" : "Run inference"}
          </Text>
        </Pressable>

        {result && (
          <View style={styles.result}>
            <Text>Head block: {result.head_block}</Text>
            <Text>Selected action k: {result.selected_action_k}</Text>
            <Text>Target block: {result.target_block}</Text>
          </View>
        )}

        {error && <Text style={styles.error}>{error}</Text>}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: "#f6f8fa", flex: 1 },
  page: { gap: 18, padding: 20 },
  title: { fontSize: 28, fontWeight: "700" },
  section: { gap: 10 },
  label: { fontSize: 16, fontWeight: "700" },
  choices: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  choice: {
    backgroundColor: "#ffffff",
    borderColor: "#8c959f",
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  choiceSelected: { backgroundColor: "#1f6feb", borderColor: "#1f6feb" },
  choiceText: { color: "#24292f", fontWeight: "600" },
  choiceTextSelected: { color: "#ffffff" },
  runButton: {
    alignItems: "center",
    backgroundColor: "#1f6feb",
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  runButtonDisabled: { backgroundColor: "#8c959f" },
  runButtonText: { color: "#ffffff", fontWeight: "700" },
  result: {
    backgroundColor: "#ffffff",
    borderColor: "#d0d7de",
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
    padding: 14,
  },
  error: { color: "#cf222e" },
});
