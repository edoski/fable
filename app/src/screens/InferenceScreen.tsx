import { Ionicons } from "@expo/vector-icons";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { formatGwei } from "../analytics";
import {
  CHAINS,
  CHAIN_DETAILS,
  HORIZONS,
  type Chain,
  type Horizon,
  type InferenceResponse,
} from "../inference";
import { colors, radii } from "../theme";

export type InferenceState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: InferenceResponse }
  | { status: "error"; message: string };

type Props = {
  chain: Chain;
  horizon: Horizon;
  state: InferenceState;
  onChainChange: (chain: Chain) => void;
  onHorizonChange: (horizon: Horizon) => void;
  onRun: () => void;
  onRunAgain: () => void;
};

function NetworkChoices({
  chain,
  disabled,
  onChange,
}: {
  chain: Chain;
  disabled: boolean;
  onChange: (chain: Chain) => void;
}) {
  return (
    <View style={styles.networkRow}>
      {CHAINS.map((choice) => {
        const active = choice === chain;
        const details = CHAIN_DETAILS[choice];
        return (
          <Pressable
            accessibilityRole="radio"
            accessibilityState={{ checked: active, disabled }}
            disabled={disabled}
            key={choice}
            onPress={() => onChange(choice)}
            style={[styles.networkCard, active && styles.networkCardActive]}
          >
            {active && (
              <Ionicons color={colors.blue} name="checkmark-circle" size={19} style={styles.check} />
            )}
            <View style={[styles.networkMark, { backgroundColor: details.color }]}>
              <Text style={styles.networkMarkText}>{details.mark}</Text>
            </View>
            <Text numberOfLines={1} style={styles.networkLabel}>
              {details.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function HorizonChoices({
  horizon,
  disabled,
  onChange,
}: {
  horizon: Horizon;
  disabled: boolean;
  onChange: (horizon: Horizon) => void;
}) {
  return (
    <View style={styles.horizonRow}>
      {HORIZONS.map((choice) => {
        const active = choice === horizon;
        return (
          <Pressable
            accessibilityRole="radio"
            accessibilityState={{ checked: active, disabled }}
            disabled={disabled}
            key={choice}
            onPress={() => onChange(choice)}
            style={[styles.horizonChoice, active && styles.horizonChoiceActive]}
          >
            <Text style={[styles.horizonText, active && styles.horizonTextActive]}>{choice}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function Setup({ chain, horizon, state, onChainChange, onHorizonChange, onRun }: Props) {
  const loading = state.status === "loading";
  return (
    <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
      <Text style={styles.title}>Live inference</Text>
      <Text style={styles.subtitle}>Choose a network and prediction horizon.</Text>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Network</Text>
        <NetworkChoices chain={chain} disabled={loading} onChange={onChainChange} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Prediction horizon</Text>
        <HorizonChoices horizon={horizon} disabled={loading} onChange={onHorizonChange} />
      </View>

      <View style={styles.note}>
        <Ionicons color={colors.teal} name="information-circle-outline" size={24} />
        <Text style={styles.noteText}>
          Predicts the lowest base fee across the next {horizon} blocks.
        </Text>
      </View>

      {state.status === "error" && (
        <View accessibilityRole="alert" style={styles.errorBox}>
          <Ionicons color={colors.red} name="alert-circle-outline" size={21} />
          <Text style={styles.errorText}>{state.message}</Text>
        </View>
      )}

      <Pressable
        accessibilityRole="button"
        accessibilityState={{ disabled: loading }}
        disabled={loading}
        onPress={onRun}
        style={[styles.primaryButton, loading && styles.primaryButtonDisabled]}
      >
        {loading && <ActivityIndicator color={colors.surface} />}
        <Text style={styles.primaryButtonText}>
          {loading ? "Running inference" : "Run inference"}
        </Text>
      </Pressable>
    </ScrollView>
  );
}

function Timeline({ result, horizon }: { result: InferenceResponse; horizon: Horizon }) {
  return (
    <View style={styles.timeline}>
      <View style={[styles.timelineCell, styles.headCell]}>
        <Text style={styles.timelineLabel}>Head</Text>
        <Text numberOfLines={1} style={styles.timelineBlock}>
          {result.head_block.toLocaleString()}
        </Text>
      </View>
      {Array.from({ length: horizon }, (_, offset) => {
        const active = offset === result.selected_action_k;
        return (
          <View key={offset} style={[styles.timelineCell, active && styles.timelineCellActive]}>
            <Text style={[styles.timelineOffset, active && styles.timelineOffsetActive]}>+{offset}</Text>
            <Ionicons
              color={active ? colors.teal : colors.muted}
              name={active ? "cube" : "cube-outline"}
              size={22}
            />
            <Text style={[styles.targetLabel, active && styles.targetLabelActive]}>
              {active ? "TARGET" : " "}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function Result({ chain, horizon, result, onRunAgain }: Props & { result: InferenceResponse }) {
  const recommendation =
    result.selected_action_k === 0
      ? "Use the next block"
      : `Wait ${result.selected_action_k} ${result.selected_action_k === 1 ? "block" : "blocks"}`;
  return (
    <ScrollView contentContainerStyle={styles.resultPage} showsVerticalScrollIndicator={false}>
      <Text style={styles.title}>Inference</Text>
      <View style={styles.recommendation}>
        <View style={styles.successIcon}>
          <Ionicons color={colors.surface} name="checkmark" size={30} />
        </View>
        <View style={styles.recommendationCopy}>
          <Text style={styles.eyebrow}>Recommendation</Text>
          <Text style={styles.recommendationText}>{recommendation}</Text>
        </View>
      </View>

      <Timeline horizon={horizon} result={result} />

      <View style={styles.metricsRow}>
        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>Target block</Text>
          <Text adjustsFontSizeToFit numberOfLines={1} style={styles.metricValue}>
            {result.target_block.toLocaleString()}
          </Text>
        </View>
        <View style={styles.metricCard}>
          <Text style={styles.metricLabel}>Predicted horizon minimum</Text>
          <Text adjustsFontSizeToFit numberOfLines={1} style={styles.metricValue}>
            {formatGwei(result.predicted_minimum_base_fee_per_gas)}
          </Text>
        </View>
      </View>

      <View style={styles.explanation}>
        <Text style={styles.explanationText}>
          The auxiliary head estimates the minimum base fee anywhere within the selected horizon.
        </Text>
      </View>

      <View style={styles.detailsCard}>
        <Text style={styles.detailsTitle}>Technical details</Text>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Network</Text>
          <Text style={styles.detailValue}>{CHAIN_DETAILS[chain].label}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Horizon</Text>
          <Text style={styles.detailValue}>{horizon} blocks</Text>
        </View>
        <View style={styles.detailRowLast}>
          <Text style={styles.detailLabel}>Action offset</Text>
          <Text style={styles.detailValue}>{result.selected_action_k}</Text>
        </View>
      </View>

      <Pressable accessibilityRole="button" onPress={onRunAgain} style={styles.primaryButton}>
        <Ionicons color={colors.surface} name="refresh" size={21} />
        <Text style={styles.primaryButtonText}>Run again</Text>
      </Pressable>
    </ScrollView>
  );
}

export function InferenceScreen(props: Props) {
  if (props.state.status === "success") {
    return <Result {...props} result={props.state.result} />;
  }
  return <Setup {...props} />;
}

const styles = StyleSheet.create({
  page: { gap: 24, padding: 20, paddingBottom: 36 },
  resultPage: { gap: 18, padding: 18, paddingBottom: 30 },
  title: { color: colors.ink, fontSize: 30, fontWeight: "800" },
  subtitle: { color: colors.muted, fontSize: 16, marginTop: -16 },
  section: { gap: 11 },
  sectionTitle: { color: colors.ink, fontSize: 17, fontWeight: "700" },
  networkRow: { flexDirection: "row", gap: 9 },
  networkCard: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    gap: 8,
    justifyContent: "center",
    minHeight: 112,
    paddingHorizontal: 4,
    position: "relative",
  },
  networkCardActive: { backgroundColor: colors.blueSoft, borderColor: colors.blue },
  check: { position: "absolute", right: 7, top: 7 },
  networkMark: {
    alignItems: "center",
    borderRadius: 21,
    height: 42,
    justifyContent: "center",
    width: 42,
  },
  networkMarkText: { color: colors.surface, fontSize: 17, fontWeight: "800" },
  networkLabel: { color: colors.ink, fontSize: 12, fontWeight: "700" },
  horizonRow: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flexDirection: "row",
    overflow: "hidden",
  },
  horizonChoice: {
    alignItems: "center",
    borderRightColor: colors.border,
    borderRightWidth: StyleSheet.hairlineWidth,
    flex: 1,
    justifyContent: "center",
    minHeight: 54,
  },
  horizonChoiceActive: { backgroundColor: colors.blue },
  horizonText: { color: colors.ink, fontSize: 18, fontWeight: "700" },
  horizonTextActive: { color: colors.surface },
  note: {
    alignItems: "center",
    backgroundColor: colors.tealSoft,
    borderColor: "#A4E5DC",
    borderRadius: radii.medium,
    borderWidth: 1,
    flexDirection: "row",
    gap: 11,
    padding: 15,
  },
  noteText: { color: colors.ink, flex: 1, fontSize: 14, lineHeight: 20 },
  errorBox: {
    alignItems: "flex-start",
    backgroundColor: colors.redSoft,
    borderColor: "#FECACA",
    borderRadius: radii.medium,
    borderWidth: 1,
    flexDirection: "row",
    gap: 9,
    padding: 13,
  },
  errorText: { color: "#B42318", flex: 1, fontSize: 13, lineHeight: 19 },
  primaryButton: {
    alignItems: "center",
    backgroundColor: colors.blue,
    borderRadius: radii.medium,
    flexDirection: "row",
    gap: 10,
    justifyContent: "center",
    minHeight: 54,
    paddingHorizontal: 18,
  },
  primaryButtonDisabled: { opacity: 0.65 },
  primaryButtonText: { color: colors.surface, fontSize: 16, fontWeight: "700" },
  recommendation: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderLeftColor: colors.teal,
    borderLeftWidth: 5,
    borderRadius: radii.large,
    borderWidth: 1,
    flexDirection: "row",
    gap: 16,
    minHeight: 112,
    padding: 18,
  },
  successIcon: {
    alignItems: "center",
    backgroundColor: colors.teal,
    borderRadius: 28,
    height: 56,
    justifyContent: "center",
    width: 56,
  },
  recommendationCopy: { flex: 1, gap: 4 },
  eyebrow: { color: colors.muted, fontSize: 13, fontWeight: "600" },
  recommendationText: { color: colors.ink, fontSize: 28, fontWeight: "800" },
  timeline: { flexDirection: "row", gap: 5 },
  timelineCell: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.small,
    borderWidth: 1,
    flex: 1,
    gap: 5,
    justifyContent: "center",
    minHeight: 86,
    minWidth: 43,
    paddingHorizontal: 2,
  },
  headCell: { flex: 1.35 },
  timelineCellActive: { backgroundColor: colors.tealSoft, borderColor: colors.teal },
  timelineLabel: { color: colors.ink, fontSize: 11, fontWeight: "700" },
  timelineBlock: { color: colors.muted, fontSize: 8 },
  timelineOffset: { color: colors.ink, fontSize: 13, fontWeight: "700" },
  timelineOffsetActive: { color: colors.teal },
  targetLabel: { color: "transparent", fontSize: 7, fontWeight: "800" },
  targetLabelActive: { color: colors.teal },
  metricsRow: { flexDirection: "row", gap: 10 },
  metricCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    gap: 8,
    minHeight: 92,
    padding: 14,
  },
  metricLabel: { color: colors.muted, fontSize: 12, fontWeight: "600" },
  metricValue: { color: colors.ink, fontSize: 21, fontWeight: "800" },
  explanation: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    padding: 13,
  },
  explanationText: { color: colors.muted, fontSize: 12, lineHeight: 18 },
  detailsCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    paddingHorizontal: 14,
  },
  detailsTitle: { color: colors.ink, fontSize: 15, fontWeight: "700", paddingVertical: 14 },
  detailRow: {
    borderTopColor: colors.border,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 12,
  },
  detailRowLast: {
    borderTopColor: colors.border,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingBottom: 14,
    paddingTop: 12,
  },
  detailLabel: { color: colors.muted, fontSize: 13 },
  detailValue: { color: colors.ink, fontSize: 13, fontWeight: "600" },
});
