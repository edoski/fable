import { Ionicons } from "@expo/vector-icons";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { formatGwei } from "../analytics";
import { HorizonSlider } from "../components/HorizonSlider";
import { NetworkIcon } from "../components/NetworkIcon";
import {
  CHAINS,
  CHAIN_DETAILS,
  type Chain,
  type ChainSnapshot,
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
  snapshot: ChainSnapshot | null;
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
              <Ionicons
                color={colors.blue}
                name="checkmark-circle"
                size={19}
                style={styles.check}
              />
            )}
            <NetworkIcon chain={choice} />
            <Text numberOfLines={1} style={styles.networkLabel}>
              {details.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function LiveConditions({ snapshot }: { snapshot: ChainSnapshot | null }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Live conditions</Text>
      <View style={styles.conditionsCard}>
        <View style={styles.conditionRow}>
          <Text style={styles.conditionLabel}>Latest block</Text>
          <Text style={styles.conditionValue}>
            {snapshot?.head_block.toLocaleString() ?? "—"}
          </Text>
        </View>
        <View style={styles.conditionRowLast}>
          <Text style={styles.conditionLabel}>Current base fee</Text>
          <Text style={styles.conditionValue}>
            {snapshot ? formatGwei(snapshot.current_base_fee_per_gas) : "—"}
          </Text>
        </View>
      </View>
    </View>
  );
}

function PredictionWindow({
  disabled,
  horizon,
  onChange,
}: {
  disabled: boolean;
  horizon: Horizon;
  onChange: (horizon: Horizon) => void;
}) {
  return (
    <View style={styles.windowCard}>
      <View style={styles.windowTrack}>
        <View style={styles.windowHead}>
          <View style={styles.windowHeadNode}>
            <Ionicons color={colors.surface} name="cube" size={13} />
          </View>
          <Text style={styles.windowHeadLabel}>Head</Text>
        </View>
        <Ionicons color={colors.blue} name="arrow-forward" size={15} />
        <View style={styles.predictionGroup}>
          <Text style={styles.predictionSpaceLabel}>Prediction space</Text>
          <View style={styles.predictionSpace}>
            <View style={styles.predictionChain}>
              <View style={styles.predictionLine} />
              {Array.from({ length: horizon }, (_, offset) => (
                <View
                  accessibilityLabel={`Future block ${offset + 1}`}
                  key={offset}
                  style={styles.predictionBlock}
                >
                  <View style={styles.predictionNode}>
                    <Ionicons
                      color={colors.blue}
                      name="cube-outline"
                      size={15}
                    />
                  </View>
                  <Text style={styles.predictionNodeLabel}>{offset + 1}</Text>
                </View>
              ))}
            </View>
          </View>
        </View>
      </View>

      <HorizonSlider
        disabled={disabled}
        onChange={onChange}
        showTicks={false}
        value={horizon}
      />
    </View>
  );
}

function ErrorDialog({
  message,
  onClose,
  onRetry,
}: {
  message: string;
  onClose: () => void;
  onRetry: () => void;
}) {
  const description = message.startsWith("Network error:")
    ? "Could not connect to the inference server. Check that the serving module is running."
    : message;
  return (
    <Modal animationType="fade" onRequestClose={onClose} transparent visible>
      <View style={styles.errorDialogRoot}>
        <Pressable
          accessibilityLabel="Dismiss inference error"
          onPress={onClose}
          style={styles.errorBackdrop}
        />
        <View accessibilityRole="alert" style={styles.errorDialog}>
          <View style={styles.errorDialogIcon}>
            <Ionicons
              color={colors.red}
              name="alert-circle-outline"
              size={28}
            />
          </View>
          <Text style={styles.errorDialogTitle}>Inference failed</Text>
          <Text style={styles.errorDialogText}>{description}</Text>
          <View style={styles.errorActions}>
            <Pressable
              accessibilityRole="button"
              onPress={onClose}
              style={styles.dismissButton}
            >
              <Text style={styles.dismissButtonText}>Dismiss</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              onPress={onRetry}
              style={styles.retryButton}
            >
              <Ionicons color={colors.surface} name="refresh" size={17} />
              <Text style={styles.retryButtonText}>Retry</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

function Setup({
  chain,
  horizon,
  snapshot,
  state,
  onChainChange,
  onHorizonChange,
  onRun,
  onRunAgain,
}: Props) {
  const loading = state.status === "loading";
  return (
    <>
      <ScrollView
        contentContainerStyle={styles.page}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.title}>Inference</Text>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Network</Text>
          <NetworkChoices
            chain={chain}
            disabled={loading}
            onChange={onChainChange}
          />
        </View>

        <LiveConditions snapshot={snapshot} />

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            Prediction window (K = {horizon})
          </Text>
          <PredictionWindow
            disabled={loading}
            horizon={horizon}
            onChange={onHorizonChange}
          />
        </View>

        <Pressable
          accessibilityRole="button"
          accessibilityState={{ disabled: loading }}
          disabled={loading}
          onPress={onRun}
          style={[
            styles.primaryButton,
            styles.setupButton,
            loading && styles.primaryButtonDisabled,
          ]}
        >
          {loading && <ActivityIndicator color={colors.surface} />}
          <Text style={styles.primaryButtonText}>
            {loading ? "Generating…" : "Get recommendation"}
          </Text>
        </Pressable>
      </ScrollView>
      {state.status === "error" && (
        <ErrorDialog
          message={state.message}
          onClose={onRunAgain}
          onRetry={onRun}
        />
      )}
    </>
  );
}

function Timeline({
  result,
  horizon,
}: {
  result: InferenceResponse;
  horizon: Horizon;
}) {
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
          <View
            key={offset}
            style={[styles.timelineCell, active && styles.timelineCellActive]}
          >
            <Text
              style={[
                styles.timelineOffset,
                active && styles.timelineOffsetActive,
              ]}
            >
              +{offset}
            </Text>
            <Ionicons
              color={active ? colors.teal : colors.muted}
              name={active ? "cube" : "cube-outline"}
              size={22}
            />
            <Text
              style={[styles.targetLabel, active && styles.targetLabelActive]}
            >
              {active ? "TARGET" : " "}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

function Result({
  chain,
  horizon,
  result,
  onRunAgain,
}: Props & { result: InferenceResponse }) {
  const recommendation =
    result.selected_action_k === 0
      ? "Use the next block"
      : `Wait ${result.selected_action_k} ${result.selected_action_k === 1 ? "block" : "blocks"}`;
  return (
    <ScrollView
      contentContainerStyle={styles.resultPage}
      showsVerticalScrollIndicator={false}
    >
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
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Action offset</Text>
          <Text style={styles.detailValue}>{result.selected_action_k}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Target block</Text>
          <Text style={styles.detailValue}>
            {result.target_block.toLocaleString()}
          </Text>
        </View>
        <View style={styles.detailRowLast}>
          <Text style={styles.detailLabel}>Predicted horizon minimum</Text>
          <Text style={styles.detailValue}>
            {formatGwei(result.predicted_minimum_base_fee_per_gas)}
          </Text>
        </View>
      </View>

      <Pressable
        accessibilityRole="button"
        onPress={onRunAgain}
        style={styles.primaryButton}
      >
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
  page: { flexGrow: 1, gap: 22, padding: 20, paddingBottom: 24 },
  resultPage: { gap: 18, padding: 18, paddingBottom: 30 },
  title: { color: colors.ink, fontSize: 30, fontWeight: "800" },
  section: { gap: 11 },
  sectionTitle: { color: colors.ink, fontSize: 17, fontWeight: "700" },
  conditionsCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    paddingHorizontal: 14,
  },
  conditionRow: {
    alignItems: "center",
    borderBottomColor: colors.border,
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 46,
  },
  conditionRowLast: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 52,
  },
  conditionLabel: { color: colors.muted, fontSize: 12 },
  conditionValue: { color: colors.ink, fontSize: 12, fontWeight: "700" },
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
  networkCardActive: {
    backgroundColor: colors.blueSoft,
    borderColor: colors.blue,
  },
  check: { position: "absolute", right: 7, top: 7 },
  networkLabel: { color: colors.ink, fontSize: 12, fontWeight: "700" },
  windowCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    gap: 12,
    padding: 14,
  },
  windowTrack: {
    alignItems: "center",
    flexDirection: "row",
    gap: 7,
  },
  windowHead: { alignItems: "center", gap: 3, paddingTop: 12, width: 38 },
  windowHeadNode: {
    alignItems: "center",
    backgroundColor: colors.navy,
    borderRadius: 8,
    height: 30,
    justifyContent: "center",
    width: 30,
  },
  windowHeadLabel: { color: colors.muted, fontSize: 9, fontWeight: "600" },
  predictionGroup: { flex: 1, gap: 3 },
  predictionSpace: {
    backgroundColor: colors.blueSoft,
    borderColor: colors.blue,
    borderRadius: radii.medium,
    borderStyle: "dashed",
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 7,
  },
  predictionSpaceLabel: {
    color: colors.blue,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.6,
    textAlign: "center",
    textTransform: "uppercase",
  },
  predictionChain: {
    flexDirection: "row",
    justifyContent: "space-between",
    position: "relative",
  },
  predictionLine: {
    backgroundColor: "#AFC8FF",
    height: 2,
    left: 13,
    position: "absolute",
    right: 13,
    top: 13,
  },
  predictionBlock: { alignItems: "center", flex: 1, gap: 1 },
  predictionNode: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.blue,
    borderRadius: 7,
    borderWidth: 1,
    height: 27,
    justifyContent: "center",
    width: 27,
  },
  predictionNodeLabel: { color: colors.blue, fontSize: 8, fontWeight: "800" },
  errorDialogRoot: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    padding: 24,
  },
  errorBackdrop: {
    backgroundColor: "rgba(7, 20, 38, 0.58)",
    bottom: 0,
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  errorDialog: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderRadius: radii.large,
    gap: 10,
    padding: 20,
    width: "100%",
  },
  errorDialogIcon: {
    alignItems: "center",
    backgroundColor: colors.redSoft,
    borderRadius: 23,
    height: 46,
    justifyContent: "center",
    width: 46,
  },
  errorDialogTitle: { color: colors.ink, fontSize: 20, fontWeight: "800" },
  errorDialogText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
    textAlign: "center",
  },
  errorActions: { flexDirection: "row", gap: 9, marginTop: 6, width: "100%" },
  dismissButton: {
    alignItems: "center",
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    justifyContent: "center",
    minHeight: 46,
  },
  dismissButtonText: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  retryButton: {
    alignItems: "center",
    backgroundColor: colors.blue,
    borderRadius: radii.medium,
    flex: 1,
    flexDirection: "row",
    gap: 7,
    justifyContent: "center",
    minHeight: 46,
  },
  retryButtonText: { color: colors.surface, fontSize: 14, fontWeight: "700" },
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
  setupButton: { marginTop: "auto" },
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
  timelineCellActive: {
    backgroundColor: colors.tealSoft,
    borderColor: colors.teal,
  },
  timelineLabel: { color: colors.ink, fontSize: 11, fontWeight: "700" },
  timelineBlock: { color: colors.muted, fontSize: 8 },
  timelineOffset: { color: colors.ink, fontSize: 13, fontWeight: "700" },
  timelineOffsetActive: { color: colors.teal },
  targetLabel: { color: "transparent", fontSize: 7, fontWeight: "800" },
  targetLabelActive: { color: colors.teal },
  detailsCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    paddingHorizontal: 14,
  },
  detailsTitle: {
    color: colors.ink,
    fontSize: 15,
    fontWeight: "700",
    paddingVertical: 14,
  },
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
