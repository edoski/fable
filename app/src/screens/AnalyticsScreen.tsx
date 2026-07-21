import { Ionicons } from "@expo/vector-icons";
import { useMemo, useState } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import {
  GRAPH_OPTIONS,
  chartData,
  formatGwei,
  formatRunDate,
  summarizeRuns,
  type GraphKind,
} from "../analytics";
import { BarChart } from "../components/BarChart";
import { ChoiceModal, PickerButton, type Choice } from "../components/ChoiceModal";
import type { InferenceRun } from "../history";
import {
  CHAINS,
  CHAIN_DETAILS,
  HORIZONS,
  type Chain,
  type Horizon,
} from "../inference";
import { colors, radii } from "../theme";

type Picker = "chain" | "horizon" | "graph" | null;

const CHAIN_OPTIONS: readonly Choice<Chain>[] = CHAINS.map((chain) => ({
  value: chain,
  label: CHAIN_DETAILS[chain].label,
}));
const HORIZON_OPTIONS: readonly Choice<Horizon>[] = HORIZONS.map((horizon) => ({
  value: horizon,
  label: `${horizon} blocks`,
}));

function MetricCard({ value, label, accent }: { value: string; label: string; accent?: boolean }) {
  return (
    <View style={styles.metricCard}>
      <Text style={[styles.metricValue, accent && styles.metricValueAccent]}>{value}</Text>
      <Text numberOfLines={2} style={styles.metricLabel}>
        {label}
      </Text>
    </View>
  );
}

function RunDetails({ run, onClose }: { run: InferenceRun | null; onClose: () => void }) {
  if (run === null) {
    return null;
  }
  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible>
      <View style={styles.dialogRoot}>
        <Pressable accessibilityLabel="Close run details" onPress={onClose} style={styles.backdrop} />
        <View style={styles.dialog}>
          <View style={styles.handle} />
          <View style={styles.dialogHeader}>
            <View>
              <Text style={styles.dialogTitle}>Run details</Text>
              <Text style={styles.dialogDate}>{formatRunDate(run.ran_at)}</Text>
            </View>
            <Pressable accessibilityLabel="Close" hitSlop={10} onPress={onClose}>
              <Ionicons color={colors.muted} name="close" size={27} />
            </Pressable>
          </View>

          <View style={styles.selectionSummary}>
            <View style={styles.selectionItem}>
              <Text style={styles.detailLabel}>Network</Text>
              <Text style={styles.detailStrong}>{CHAIN_DETAILS[run.chain].label}</Text>
            </View>
            <View style={styles.selectionItem}>
              <Text style={styles.detailLabel}>Horizon</Text>
              <Text style={styles.detailStrong}>{run.K} blocks</Text>
            </View>
          </View>

          <Text style={styles.groupTitle}>Prediction</Text>
          <View style={styles.detailsCard}>
            <Detail label="Head block" value={run.head_block.toLocaleString()} />
            <Detail label="Action offset" value={String(run.selected_action_k)} />
            <Detail label="Target block" value={run.target_block.toLocaleString()} />
            <Detail
              label="Predicted horizon minimum"
              last
              value={formatGwei(run.predicted_minimum_base_fee_per_gas)}
            />
          </View>
          <Text style={styles.disclaimer}>
            Base fee per gas. The regression value estimates the minimum anywhere within the horizon.
          </Text>
          <Pressable accessibilityRole="button" onPress={onClose} style={styles.closeButton}>
            <Text style={styles.closeButtonText}>Close</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

function Detail({ label, value, last = false }: { label: string; value: string; last?: boolean }) {
  return (
    <View style={[styles.detailRow, last && styles.detailRowLast]}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
    </View>
  );
}

export function AnalyticsScreen({
  runs,
  chain,
  horizon,
  storageError,
  onChainChange,
  onHorizonChange,
}: {
  runs: readonly InferenceRun[];
  chain: Chain;
  horizon: Horizon;
  storageError: string | null;
  onChainChange: (chain: Chain) => void;
  onHorizonChange: (horizon: Horizon) => void;
}) {
  const [picker, setPicker] = useState<Picker>(null);
  const [graph, setGraph] = useState<GraphKind>("offsets");
  const [selectedRun, setSelectedRun] = useState<InferenceRun | null>(null);
  const filteredRuns = useMemo(
    () => runs.filter((run) => run.chain === chain && run.K === horizon),
    [runs, chain, horizon],
  );
  const summary = useMemo(() => summarizeRuns(filteredRuns), [filteredRuns]);
  const data = useMemo(() => chartData(graph, filteredRuns, horizon), [graph, filteredRuns, horizon]);
  const graphLabel = GRAPH_OPTIONS.find((option) => option.value === graph)?.label ?? "Graph";

  return (
    <>
      <ScrollView contentContainerStyle={styles.page} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Analytics</Text>
        <View style={styles.filters}>
          <PickerButton label={CHAIN_DETAILS[chain].label} onPress={() => setPicker("chain")} />
          <PickerButton label={`${horizon} blocks`} onPress={() => setPicker("horizon")} />
        </View>

        {storageError && (
          <View accessibilityRole="alert" style={styles.storageError}>
            <Text style={styles.storageErrorText}>{storageError}</Text>
          </View>
        )}

        <View style={styles.metrics}>
          <MetricCard label="Total runs" value={String(summary.totalRuns)} />
          <MetricCard
            accent
            label="Immediate action"
            value={summary.immediatePercent === null ? "—" : `${summary.immediatePercent.toFixed(0)}%`}
          />
          <MetricCard
            label="Average offset"
            value={summary.averageOffset === null ? "—" : summary.averageOffset.toFixed(1)}
          />
        </View>

        <View style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Text style={styles.chartTitle}>
              {graph === "offsets"
                ? "Selected action distribution"
                : graph === "fees"
                  ? "Predicted horizon minimum"
                  : "Inference activity"}
            </Text>
            <View style={styles.graphPicker}>
              <PickerButton label={graphLabel} onPress={() => setPicker("graph")} />
            </View>
          </View>
          <BarChart data={data} />
        </View>

        <Text style={styles.sectionTitle}>Recent runs</Text>
        <View style={styles.runList}>
          {filteredRuns.length === 0 ? (
            <View style={styles.emptyRuns}>
              <Text style={styles.emptyRunsTitle}>No matching runs</Text>
              <Text style={styles.emptyRunsText}>Run an inference with these filters to begin.</Text>
            </View>
          ) : (
            filteredRuns.slice(0, 10).map((run, index) => (
              <Pressable
                accessibilityHint="Opens run details"
                accessibilityRole="button"
                key={run.id}
                onPress={() => setSelectedRun(run)}
                style={[styles.runRow, index === filteredRuns.length - 1 && styles.runRowLast]}
              >
                <View style={styles.runIcon}>
                  <Ionicons color={colors.blue} name="git-branch-outline" size={22} />
                </View>
                <View style={styles.runCopy}>
                  <Text style={styles.runDate}>{formatRunDate(run.ran_at)}</Text>
                  <Text style={styles.runMeta}>Target {run.target_block.toLocaleString()}</Text>
                  <Text style={styles.runMeta}>Offset {run.selected_action_k}</Text>
                </View>
                <Ionicons color={colors.muted} name="chevron-forward" size={21} />
              </Pressable>
            ))
          )}
        </View>
      </ScrollView>

      <ChoiceModal
        onClose={() => setPicker(null)}
        onSelect={onChainChange}
        options={CHAIN_OPTIONS}
        selected={chain}
        title="Network"
        visible={picker === "chain"}
      />
      <ChoiceModal
        onClose={() => setPicker(null)}
        onSelect={onHorizonChange}
        options={HORIZON_OPTIONS}
        selected={horizon}
        title="Prediction horizon"
        visible={picker === "horizon"}
      />
      <ChoiceModal
        onClose={() => setPicker(null)}
        onSelect={setGraph}
        options={GRAPH_OPTIONS}
        selected={graph}
        title="Graph"
        visible={picker === "graph"}
      />
      <RunDetails onClose={() => setSelectedRun(null)} run={selectedRun} />
    </>
  );
}

const styles = StyleSheet.create({
  page: { gap: 20, padding: 18, paddingBottom: 30 },
  title: { color: colors.ink, fontSize: 30, fontWeight: "800" },
  filters: { flexDirection: "row", gap: 10 },
  storageError: {
    backgroundColor: colors.redSoft,
    borderColor: "#FECACA",
    borderRadius: radii.medium,
    borderWidth: 1,
    padding: 12,
  },
  storageErrorText: { color: "#B42318", fontSize: 12 },
  metrics: { flexDirection: "row", gap: 9 },
  metricCard: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    gap: 5,
    justifyContent: "center",
    minHeight: 98,
    padding: 9,
  },
  metricValue: { color: colors.blue, fontSize: 25, fontWeight: "800" },
  metricValueAccent: { color: colors.teal },
  metricLabel: { color: colors.muted, fontSize: 10, lineHeight: 14, textAlign: "center" },
  chartCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    gap: 14,
    padding: 14,
  },
  chartHeader: { alignItems: "center", flexDirection: "row", gap: 10, justifyContent: "space-between" },
  chartTitle: { color: colors.ink, flex: 1, fontSize: 15, fontWeight: "700" },
  graphPicker: { maxWidth: 142, minWidth: 118 },
  sectionTitle: { color: colors.ink, fontSize: 20, fontWeight: "700", marginBottom: -8 },
  runList: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    overflow: "hidden",
  },
  runRow: {
    alignItems: "center",
    borderBottomColor: colors.border,
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    gap: 12,
    minHeight: 78,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  runRowLast: { borderBottomWidth: 0 },
  runIcon: {
    alignItems: "center",
    backgroundColor: colors.blueSoft,
    borderRadius: radii.small,
    height: 42,
    justifyContent: "center",
    width: 42,
  },
  runCopy: { flex: 1, gap: 2 },
  runDate: { color: colors.ink, fontSize: 13, fontWeight: "700" },
  runMeta: { color: colors.muted, fontSize: 12 },
  emptyRuns: { alignItems: "center", gap: 4, padding: 28 },
  emptyRunsTitle: { color: colors.ink, fontSize: 15, fontWeight: "700" },
  emptyRunsText: { color: colors.muted, fontSize: 12, textAlign: "center" },
  dialogRoot: { flex: 1, justifyContent: "flex-end" },
  backdrop: {
    backgroundColor: "rgba(7, 20, 38, 0.58)",
    bottom: 0,
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  dialog: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    gap: 14,
    paddingBottom: 28,
    paddingHorizontal: 18,
    paddingTop: 9,
  },
  handle: {
    alignSelf: "center",
    backgroundColor: colors.border,
    borderRadius: 3,
    height: 5,
    width: 48,
  },
  dialogHeader: { alignItems: "flex-start", flexDirection: "row", justifyContent: "space-between" },
  dialogTitle: { color: colors.ink, fontSize: 24, fontWeight: "800" },
  dialogDate: { color: colors.muted, fontSize: 13, marginTop: 2 },
  selectionSummary: {
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flexDirection: "row",
    padding: 12,
  },
  selectionItem: { flex: 1, gap: 3 },
  detailStrong: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  groupTitle: { color: colors.blue, fontSize: 15, fontWeight: "700" },
  detailsCard: {
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    overflow: "hidden",
  },
  detailRow: {
    borderBottomColor: colors.border,
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 12,
    paddingVertical: 11,
  },
  detailRowLast: { borderBottomWidth: 0 },
  detailLabel: { color: colors.muted, fontSize: 12 },
  detailValue: { color: colors.ink, fontSize: 12, fontWeight: "600" },
  disclaimer: { color: colors.muted, fontSize: 11, lineHeight: 16 },
  closeButton: {
    alignItems: "center",
    backgroundColor: colors.blue,
    borderRadius: radii.medium,
    justifyContent: "center",
    minHeight: 50,
  },
  closeButtonText: { color: colors.surface, fontSize: 15, fontWeight: "700" },
});
