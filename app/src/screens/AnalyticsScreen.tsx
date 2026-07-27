import { Ionicons } from "@expo/vector-icons";
import { useState } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  Text,
  View,
} from "react-native";
import { BarChart as GiftedBarChart } from "react-native-gifted-charts";

import {
  GRAPH_OPTIONS,
  feeComparisonData,
  formatGwei,
  formatRunDate,
  recommendedWaitData,
  realizedSavingsPercent,
  runsForSelection,
  savingsByWaitData,
  summarizeRuns,
} from "../analytics";
import { DetailRow } from "../components/DetailRow";
import { HorizonSlider } from "../components/HorizonSlider";
import { NetworkIcon } from "../components/NetworkIcon";
import { CHAINS, CHAIN_DETAILS, type Chain, type Horizon } from "../domain";
import type { InferenceRun } from "../history";
import { styles } from "../styles";
import { colors, radii } from "../theme";

function SummaryCard({
  value,
  label,
  accent = false,
}: {
  value: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <View style={[styles.surface, styles.summaryCard]}>
      <Text style={[styles.summaryValue, accent && styles.summaryValueAccent]}>
        {value}
      </Text>
      <Text numberOfLines={1} style={styles.summaryLabel}>
        {label}
      </Text>
    </View>
  );
}

function formatSavings(value: number): string {
  return `${value.toFixed(1)}%`;
}

const CHART_HEIGHT = 138;

function niceStep(range: number): number {
  const rough = range / 3;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-9)));
  const normalized = rough / magnitude;
  const multiplier =
    normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function EmptyGraph({ outcomes }: { outcomes: boolean }) {
  return (
    <View style={styles.emptyGraph}>
      <Text style={styles.emptyGraphTitle}>
        {outcomes ? "No outcomes yet" : "No runs yet"}
      </Text>
      <Text style={styles.emptyGraphText}>
        {outcomes
          ? "Resolved inferences will populate this graph."
          : "Runs will populate this graph."}
      </Text>
    </View>
  );
}

function chartScale(values: readonly number[]) {
  const rawMinimum = Math.min(0, ...values);
  const rawMaximum = Math.max(0, ...values);
  const step = niceStep(
    rawMinimum === rawMaximum ? 1 : rawMaximum - rawMinimum,
  );
  const minimum = Math.floor(rawMinimum / step) * step;
  const maximum = Math.max(step, Math.ceil(rawMaximum / step) * step);
  const positiveSections = Math.round(maximum / step);
  const negativeSections = Math.round(Math.abs(minimum) / step);

  return {
    maximum,
    minimum,
    negativeSections,
    positiveSections,
    step,
    stepHeight:
      CHART_HEIGHT / Math.max(positiveSections + negativeSections, 1),
  };
}

function RecommendedWaitChart({
  runs,
  horizon,
}: {
  runs: readonly InferenceRun[];
  horizon: Horizon;
}) {
  const data = recommendedWaitData(runs, horizon);
  if (data.length === 0) {
    return <EmptyGraph outcomes={false} />;
  }
  const scale = chartScale(data.map((item) => item.value ?? 0));

  return (
    <View style={styles.graph}>
      <GiftedBarChart
        barBorderRadius={radii.small / 2}
        data={data.map((item) => ({
          frontColor: colors.blue,
          label: item.label,
          value: item.value ?? 0,
        }))}
        disablePress
        endSpacing={10}
        initialSpacing={10}
        maxValue={scale.maximum}
        noOfSections={scale.positiveSections}
        rulesColor={colors.border}
        stepHeight={scale.stepHeight}
        stepValue={scale.step}
        xAxisColor={colors.muted}
        xAxisLabelsAtBottom
        xAxisLabelsHeight={14}
        xAxisLabelTextStyle={styles.graphAxisText}
        yAxisLabelWidth={34}
        yAxisTextStyle={styles.graphAxisText}
        yAxisThickness={0}
      />
      <Text style={styles.graphXAxisTitle}>Wait (blocks)</Text>
    </View>
  );
}

function SavingsByWaitChart({
  runs,
  horizon,
}: {
  runs: readonly InferenceRun[];
  horizon: Horizon;
}) {
  const data = savingsByWaitData(runs, horizon);
  const values = data.flatMap((item) =>
    item.value === null ? [] : [item.value],
  );
  if (values.length === 0) {
    return <EmptyGraph outcomes />;
  }
  const scale = chartScale(values);

  return (
    <View style={styles.graph}>
      <GiftedBarChart
        barBorderRadius={radii.small / 2}
        data={data.map((item) => ({
          frontColor:
            item.value !== null && item.value < 0 ? colors.red : colors.teal,
          label: item.label,
          value: item.value ?? 0,
        }))}
        disablePress
        endSpacing={10}
        formatYLabel={(label) => `${Number(label).toFixed(0)}%`}
        initialSpacing={10}
        maxValue={scale.maximum}
        mostNegativeValue={scale.minimum}
        negativeStepValue={scale.step}
        noOfSections={scale.positiveSections}
        noOfSectionsBelowXAxis={scale.negativeSections}
        rulesColor={colors.border}
        stepHeight={scale.stepHeight}
        stepValue={scale.step}
        xAxisColor={colors.muted}
        xAxisLabelsAtBottom
        xAxisLabelsHeight={14}
        xAxisLabelTextStyle={styles.graphAxisText}
        yAxisLabelWidth={34}
        yAxisTextStyle={styles.graphAxisText}
        yAxisThickness={0}
      />
      <Text style={styles.graphXAxisTitle}>Wait (blocks)</Text>
    </View>
  );
}

function BaseFeeByWaitChart({
  runs,
  horizon,
}: {
  runs: readonly InferenceRun[];
  horizon: Horizon;
}) {
  const data = feeComparisonData(runs, horizon);
  if (data.length === 0) {
    return <EmptyGraph outcomes />;
  }
  const maximumValue = Math.max(
    ...data.flatMap((item) => [item.immediate, item.fable]),
  );
  const step = niceStep(maximumValue);
  const maximum = Math.ceil(maximumValue / step) * step;
  const sections = Math.round(maximum / step);

  return (
    <View style={styles.graph}>
      <GiftedBarChart
        barBorderRadius={radii.small / 2}
        barWidth={18}
        data={data.flatMap((item, index) => [
          {
            frontColor: colors.amberSoft,
            label: item.label,
            labelWidth: 36,
            spacing: 4,
            value: item.immediate,
          },
          {
            frontColor: colors.blue,
            spacing: index === data.length - 1 ? 0 : 20,
            value: item.fable,
          },
        ])}
        disablePress
        endSpacing={10}
        formatYLabel={(label) => {
          const value = Number(label);
          return value >= 10 ? value.toFixed(0) : value.toFixed(1);
        }}
        initialSpacing={10}
        maxValue={maximum}
        noOfSections={sections}
        rulesColor={colors.border}
        spacing={0}
        stepHeight={CHART_HEIGHT / sections}
        stepValue={step}
        xAxisColor={colors.muted}
        xAxisLabelsAtBottom
        xAxisLabelsHeight={14}
        xAxisLabelTextStyle={styles.graphAxisText}
        yAxisLabelWidth={34}
        yAxisTextStyle={styles.graphAxisText}
        yAxisThickness={0}
      />
      <Text style={styles.graphXAxisTitle}>Recommended wait (blocks)</Text>
    </View>
  );
}

function runSummary(run: InferenceRun): string {
  const wait =
    run.selected_action_k === 0
      ? "Act now"
      : `Wait ${run.selected_action_k} block${run.selected_action_k === 1 ? "" : "s"}`;
  const savings = realizedSavingsPercent(run);
  if (run.outcome === undefined) {
    return `${wait} · Pending`;
  }
  if (savings === null) {
    return `${wait} · Unavailable`;
  }
  const outcome =
    savings >= 0
      ? `Saved ${formatSavings(savings)}`
      : `${formatSavings(Math.abs(savings))} higher`;
  return `${wait} · ${outcome}`;
}

function NetworkPicker({
  selected,
  onClose,
  onSelect,
}: {
  selected: Chain;
  onClose: () => void;
  onSelect: (chain: Chain) => void;
}) {
  return (
    <Modal animationType="fade" onRequestClose={onClose} transparent visible>
      <View style={styles.dialogRoot}>
        <Pressable
          accessibilityLabel="Close network picker"
          onPress={onClose}
          style={styles.backdrop}
        />
        <View style={[styles.dialog, styles.sheet, styles.networkSheet]}>
          <View style={styles.networkSheetHeader}>
            <Text style={styles.networkSheetTitle}>Select network</Text>
            <Pressable
              accessibilityLabel="Close"
              hitSlop={10}
              onPress={onClose}
            >
              <Ionicons color={colors.muted} name="close" size={25} />
            </Pressable>
          </View>
          <View style={styles.networkOptions}>
            {CHAINS.map((chain) => {
              const active = chain === selected;
              return (
                <Pressable
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  key={chain}
                  onPress={() => onSelect(chain)}
                  style={[
                    styles.networkCard,
                    styles.networkOption,
                    active && styles.networkCardActive,
                  ]}
                >
                  <NetworkIcon chain={chain} size={26} />
                  <Text
                    style={[
                      styles.networkOptionText,
                      active && styles.networkOptionTextActive,
                    ]}
                  >
                    {CHAIN_DETAILS[chain].label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>
      </View>
    </Modal>
  );
}

function RunDetails({
  run,
  onClose,
}: {
  run: InferenceRun | null;
  onClose: () => void;
}) {
  if (run === null) {
    return null;
  }
  const savings = realizedSavingsPercent(run);
  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible>
      <View style={styles.dialogRoot}>
        <Pressable
          accessibilityLabel="Close run details"
          onPress={onClose}
          style={styles.backdrop}
        />
        <View style={[styles.dialog, styles.sheet, styles.runDialog]}>
          <View style={styles.handle} />
          <View style={styles.dialogHeader}>
            <View>
              <Text style={styles.dialogTitle}>Run details</Text>
              <Text style={styles.dialogDate}>{formatRunDate(run.ran_at)}</Text>
            </View>
            <Pressable
              accessibilityLabel="Close"
              hitSlop={10}
              onPress={onClose}
            >
              <Ionicons color={colors.muted} name="close" size={27} />
            </Pressable>
          </View>

          <View style={styles.selectionSummary}>
            <View style={styles.selectionItem}>
              <Text style={styles.detailLabel}>Network</Text>
              <Text style={styles.detailStrong}>
                {CHAIN_DETAILS[run.chain].label}
              </Text>
            </View>
            <View style={styles.selectionItem}>
              <Text style={styles.detailLabel}>Horizon</Text>
              <Text style={styles.detailStrong}>{run.K} blocks</Text>
            </View>
          </View>

          <Text style={styles.groupTitle}>Prediction</Text>
          <View style={[styles.surface, styles.detailsCard]}>
            <DetailRow
              label="Head block"
              value={run.head_block.toLocaleString()}
            />
            <DetailRow
              label="Action offset"
              value={String(run.selected_action_k)}
            />
            <DetailRow
              label="Target block"
              value={run.target_block.toLocaleString()}
            />
            <DetailRow
              label="Predicted base fee"
              last
              value={formatGwei(run.predicted_minimum_base_fee_per_gas)}
            />
          </View>
          <Text style={styles.groupTitle}>Outcome</Text>
          <View style={[styles.surface, styles.detailsCard]}>
            <DetailRow
              label="Act-now base fee"
              value={
                run.outcome === undefined
                  ? "Pending"
                  : formatGwei(run.outcome.immediate_base_fee_per_gas)
              }
            />
            <DetailRow
              label="Selected base fee"
              value={
                run.outcome === undefined
                  ? "Pending"
                  : formatGwei(run.outcome.selected_base_fee_per_gas)
              }
            />
            <DetailRow
              label="Realized savings"
              last
              value={
                run.outcome === undefined
                  ? "Pending"
                  : savings === null
                    ? "Unavailable"
                    : formatSavings(savings)
              }
            />
          </View>
          <Pressable
            accessibilityRole="button"
            onPress={onClose}
            style={[styles.button, styles.closeButton]}
          >
            <Text style={styles.buttonText}>Close</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

export function AnalyticsScreen({
  runs,
  chain,
  horizon,
  onChainChange,
  storageError,
}: {
  runs: readonly InferenceRun[];
  chain: Chain;
  horizon: Horizon;
  onChainChange: (chain: Chain) => void;
  storageError: string | null;
}) {
  const [analyticsHorizon, setAnalyticsHorizon] =
    useState<Horizon>(horizon);
  const [networkPickerOpen, setNetworkPickerOpen] = useState(false);
  const [selectedRun, setSelectedRun] = useState<InferenceRun | null>(null);
  const graphRuns = runsForSelection(runs, chain, analyticsHorizon);
  const summary = summarizeRuns(graphRuns);

  return (
    <>
      <ScrollView
        contentContainerStyle={styles.page}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.titleRow}>
          <Text style={styles.title}>Analytics</Text>
          <Pressable
            accessibilityHint="Opens network picker"
            accessibilityRole="button"
            onPress={() => setNetworkPickerOpen(true)}
            style={styles.networkBadge}
          >
            <NetworkIcon chain={chain} size={14} />
            <Text style={styles.networkBadgeText}>
              {CHAIN_DETAILS[chain].label}
            </Text>
            <Ionicons color={colors.blue} name="chevron-down" size={14} />
          </Pressable>
        </View>

        {storageError && (
          <View accessibilityRole="alert" style={styles.storageError}>
            <Text style={styles.storageErrorText}>{storageError}</Text>
          </View>
        )}

        <View style={styles.summarySection}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <View style={styles.summaryCards}>
            <SummaryCard
              accent
              label="Avg savings"
              value={
                summary.averageSavingsPercent === null
                  ? "—"
                  : formatSavings(summary.averageSavingsPercent)
              }
            />
            <SummaryCard
              label="Win rate"
              value={
                summary.winPercent === null
                  ? "—"
                  : `${summary.winPercent.toFixed(0)}%`
              }
            />
            <SummaryCard
              label="Avg wait (blocks)"
              value={
                summary.averageWait === null
                  ? "—"
                  : summary.averageWait.toFixed(1)
              }
            />
          </View>
        </View>

        <View style={styles.graphSection}>
          <View style={styles.graphFilter}>
            <Text style={styles.sectionTitle}>
              Prediction window (K = {analyticsHorizon})
            </Text>
            <View style={[styles.surface, styles.graphSliderCard]}>
              <HorizonSlider
                onChange={setAnalyticsHorizon}
                value={analyticsHorizon}
              />
            </View>
          </View>
          <View style={styles.chartCards}>
            {GRAPH_OPTIONS.map((graph) => (
              <View
                key={graph.value}
                style={[styles.surface, styles.chartCard]}
              >
                <View style={styles.chartHeader}>
                  <Text style={styles.chartTitle}>{graph.label}</Text>
                  {graph.value === "fees" && (
                    <View style={styles.graphLegend}>
                      <View
                        style={[
                          styles.graphLegendDot,
                          styles.graphImmediateDot,
                        ]}
                      />
                      <Text style={styles.graphLegendLabel}>Act now</Text>
                      <View
                        style={[
                          styles.graphLegendDot,
                          styles.graphFableDot,
                        ]}
                      />
                      <Text style={styles.graphLegendLabel}>FABLE</Text>
                    </View>
                  )}
                </View>
                {graph.value === "waits" && (
                  <RecommendedWaitChart
                    horizon={analyticsHorizon}
                    runs={graphRuns}
                  />
                )}
                {graph.value === "savings" && (
                  <SavingsByWaitChart
                    horizon={analyticsHorizon}
                    runs={graphRuns}
                  />
                )}
                {graph.value === "fees" && (
                  <BaseFeeByWaitChart
                    horizon={analyticsHorizon}
                    runs={graphRuns}
                  />
                )}
              </View>
            ))}
          </View>
        </View>

        <Text style={styles.sectionTitle}>Runs ({graphRuns.length})</Text>
        <View style={[styles.surface, styles.runList]}>
          {graphRuns.length === 0 ? (
            <View style={styles.emptyRuns}>
              <Text style={styles.emptyRunsTitle}>No runs yet</Text>
              <Text style={styles.emptyRunsText}>
                No runs match this prediction window.
              </Text>
            </View>
          ) : (
            <ScrollView
              nestedScrollEnabled
              showsVerticalScrollIndicator={graphRuns.length > 4}
              style={styles.runScroller}
            >
              {graphRuns.map((run, index) => (
                <Pressable
                  accessibilityHint="Opens run details"
                  accessibilityRole="button"
                  key={run.id}
                  onPress={() => setSelectedRun(run)}
                  style={[
                    styles.runRow,
                    index === graphRuns.length - 1 && styles.runRowLast,
                  ]}
                >
                  <View style={styles.runIcon}>
                    <Ionicons
                      color={colors.blue}
                      name="git-branch-outline"
                      size={22}
                    />
                  </View>
                  <View style={styles.runCopy}>
                    <Text style={styles.runDate}>
                      {formatRunDate(run.ran_at)}
                    </Text>
                    <Text numberOfLines={1} style={styles.runMeta}>
                      {runSummary(run)}
                    </Text>
                  </View>
                  <Ionicons
                    color={colors.muted}
                    name="chevron-forward"
                    size={21}
                  />
                </Pressable>
              ))}
            </ScrollView>
          )}
        </View>
      </ScrollView>

      <RunDetails onClose={() => setSelectedRun(null)} run={selectedRun} />
      {networkPickerOpen && (
        <NetworkPicker
          onClose={() => setNetworkPickerOpen(false)}
          onSelect={(nextChain) => {
            onChainChange(nextChain);
            setNetworkPickerOpen(false);
          }}
          selected={chain}
        />
      )}
    </>
  );
}
