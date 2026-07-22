import { Ionicons } from "@expo/vector-icons";
import { useMemo, useRef, useState } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";
import {
  BarChart as GiftedBarChart,
} from "react-native-gifted-charts";

import {
  GRAPH_OPTIONS,
  feeComparisonData,
  formatGwei,
  formatRunDate,
  recommendedWaitData,
  realizedSavingsPercent,
  savingsByWaitData,
  summarizeRuns,
  type GraphKind,
} from "../analytics";
import { HorizonSlider } from "../components/HorizonSlider";
import { NetworkIcon } from "../components/NetworkIcon";
import type { InferenceRun } from "../history";
import { CHAINS, CHAIN_DETAILS, type Chain, type Horizon } from "../inference";
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
    <View style={styles.summaryCard}>
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

const PLOT_HEIGHT = 138;

function niceStep(range: number): number {
  const rough = range / 3;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-9)));
  const normalized = rough / magnitude;
  const multiplier =
    normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function EmptyGraph() {
  return (
    <View style={styles.emptyGraph}>
      <Text style={styles.emptyGraphTitle}>No outcomes yet</Text>
      <Text style={styles.emptyGraphText}>
        Resolved inferences will populate this graph.
      </Text>
    </View>
  );
}

function AnalyticsGraph({
  kind,
  runs,
  horizon,
}: {
  kind: GraphKind;
  runs: readonly InferenceRun[];
  horizon: Horizon;
}) {
  const { width: screenWidth } = useWindowDimensions();

  if (kind !== "fees") {
    const data =
      kind === "waits"
        ? recommendedWaitData(runs, horizon)
        : savingsByWaitData(runs, horizon);
    const values = data.flatMap((item) =>
      item.value === null ? [] : [item.value],
    );
    if (values.length === 0) {
      return <EmptyGraph />;
    }

    const rawMinimum = Math.min(0, ...values);
    const rawMaximum = Math.max(0, ...values);
    const step = niceStep(
      rawMinimum === rawMaximum ? 1 : rawMaximum - rawMinimum,
    );
    const minimum = Math.floor(rawMinimum / step) * step;
    const maximum = Math.max(step, Math.ceil(rawMaximum / step) * step);
    const positiveSections = Math.round(maximum / step);
    const negativeSections = Math.round(Math.abs(minimum) / step);
    const stepHeight =
      PLOT_HEIGHT / Math.max(positiveSections + negativeSections, 1);
    const chartWidth = Math.min(screenWidth - 128, 276);
    const barWidth = Math.min(28, chartWidth / Math.max(data.length * 2, 1));
    const spacing =
      data.length < 2
        ? 0
        : (chartWidth - barWidth * data.length - 20) / (data.length - 1);
    const savingsGraph = kind === "savings";

    return (
      <View style={styles.graph}>
        <View style={styles.graphPlotRow}>
          <View style={styles.graphYAxisTitleSlot}>
            <Text numberOfLines={1} style={styles.graphAxisTitle}>
              {savingsGraph ? "Avg savings" : "Runs"}
            </Text>
          </View>
          <GiftedBarChart
            barBorderRadius={radii.small / 2}
            barWidth={barWidth}
            data={data.map((item) => ({
              value: item.value ?? 0,
              label: item.label,
              frontColor:
                item.value !== null && item.value < 0
                  ? colors.red
                  : savingsGraph
                    ? colors.teal
                    : colors.blue,
            }))}
            disablePress
            disableScroll
            endSpacing={10}
            formatYLabel={(label) =>
              savingsGraph ? `${Number(label).toFixed(0)}%` : label
            }
            initialSpacing={10}
            maxValue={maximum}
            mostNegativeValue={minimum}
            negativeStepValue={step}
            noOfSections={positiveSections}
            noOfSectionsBelowXAxis={negativeSections}
            rulesColor={colors.border}
            rulesThickness={1}
            spacing={spacing}
            stepHeight={stepHeight}
            stepValue={step}
            width={chartWidth}
            xAxisColor={colors.muted}
            xAxisLabelsAtBottom
            xAxisLabelsHeight={14}
            xAxisLabelTextStyle={styles.graphAxisText}
            xAxisThickness={1}
            yAxisColor="transparent"
            yAxisLabelWidth={34}
            yAxisTextStyle={styles.graphAxisText}
            yAxisThickness={0}
          />
        </View>
        <Text style={styles.graphXAxisTitle}>Wait (blocks)</Text>
      </View>
    );
  }

  const data = feeComparisonData(runs, horizon);
  if (data.length === 0) {
    return <EmptyGraph />;
  }
  const maximumValue = Math.max(
    ...data.flatMap((item) => [item.immediate, item.fable]),
  );
  const step = niceStep(maximumValue);
  const maximum = Math.ceil(maximumValue / step) * step;
  const chartWidth = Math.min(screenWidth - 136, 266);
  const barWidth = 18;
  const pairGap = 4;
  const pairWidth = barWidth * 2 + pairGap;
  const groupGap =
    data.length < 2
      ? 0
      : (chartWidth - 20 - pairWidth * data.length) / (data.length - 1);
  const initialSpacing =
    data.length === 1 ? (chartWidth - pairWidth) / 2 : 10;
  const sections = Math.round(maximum / step);

  return (
    <View style={styles.graph}>
      <View style={styles.graphPlotRow}>
        <View style={styles.graphYAxisTitleSlot}>
          <Text numberOfLines={1} style={styles.graphAxisTitleWide}>
            Base fee (Gwei)
          </Text>
        </View>
        <GiftedBarChart
          barBorderRadius={radii.small / 2}
          barWidth={barWidth}
          data={data.flatMap((item, index) => [
            {
              value: item.immediate,
              label: item.label,
              labelWidth: barWidth * 2,
              frontColor: colors.amberSoft,
              spacing: pairGap,
            },
            {
              value: item.fable,
              frontColor: colors.blue,
              spacing: index === data.length - 1 ? 0 : groupGap,
            },
          ])}
          disablePress
          disableScroll
          endSpacing={10}
          formatYLabel={(label) => {
            const value = Number(label);
            return value >= 10 ? value.toFixed(0) : value.toFixed(1);
          }}
          initialSpacing={initialSpacing}
          maxValue={maximum}
          noOfSections={sections}
          rulesColor={colors.border}
          rulesThickness={1}
          spacing={0}
          stepHeight={PLOT_HEIGHT / sections}
          stepValue={step}
          width={chartWidth}
          xAxisColor={colors.muted}
          xAxisLabelsAtBottom
          xAxisLabelsHeight={14}
          xAxisLabelTextStyle={styles.graphAxisText}
          xAxisThickness={1}
          yAxisColor="transparent"
          yAxisLabelWidth={34}
          yAxisTextStyle={styles.graphAxisText}
          yAxisThickness={0}
        />
      </View>
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
  if (savings === null) {
    return `${wait} · Pending`;
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
        <View style={styles.networkSheet}>
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
                    styles.networkOption,
                    active && styles.networkOptionActive,
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
  return (
    <Modal animationType="slide" onRequestClose={onClose} transparent visible>
      <View style={styles.dialogRoot}>
        <Pressable
          accessibilityLabel="Close run details"
          onPress={onClose}
          style={styles.backdrop}
        />
        <View style={styles.dialog}>
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
          <View style={styles.detailsCard}>
            <Detail
              label="Head block"
              value={run.head_block.toLocaleString()}
            />
            <Detail
              label="Action offset"
              value={String(run.selected_action_k)}
            />
            <Detail
              label="Target block"
              value={run.target_block.toLocaleString()}
            />
            <Detail
              label="Predicted base fee"
              last
              value={formatGwei(run.predicted_minimum_base_fee_per_gas)}
            />
          </View>
          <Text style={styles.groupTitle}>Outcome</Text>
          <View style={styles.detailsCard}>
            <Detail
              label="Act-now base fee"
              value={
                run.outcome === undefined
                  ? "Pending"
                  : formatGwei(run.outcome.immediate_base_fee_per_gas)
              }
            />
            <Detail
              label="Selected base fee"
              value={
                run.outcome === undefined
                  ? "Pending"
                  : formatGwei(run.outcome.selected_base_fee_per_gas)
              }
            />
            <Detail
              label="Realized savings"
              last
              value={
                run.outcome === undefined
                  ? "Pending"
                  : formatSavings(realizedSavingsPercent(run) ?? 0)
              }
            />
          </View>
          <Pressable
            accessibilityRole="button"
            onPress={onClose}
            style={styles.closeButton}
          >
            <Text style={styles.closeButtonText}>Close</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

function Detail({
  label,
  value,
  last = false,
}: {
  label: string;
  value: string;
  last?: boolean;
}) {
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
  onChainChange,
  storageError,
}: {
  runs: readonly InferenceRun[];
  chain: Chain;
  horizon: Horizon;
  onChainChange: (chain: Chain) => void;
  storageError: string | null;
}) {
  const [carouselWidth, setCarouselWidth] = useState(0);
  const [graphIndex, setGraphIndex] = useState(0);
  const [graphHorizon, setGraphHorizon] = useState<Horizon>(horizon);
  const [networkPickerOpen, setNetworkPickerOpen] = useState(false);
  const [selectedRun, setSelectedRun] = useState<InferenceRun | null>(null);
  const carousel = useRef<ScrollView>(null);
  const chartStep = carouselWidth + 12;
  const networkRuns = useMemo(
    () => runs.filter((run) => run.chain === chain),
    [runs, chain],
  );
  const graphRuns = useMemo(
    () => networkRuns.filter((run) => run.K === graphHorizon),
    [networkRuns, graphHorizon],
  );
  const summary = useMemo(() => summarizeRuns(networkRuns), [networkRuns]);
  const graphs = GRAPH_OPTIONS;
  const visibleRuns = graphRuns.slice(0, 10);

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
                summary.averageOffset === null
                  ? "—"
                  : summary.averageOffset.toFixed(1)
              }
            />
          </View>
        </View>

        <View style={styles.graphSection}>
          <View style={styles.graphFilter}>
            <Text style={styles.sectionTitle}>
              Prediction window (K = {graphHorizon})
            </Text>
            <View style={styles.graphSliderCard}>
              <HorizonSlider
                onChange={setGraphHorizon}
                showTicks={false}
                value={graphHorizon}
              />
            </View>
          </View>
          <View
            onLayout={(event) =>
              setCarouselWidth(Math.round(event.nativeEvent.layout.width))
            }
            style={styles.carouselSection}
          >
            <ScrollView
              accessibilityLabel="Analytics graphs"
              decelerationRate="fast"
              disableIntervalMomentum
              horizontal
              onMomentumScrollEnd={(event) =>
                setGraphIndex(
                  Math.round(event.nativeEvent.contentOffset.x / chartStep),
                )
              }
              ref={carousel}
              showsHorizontalScrollIndicator={false}
              snapToAlignment="start"
              snapToInterval={chartStep}
              contentContainerStyle={styles.carouselContent}
              style={styles.carouselViewport}
            >
              {carouselWidth > 0 &&
                graphs.map((graph) => (
                  <View
                    key={graph.value}
                    style={[styles.chartCard, { width: carouselWidth }]}
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
                    <AnalyticsGraph
                      horizon={graphHorizon}
                      kind={graph.value}
                      runs={graphRuns}
                    />
                  </View>
                ))}
            </ScrollView>
            <View accessibilityRole="tablist" style={styles.carouselDots}>
              {graphs.map((graph, index) => (
                <Pressable
                  accessibilityLabel={`Show ${graph.label}`}
                  accessibilityRole="tab"
                  accessibilityState={{ selected: graphIndex === index }}
                  hitSlop={8}
                  key={graph.value}
                  onPress={() => {
                    setGraphIndex(index);
                    carousel.current?.scrollTo({
                      animated: true,
                      x: index * chartStep,
                    });
                  }}
                  style={[
                    styles.carouselDot,
                    graphIndex === index && styles.carouselDotActive,
                  ]}
                />
              ))}
            </View>
          </View>
        </View>

        <Text style={styles.sectionTitle}>Runs ({graphRuns.length})</Text>
        <View style={styles.runList}>
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
              showsVerticalScrollIndicator={visibleRuns.length > 4}
              style={styles.runScroller}
            >
              {visibleRuns.map((run, index) => (
                <Pressable
                  accessibilityHint="Opens run details"
                  accessibilityRole="button"
                  key={run.id}
                  onPress={() => setSelectedRun(run)}
                  style={[
                    styles.runRow,
                    index === visibleRuns.length - 1 && styles.runRowLast,
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

const styles = StyleSheet.create({
  page: { gap: 20, padding: 18, paddingBottom: 30 },
  titleRow: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  title: { color: colors.ink, fontSize: 30, fontWeight: "800" },
  networkBadge: {
    alignItems: "center",
    backgroundColor: colors.blueSoft,
    borderRadius: 14,
    flexDirection: "row",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  networkBadgeText: { color: colors.blue, fontSize: 11, fontWeight: "700" },
  networkSheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    gap: 16,
    paddingBottom: 30,
    paddingHorizontal: 18,
    paddingTop: 18,
  },
  networkSheetHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  networkSheetTitle: { color: colors.ink, fontSize: 20, fontWeight: "800" },
  networkOptions: { flexDirection: "row", gap: 9 },
  networkOption: {
    alignItems: "center",
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    gap: 8,
    justifyContent: "center",
    minHeight: 88,
    padding: 8,
  },
  networkOptionActive: {
    backgroundColor: colors.blueSoft,
    borderColor: colors.blue,
  },
  networkOptionText: { color: colors.muted, fontSize: 11, fontWeight: "700" },
  networkOptionTextActive: { color: colors.blue },
  storageError: {
    backgroundColor: colors.redSoft,
    borderColor: "#FECACA",
    borderRadius: radii.medium,
    borderWidth: 1,
    padding: 12,
  },
  storageErrorText: { color: "#B42318", fontSize: 12 },
  summarySection: { gap: 10 },
  summaryCards: { flexDirection: "row", gap: 9 },
  summaryCard: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    gap: 5,
    justifyContent: "center",
    minHeight: 98,
    padding: 8,
  },
  summaryLabel: { color: colors.muted, fontSize: 10, textAlign: "center" },
  summaryValue: { color: colors.blue, fontSize: 20, fontWeight: "800" },
  summaryValueAccent: { color: colors.teal },
  graphSection: { gap: 14 },
  graphFilter: { gap: 10 },
  graphSliderCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  carouselSection: {
    alignItems: "center",
    gap: 10,
    overflow: "hidden",
    width: "100%",
  },
  carouselViewport: { width: "100%" },
  carouselContent: { gap: 12 },
  chartCard: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    gap: 14,
    padding: 14,
  },
  chartHeader: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  chartTitle: { color: colors.ink, fontSize: 15, fontWeight: "700" },
  graph: { gap: 2 },
  graphPlotRow: { alignItems: "center", flexDirection: "row" },
  graphYAxisTitleSlot: {
    alignItems: "center",
    alignSelf: "stretch",
    justifyContent: "center",
    width: 18,
  },
  graphAxisTitle: {
    color: colors.muted,
    fontSize: 9,
    fontWeight: "600",
    textAlign: "center",
    transform: [{ rotate: "-90deg" }],
    width: 96,
  },
  graphAxisTitleWide: {
    color: colors.muted,
    fontSize: 9,
    fontWeight: "600",
    textAlign: "center",
    transform: [{ rotate: "-90deg" }],
    width: 110,
  },
  graphAxisText: { color: colors.muted, fontSize: 9 },
  graphXAxisTitle: {
    color: colors.muted,
    fontSize: 9,
    fontWeight: "600",
    marginLeft: 18,
    marginTop: 3,
    textAlign: "center",
  },
  graphLegend: {
    alignItems: "center",
    flexDirection: "row",
    gap: 4,
  },
  graphLegendDot: { borderRadius: 4, height: 7, marginLeft: 5, width: 7 },
  graphImmediateDot: { backgroundColor: colors.amberSoft },
  graphFableDot: { backgroundColor: colors.blue },
  graphLegendLabel: { color: colors.muted, fontSize: 8 },
  emptyGraph: {
    alignItems: "center",
    height: 184,
    justifyContent: "center",
    padding: 24,
  },
  emptyGraphTitle: { color: colors.ink, fontSize: 16, fontWeight: "700" },
  emptyGraphText: {
    color: colors.muted,
    fontSize: 13,
    marginTop: 5,
    textAlign: "center",
  },
  carouselDots: { alignItems: "center", flexDirection: "row", gap: 6 },
  carouselDot: {
    backgroundColor: colors.border,
    borderRadius: 4,
    height: 7,
    width: 7,
  },
  carouselDotActive: { backgroundColor: colors.blue },
  sectionTitle: {
    color: colors.ink,
    fontSize: 17,
    fontWeight: "700",
  },
  runList: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.large,
    borderWidth: 1,
    overflow: "hidden",
  },
  runScroller: { maxHeight: 272 },
  runRow: {
    alignItems: "center",
    borderBottomColor: colors.border,
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    gap: 12,
    minHeight: 68,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  runRowLast: { borderBottomWidth: 0 },
  runIcon: {
    alignItems: "center",
    backgroundColor: colors.blueSoft,
    borderRadius: radii.small,
    height: 38,
    justifyContent: "center",
    width: 38,
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
  dialogHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    justifyContent: "space-between",
  },
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
  closeButton: {
    alignItems: "center",
    backgroundColor: colors.blue,
    borderRadius: radii.medium,
    justifyContent: "center",
    minHeight: 50,
  },
  closeButtonText: { color: colors.surface, fontSize: 15, fontWeight: "700" },
});
