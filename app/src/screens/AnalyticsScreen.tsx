import { Ionicons } from "@expo/vector-icons";
import { type PropsWithChildren, type ReactNode, useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { BarChart as GiftedBarChart } from "react-native-gifted-charts";

import {
  formatGwei,
  formatRunDate,
  realizedSavingsPercent,
  runsForSelection,
  summarizeRuns,
  type WaitBucket,
  waitBuckets,
} from "../analytics";
import { DetailRow } from "../components/DetailRow";
import { HorizonSlider } from "../components/HorizonSlider";
import { NetworkIcon } from "../components/NetworkIcon";
import { Overlay } from "../components/Overlay";
import { CHAINS, CHAIN_LABELS, type Chain, type Horizon } from "../domain";
import type { InferenceRun } from "../history";
import { styles } from "../styles";
import { colors, radii } from "../theme";

function SummaryCard({
  format,
  value,
  label,
  accent = false,
}: {
  format: (value: number) => string;
  value: number | null;
  label: string;
  accent?: boolean;
}) {
  return (
    <View style={[styles.surface, styles.summaryCard]}>
      <Text style={[styles.summaryValue, accent && styles.accentText]}>
        {value === null ? "—" : format(value)}
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
const AXIS_PROPS = {
  barBorderRadius: radii.small / 2,
  disablePress: true,
  endSpacing: 10,
  initialSpacing: 10,
  rulesColor: colors.border,
  xAxisColor: colors.muted,
  xAxisLabelsAtBottom: true,
  xAxisLabelsHeight: 14,
  xAxisLabelTextStyle: styles.graphAxisText,
  yAxisLabelWidth: 34,
  yAxisTextStyle: styles.graphAxisText,
  yAxisThickness: 0,
} as const;

function niceStep(range: number): number {
  const rough = range / 3;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-9)));
  const normalized = rough / magnitude;
  const multiplier =
    normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
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
    maxValue: maximum,
    mostNegativeValue: minimum,
    negativeStepValue: step,
    noOfSections: positiveSections,
    noOfSectionsBelowXAxis: negativeSections,
    stepHeight:
      CHART_HEIGHT / Math.max(positiveSections + negativeSections, 1),
    stepValue: step,
  };
}

function ChartCard({
  children,
  empty,
  legend,
  title,
  xAxisTitle,
}: PropsWithChildren<{
  empty: "runs" | "outcomes" | null;
  legend?: ReactNode;
  title: string;
  xAxisTitle: string;
}>) {
  return (
    <View style={[styles.surface, styles.chartCard]}>
      <View style={styles.headerRow}>
        <Text style={styles.chartTitle}>{title}</Text>
        {legend}
      </View>
      {empty === null ? (
        <View style={styles.graph}>
          {children}
          <Text style={styles.graphXAxisTitle}>{xAxisTitle}</Text>
        </View>
      ) : (
        <View style={styles.emptyGraph}>
          <Text style={styles.emptyGraphTitle}>
            {empty === "outcomes" ? "No outcomes yet" : "No runs yet"}
          </Text>
          <Text style={styles.emptyGraphText}>
            {empty === "outcomes"
              ? "Resolved inferences will populate this graph."
              : "Runs will populate this graph."}
          </Text>
        </View>
      )}
    </View>
  );
}

function RecommendedWaitChart({
  buckets,
}: {
  buckets: readonly WaitBucket[];
}) {
  return (
    <ChartCard
      empty={buckets.length === 0 ? "runs" : null}
      title="Recommended wait distribution"
      xAxisTitle="Wait (blocks)"
    >
      <GiftedBarChart
        {...AXIS_PROPS}
        {...chartScale(buckets.map((bucket) => bucket.runCount))}
        data={buckets.map((bucket) => ({
          frontColor: colors.blue,
          label: bucket.label,
          value: bucket.runCount,
        }))}
      />
    </ChartCard>
  );
}

function SavingsByWaitChart({
  buckets,
}: {
  buckets: readonly WaitBucket[];
}) {
  const values = buckets.flatMap((bucket) =>
    bucket.savingsPercent === null ? [] : [bucket.savingsPercent],
  );
  return (
    <ChartCard
      empty={values.length === 0 ? "outcomes" : null}
      title="Savings by wait (%)"
      xAxisTitle="Wait (blocks)"
    >
      <GiftedBarChart
        {...AXIS_PROPS}
        {...chartScale(values)}
        data={buckets.map((bucket) => ({
          frontColor:
            bucket.savingsPercent !== null && bucket.savingsPercent < 0
              ? colors.red
              : colors.teal,
          label: bucket.label,
          value: bucket.savingsPercent ?? 0,
        }))}
        formatYLabel={(label) => `${Number(label).toFixed(0)}%`}
      />
    </ChartCard>
  );
}

function BaseFeeByWaitChart({
  buckets,
}: {
  buckets: readonly WaitBucket[];
}) {
  const data = buckets.filter(
    (
      bucket,
    ): bucket is WaitBucket & {
      fableGwei: number;
      immediateGwei: number;
    } => bucket.fableGwei !== null && bucket.immediateGwei !== null,
  );
  return (
    <ChartCard
      legend={
        <View style={styles.graphLegend}>
          <View
            style={[styles.graphLegendDot, styles.graphImmediateDot]}
          />
          <Text style={styles.graphLegendLabel}>Act now</Text>
          <View style={[styles.graphLegendDot, styles.graphFableDot]} />
          <Text style={styles.graphLegendLabel}>FABLE</Text>
        </View>
      }
      title="Base fee by wait (Gwei)"
      empty={data.length === 0 ? "outcomes" : null}
      xAxisTitle="Recommended wait (blocks)"
    >
      <GiftedBarChart
        {...AXIS_PROPS}
        {...chartScale(
          data.flatMap((bucket) => [bucket.immediateGwei, bucket.fableGwei]),
        )}
        barWidth={18}
        data={data.flatMap((bucket, index) => [
          {
            frontColor: colors.amberSoft,
            label: bucket.label,
            labelWidth: 36,
            spacing: 4,
            value: bucket.immediateGwei,
          },
          {
            frontColor: colors.blue,
            spacing: index === data.length - 1 ? 0 : 20,
            value: bucket.fableGwei,
          },
        ])}
        formatYLabel={(label) => {
          const value = Number(label);
          return value >= 10 ? value.toFixed(0) : value.toFixed(1);
        }}
        spacing={0}
      />
    </ChartCard>
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
    <Overlay
      animationType="fade"
      backdropLabel="Close network picker"
      onClose={onClose}
    >
      <View style={[styles.dialog, styles.sheet, styles.networkSheet]}>
        <View style={styles.headerRow}>
          <Text style={styles.networkSheetTitle}>Select network</Text>
          <Pressable
            accessibilityLabel="Close"
            hitSlop={10}
            onPress={onClose}
          >
            <Ionicons color={colors.muted} name="close" size={25} />
          </Pressable>
        </View>
        <View style={styles.cardRow}>
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
                  {CHAIN_LABELS[chain]}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>
    </Overlay>
  );
}

function RunDetails({
  run,
  onClose,
}: {
  run: InferenceRun;
  onClose: () => void;
}) {
  const savings = realizedSavingsPercent(run);
  return (
    <Overlay
      animationType="slide"
      backdropLabel="Close run details"
      onClose={onClose}
    >
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
              {CHAIN_LABELS[run.chain]}
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
    </Overlay>
  );
}

export function AnalyticsScreen({
  runs,
  chain,
  horizon,
  loadError,
  onChainChange,
}: {
  runs: readonly InferenceRun[];
  chain: Chain;
  horizon: Horizon;
  loadError: string | null;
  onChainChange: (chain: Chain) => void;
}) {
  const [analyticsHorizon, setAnalyticsHorizon] =
    useState<Horizon>(horizon);
  const [networkPickerOpen, setNetworkPickerOpen] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const graphRuns = runsForSelection(runs, chain, analyticsHorizon);
  const buckets = waitBuckets(graphRuns, analyticsHorizon);
  const summary = summarizeRuns(graphRuns);

  return (
    <>
      <ScrollView
        contentContainerStyle={styles.page}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={styles.title}>Analytics</Text>
          <Pressable
            accessibilityHint="Opens network picker"
            accessibilityRole="button"
            onPress={() => setNetworkPickerOpen(true)}
            style={styles.networkBadge}
          >
            <NetworkIcon chain={chain} size={14} />
            <Text style={styles.networkBadgeText}>
              {CHAIN_LABELS[chain]}
            </Text>
            <Ionicons color={colors.blue} name="chevron-down" size={14} />
          </Pressable>
        </View>

        {loadError && (
          <View accessibilityRole="alert" style={styles.storageError}>
            <Text style={styles.storageErrorText}>{loadError}</Text>
          </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <View style={styles.cardRow}>
            <SummaryCard
              accent
              format={formatSavings}
              label="Avg savings"
              value={summary.averageSavingsPercent}
            />
            <SummaryCard
              format={(value) => `${value.toFixed(0)}%`}
              label="Win rate"
              value={summary.winPercent}
            />
            <SummaryCard
              format={(value) => value.toFixed(1)}
              label="Avg wait (blocks)"
              value={summary.averageWait}
            />
          </View>
        </View>

        <View style={styles.graphSection}>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              Horizon (K = {analyticsHorizon})
            </Text>
            <View style={[styles.surface, styles.graphSliderCard]}>
              <HorizonSlider
                onChange={setAnalyticsHorizon}
                value={analyticsHorizon}
              />
            </View>
          </View>
          <View style={styles.chartCards}>
            <RecommendedWaitChart buckets={buckets} />
            <SavingsByWaitChart buckets={buckets} />
            <BaseFeeByWaitChart buckets={buckets} />
          </View>
        </View>

        <Text style={styles.sectionTitle}>Runs ({graphRuns.length})</Text>
        <View style={[styles.surface, styles.clippedCard]}>
          {graphRuns.length === 0 ? (
            <View style={styles.emptyRuns}>
              <Text style={styles.emptyRunsTitle}>No runs yet</Text>
              <Text style={styles.emptyRunsText}>
                No runs match this horizon.
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
                  onPress={() => setSelectedRunId(run.id)}
                  style={[
                    styles.runRow,
                    index === graphRuns.length - 1 && styles.lastRow,
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

      {selectedRun !== null && (
        <RunDetails onClose={() => setSelectedRunId(null)} run={selectedRun} />
      )}
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
