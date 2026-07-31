import { Ionicons } from "@expo/vector-icons";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  Text,
  View,
} from "react-native";

import { formatGwei } from "../analytics";
import { DetailRow } from "../components/DetailRow";
import { HorizonSlider } from "../components/HorizonSlider";
import { NetworkIcon } from "../components/NetworkIcon";
import { Overlay } from "../components/Overlay";
import {
  CHAINS,
  CHAIN_LABELS,
  type Chain,
  type Horizon,
} from "../domain";
import type {
  ChainSnapshot,
  InferenceResult,
} from "../inference";
import { styles } from "../styles";
import { colors } from "../theme";

export type InferenceState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: InferenceResult }
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
    <View style={styles.cardRow}>
      {CHAINS.map((choice) => {
        const active = choice === chain;
        const label = CHAIN_LABELS[choice];
        return (
          <Pressable
            accessibilityRole="radio"
            accessibilityState={{ checked: active, disabled }}
            disabled={disabled}
            key={choice}
            onPress={() => onChange(choice)}
            style={[
              styles.networkCard,
              styles.inferenceNetworkCard,
              active && styles.networkCardActive,
            ]}
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
              {label}
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
      <View style={[styles.surface, styles.clippedCard]}>
        <DetailRow
          label="Latest block"
          value={snapshot?.head_block.toLocaleString() ?? "—"}
        />
        <DetailRow
          label="Current base fee"
          last
          value={snapshot ? formatGwei(snapshot.current_base_fee_per_gas) : "—"}
        />
      </View>
    </View>
  );
}

function HorizonSelector({
  disabled,
  horizon,
  onChange,
}: {
  disabled: boolean;
  horizon: Horizon;
  onChange: (horizon: Horizon) => void;
}) {
  return (
    <View style={[styles.surface, styles.windowCard]}>
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
  return (
    <Overlay
      animationType="fade"
      backdropLabel="Dismiss inference error"
      centered
      onClose={onClose}
    >
      <View
        accessibilityRole="alert"
        style={[styles.dialog, styles.errorDialog]}
      >
        <View style={styles.errorDialogIcon}>
          <Ionicons
            color={colors.red}
            name="alert-circle-outline"
            size={28}
          />
        </View>
        <Text style={styles.errorDialogTitle}>Inference failed</Text>
        <Text style={styles.errorDialogText}>{message}</Text>
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
            style={[styles.button, styles.retryButton]}
          >
            <Ionicons color={colors.surface} name="refresh" size={17} />
            <Text style={styles.buttonText}>Retry</Text>
          </Pressable>
        </View>
      </View>
    </Overlay>
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
            Horizon (K = {horizon})
          </Text>
          <HorizonSelector
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
            styles.button,
            styles.primaryButton,
            styles.setupButton,
            loading && styles.buttonDisabled,
          ]}
        >
          {loading && <ActivityIndicator color={colors.surface} />}
          <Text style={styles.buttonText}>
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
}: {
  result: InferenceResult;
}) {
  return (
    <View style={styles.timeline}>
      <View style={[styles.timelineCell, styles.timelineHeadCell]}>
        <Text style={styles.timelineLabel}>Head</Text>
        <Text numberOfLines={1} style={styles.timelineBlock}>
          {result.head_block.toLocaleString()}
        </Text>
      </View>
      {Array.from({ length: result.K }, (_, offset) => {
        const active = offset === result.selected_action_k;
        return (
          <View
            key={offset}
            style={[styles.timelineCell, active && styles.timelineCellActive]}
          >
            <Text
              style={[
                styles.timelineOffset,
                active && styles.accentText,
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
              style={[
                styles.timelineTargetLabel,
                active && styles.accentText,
              ]}
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
  result,
  onRunAgain,
}: {
  result: InferenceResult;
  onRunAgain: () => void;
}) {
  const recommendation =
    result.selected_action_k === 0
      ? "Use the next block"
      : `Wait ${result.selected_action_k} ${result.selected_action_k === 1 ? "block" : "blocks"}`;
  return (
    <ScrollView
      contentContainerStyle={styles.page}
      showsVerticalScrollIndicator={false}
    >
      <Text style={styles.title}>Inference</Text>
      <View style={[styles.surface, styles.recommendation]}>
        <View style={styles.successIcon}>
          <Ionicons color={colors.surface} name="checkmark" size={30} />
        </View>
        <View style={styles.recommendationCopy}>
          <Text style={styles.eyebrow}>Recommendation</Text>
          <Text style={styles.recommendationText}>{recommendation}</Text>
        </View>
      </View>

      <Timeline result={result} />

      <View style={[styles.surface, styles.detailsCard]}>
        <Text style={styles.detailsTitle}>Technical details</Text>
        <DetailRow label="Network" value={CHAIN_LABELS[result.chain]} />
        <DetailRow label="Horizon" value={`${result.K} blocks`} />
        <DetailRow
          label="Action offset"
          value={String(result.selected_action_k)}
        />
        <DetailRow
          label="Target block"
          value={result.target_block.toLocaleString()}
        />
        <DetailRow
          label="Predicted horizon minimum"
          last
          value={formatGwei(result.predicted_minimum_base_fee_per_gas)}
        />
      </View>

      <Pressable
        accessibilityRole="button"
        onPress={onRunAgain}
        style={[styles.button, styles.primaryButton]}
      >
        <Ionicons color={colors.surface} name="refresh" size={21} />
        <Text style={styles.buttonText}>Run again</Text>
      </Pressable>
    </ScrollView>
  );
}

export function InferenceScreen(props: Props) {
  if (props.state.status === "success") {
    return (
      <Result
        onRunAgain={props.onRunAgain}
        result={props.state.result}
      />
    );
  }
  return <Setup {...props} />;
}
