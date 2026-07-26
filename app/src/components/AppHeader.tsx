import { StyleSheet, Text, View } from "react-native";

import type { RpcStatus } from "../engineLifecycle";
import { CHAIN_DETAILS, type Chain } from "../domain";
import { colors } from "../theme";

const STATUS = {
  checking: { color: colors.amber, label: "CHECKING" },
  live: { color: colors.green, label: "LIVE" },
  offline: { color: colors.red, label: "OFFLINE" },
} as const;

export function AppHeader({
  chain,
  status,
}: {
  chain: Chain;
  status: RpcStatus;
}) {
  const presentation = STATUS[status];
  const network = CHAIN_DETAILS[chain];
  return (
    <View style={styles.header}>
      <Text style={styles.brand}>FABLE</Text>
      <View
        accessibilityLabel={`${network.label} RPC ${presentation.label.toLowerCase()}`}
        accessibilityRole="text"
        style={styles.status}
      >
        <View style={[styles.dot, { backgroundColor: presentation.color }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    alignItems: "center",
    backgroundColor: colors.navy,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 58,
    paddingHorizontal: 20,
  },
  brand: {
    color: colors.surface,
    fontSize: 21,
    fontWeight: "800",
    letterSpacing: 1.5,
  },
  status: { alignItems: "center", justifyContent: "center" },
  dot: { borderRadius: 6, height: 10, width: 10 },
});
