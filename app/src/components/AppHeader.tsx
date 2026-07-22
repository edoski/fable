import { StyleSheet, Text, View } from "react-native";

import { CHAIN_DETAILS, type Chain } from "../inference";
import { colors } from "../theme";

export type ServiceStatus = "checking" | "live" | "offline";

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
  status: ServiceStatus;
}) {
  const presentation = STATUS[status];
  const network = CHAIN_DETAILS[chain];
  return (
    <View style={styles.header}>
      <Text style={styles.brand}>FABLE</Text>
      <View
        accessibilityLabel={`${network.label} inference service ${presentation.label.toLowerCase()}`}
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
