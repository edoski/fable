import { StyleSheet, Text, View } from "react-native";

import { colors } from "../theme";

export type RpcStatus = "checking" | "live" | "offline";

const STATUS = {
  checking: colors.amber,
  live: colors.green,
  offline: colors.red,
} as const;

export function AppHeader({
  status,
}: {
  status: RpcStatus;
}) {
  return (
    <View style={styles.header}>
      <Text style={styles.brand}>KAIROS</Text>
      <View style={[styles.dot, { backgroundColor: STATUS[status] }]} />
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
  dot: { borderRadius: 6, height: 10, width: 10 },
});
