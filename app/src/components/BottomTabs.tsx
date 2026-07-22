import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { colors } from "../theme";

export type AppTab = "inference" | "analytics";

const TABS = [
  { value: "inference", label: "Inference", icon: "pulse-outline" },
  { value: "analytics", label: "Analytics", icon: "bar-chart-outline" },
] as const;

export function BottomTabs({
  selected,
  onSelect,
}: {
  selected: AppTab;
  onSelect: (tab: AppTab) => void;
}) {
  return (
    <View style={styles.tabs}>
      {TABS.map((tab) => {
        const active = selected === tab.value;
        return (
          <Pressable
            accessibilityRole="tab"
            accessibilityState={{ selected: active }}
            key={tab.value}
            onPress={() => onSelect(tab.value)}
            style={styles.tab}
          >
            <View
              style={[styles.indicator, active && styles.indicatorActive]}
            />
            <Ionicons
              color={active ? colors.blue : colors.muted}
              name={tab.icon}
              size={23}
            />
            <Text style={[styles.label, active && styles.labelActive]}>
              {tab.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  tabs: {
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: "row",
    minHeight: 52,
  },
  tab: { alignItems: "center", flex: 1, gap: 3, justifyContent: "center" },
  indicator: {
    backgroundColor: "transparent",
    borderRadius: 2,
    height: 3,
    width: 76,
  },
  indicatorActive: { backgroundColor: colors.blue },
  label: { color: colors.muted, fontSize: 12, fontWeight: "600" },
  labelActive: { color: colors.blue },
});
