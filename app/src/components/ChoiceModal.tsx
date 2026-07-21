import { Ionicons } from "@expo/vector-icons";
import { Modal, Pressable, StyleSheet, Text, View } from "react-native";

import { colors, radii } from "../theme";

export type Choice<T extends string | number> = { value: T; label: string };

export function PickerButton({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.pickerButton}>
      <Text numberOfLines={1} style={styles.pickerLabel}>
        {label}
      </Text>
      <Ionicons color={colors.muted} name="chevron-down" size={18} />
    </Pressable>
  );
}

export function ChoiceModal<T extends string | number>({
  title,
  visible,
  options,
  selected,
  onClose,
  onSelect,
}: {
  title: string;
  visible: boolean;
  options: readonly Choice<T>[];
  selected: T;
  onClose: () => void;
  onSelect: (value: T) => void;
}) {
  return (
    <Modal animationType="fade" onRequestClose={onClose} transparent visible={visible}>
      <View style={styles.modalRoot}>
        <Pressable accessibilityLabel="Close picker" onPress={onClose} style={styles.backdrop} />
        <View style={styles.sheet}>
          <View style={styles.handle} />
          <Text style={styles.title}>{title}</Text>
          {options.map((option) => {
            const active = selected === option.value;
            return (
              <Pressable
                accessibilityRole="radio"
                accessibilityState={{ checked: active }}
                key={String(option.value)}
                onPress={() => {
                  onSelect(option.value);
                  onClose();
                }}
                style={[styles.option, active && styles.optionActive]}
              >
                <Text style={[styles.optionText, active && styles.optionTextActive]}>
                  {option.label}
                </Text>
                {active && <Ionicons color={colors.blue} name="checkmark-circle" size={22} />}
              </Pressable>
            );
          })}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  pickerButton: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flex: 1,
    flexDirection: "row",
    gap: 8,
    justifyContent: "space-between",
    minHeight: 48,
    paddingHorizontal: 14,
  },
  pickerLabel: { color: colors.ink, flex: 1, fontSize: 14, fontWeight: "600" },
  modalRoot: { flex: 1, justifyContent: "flex-end" },
  backdrop: {
    backgroundColor: "rgba(7, 20, 38, 0.48)",
    bottom: 0,
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  sheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    gap: 8,
    paddingBottom: 28,
    paddingHorizontal: 20,
    paddingTop: 10,
  },
  handle: {
    alignSelf: "center",
    backgroundColor: colors.border,
    borderRadius: 3,
    height: 5,
    marginBottom: 8,
    width: 48,
  },
  title: { color: colors.ink, fontSize: 20, fontWeight: "700", marginBottom: 4 },
  option: {
    alignItems: "center",
    borderColor: colors.border,
    borderRadius: radii.medium,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 50,
    paddingHorizontal: 14,
  },
  optionActive: { backgroundColor: colors.blueSoft, borderColor: colors.blue },
  optionText: { color: colors.ink, fontSize: 15, fontWeight: "600" },
  optionTextActive: { color: colors.blue },
});
