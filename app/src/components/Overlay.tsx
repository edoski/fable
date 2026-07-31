import type { PropsWithChildren } from "react";
import {
  Modal,
  Pressable,
  StyleSheet,
  View,
  type ModalProps,
} from "react-native";

import { colors } from "../theme";

type Props = PropsWithChildren<{
  animationType: ModalProps["animationType"];
  backdropLabel: string;
  centered?: boolean;
  onClose: () => void;
}>;

export function Overlay({
  animationType,
  backdropLabel,
  centered = false,
  children,
  onClose,
}: Props) {
  return (
    <Modal
      animationType={animationType}
      onRequestClose={onClose}
      transparent
      visible
    >
      <View style={[styles.root, centered ? styles.centered : styles.bottom]}>
        <Pressable
          accessibilityLabel={backdropLabel}
          onPress={onClose}
          style={styles.backdrop}
        />
        {children}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  bottom: { justifyContent: "flex-end" },
  centered: {
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  backdrop: {
    backgroundColor: colors.overlay,
    bottom: 0,
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
});
