import Slider from "@react-native-community/slider";
import { StyleSheet, View } from "react-native";

import type { Horizon } from "../domain";
import { colors } from "../theme";

export function HorizonSlider({
  disabled = false,
  onChange,
  value,
}: {
  disabled?: boolean;
  onChange: (value: Horizon) => void;
  value: Horizon;
}) {
  return (
    <View style={styles.control}>
      <Slider
        accessibilityLabel="Prediction horizon"
        accessibilityValue={{
          max: 5,
          min: 2,
          now: value,
          text: `${value} blocks`,
        }}
        disabled={disabled}
        maximumTrackTintColor={colors.border}
        maximumValue={5}
        minimumTrackTintColor={colors.blue}
        minimumValue={2}
        onValueChange={(nextValue) => onChange(nextValue as Horizon)}
        step={1}
        style={styles.slider}
        thumbTintColor={colors.blue}
        value={value}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  control: { gap: 1 },
  slider: { height: 28, width: "100%" },
});
