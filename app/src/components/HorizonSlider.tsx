import Slider from "@react-native-community/slider";
import { StyleSheet, Text, View } from "react-native";

import { HORIZONS, type Horizon } from "../inference";
import { colors } from "../theme";

export function HorizonSlider({
  disabled = false,
  onChange,
  showTicks = true,
  value,
}: {
  disabled?: boolean;
  onChange: (value: Horizon) => void;
  showTicks?: boolean;
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
      {showTicks && (
        <View style={styles.ticks}>
          {HORIZONS.map((horizon) => (
            <Text
              key={horizon}
              style={[styles.tick, horizon === value && styles.tickActive]}
            >
              {horizon}
            </Text>
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  control: { gap: 1 },
  slider: { height: 28, width: "100%" },
  ticks: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 7,
  },
  tick: { color: colors.muted, fontSize: 10, fontWeight: "600" },
  tickActive: { color: colors.blue, fontWeight: "800" },
});
