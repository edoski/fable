import Slider from "@react-native-community/slider";

import { HORIZONS, type Horizon } from "../domain";
import { styles } from "../styles";
import { colors } from "../theme";

export function HorizonSlider({
  disabled,
  onChange,
  value,
}: {
  disabled?: boolean;
  onChange: (value: Horizon) => void;
  value: Horizon;
}) {
  const minimum = HORIZONS[0];
  const maximum = HORIZONS[HORIZONS.length - 1];

  return (
    <Slider
      disabled={disabled}
      maximumTrackTintColor={colors.border}
      maximumValue={maximum}
      minimumTrackTintColor={colors.blue}
      minimumValue={minimum}
      onSlidingComplete={(nextValue: number) =>
        onChange(nextValue as Horizon)
      }
      step={1}
      style={styles.horizonSlider}
      thumbTintColor={colors.blue}
      value={value}
    />
  );
}
