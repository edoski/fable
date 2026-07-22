import Svg, { Path } from "react-native-svg";

import type { Chain } from "../inference";

export function NetworkIcon({
  chain,
  color,
  size = 42,
}: {
  chain: Chain;
  color?: string;
  size?: number;
}) {
  if (chain === "ethereum") {
    const fill = color ?? "#627EEA";
    return (
      <Svg
        accessible={false}
        height={size}
        viewBox="0 0 1920 1920"
        width={size}
      >
        <Path
          d="m959.8 730.9-539.8 245.4 539.8 319.1 539.9-319.1z"
          fill={fill}
          opacity={0.6}
        />
        <Path
          d="m420.2 976.3 539.8 319.1v-564.5-650.3z"
          fill={fill}
          opacity={0.45}
        />
        <Path
          d="m960 80.6v650.3 564.5l539.8-319.1z"
          fill={fill}
          opacity={0.8}
        />
        <Path d="m420 1078.7 539.8 760.7v-441.8z" fill={fill} opacity={0.45} />
        <Path d="m959.8 1397.6v441.8l540.2-760.7z" fill={fill} opacity={0.8} />
      </Svg>
    );
  }

  if (chain === "polygon") {
    return (
      <Svg
        accessible={false}
        height={size}
        viewBox="0 0 24.3 24.3"
        width={size}
      >
        <Path
          d="M17.41 1.223 10.595 5.137v12.216l-3.76 2.18-3.784-2.181V12.99l3.784-2.16 2.432 1.41V8.712L6.813 7.319 0 11.277v7.83l6.836 3.937 6.814-3.937V6.893l3.783-2.182 3.782 2.182v4.342l-3.782 2.201-2.454-1.423v3.511l2.431 1.402 6.881-3.914V5.137L17.41 1.223Z"
          fill={color ?? "#8247E5"}
        />
      </Svg>
    );
  }

  return (
    <Svg accessible={false} height={size} viewBox="0 0 722 628" width={size}>
      <Path
        d="M548.831 381.485c11.184-19.05 38.961-19.05 50.022 0l118.85 203.04c11.184 19.05-2.827 42.771-25.073 42.771H454.932c-22.246 0-36.135-23.721-25.073-42.771l118.972-203.04Z"
        fill={color ?? "#E84142"}
      />
      <Path
        d="m477.034 246.295-97.464-169.978c-10.939-19.05-38.224-19.05-49.162 0L4.218 584.407c-10.939 19.05 2.704 42.894 24.581 42.894h194.804c21.755 0 41.788-11.676 52.604-30.604l200.704-350.402h.123Z"
        fill={color ?? "#E84142"}
      />
    </Svg>
  );
}
