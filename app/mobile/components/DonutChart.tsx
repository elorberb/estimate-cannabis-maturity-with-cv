import { View, Text, StyleSheet } from "react-native";
import Svg, { Circle } from "react-native-svg";
import { useTheme } from "../contexts/ThemeContext";

type Segment = {
  value: number;
  color: string;
};

type Props = {
  segments: Segment[];
  size?: number;
  strokeWidth?: number;
  centerLabel?: string;
  centerSublabel?: string;
};

export function DonutChart({
  segments,
  size = 160,
  strokeWidth = 22,
  centerLabel,
  centerSublabel,
}: Props) {
  const { Colors } = useTheme();
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const cx = size / 2;
  const cy = size / 2;
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;

  let cumOffset = 0;

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Svg
        width={size}
        height={size}
        style={[StyleSheet.absoluteFill, { transform: [{ rotate: "-90deg" }] }]}
      >
        <Circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={Colors.surfaceHighest}
          strokeWidth={strokeWidth}
        />
        {segments.map((seg, i) => {
          const segLen = (seg.value / total) * circumference;
          const offset = cumOffset;
          cumOffset += segLen;
          if (seg.value <= 0) return null;
          return (
            <Circle
              key={i}
              cx={cx}
              cy={cy}
              r={radius}
              fill="none"
              stroke={seg.color}
              strokeWidth={strokeWidth}
              strokeLinecap="butt"
              strokeDasharray={`${segLen} ${circumference}`}
              strokeDashoffset={-offset}
            />
          );
        })}
      </Svg>
      {centerLabel && (
        <View style={styles.center}>
          <Text style={[styles.centerLabel, { color: Colors.textPrimary }]}>{centerLabel}</Text>
          {centerSublabel && (
            <Text style={[styles.centerSublabel, { color: Colors.textMuted }]}>{centerSublabel}</Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
  },
  center: {
    alignItems: "center",
  },
  centerLabel: {
    fontSize: 30,
    fontWeight: "800",
    letterSpacing: -1,
  },
  centerSublabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1,
    textTransform: "uppercase",
    marginTop: 2,
  },
});
