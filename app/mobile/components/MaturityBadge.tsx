import { View, Text, StyleSheet } from "react-native";
import { useTheme } from "../contexts/ThemeContext";

type Props = {
  stage: string;
  size?: "sm" | "md";
};

export function MaturityBadge({ stage, size = "md" }: Props) {
  const { getMaturityColors } = useTheme();
  const colors = getMaturityColors(stage);

  return (
    <View
      style={[
        styles.badge,
        { backgroundColor: colors.background },
        size === "sm" && styles.badgeSm,
      ]}
    >
      <Text
        style={[
          styles.text,
          { color: colors.text },
          size === "sm" && styles.textSm,
        ]}
      >
        {colors.label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 999,
  },
  badgeSm: {
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  text: {
    fontSize: 13,
    fontWeight: "600",
    letterSpacing: 0.3,
  },
  textSm: {
    fontSize: 11,
  },
});
