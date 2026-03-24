import { Pressable, Text, StyleSheet, ViewStyle } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { Gradients, Colors } from "../constants/theme";

type Props = {
  label: string;
  onPress: () => void;
  icon?: keyof typeof Ionicons.glyphMap;
  style?: ViewStyle;
  fullWidth?: boolean;
};

export function GradientButton({ label, onPress, icon, style, fullWidth = true }: Props) {
  return (
    <Pressable onPress={onPress} style={[fullWidth && styles.fullWidth, style]}>
      {({ pressed }) => (
        <LinearGradient
          colors={Gradients.vitality}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={[styles.gradient, pressed && styles.pressed]}
        >
          {icon && <Ionicons name={icon} size={18} color={Colors.accentText} />}
          <Text style={styles.label}>{label}</Text>
        </LinearGradient>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  fullWidth: {
    width: "100%",
  },
  gradient: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 15,
    paddingHorizontal: 24,
    borderRadius: 999,
    gap: 8,
  },
  pressed: {
    opacity: 0.88,
  },
  label: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.3,
  },
});
