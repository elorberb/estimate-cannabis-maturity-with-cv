import { View, Text, Pressable, StyleSheet } from "react-native";
import { useRouter, usePathname } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Colors } from "../constants/theme";

const NAV_ITEMS = [
  { label: "Home", icon: "leaf-outline" as const, activeIcon: "leaf" as const, route: "/home" },
  { label: "Analysis", icon: "camera-outline" as const, activeIcon: "camera" as const, route: "/camera" },
  { label: "History", icon: "time-outline" as const, activeIcon: "time" as const, route: "/history" },
  { label: "Settings", icon: "settings-outline" as const, activeIcon: "settings" as const, route: "/settings" },
];

export function BottomNav() {
  const router = useRouter();
  const pathname = usePathname();

  return (
    <View style={styles.container}>
      {NAV_ITEMS.map((item) => {
        const isActive = pathname === item.route;
        return (
          <Pressable
            key={item.route}
            style={[styles.item, isActive && styles.itemActive]}
            onPress={() => router.push(item.route as never)}
          >
            <Ionicons
              name={isActive ? item.activeIcon : item.icon}
              size={22}
              color={isActive ? Colors.accent : Colors.textMuted}
            />
            <Text style={[styles.label, isActive && styles.labelActive]}>
              {item.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    backgroundColor: "rgba(23,31,54,0.92)",
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    paddingBottom: 24,
    paddingTop: 10,
  },
  item: {
    flex: 1,
    alignItems: "center",
    gap: 4,
    paddingVertical: 4,
    borderRadius: 12,
  },
  itemActive: {
    backgroundColor: Colors.accentSurface,
  },
  label: {
    fontSize: 10,
    fontWeight: "700",
    color: Colors.textMuted,
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  labelActive: {
    color: Colors.accent,
  },
});
