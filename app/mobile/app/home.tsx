import { View, Text, Pressable, StyleSheet } from "react-native";
import { useRouter } from "expo-router";

export default function HomeScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Pressable style={styles.logoutButton} onPress={() => router.replace("/")}>
        <Text style={styles.logoutText}>Log out</Text>
      </Pressable>

      <Text style={styles.title}>Snap & Analyze</Text>
      <Text style={styles.subtitle}>
        Estimate cannabis maturity from a single macro photo.
      </Text>
      <Pressable style={styles.primaryButton} onPress={() => router.push("/camera")}>
        <Text style={styles.primaryButtonText}>Analyze photo</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 24,
    backgroundColor: "#020617",
  },
  logoutButton: {
    position: "absolute",
    top: 60,
    right: 24,
  },
  logoutText: {
    fontSize: 15,
    color: "#6b7280",
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#e5e7eb",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: "#9ca3af",
    textAlign: "center",
    marginBottom: 32,
  },
  primaryButton: {
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 999,
    backgroundColor: "#22c55e",
  },
  primaryButtonText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#022c22",
  },
});
