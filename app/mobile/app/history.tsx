import { View, Text, Pressable, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Colors } from "../constants/theme";

export default function HistoryScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Pressable onPress={() => router.back()}>
          <Text style={styles.backText}>← Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>My Analyses</Text>
        <View style={styles.headerSpacer} />
      </View>

      <View style={styles.emptyState}>
        <View style={styles.emptyIconContainer}>
          <View style={styles.emptyIconInner} />
        </View>
        <Text style={styles.emptyTitle}>No analyses yet</Text>
        <Text style={styles.emptySub}>
          Complete a scan and save the result to build your analysis history.
        </Text>
        <Pressable
          style={styles.ctaButton}
          onPress={() => router.push("/camera")}
        >
          <Text style={styles.ctaButtonText}>Start First Scan</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.surfaceElevated,
  },
  backText: {
    fontSize: 15,
    color: Colors.accent,
    fontWeight: "500",
    width: 60,
  },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  headerSpacer: {
    width: 60,
  },
  emptyState: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 40,
    gap: 10,
  },
  emptyIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 20,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
  },
  emptyIconInner: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: Colors.surfaceElevated,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  emptyTitle: {
    fontSize: 19,
    fontWeight: "700",
    color: Colors.textPrimary,
    textAlign: "center",
  },
  emptySub: {
    fontSize: 14,
    color: Colors.textMuted,
    textAlign: "center",
    lineHeight: 21,
  },
  ctaButton: {
    marginTop: 20,
    backgroundColor: Colors.accent,
    paddingHorizontal: 28,
    paddingVertical: 13,
    borderRadius: 10,
  },
  ctaButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.2,
  },
});
