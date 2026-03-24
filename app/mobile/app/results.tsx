import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Colors, Gradients, MaturityColors } from "../constants/theme";
import { MaturityBadge } from "../components/MaturityBadge";
import { BottomNav } from "../components/BottomNav";

export const MOCK_RESULT = {
  maturity_stage: "peak",
  recommendation:
    "Trichome analysis indicates optimal THC concentration. 73% of glandular heads have reached a milky/cloudy state with 15% turning amber. Harvest within the next 48 hours for maximum potency and terpene profile retention.",
  trichome: { clear: 12, cloudy: 73, amber: 15, total: 247 },
  stigma: { orange: 71, green: 29, total: 38 },
};

function SegmentedBar({
  segments,
}: {
  segments: { color: string; value: number; label: string }[];
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  return (
    <View>
      <View style={barStyles.track}>
        {segments.map((seg, i) => (
          <View
            key={i}
            style={[
              barStyles.segment,
              { flex: seg.value / total, backgroundColor: seg.color },
            ]}
          />
        ))}
      </View>
      <View style={barStyles.stats}>
        {segments.map((seg, i) => (
          <View key={i} style={barStyles.statCol}>
            <Text style={[barStyles.statLabel, { color: seg.color }]}>
              {seg.label.toUpperCase()}
            </Text>
            <Text style={barStyles.statValue}>{seg.value}%</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const barStyles = StyleSheet.create({
  track: {
    flexDirection: "row",
    height: 12,
    borderRadius: 6,
    overflow: "hidden",
    backgroundColor: Colors.surfaceHighest,
  },
  segment: {
    height: "100%",
  },
  stats: {
    flexDirection: "row",
    marginTop: 12,
  },
  statCol: {
    flex: 1,
    alignItems: "center",
    borderLeftWidth: 1,
    borderLeftColor: "rgba(65,71,91,0.2)",
    paddingLeft: 8,
  },
  statLabel: {
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginBottom: 2,
  },
  statValue: {
    fontSize: 20,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
});

export default function ResultsScreen() {
  const router = useRouter();
  const stage = MOCK_RESULT.maturity_stage as keyof typeof MaturityColors;
  const stageColors = MaturityColors[stage];

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.content}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Pressable style={styles.backButton} onPress={() => router.back()}>
              <Ionicons name="arrow-back" size={20} color={Colors.accent} />
            </Pressable>
            <Text style={styles.headerTitle}>Analysis Results</Text>
          </View>
          <View style={styles.headerRight}>
            <Ionicons name="share-outline" size={20} color={Colors.textSecondary} />
            <Ionicons name="ellipsis-vertical" size={20} color={Colors.textSecondary} style={{ marginLeft: 16 }} />
          </View>
        </View>

        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.maturityCard}>
            <LinearGradient
              colors={Gradients.vitality}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.maturityAccentBar}
            />
            <View style={styles.maturityBody}>
              <View style={styles.maturityTop}>
                <View>
                  <Text style={styles.maturityLabel}>MATURITY STAGE</Text>
                  <Text style={[styles.maturityStage, { color: stageColors.text }]}>
                    {stageColors.label}
                  </Text>
                </View>
                <View style={styles.confidenceGroup}>
                  <View style={styles.readyBadge}>
                    <Text style={styles.readyBadgeText}>READY</Text>
                  </View>
                  <View style={styles.confidencePill}>
                    <View style={styles.confidenceDot} />
                    <Text style={styles.confidenceText}>94% AI Confidence</Text>
                  </View>
                </View>
              </View>
              <MaturityBadge stage={MOCK_RESULT.maturity_stage} size="sm" />
            </View>
          </View>

          <View style={styles.recommendationCard}>
            <View style={styles.cardLabelRow}>
              <Ionicons name="bulb-outline" size={16} color={Colors.accent} />
              <Text style={styles.cardLabel}>HARVEST RECOMMENDATION</Text>
            </View>
            <Text style={styles.recommendationText}>
              {MOCK_RESULT.recommendation}
            </Text>
            <View style={styles.recBlobBg} />
          </View>

          <View style={styles.imagePlaceholder}>
            <View style={styles.imagePlaceholderIcon}>
              <Ionicons name="image-outline" size={32} color={Colors.textMuted} />
            </View>
            <Text style={styles.imagePlaceholderText}>Annotated Image</Text>
            <Text style={styles.imagePlaceholderSub}>
              Trichome detections highlighted in real-time
            </Text>
          </View>

          <View style={styles.sectionHeader}>
            <Ionicons name="bar-chart-outline" size={18} color={Colors.accent} />
            <Text style={styles.sectionTitle}>Analysis Details</Text>
          </View>

          <View style={styles.analysisCard}>
            <View style={styles.cardTopRow}>
              <View style={styles.cardLabelRow}>
                <View style={[styles.dot, { backgroundColor: "#e2e8f0" }]} />
                <Text style={styles.cardLabel}>TRICHOME DISTRIBUTION</Text>
              </View>
              <View style={styles.countPill}>
                <Text style={styles.countPillText}>{MOCK_RESULT.trichome.total} DETECTED</Text>
              </View>
            </View>
            <SegmentedBar
              segments={[
                { color: "#6f758b", value: MOCK_RESULT.trichome.clear, label: "Clear" },
                { color: "#dfe4fe", value: MOCK_RESULT.trichome.cloudy, label: "Cloudy" },
                { color: "#d97706", value: MOCK_RESULT.trichome.amber, label: "Amber" },
              ]}
            />
          </View>

          <View style={styles.analysisCard}>
            <View style={styles.cardTopRow}>
              <View style={styles.cardLabelRow}>
                <View style={[styles.dot, { backgroundColor: Colors.tertiary }]} />
                <Text style={styles.cardLabel}>PISTIL MATURATION</Text>
              </View>
              <View style={styles.countPill}>
                <Text style={styles.countPillText}>{MOCK_RESULT.stigma.total} DETECTED</Text>
              </View>
            </View>
            <SegmentedBar
              segments={[
                { color: "#d97706", value: MOCK_RESULT.stigma.orange, label: "Orange" },
                { color: Colors.accent, value: MOCK_RESULT.stigma.green, label: "Green" },
              ]}
            />
          </View>

          <View style={styles.reviewBanner}>
            <Ionicons name="pencil-outline" size={20} color={Colors.warning} />
            <Text style={styles.reviewBannerText}>Want to adjust values?</Text>
            <Pressable style={styles.reviewButton} onPress={() => router.push("/review")}>
              <Text style={styles.reviewButtonText}>Manual Edit</Text>
            </Pressable>
          </View>

          <View style={styles.actionRow}>
            <Pressable style={styles.saveButtonWrapper} onPress={() => router.replace("/home")}>
              {({ pressed }) => (
                <LinearGradient
                  colors={Gradients.vitality}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={[styles.saveButton, pressed && styles.pressed]}
                >
                  <Ionicons name="checkmark-circle-outline" size={18} color={Colors.accentText} />
                  <Text style={styles.saveButtonText}>Save Result</Text>
                </LinearGradient>
              )}
            </Pressable>

            <Pressable
              style={styles.rescanButton}
              onPress={() => router.replace("/camera")}
            >
              <Ionicons name="camera-outline" size={18} color={Colors.accent} />
            </Pressable>
          </View>
        </ScrollView>

        <BottomNav />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 12,
    backgroundColor: Colors.surface,
  },
  headerLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  backButton: {
    padding: 2,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.accent,
    letterSpacing: -0.2,
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
  },
  scroll: {
    paddingHorizontal: 20,
    paddingBottom: 24,
    paddingTop: 16,
  },
  maturityCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    overflow: "hidden",
    marginBottom: 12,
  },
  maturityAccentBar: {
    height: 3,
    width: "100%",
  },
  maturityBody: {
    padding: 20,
    gap: 12,
  },
  maturityTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  maturityLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textSecondary,
    letterSpacing: 1.5,
    marginBottom: 4,
  },
  maturityStage: {
    fontSize: 40,
    fontWeight: "800",
    letterSpacing: -1.5,
  },
  confidenceGroup: {
    alignItems: "flex-end",
    gap: 6,
  },
  readyBadge: {
    backgroundColor: Colors.accentSurface,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: "rgba(107,255,143,0.2)",
  },
  readyBadgeText: {
    fontSize: 9,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 1.5,
  },
  confidencePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: Colors.surfaceHighest,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
  },
  confidenceDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.accent,
  },
  confidenceText: {
    fontSize: 10,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  recommendationCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    gap: 10,
    overflow: "hidden",
  },
  cardLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  cardLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textSecondary,
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
  recommendationText: {
    fontSize: 14,
    color: Colors.textPrimary,
    lineHeight: 22,
  },
  recBlobBg: {
    position: "absolute",
    top: 0,
    right: 0,
    padding: 16,
    opacity: 0.06,
  },
  imagePlaceholder: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: Colors.border,
    borderStyle: "dashed",
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 24,
    gap: 8,
  },
  imagePlaceholderIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.surfaceHighest,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  imagePlaceholderText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  imagePlaceholderSub: {
    fontSize: 12,
    color: Colors.textMuted,
    textAlign: "center",
    paddingHorizontal: 32,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  analysisCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 20,
    marginBottom: 12,
    gap: 16,
  },
  cardTopRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  countPill: {
    backgroundColor: Colors.surfaceHighest,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
  },
  countPillText: {
    fontSize: 9,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: 0.5,
  },
  reviewBanner: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.warningSurface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(120,53,15,0.3)",
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  reviewBannerText: {
    flex: 1,
    fontSize: 13,
    fontWeight: "600",
    color: "#fde68a",
  },
  reviewButton: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 8,
    backgroundColor: "rgba(217,119,6,0.15)",
  },
  reviewButtonText: {
    fontSize: 11,
    fontWeight: "800",
    color: Colors.warning,
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
  },
  saveButtonWrapper: {
    flex: 1,
  },
  saveButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 15,
    borderRadius: 999,
    gap: 8,
  },
  pressed: {
    opacity: 0.88,
  },
  saveButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
  },
  rescanButton: {
    width: 52,
    height: 52,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: Colors.accent,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(107,255,143,0.05)",
  },
});
