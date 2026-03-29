import { useState, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Image,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { ApiClient } from "../api/client";
import { useTheme } from "../contexts/ThemeContext";
import { PlantAnalysisItem } from "../api/types";

type ImageView = "original" | "detected";

function AnalysisImageBlock({
  imageUrl,
  annotatedUrl,
  blockStyles,
}: {
  imageUrl: string;
  annotatedUrl: string | null;
  blockStyles: ReturnType<typeof createStyles>;
}) {
  const { Colors } = useTheme();
  const hasDetected = !!annotatedUrl;
  const [view, setView] = useState<ImageView>(hasDetected ? "detected" : "original");
  const activeUrl = view === "detected" && annotatedUrl ? annotatedUrl : imageUrl;

  return (
    <View>
      <Image source={{ uri: activeUrl }} style={blockStyles.analysisImage} resizeMode="cover" />
      {hasDetected && (
        <View style={blockStyles.imageToggleRow}>
          <Pressable
            style={[blockStyles.imageToggleBtn, view === "original" && blockStyles.imageToggleBtnActive]}
            onPress={() => setView("original")}
          >
            <Ionicons name="image-outline" size={12} color={view === "original" ? Colors.accentText : Colors.textMuted} />
            <Text style={[blockStyles.imageToggleText, view === "original" && blockStyles.imageToggleTextActive]}>
              Original
            </Text>
          </Pressable>
          <Pressable
            style={[blockStyles.imageToggleBtn, view === "detected" && blockStyles.imageToggleBtnActive]}
            onPress={() => setView("detected")}
          >
            <Ionicons name="scan-outline" size={12} color={view === "detected" ? Colors.accentText : Colors.textMuted} />
            <Text style={[blockStyles.imageToggleText, view === "detected" && blockStyles.imageToggleTextActive]}>
              Detected
            </Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

export default function PlantDetailScreen() {
  const router = useRouter();
  const { Colors, getMaturityColors } = useTheme();
  const { plantId, plantName } = useLocalSearchParams<{ plantId: string; plantName: string }>();
  const [analyses, setAnalyses] = useState<PlantAnalysisItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalyses = useCallback(async () => {
    if (!plantId) return;
    setLoading(true);
    setError(null);
    try {
      const history = await ApiClient.listPlantAnalyses(plantId);
      setAnalyses(history.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  }, [plantId]);

  useFocusEffect(
    useCallback(() => {
      fetchAnalyses();
    }, [fetchAnalyses])
  );

  const styles = createStyles(Colors);

  const handleDeleteAnalysis = (analysisId: string) => {
    Alert.alert("Delete this scan?", "This cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            await ApiClient.deleteAnalysis(analysisId);
            setAnalyses((prev) => prev.filter((a) => a.id !== analysisId));
          } catch {
            Alert.alert("Error", "Failed to delete scan.");
          }
        },
      },
    ]);
  };

  const renderAnalysisItem = ({ item, index }: { item: PlantAnalysisItem; index: number }) => {
    const maturity = getMaturityColors(item.maturity_stage);
    const date = new Date(item.created_at);
    const dateLabel = date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    const timeLabel = date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });

    const trichomeTotal = item.trichome_distribution
      ? Object.values(item.trichome_distribution).reduce((a, b) => a + b, 0)
      : 0;
    const cloudy = item.trichome_distribution?.cloudy ?? 0;
    const amber = item.trichome_distribution?.amber ?? 0;
    const clear = item.trichome_distribution?.clear ?? 0;

    return (
      <View style={styles.timelineItem}>
        <View style={styles.timelineConnector}>
          <View style={[styles.timelineDot, { backgroundColor: maturity.text }]} />
          {index < analyses.length - 1 && <View style={styles.timelineLine} />}
        </View>

        <View style={styles.analysisCard}>
          <View style={styles.cardTopRow}>
            <View style={styles.dateBlock}>
              <Text style={styles.dateText}>{dateLabel}</Text>
              <Text style={styles.timeText}>{timeLabel}</Text>
            </View>
            <View style={styles.cardTopRight}>
              <View style={[styles.maturityBadge, { backgroundColor: maturity.background }]}>
                <Text style={[styles.maturityText, { color: maturity.text }]}>
                  {maturity.label}
                </Text>
              </View>
              <Pressable onPress={() => handleDeleteAnalysis(item.id)} hitSlop={8}>
                <Ionicons name="trash-outline" size={15} color={Colors.danger} />
              </Pressable>
            </View>
          </View>

          {item.image_url ? (
            <AnalysisImageBlock imageUrl={item.image_url} annotatedUrl={item.annotated_image_url} blockStyles={styles} />
          ) : item.annotated_image_url ? (
            <Image source={{ uri: item.annotated_image_url }} style={styles.analysisImage} resizeMode="cover" />
          ) : null}

          {item.trichome_distribution && trichomeTotal > 0 && (
            <View style={styles.statsRow}>
              <View style={styles.statChip}>
                <View style={[styles.statDot, { backgroundColor: "#e5e7eb" }]} />
                <Text style={styles.statValue}>{Math.round((clear / trichomeTotal) * 100)}%</Text>
                <Text style={styles.statLabel}>clear</Text>
              </View>
              <View style={styles.statChip}>
                <View style={[styles.statDot, { backgroundColor: "#a78bfa" }]} />
                <Text style={styles.statValue}>{Math.round((cloudy / trichomeTotal) * 100)}%</Text>
                <Text style={styles.statLabel}>cloudy</Text>
              </View>
              <View style={styles.statChip}>
                <View style={[styles.statDot, { backgroundColor: "#f59e0b" }]} />
                <Text style={styles.statValue}>{Math.round((amber / trichomeTotal) * 100)}%</Text>
                <Text style={styles.statLabel}>amber</Text>
              </View>
              <View style={styles.statChipTotal}>
                <Text style={styles.statValue}>{trichomeTotal}</Text>
                <Text style={styles.statLabel}>total</Text>
              </View>
            </View>
          )}

          {item.stigma_ratios && (
            <View style={styles.stigmaRow}>
              <Text style={styles.stigmaLabel}>Stigma</Text>
              <View style={styles.stigmaBar}>
                <View
                  style={[
                    styles.stigmaSegment,
                    {
                      flex: item.stigma_ratios.orange,
                      backgroundColor: "#f97316",
                    },
                  ]}
                />
                <View
                  style={[
                    styles.stigmaSegment,
                    {
                      flex: item.stigma_ratios.green,
                      backgroundColor: Colors.accent,
                    },
                  ]}
                />
              </View>
              <Text style={styles.stigmaRatioText}>
                {Math.round(item.stigma_ratios.orange * 100)}% orange
              </Text>
            </View>
          )}

          <Text style={styles.recommendation} numberOfLines={2}>
            {item.recommendation}
          </Text>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
        </Pressable>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle} numberOfLines={1}>{plantName ?? "Plant"}</Text>
          <Text style={styles.headerSub}>{analyses.length} scan{analyses.length !== 1 ? "s" : ""}</Text>
        </View>
        <View style={{ width: 36 }} />
      </View>

      {loading && (
        <View style={styles.centeredState}>
          <ActivityIndicator size="large" color={Colors.accent} />
          <Text style={styles.stateText}>Loading history...</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centeredState}>
          <Ionicons name="warning-outline" size={40} color={Colors.danger} />
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={fetchAnalyses}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && analyses.length === 0 && (
        <View style={styles.centeredState}>
          <Ionicons name="scan-outline" size={40} color={Colors.textMuted} />
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptySub}>
            Analyze this plant to start building its history.
          </Text>
          <Pressable style={styles.ctaButton} onPress={() => router.push("/camera")}>
            <Text style={styles.ctaButtonText}>Scan Now</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && analyses.length > 0 && (
        <FlatList
          data={analyses}
          keyExtractor={(item) => item.id}
          renderItem={renderAnalysisItem}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
        />
      )}
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
    gap: 12,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: Colors.surface,
    alignItems: "center",
    justifyContent: "center",
  },
  headerCenter: {
    flex: 1,
    gap: 2,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  headerSub: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  centeredState: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 40,
    gap: 12,
  },
  stateText: {
    fontSize: 14,
    color: Colors.textMuted,
    marginTop: 8,
  },
  errorText: {
    fontSize: 14,
    color: Colors.danger,
    textAlign: "center",
  },
  retryButton: {
    backgroundColor: Colors.surfaceElevated,
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  retryText: {
    fontSize: 14,
    fontWeight: "600",
    color: Colors.textPrimary,
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
    marginTop: 12,
    backgroundColor: Colors.accent,
    paddingHorizontal: 28,
    paddingVertical: 13,
    borderRadius: 10,
  },
  ctaButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
  },
  listContent: {
    padding: 16,
    paddingBottom: 40,
  },
  timelineItem: {
    flexDirection: "row",
    gap: 12,
  },
  timelineConnector: {
    alignItems: "center",
    width: 20,
    paddingTop: 4,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  timelineLine: {
    flex: 1,
    width: 2,
    backgroundColor: Colors.border,
    marginTop: 4,
    marginBottom: -4,
  },
  analysisCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 14,
    marginBottom: 16,
    gap: 12,
  },
  cardTopRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  cardTopRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  dateBlock: {
    gap: 2,
  },
  dateText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  timeText: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  maturityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  maturityText: {
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  analysisImage: {
    width: "100%",
    height: 180,
    borderRadius: 10,
    backgroundColor: Colors.surfaceElevated,
  },
  imageToggleRow: {
    flexDirection: "row",
    gap: 6,
    marginTop: 8,
  },
  imageToggleBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: Colors.surfaceHighest,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  imageToggleBtnActive: {
    backgroundColor: Colors.accent,
    borderColor: Colors.accent,
  },
  imageToggleText: {
    fontSize: 11,
    fontWeight: "700",
    color: Colors.textMuted,
    letterSpacing: 0.3,
  },
  imageToggleTextActive: {
    color: Colors.accentText,
  },
  statsRow: {
    flexDirection: "row",
    gap: 8,
  },
  statChip: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
    gap: 4,
  },
  statChipTotal: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
    gap: 4,
  },
  statDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  statValue: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  statLabel: {
    fontSize: 11,
    color: Colors.textMuted,
  },
  stigmaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  stigmaLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: Colors.textMuted,
    width: 44,
  },
  stigmaBar: {
    flex: 1,
    height: 8,
    borderRadius: 4,
    flexDirection: "row",
    overflow: "hidden",
    backgroundColor: Colors.surfaceElevated,
  },
  stigmaSegment: {
    height: "100%",
  },
  stigmaRatioText: {
    fontSize: 12,
    color: Colors.textSecondary,
    width: 70,
    textAlign: "right",
  },
  recommendation: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
    fontStyle: "italic",
  },
}); }
