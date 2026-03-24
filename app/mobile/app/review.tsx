import { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ScrollView,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Colors, getMaturityColors } from "../constants/theme";
import { MaturityBadge } from "../components/MaturityBadge";

const STEP = 5;

type TrichomeKey = "clear" | "cloudy" | "amber";
type StigmaKey = "orange" | "green";

function calculateMaturity(clear: number, cloudy: number, amber: number): string {
  if (amber > 50) return "late";
  if (amber > 30) return "mature";
  if (cloudy > 60) return "peak";
  if (cloudy > 30) return "developing";
  return "early";
}

function DistributionBar({
  segments,
}: {
  segments: { color: string; value: number }[];
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  return (
    <View style={barStyles.track}>
      {segments.map((seg, i) => (
        <View
          key={i}
          style={[
            barStyles.fill,
            { flex: seg.value / total, backgroundColor: seg.color },
          ]}
        />
      ))}
    </View>
  );
}

const barStyles = StyleSheet.create({
  track: {
    flexDirection: "row",
    height: 6,
    borderRadius: 3,
    overflow: "hidden",
    backgroundColor: Colors.surfaceElevated,
    marginTop: 14,
    marginBottom: 4,
  },
  fill: {
    height: "100%",
  },
});

function AdjusterRow({
  label,
  dotColor,
  value,
  onDecrement,
  onIncrement,
}: {
  label: string;
  dotColor: string;
  value: number;
  onDecrement: () => void;
  onIncrement: () => void;
}) {
  return (
    <View style={adjusterStyles.row}>
      <View style={adjusterStyles.labelGroup}>
        <View style={[adjusterStyles.dot, { backgroundColor: dotColor }]} />
        <Text style={adjusterStyles.label}>{label}</Text>
      </View>
      <View style={adjusterStyles.controls}>
        <Pressable style={adjusterStyles.stepButton} onPress={onDecrement}>
          <Text style={adjusterStyles.stepButtonText}>−</Text>
        </Pressable>
        <Text style={adjusterStyles.value}>{value}%</Text>
        <Pressable style={adjusterStyles.stepButton} onPress={onIncrement}>
          <Text style={adjusterStyles.stepButtonText}>+</Text>
        </Pressable>
      </View>
    </View>
  );
}

const adjusterStyles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 10,
  },
  labelGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  label: {
    fontSize: 14,
    fontWeight: "500",
    color: Colors.textPrimary,
  },
  controls: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
  },
  stepButton: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: Colors.surfaceElevated,
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
  },
  stepButtonText: {
    fontSize: 18,
    color: Colors.textPrimary,
    fontWeight: "600",
    lineHeight: 22,
  },
  value: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
    width: 44,
    textAlign: "center",
  },
});

export default function ReviewScreen() {
  const router = useRouter();

  const [trichome, setTrichome] = useState({ clear: 12, cloudy: 73, amber: 15 });
  const [stigma, setStigma] = useState({ orange: 71, green: 29 });

  const adjustTrichome = (key: TrichomeKey, delta: number) => {
    setTrichome((prev) => ({
      ...prev,
      [key]: Math.max(0, Math.min(100, prev[key] + delta)),
    }));
  };

  const adjustStigma = (key: StigmaKey, delta: number) => {
    setStigma((prev) => ({
      ...prev,
      [key]: Math.max(0, Math.min(100, prev[key] + delta)),
    }));
  };

  const calculatedStage = calculateMaturity(
    trichome.clear,
    trichome.cloudy,
    trichome.amber
  );
  const stageColors = getMaturityColors(calculatedStage);

  const handleConfirm = () => {
    Alert.alert(
      "Save adjusted result?",
      `Maturity stage: ${stageColors.label}`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Save",
          onPress: () => router.replace("/home"),
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Pressable onPress={() => router.back()}>
          <Text style={styles.backText}>← Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Review Results</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>Manual adjustment</Text>
          <Text style={styles.infoText}>
            Modify the trichome and pistil values to better reflect what you observed. The maturity stage will update automatically.
          </Text>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionLabel}>TRICHOME DISTRIBUTION</Text>

          <DistributionBar
            segments={[
              { color: "#64748b", value: trichome.clear },
              { color: "#e2e8f0", value: trichome.cloudy },
              { color: "#f59e0b", value: trichome.amber },
            ]}
          />

          <View style={styles.divider} />

          <AdjusterRow
            label="Clear"
            dotColor="#64748b"
            value={trichome.clear}
            onDecrement={() => adjustTrichome("clear", -STEP)}
            onIncrement={() => adjustTrichome("clear", STEP)}
          />
          <View style={styles.rowDivider} />
          <AdjusterRow
            label="Cloudy"
            dotColor="#e2e8f0"
            value={trichome.cloudy}
            onDecrement={() => adjustTrichome("cloudy", -STEP)}
            onIncrement={() => adjustTrichome("cloudy", STEP)}
          />
          <View style={styles.rowDivider} />
          <AdjusterRow
            label="Amber"
            dotColor="#f59e0b"
            value={trichome.amber}
            onDecrement={() => adjustTrichome("amber", -STEP)}
            onIncrement={() => adjustTrichome("amber", STEP)}
          />
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionLabel}>PISTIL / STIGMA ANALYSIS</Text>

          <DistributionBar
            segments={[
              { color: "#f59e0b", value: stigma.orange },
              { color: "#4ade80", value: stigma.green },
            ]}
          />

          <View style={styles.divider} />

          <AdjusterRow
            label="Orange"
            dotColor="#f59e0b"
            value={stigma.orange}
            onDecrement={() => adjustStigma("orange", -STEP)}
            onIncrement={() => adjustStigma("orange", STEP)}
          />
          <View style={styles.rowDivider} />
          <AdjusterRow
            label="Green"
            dotColor="#4ade80"
            value={stigma.green}
            onDecrement={() => adjustStigma("green", -STEP)}
            onIncrement={() => adjustStigma("green", STEP)}
          />
        </View>

        <View style={styles.resultCard}>
          <Text style={styles.sectionLabel}>RECALCULATED RESULT</Text>
          <View style={styles.resultRow}>
            <View style={styles.resultLeft}>
              <Text style={styles.resultStageName}>{stageColors.label}</Text>
              <MaturityBadge stage={calculatedStage} size="sm" />
            </View>
            <View
              style={[
                styles.resultAccent,
                { backgroundColor: stageColors.background },
              ]}
            >
              <Text style={[styles.resultAccentText, { color: stageColors.text }]}>
                {stageColors.label[0]}
              </Text>
            </View>
          </View>
          <Text style={styles.resultNote}>
            Based on your adjusted values above.
          </Text>
        </View>

        <Pressable style={styles.confirmButton} onPress={handleConfirm}>
          <Text style={styles.confirmButtonText}>Confirm & Save</Text>
        </Pressable>

        <Pressable
          style={styles.keepAiButton}
          onPress={() => router.back()}
        >
          <Text style={styles.keepAiText}>Keep original AI result</Text>
        </Pressable>
      </ScrollView>
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
  scroll: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 48,
  },
  infoCard: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
    marginBottom: 16,
    gap: 6,
    borderLeftWidth: 3,
    borderLeftColor: Colors.accent,
  },
  infoTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  infoText: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
  },
  sectionCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 18,
    marginBottom: 14,
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: "700",
    color: Colors.textMuted,
    letterSpacing: 1.2,
  },
  divider: {
    height: 1,
    backgroundColor: Colors.border,
    marginTop: 16,
    marginBottom: 4,
  },
  rowDivider: {
    height: 1,
    backgroundColor: Colors.surfaceElevated,
  },
  resultCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 18,
    marginBottom: 20,
    gap: 12,
  },
  resultRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 10,
  },
  resultLeft: {
    gap: 8,
  },
  resultStageName: {
    fontSize: 28,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
  resultAccent: {
    width: 52,
    height: 52,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
  },
  resultAccentText: {
    fontSize: 24,
    fontWeight: "800",
  },
  resultNote: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  confirmButton: {
    width: "100%",
    paddingVertical: 16,
    borderRadius: 12,
    backgroundColor: Colors.accent,
    alignItems: "center",
    marginBottom: 14,
  },
  confirmButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.2,
  },
  keepAiButton: {
    alignItems: "center",
    paddingVertical: 8,
  },
  keepAiText: {
    fontSize: 14,
    color: Colors.textMuted,
  },
});
