import { View, Text, Image, Pressable, StyleSheet } from "react-native";
import { MaturityBadge } from "./MaturityBadge";
import { Colors } from "../constants/theme";
import { AnalysisListItem } from "../api/types";

type Props = {
  analysis: AnalysisListItem;
  onPress: () => void;
};

export function AnalysisCard({ analysis, onPress }: Props) {
  const date = new Date(analysis.created_at).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <Pressable style={styles.card} onPress={onPress}>
      <View style={styles.thumbnail}>
        {analysis.image_url ? (
          <Image source={{ uri: analysis.image_url }} style={styles.image} />
        ) : (
          <View style={styles.imagePlaceholder}>
            <View style={styles.imagePlaceholderInner} />
          </View>
        )}
      </View>
      <View style={styles.content}>
        <Text style={styles.date}>{date}</Text>
        <MaturityBadge stage={analysis.maturity_stage} size="sm" />
        <Text style={styles.recommendation} numberOfLines={2}>
          {analysis.recommendation}
        </Text>
      </View>
      <Text style={styles.chevron}>›</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 12,
    marginBottom: 10,
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 10,
    overflow: "hidden",
    marginRight: 14,
  },
  image: {
    width: "100%",
    height: "100%",
  },
  imagePlaceholder: {
    width: "100%",
    height: "100%",
    backgroundColor: Colors.surfaceElevated,
    alignItems: "center",
    justifyContent: "center",
  },
  imagePlaceholderInner: {
    width: 24,
    height: 24,
    borderRadius: 6,
    backgroundColor: Colors.border,
  },
  content: {
    flex: 1,
    gap: 5,
  },
  date: {
    fontSize: 11,
    color: Colors.textMuted,
    fontWeight: "500",
  },
  recommendation: {
    fontSize: 12,
    color: Colors.textSecondary,
    lineHeight: 17,
  },
  chevron: {
    fontSize: 20,
    color: Colors.textMuted,
    marginLeft: 8,
  },
});
