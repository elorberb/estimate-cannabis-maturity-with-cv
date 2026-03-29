import { useState, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { ApiClient } from "../api/client";
import { PlantResponse } from "../api/types";
import { useTheme } from "../contexts/ThemeContext";

export default function HistoryScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const [plants, setPlants] = useState<PlantResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPlants = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await ApiClient.listPlants();
      setPlants(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load plants");
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchPlants();
    }, [fetchPlants])
  );

  const handleDeletePlant = (plant: PlantResponse) => {
    Alert.alert(
      `Delete "${plant.name}"?`,
      "This will permanently delete this plant and all its linked data.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              await ApiClient.deletePlant(plant.id);
              setPlants((prev) => prev.filter((p) => p.id !== plant.id));
            } catch {
              Alert.alert("Error", "Failed to delete plant.");
            }
          },
        },
      ]
    );
  };

  const renderPlantCard = ({ item }: { item: PlantResponse }) => {
    const createdDate = new Date(item.created_at).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });

    const statusColor =
      item.status === "active"
        ? Colors.accent
        : item.status === "harvested"
          ? Colors.warning
          : Colors.textMuted;

    return (
      <Pressable
        style={({ pressed }) => [styles.plantCard, pressed && { opacity: 0.8 }]}
        onPress={() => router.push({ pathname: "/plant-detail", params: { plantId: item.id, plantName: item.name } })}
      >
        <View style={styles.plantCardLeft}>
          <View style={styles.plantIconContainer}>
            <Ionicons name="leaf" size={22} color={Colors.accent} />
          </View>
        </View>
        <View style={styles.plantCardCenter}>
          <Text style={styles.plantName} numberOfLines={1}>{item.name}</Text>
          <Text style={styles.plantDate}>{createdDate}</Text>
        </View>
        <View style={styles.plantCardRight}>
          <View style={[styles.statusBadge, { borderColor: statusColor }]}>
            <Text style={[styles.statusText, { color: statusColor }]}>
              {item.status}
            </Text>
          </View>
          <Pressable style={styles.deleteButton} onPress={() => handleDeletePlant(item)} hitSlop={8}>
            <Ionicons name="trash-outline" size={16} color={Colors.danger} />
          </Pressable>
        </View>
      </Pressable>
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>My Plants</Text>
      </View>

      {loading && (
        <View style={styles.centeredState}>
          <ActivityIndicator size="large" color={Colors.accent} />
          <Text style={styles.stateText}>Loading plants...</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centeredState}>
          <Ionicons name="warning-outline" size={40} color={Colors.danger} />
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={fetchPlants}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && plants.length === 0 && (
        <View style={styles.centeredState}>
          <View style={styles.emptyIconContainer}>
            <Ionicons name="leaf-outline" size={36} color={Colors.textMuted} />
          </View>
          <Text style={styles.emptyTitle}>No plants yet</Text>
          <Text style={styles.emptySub}>
            Complete a scan and save the result to start tracking a plant.
          </Text>
          <Pressable style={styles.ctaButton} onPress={() => router.push("/camera")}>
            <Text style={styles.ctaButtonText}>Start First Scan</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && plants.length > 0 && (
        <FlatList
          data={plants}
          keyExtractor={(item) => item.id}
          renderItem={renderPlantCard}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={
            <Text style={styles.listHeader}>{plants.length} plant{plants.length !== 1 ? "s" : ""}</Text>
          }
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
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
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
    lineHeight: 20,
  },
  retryButton: {
    marginTop: 8,
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
    letterSpacing: 0.2,
  },
  listContent: {
    padding: 16,
    gap: 10,
  },
  listHeader: {
    fontSize: 13,
    fontWeight: "600",
    color: Colors.textMuted,
    marginBottom: 10,
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  plantCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 14,
    gap: 12,
    marginBottom: 10,
  },
  plantCardLeft: {},
  plantIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: Colors.accentSurface,
    alignItems: "center",
    justifyContent: "center",
  },
  plantCardCenter: {
    flex: 1,
    gap: 2,
  },
  plantName: {
    fontSize: 16,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  plantDate: {
    fontSize: 12,
    color: Colors.textMuted,
    marginTop: 2,
  },
  plantCardRight: {
    alignItems: "flex-end",
    gap: 8,
  },
  deleteButton: {
    padding: 4,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    borderWidth: 1,
  },
  statusText: {
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
}); }
