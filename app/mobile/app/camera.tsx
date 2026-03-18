import { View, Text, Pressable, Image, StyleSheet, Alert } from "react-native";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import { useState } from "react";

export default function CameraScreen() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Camera access is needed to take photos.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });
    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const chooseFromGallery = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission required", "Gallery access is needed to pick photos.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });
    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  return (
    <View style={styles.container}>
      <Pressable style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backButtonText}>← Back</Text>
      </Pressable>

      <Text style={styles.title}>New Analysis</Text>
      <Text style={styles.subtitle}>
        Hold your phone 2–3 cm from the trichomes for best results.
      </Text>

      {selectedImage ? (
        <Image source={{ uri: selectedImage }} style={styles.preview} />
      ) : (
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>No photo selected</Text>
        </View>
      )}

      <Pressable style={styles.primaryButton} onPress={takePhoto}>
        <Text style={styles.primaryButtonText}>Take Photo</Text>
      </Pressable>

      <Pressable style={styles.secondaryButton} onPress={chooseFromGallery}>
        <Text style={styles.secondaryButtonText}>Choose from Gallery</Text>
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
  backButton: {
    position: "absolute",
    top: 60,
    left: 24,
  },
  backButtonText: {
    color: "#22c55e",
    fontSize: 16,
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    color: "#e5e7eb",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: "#9ca3af",
    textAlign: "center",
    marginBottom: 32,
  },
  placeholder: {
    width: "100%",
    height: 240,
    backgroundColor: "#0f172a",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 32,
  },
  placeholderText: {
    color: "#6b7280",
    fontSize: 15,
  },
  preview: {
    width: "100%",
    height: 240,
    borderRadius: 16,
    marginBottom: 32,
    resizeMode: "cover",
  },
  primaryButton: {
    width: "100%",
    paddingVertical: 14,
    borderRadius: 999,
    backgroundColor: "#22c55e",
    alignItems: "center",
    marginBottom: 12,
  },
  primaryButtonText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#022c22",
  },
  secondaryButton: {
    width: "100%",
    paddingVertical: 14,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#22c55e",
    alignItems: "center",
  },
  secondaryButtonText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#22c55e",
  },
});
