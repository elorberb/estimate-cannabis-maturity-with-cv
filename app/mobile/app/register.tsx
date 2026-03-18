import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { useRouter } from "expo-router";

export default function RegisterScreen() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <View style={styles.inner}>
        <Text style={styles.title}>Create Account</Text>
        <Text style={styles.subtitle}>Join to start tracking your harvest.</Text>

        <TextInput
          style={styles.input}
          placeholder="Name"
          placeholderTextColor="#6b7280"
          value={name}
          onChangeText={setName}
          autoCapitalize="words"
        />
        <TextInput
          style={styles.input}
          placeholder="Email"
          placeholderTextColor="#6b7280"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
        <TextInput
          style={styles.input}
          placeholder="Password"
          placeholderTextColor="#6b7280"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <Pressable style={styles.primaryButton} onPress={() => router.replace("/home")}>
          <Text style={styles.primaryButtonText}>Create Account</Text>
        </Pressable>

        <Pressable onPress={() => router.back()}>
          <Text style={styles.backText}>Already have an account? Sign in</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020617",
  },
  inner: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 24,
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
    marginBottom: 40,
  },
  input: {
    width: "100%",
    height: 50,
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#1e293b",
    borderRadius: 12,
    paddingHorizontal: 16,
    color: "#e5e7eb",
    fontSize: 16,
    marginBottom: 12,
  },
  primaryButton: {
    width: "100%",
    paddingVertical: 14,
    borderRadius: 999,
    backgroundColor: "#22c55e",
    alignItems: "center",
    marginTop: 8,
    marginBottom: 24,
  },
  primaryButtonText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#022c22",
  },
  backText: {
    fontSize: 15,
    color: "#6b7280",
    textDecorationLine: "underline",
  },
});
