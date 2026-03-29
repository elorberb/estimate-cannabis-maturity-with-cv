import { View } from "react-native";
import { Stack, usePathname } from "expo-router";
import { ThemeProvider } from "../contexts/ThemeContext";
import { BottomNav } from "../components/BottomNav";

const HIDE_NAV_ROUTES = ["/", "/register", "/review", "/trichome-samples", "/stigma-samples", "/save-flower"];

function AppShell() {
  const pathname = usePathname();
  const showNav = !HIDE_NAV_ROUTES.includes(pathname);

  return (
    <View style={{ flex: 1 }}>
      <Stack style={{ flex: 1 }} screenOptions={{ headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="register" />
        <Stack.Screen name="home" />
        <Stack.Screen name="camera" />
        <Stack.Screen name="results" />
        <Stack.Screen name="trichome-samples" />
        <Stack.Screen name="stigma-samples" />
        <Stack.Screen name="save-flower" />
        <Stack.Screen name="review" />
        <Stack.Screen name="history" />
        <Stack.Screen name="plant-detail" />
        <Stack.Screen name="how-it-works" />
        <Stack.Screen name="settings" />
      </Stack>
      {showNav && <BottomNav />}
    </View>
  );
}

export default function RootLayout() {
  return (
    <ThemeProvider>
      <AppShell />
    </ThemeProvider>
  );
}
