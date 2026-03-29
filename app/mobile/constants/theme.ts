export type ColorScheme = "dark" | "light";

const DarkColors = {
  background: "#070d1f",
  surface: "#0c1326",
  surfaceElevated: "#171f36",
  surfaceHighest: "#1c253e",
  border: "#41475b",
  borderSubtle: "rgba(65,71,91,0.15)",
  textPrimary: "#dfe4fe",
  textSecondary: "#a5aac2",
  textMuted: "#6f758b",
  accent: "#6bff8f",
  accentDark: "#0abc56",
  accentSurface: "rgba(107,255,143,0.10)",
  accentText: "#004a1d",
  danger: "#ff7351",
  dangerSurface: "#b92902",
  warning: "#d97706",
  warningSurface: "rgba(120,53,15,0.20)",
  tertiary: "#7de9ff",
  tertiarySurface: "rgba(0,224,253,0.10)",
};

const LightColors = {
  background: "#f1f5f9",
  surface: "#ffffff",
  surfaceElevated: "#f8fafc",
  surfaceHighest: "#e8eef5",
  border: "#cbd5e1",
  borderSubtle: "rgba(148,163,184,0.20)",
  textPrimary: "#0f172a",
  textSecondary: "#475569",
  textMuted: "#94a3b8",
  accent: "#16a34a",
  accentDark: "#15803d",
  accentSurface: "rgba(22,163,74,0.10)",
  accentText: "#ffffff",
  danger: "#dc2626",
  dangerSurface: "#fecaca",
  warning: "#d97706",
  warningSurface: "rgba(217,119,6,0.10)",
  tertiary: "#0284c7",
  tertiarySurface: "rgba(2,132,199,0.10)",
};

export const Colors = DarkColors;

export const ThemeColors: Record<ColorScheme, typeof DarkColors> = {
  dark: DarkColors,
  light: LightColors,
};

export const Gradients = {
  vitality: ["#6bff8f", "#0abc56"] as [string, string],
};

export const GradientsByTheme: Record<ColorScheme, { vitality: [string, string] }> = {
  dark: { vitality: ["#6bff8f", "#0abc56"] },
  light: { vitality: ["#16a34a", "#15803d"] },
};

export type MaturityStage = "early" | "developing" | "peak" | "mature" | "late";

export const MaturityColors: Record<MaturityStage, { background: string; text: string; label: string }> = {
  early: { background: "#1e3a5f", text: "#60a5fa", label: "Early" },
  developing: { background: "#2e1065", text: "#a78bfa", label: "Developing" },
  peak: { background: "#052e16", text: "#6bff8f", label: "Peak" },
  mature: { background: "#451a03", text: "#fbbf24", label: "Mature" },
  late: { background: "#450a0a", text: "#f87171", label: "Late" },
};

export const MaturityColorsByTheme: Record<ColorScheme, Record<MaturityStage, { background: string; text: string; label: string }>> = {
  dark: MaturityColors,
  light: {
    early: { background: "#dbeafe", text: "#1d4ed8", label: "Early" },
    developing: { background: "#ede9fe", text: "#6d28d9", label: "Developing" },
    peak: { background: "#dcfce7", text: "#15803d", label: "Peak" },
    mature: { background: "#fef3c7", text: "#b45309", label: "Mature" },
    late: { background: "#fee2e2", text: "#b91c1c", label: "Late" },
  },
};

export function getMaturityColors(stage: string): { background: string; text: string; label: string } {
  return MaturityColors[stage as MaturityStage] ?? MaturityColors.early;
}
