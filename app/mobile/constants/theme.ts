export const Colors = {
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

export const Gradients = {
  vitality: ["#6bff8f", "#0abc56"] as [string, string],
};

export type MaturityStage = "early" | "developing" | "peak" | "mature" | "late";

export const MaturityColors: Record<MaturityStage, { background: string; text: string; label: string }> = {
  early: { background: "#1e3a5f", text: "#60a5fa", label: "Early" },
  developing: { background: "#2e1065", text: "#a78bfa", label: "Developing" },
  peak: { background: "#052e16", text: "#6bff8f", label: "Peak" },
  mature: { background: "#451a03", text: "#fbbf24", label: "Mature" },
  late: { background: "#450a0a", text: "#f87171", label: "Late" },
};

export function getMaturityColors(stage: string): { background: string; text: string; label: string } {
  return MaturityColors[stage as MaturityStage] ?? MaturityColors.early;
}
