import {
  AnalyzeResponse,
  AnalysisListResponse,
  HealthResponse,
} from "./types";

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export const ApiClient = {
  async analyzeImage(fileUri: string, deviceId: string): Promise<AnalyzeResponse> {
    const form = new FormData();
    form.append("file", {
      uri: fileUri,
      name: "photo.jpg",
      type: "image/jpeg",
    } as unknown as Blob);

    return request<AnalyzeResponse>(`/api/v1/analyze?device_id=${encodeURIComponent(deviceId)}`, {
      method: "POST",
      body: form,
    });
  },

  async getAnalysis(id: string): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/api/v1/analyses/${id}`);
  },

  async listAnalyses(): Promise<AnalysisListResponse> {
    return request<AnalysisListResponse>("/api/v1/analyses");
  },

  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/v1/health");
  },
};
