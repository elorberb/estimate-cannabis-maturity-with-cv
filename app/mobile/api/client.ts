import {
  AnalyzeResponse,
  AnalysisListResponse,
  HealthResponse,
} from "./types";

const API_BASE_URL = "http://192.168.1.213:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export const ApiClient = {
  async analyzeImage(imageUrl: string): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>("/analyze", {
      method: "POST",
      body: JSON.stringify({ image_url: imageUrl }),
    });
  },

  async getAnalysis(id: string): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/analyses/${id}`);
  },

  async listAnalyses(): Promise<AnalysisListResponse> {
    return request<AnalysisListResponse>("/analyses");
  },

  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },
};

