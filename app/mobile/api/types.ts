export type TrichomeDistribution = {
  clear: number;
  cloudy: number;
  amber: number;
  total: number;
};

export type StigmaRatios = {
  avg_green_ratio: number;
  avg_orange_ratio: number;
  total_count: number;
};

export type DetectionItem = {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  trichome_type: "clear" | "cloudy" | "amber";
  confidence: number;
};

export type AnalyzeResponse = {
  id: string;
  created_at: string;
  image_url: string;
  annotated_image_url: string | null;
  trichome_distribution: TrichomeDistribution;
  stigma_ratios: StigmaRatios;
  maturity_stage: string;
  recommendation: string;
  detections: DetectionItem[];
  counts: Record<string, number>;
};

export type AnalysisListItem = {
  id: string;
  created_at: string;
  maturity_stage: string;
  recommendation: string;
  image_url: string;
};

export type AnalysisListResponse = {
  items: AnalysisListItem[];
  total: number;
};

export type HealthResponse = {
  status: string;
};
