export interface Phase {
  phase_name: string;
  metrics: {
    macro_f1: number;
    accuracy: number;
    per_class_f1: number[];
  };
  timestamp: string;
  description?: string;
}

export interface MetricsResponse {
  generated_at: string;
  macro_f1: number;
  weighted_f1?: number;
  accuracy: number;
  per_class_f1: Record<string, number>;
  plots: Record<string, string>;
  artifact_path: string;
  phases?: Phase[];
}

export interface FeatureImportanceRow {
  Feature: string;
  Importance?: number;
  Importance_Pct?: number;
}

export interface PredictResponse {
  predicted_class: string;
  probabilities: Record<string, number>;
  raw_scores: number[];
  feature_count: number;
  missing_features: number;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  model_name?: string;
  model_loaded_at?: string;
  model_path?: string;
  metrics_updated_at?: string;
  artifact_available: boolean;
  pipeline_job: Record<string, unknown>;
}

export interface ModelInfo {
  filename: string;
  model_name: string;
  file_size: number;
  file_mtime: string;
  is_current: boolean;
  error?: string;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  current_model?: string;
}

export interface SwitchModelResponse {
  status: string;
  model_name: string;
  filename: string;
  loaded_at: string;
  message: string;
}


