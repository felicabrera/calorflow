// API Types and Interfaces

export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  version: string;
  models_available: {
    FCC: boolean;
    CCR: boolean;
  };
}

export interface ProcessInfo {
  name: string;
  description: string;
  targets: string[];
  data_files: {
    predictors: string;
    targets: string[];
  };
}

export interface TrainingRequest {
  process_name: 'FCC' | 'CCR';
  n_trials?: number;
  cv_folds?: number;
  use_optuna?: boolean;
  random_state?: number;
}

export interface TrainingResponse {
  success: boolean;
  process_name: string;
  best_models: {
    PCI: string;
    H2: string;
  };
  cv_scores: {
    PCI: number;
    H2: number;
  };
  training_time: number;
  timestamp: string;
}

export interface PredictionRequest {
  process_name: 'FCC' | 'CCR';
  features: Record<string, number>[];
}

export interface PredictionResponse {
  success: boolean;
  predictions: {
    PCI: number[];
    H2: number[];
  };
  n_predictions: number;
  process_name: string;
  timestamp: string;
}

export interface DataQualityRequest {
  data_path: string;
  target_cols?: string[];
}

export interface DataQualityResponse {
  success: boolean;
  n_samples: number;
  n_features: number;
  missing_values: Record<string, number>;
  missing_percentage: number;
  duplicates: number;
  numeric_features: number;
  categorical_features: number;
  datetime_features: number;
  quality_score: number;
  recommendations: string[];
}

export interface ErrorResponse {
  success: false;
  error: string;
}
