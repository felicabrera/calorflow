import type {
  HealthCheckResponse,
  ProcessInfo,
  TrainingRequest,
  TrainingResponse,
  PredictionRequest,
  PredictionResponse,
  DataQualityRequest,
  DataQualityResponse,
  ErrorResponse
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class CalorflowAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Health Check
  async healthCheck(): Promise<HealthCheckResponse> {
    return this.request<HealthCheckResponse>('/health');
  }

  // Get Process Info
  async getProcessInfo(processName: 'FCC' | 'CCR'): Promise<ProcessInfo> {
    return this.request<ProcessInfo>(`/process-info/${processName}`);
  }

  // List Available Models
  async listModels(): Promise<{ FCC: boolean; CCR: boolean }> {
    return this.request<{ FCC: boolean; CCR: boolean }>('/models');
  }

  // Train Model
  async trainModel(data: TrainingRequest): Promise<TrainingResponse | ErrorResponse> {
    return this.request<TrainingResponse>('/train', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Single Prediction
  async predict(data: PredictionRequest): Promise<PredictionResponse | ErrorResponse> {
    return this.request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Check Data Quality
  async checkDataQuality(data: DataQualityRequest): Promise<DataQualityResponse | ErrorResponse> {
    return this.request<DataQualityResponse>('/data-quality', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Upload File
  async uploadFile(file: File): Promise<{ success: boolean; filename: string; path: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload-data`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return await response.json();
  }

  // Download Prediction
  async downloadPrediction(filename: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/download-prediction/${filename}`);
    
    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return await response.blob();
  }
}

export const api = new CalorflowAPI();
export default api;
