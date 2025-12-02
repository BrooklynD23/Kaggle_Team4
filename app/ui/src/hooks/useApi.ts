import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import api from '../api/client';
import type {
  FeatureImportanceRow,
  HealthResponse,
  MetricsResponse,
  ModelsListResponse,
  PredictResponse,
  SwitchModelResponse
} from '../api/types';

const fetchMetrics = async (): Promise<MetricsResponse> => {
  const { data } = await api.get<MetricsResponse>('/metrics/latest');
  return data;
};

const fetchFeatures = async (): Promise<FeatureImportanceRow[]> => {
  const { data } = await api.get<{ features: FeatureImportanceRow[] }>('/insights/feature-importance');
  return data.features ?? [];
};

const fetchHealth = async (): Promise<HealthResponse> => {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
};

export const useMetrics = () =>
  useQuery({
    queryKey: ['metrics'],
    queryFn: fetchMetrics,
    refetchInterval: 30_000
  });

export const useFeatureImportance = () =>
  useQuery({
    queryKey: ['feature-importance'],
    queryFn: fetchFeatures,
    refetchInterval: 60_000
  });

export const useHealth = () =>
  useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30_000
  });

export const usePredict = () =>
  useMutation({
    mutationFn: async (features: Record<string, number>) => {
      const { data } = await api.post<PredictResponse>('/predict', { features });
      return data;
    }
  });

const fetchModels = async (): Promise<ModelsListResponse> => {
  const { data } = await api.get<ModelsListResponse>('/models');
  return data;
};

export const useModels = () =>
  useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    refetchInterval: 30_000
  });

export const useSwitchModel = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (filename: string) => {
      const { data } = await api.post<SwitchModelResponse>('/models/switch', { filename });
      return data;
    },
    onSuccess: () => {
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['models'] });
      queryClient.invalidateQueries({ queryKey: ['health'] });
    }
  });
};


