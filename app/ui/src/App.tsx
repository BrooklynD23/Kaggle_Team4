import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useFeatureImportance, useHealth, useMetrics } from './hooks/useApi';
import { MetricsGrid } from './components/MetricsGrid';
import { FeatureList } from './components/FeatureList';
import { Timeline } from './components/Timeline';
import { PlotGallery } from './components/PlotGallery';
import { PredictionPanel } from './components/PredictionPanel';
import { ModelSelector } from './components/ModelSelector';

const queryClient = new QueryClient();

function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useMetrics();
  const { data: features } = useFeatureImportance();
  const { data: health } = useHealth();

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Student Success Command Center</h1>
          <p>Live metrics & demo controls backed by the FastAPI service.</p>
        </div>
        <div className="header-controls">
          <ModelSelector />
          <div className="health">
            <span className={`status-dot ${health?.status === 'ok' ? 'ok' : 'warn'}`} />
            <div>
              <strong>{health?.model_name ?? 'Model not loaded'}</strong>
              <small>Loaded: {health?.model_loaded_at ? new Date(health.model_loaded_at).toLocaleString() : '—'}</small>
            </div>
          </div>
        </div>
      </header>
      <main className="layout">
        <div className="column">
          <MetricsGrid metrics={metrics} />
          {!metricsLoading && <Timeline phases={metrics?.phases} />}
          <FeatureList features={features} />
        </div>
        <div className="column">
          <PlotGallery plots={metrics?.plots} />
          <PredictionPanel />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}


