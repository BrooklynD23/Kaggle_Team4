import type { MetricsResponse } from '../api/types';
import { KPICard } from './KPICard';

type Props = {
  metrics?: MetricsResponse;
};

export function MetricsGrid({ metrics }: Props) {
  return (
    <section className="panel">
      <header>
        <h3>Latest Metrics</h3>
        <span className="subtle">{metrics?.generated_at ? new Date(metrics.generated_at).toLocaleString() : '—'}</span>
      </header>
      <div className="metrics-grid">
        <KPICard label="Macro F1" value={metrics?.macro_f1} />
        <KPICard label="Weighted F1" value={metrics?.weighted_f1} />
        <KPICard label="Accuracy" value={metrics?.accuracy} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
        {Object.entries(metrics?.per_class_f1 ?? {}).map(([label, value]) => (
          <KPICard key={label} label={`${label} F1`} value={value} />
        ))}
      </div>
    </section>
  );
}


