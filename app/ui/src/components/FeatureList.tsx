import type { FeatureImportanceRow } from '../api/types';

type Props = {
  features?: FeatureImportanceRow[];
  maxItems?: number;
};

export function FeatureList({ features, maxItems = 10 }: Props) {
  const top = (features ?? []).slice(0, maxItems);

  return (
    <section className="panel">
      <header>
        <h3>Top Drivers</h3>
        <span className="subtle">updated automatically after each run</span>
      </header>
      {top.length === 0 ? (
        <p className="empty-state">Run the pipeline to populate feature importance.</p>
      ) : (
        <ul className="feature-list">
          {top.map((row) => (
            <li key={row.Feature}>
              <span>{row.Feature}</span>
              <strong>{(row.Importance_Pct ?? row.Importance ?? 0).toFixed(2)}%</strong>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}


