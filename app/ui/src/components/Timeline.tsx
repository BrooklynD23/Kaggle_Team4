import type { Phase } from '../api/types';

type Props = {
  phases?: Phase[];
};

export function Timeline({ phases }: Props) {
  if (!phases || phases.length === 0) {
    return null;
  }

  return (
    <section className="panel">
      <header>
        <h3>Experiment Timeline</h3>
      </header>
      <ul className="timeline">
        {phases.map((phase) => (
          <li key={`${phase.phase_name}-${phase.timestamp}`}>
            <div className="timeline-header">
              <strong>{phase.phase_name}</strong>
              <span className="subtle">{new Date(phase.timestamp).toLocaleString()}</span>
            </div>
            {phase.description && <p>{phase.description}</p>}
            <div className="timeline-metrics">
              <span>Macro F1: {phase.metrics.macro_f1?.toFixed(3)}</span>
              <span>Accuracy: {phase.metrics.accuracy?.toFixed(3)}</span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}


