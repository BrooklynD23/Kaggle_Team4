import { apiBaseUrl } from '../api/client';

type Props = {
  plots?: Record<string, string>;
};

export function PlotGallery({ plots }: Props) {
  if (!plots || Object.keys(plots).length === 0) {
    return (
      <section className="panel">
        <header>
          <h3>Visuals</h3>
        </header>
        <p className="empty-state">
          No plots yet. Run the pipeline and `py -m src.evaluation.visuals` to generate PNG/SVG assets.
        </p>
      </section>
    );
  }

  return (
    <section className="panel">
      <header>
        <h3>Visuals</h3>
        <span className="subtle">Rendering directly from artifacts/plots</span>
      </header>
      <div className="plot-grid">
        {Object.entries(plots).map(([key, relativePath]) => (
          <figure key={key}>
            <img src={`${apiBaseUrl?.replace(/\/$/, '')}/${relativePath}`} alt={key} loading="lazy" />
            <figcaption>{key.replace(/_/g, ' ')}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}


