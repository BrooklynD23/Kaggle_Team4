import { useState } from 'react';
import { apiBaseUrl } from '../api/client';

type Props = {
  plots?: Record<string, string>;
};

export function PlotGallery({ plots }: Props) {
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);

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
    <>
      <section className="panel">
        <header>
          <h3>Visuals</h3>
          <span className="subtle">Click to expand</span>
        </header>
        <div className="plot-grid">
          {Object.entries(plots).map(([key, relativePath]) => {
            const src = `${apiBaseUrl?.replace(/\/$/, '')}/${relativePath}`;
            return (
              <figure
                key={key}
                onClick={() => setSelectedImage({ src, alt: key })}
                style={{ cursor: 'pointer' }}
              >
                <img src={src} alt={key} loading="lazy" />
                <figcaption>{key.replace(/_/g, ' ')}</figcaption>
              </figure>
            );
          })}
        </div>
      </section>

      {/* Modal for full-screen view */}
      {selectedImage && (
        <div
          className="modal-overlay"
          onClick={() => setSelectedImage(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            zIndex: 9999,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '2rem'
          }}
        >
          <button
            onClick={() => setSelectedImage(null)}
            style={{
              position: 'absolute',
              top: '1rem',
              right: '2rem',
              background: 'transparent',
              border: 'none',
              color: 'white',
              fontSize: '3rem',
              cursor: 'pointer',
              lineHeight: 1
            }}
          >
            &times;
          </button>
          <img
            src={selectedImage.src}
            alt={selectedImage.alt}
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              borderRadius: '8px',
              boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
            }}
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the image itself
          />
        </div>
      )}
    </>
  );
}


