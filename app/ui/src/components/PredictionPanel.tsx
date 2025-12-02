import { useRef, useState } from 'react';
import { usePredict } from '../hooks/useApi';

const FEATURE_FIELDS = [
  { key: 'grade_improvement', label: 'Grade Improvement (Sem2 - Sem1)', placeholder: '-5 to +5' },
  { key: 'grade_per_unit_sem1', label: 'Grade per Unit (Sem1)', placeholder: '0 - 5' },
  { key: 'Curricular units 1st sem (grade)', label: 'Sem1 Grade', placeholder: '0 - 20' },
  { key: "Father's occupation", label: "Father's Occupation (encoded)", placeholder: '1 - 40' },
  { key: 'Course', label: 'Course Code', placeholder: '100 - 999' }
];

// Preset test cases for quick demo
const PRESET_CASES = [
  {
    name: '📗 High-Performing Graduate',
    description: 'Strong academic performance',
    features: {
      grade_improvement: 3.0,
      grade_per_unit_sem1: 14.5,
      'Curricular units 1st sem (grade)': 15.0,
      "Father's occupation": 5,
      Course: 171,
      approval_rate_sem1: 1.0,
      approval_rate_overall: 1.0,
      total_approved_units: 6,
      financial_risk: 0,
      high_risk_student: 0
    }
  },
  {
    name: '📕 At-Risk Dropout',
    description: 'Warning signs present',
    features: {
      grade_improvement: -4.0,
      grade_per_unit_sem1: 5.0,
      'Curricular units 1st sem (grade)': 7.0,
      "Father's occupation": 18,
      Course: 9500,
      approval_rate_sem1: 0.4,
      approval_rate_overall: 0.35,
      total_approved_units: 2,
      financial_risk: 1,
      high_risk_student: 1
    }
  },
  {
    name: '📙 Enrolled - Mixed Signals',
    description: 'Uncertain trajectory',
    features: {
      grade_improvement: 0.0,
      grade_per_unit_sem1: 11.0,
      'Curricular units 1st sem (grade)': 12.0,
      "Father's occupation": 10,
      Course: 9238,
      approval_rate_sem1: 0.67,
      approval_rate_overall: 0.65,
      total_approved_units: 4,
      financial_risk: 0,
      high_risk_student: 0
    }
  }
];

type FormState = Record<string, number>;

const initialState = FEATURE_FIELDS.reduce<FormState>((acc, field) => {
  acc[field.key] = 0;
  return acc;
}, {});

export function PredictionPanel() {
  const [form, setForm] = useState<FormState>(initialState);
  const [extras, setExtras] = useState('');
  const [loadedFileName, setLoadedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mutation = usePredict();

  const handleChange = (key: string, value: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: Number(value)
    }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsed = JSON.parse(content);
        
        // Check if it's our mock_test_data format with test_cases array
        if (parsed.test_cases && Array.isArray(parsed.test_cases)) {
          // Use the first test case's features
          const features = parsed.test_cases[0].features;
          applyFeatures(features);
          setLoadedFileName(`${file.name} (${parsed.test_cases[0].name})`);
        } else if (parsed.features) {
          // Single test case with features property
          applyFeatures(parsed.features);
          setLoadedFileName(file.name);
        } else {
          // Direct features object
          applyFeatures(parsed);
          setLoadedFileName(file.name);
        }
      } catch (error) {
        alert('Invalid JSON file. Please upload a valid JSON file with features.');
      }
    };
    reader.readAsText(file);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const applyFeatures = (features: Record<string, number>) => {
    // Update form fields
    const newForm = { ...initialState };
    FEATURE_FIELDS.forEach((field) => {
      if (features[field.key] !== undefined) {
        newForm[field.key] = features[field.key];
      }
    });
    setForm(newForm);

    // Put remaining features in extras textarea
    const extraFeatures: Record<string, number> = {};
    Object.entries(features).forEach(([key, value]) => {
      if (!FEATURE_FIELDS.some((f) => f.key === key)) {
        extraFeatures[key] = value;
      }
    });
    if (Object.keys(extraFeatures).length > 0) {
      setExtras(JSON.stringify(extraFeatures, null, 2));
    }
  };

  const handlePresetSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(event.target.value, 10);
    if (isNaN(index) || index < 0) return;
    
    const preset = PRESET_CASES[index];
    applyFeatures(preset.features);
    setLoadedFileName(`Preset: ${preset.name}`);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    let payload: Record<string, number> = { ...form };

    if (extras.trim()) {
      try {
        const parsed = JSON.parse(extras);
        payload = { ...payload, ...parsed };
      } catch (error) {
        alert('Invalid JSON payload.');
        return;
      }
    }

    mutation.mutate(payload);
  };

  const handleClear = () => {
    setForm(initialState);
    setExtras('');
    setLoadedFileName(null);
    mutation.reset();
  };

  return (
    <section className="panel">
      <header>
        <h3>Try a Prediction</h3>
        <span className="subtle">Upload JSON, select preset, or fill manually.</span>
      </header>

      {/* File Upload & Preset Selection */}
      <div className="upload-controls">
        <label className="file-upload-btn">
          📁 Upload JSON
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            onChange={handleFileUpload}
            hidden
          />
        </label>
        
        <select onChange={handlePresetSelect} defaultValue="">
          <option value="" disabled>🎯 Quick Test Cases</option>
          {PRESET_CASES.map((preset, idx) => (
            <option key={idx} value={idx}>
              {preset.name} — {preset.description}
            </option>
          ))}
        </select>

        <button type="button" className="clear-btn" onClick={handleClear}>
          ↺ Clear
        </button>
      </div>

      {loadedFileName && (
        <div className="loaded-indicator">
          ✓ Loaded: <strong>{loadedFileName}</strong>
        </div>
      )}

      <form className="prediction-form" onSubmit={handleSubmit}>
        {FEATURE_FIELDS.map((field) => (
          <label key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              step="0.01"
              value={form[field.key]}
              placeholder={field.placeholder}
              onChange={(event) => handleChange(field.key, event.target.value)}
            />
          </label>
        ))}
        <label>
          <span>Additional features (JSON)</span>
          <textarea
            value={extras}
            placeholder='{"approval_rate_sem1": 0.8}'
            onChange={(event) => setExtras(event.target.value)}
            rows={4}
          />
        </label>
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Scoring…' : 'Predict Outcome'}
        </button>
      </form>

      {mutation.data && (
        <div className="prediction-result">
          <h4>Prediction Result</h4>
          <p className={`prediction-class outcome-${mutation.data.predicted_class.toLowerCase()}`}>
            {mutation.data.predicted_class}
          </p>
          <div className="probability-bars">
            {Object.entries(mutation.data.probabilities).map(([label, value]) => (
              <div key={label} className="prob-row">
                <span className="prob-label">{label}</span>
                <div className="prob-bar-container">
                  <div 
                    className={`prob-bar bar-${label.toLowerCase()}`} 
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
                <span className="prob-value">{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <small className="feature-info">
            Features used: {mutation.data.feature_count - mutation.data.missing_features} / {mutation.data.feature_count}
            {mutation.data.missing_features > 0 && ` (${mutation.data.missing_features} auto-filled with 0)`}
          </small>
        </div>
      )}
    </section>
  );
}
