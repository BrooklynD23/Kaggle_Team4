import { useModels, useSwitchModel } from '../hooks/useApi';

export function ModelSelector() {
  const { data: modelsData, isLoading } = useModels();
  const switchModel = useSwitchModel();

  const handleModelChange = (filename: string) => {
    if (filename && filename !== modelsData?.current_model) {
      switchModel.mutate(filename);
    }
  };

  if (isLoading) {
    return (
      <div className="model-selector">
        <span className="subtle">Loading models...</span>
      </div>
    );
  }

  if (!modelsData || modelsData.models.length === 0) {
    return (
      <div className="model-selector">
        <span className="subtle">No models available</span>
      </div>
    );
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateString: string): string => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="model-selector">
      <label>
        <span className="model-selector-label">🤖 Active Model:</span>
        <select
          value={modelsData.current_model || ''}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={switchModel.isPending}
          className="model-select"
        >
          {modelsData.models.map((model) => (
            <option key={model.filename} value={model.filename}>
              {model.model_name} ({formatDate(model.file_mtime)}) - {formatFileSize(model.file_size)}
              {model.is_current ? ' ⭐' : ''}
            </option>
          ))}
        </select>
      </label>
      {switchModel.isPending && (
        <span className="model-switching">Switching model...</span>
      )}
      {switchModel.isSuccess && (
        <span className="model-switched">✓ {switchModel.data?.message}</span>
      )}
      {switchModel.isError && (
        <span className="model-error">✗ Failed to switch model</span>
      )}
      <div className="model-info">
        <small>
          {modelsData.models.length} model{modelsData.models.length !== 1 ? 's' : ''} available
        </small>
      </div>
    </div>
  );
}

