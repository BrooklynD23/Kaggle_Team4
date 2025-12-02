type Props = {
  label: string;
  value?: number;
  helper?: string;
  formatter?: (value: number) => string;
};

const defaultFormatter = (value: number) => value.toFixed(3);

export function KPICard({ label, value, helper, formatter = defaultFormatter }: Props) {
  return (
    <div className="kpi-card">
      <p className="kpi-label">{label}</p>
      <h2 className="kpi-value">{value !== undefined ? formatter(value) : '—'}</h2>
      {helper && <span className="kpi-helper">{helper}</span>}
    </div>
  );
}


