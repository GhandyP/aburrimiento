import type { Indicator } from "../generated/schema";
import { VALUE_RANGE } from "../generated/schema";

type FieldProps = {
  indicator: Indicator;
  value: number;
  error?: string;
  onChange: (value: number) => void;
};

export function Field({ indicator, value, error, onChange }: FieldProps) {
  const inputId = `indicator-${indicator.id}`;
  const errorId = `${inputId}-error`;
  return (
    <div className="field">
      <label htmlFor={inputId}>{indicator.labels.es}</label>
      <div className="field-controls">
        <input
          id={inputId}
          type="range"
          min={VALUE_RANGE.min}
          max={VALUE_RANGE.max}
          step="any"
          value={Number.isFinite(value) ? value : VALUE_RANGE.min}
          onChange={(event) => onChange(Number(event.target.value))}
          aria-describedby={error ? errorId : undefined}
        />
        <input
          className="number-input"
          type="number"
          min={VALUE_RANGE.min}
          max={VALUE_RANGE.max}
          step="any"
          value={Number.isNaN(value) ? "" : value}
          onChange={(event) => onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))}
          aria-label={`${indicator.labels.es} exact value`}
          aria-describedby={error ? errorId : undefined}
        />
      </div>
      {error && <p className="field-error" id={errorId}>{error}</p>}
    </div>
  );
}
