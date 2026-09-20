import { INDICATORS, type Indicator, type IndicatorId } from "../generated/schema";
import { Field } from "./Field";

type GroupProps = {
  group: { id: string; labels: { es: string } };
  values: Record<string, number>;
  errors: Record<string, string>;
  onChange: (indicator: Indicator, value: number) => void;
};

export function Group({ group, values, errors, onChange }: GroupProps) {
  const indicators = INDICATORS.filter((indicator) => indicator.groupId === group.id);
  return (
    <fieldset className="indicator-group">
      <legend>{group.labels.es}</legend>
      {indicators.map((indicator) => (
        <Field
          key={indicator.id}
          indicator={indicator}
          value={values[indicator.id as IndicatorId] ?? 0.5}
          error={errors[indicator.id]}
          onChange={(value) => onChange(indicator, value)}
        />
      ))}
    </fieldset>
  );
}
