import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { analyze, checkHealth, type AnalysisResult } from "./api";
import { API_BASE_URL, API_BASE_URL_REASON } from "./config";
import { GROUPS, INDICATOR_IDS, VALUE_RANGE, type IndicatorId } from "./generated/schema";
import { Banner } from "./components/Banner";
import { Group } from "./components/Group";
import { ResultCard } from "./components/ResultCard";

const midpoint = (VALUE_RANGE.min + VALUE_RANGE.max) / 2;
const initialValues = Object.fromEntries(INDICATOR_IDS.map((id) => [id, midpoint])) as Record<IndicatorId, number>;

function App() {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [healthWarning, setHealthWarning] = useState<string | null>(null);

  useEffect(() => {
    void checkHealth().then((health) => {
      if (!health.reachable) setHealthWarning(health.detail);
    });
  }, []);

  const updateValue = (indicatorId: IndicatorId, value: number) => {
    setValues((current) => ({ ...current, [indicatorId]: value }));
    setErrors((current) => ({ ...current, [indicatorId]: "" }));
    setResult(null);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const clientErrors: Record<string, string> = {};
    for (const indicatorId of INDICATOR_IDS) {
      const value = values[indicatorId];
      if (!Number.isFinite(value) || value < VALUE_RANGE.min || value > VALUE_RANGE.max) {
        clientErrors[indicatorId] = `Enter a value between ${VALUE_RANGE.min} and ${VALUE_RANGE.max}.`;
      }
    }
    if (Object.keys(clientErrors).length > 0) {
      setErrors(clientErrors);
      return;
    }
    setBusy(true);
    setErrors({});
    setResult(null);
    const response = await analyze(values);
    setResult(response);
    if (response.kind === "validation_error") setErrors(response.fieldErrors);
    setBusy(false);
  };

  const networkMessage = result?.kind === "network_error" ? result.message : null;
  const validationMessage = result?.kind === "validation_error" && Object.keys(result.fieldErrors).length === 0 ? result.message : null;

  return (
    <main className="app-shell">
      <header>
        <p className="eyebrow">Aburrimiento</p>
        <h1>Group boredom analyzer</h1>
        <p className="intro">Assess the signals below to receive a schema-based analysis.</p>
      </header>
      {healthWarning && <Banner onDismiss={() => setHealthWarning(null)}>{healthWarning} Configured API: {API_BASE_URL}.</Banner>}
      {API_BASE_URL_REASON === "invalid" && <Banner tone="info">The API URL configuration is invalid; using {API_BASE_URL}.</Banner>}
      {networkMessage && <Banner tone="error">{networkMessage} Configured API: {API_BASE_URL}.</Banner>}
      {validationMessage && <Banner tone="error">{validationMessage}</Banner>}
      <form onSubmit={submit} noValidate>
        {GROUPS.map((group) => (
          <Group
            key={group.id}
            group={group}
            values={values}
            errors={errors}
            onChange={(indicator, value) => updateValue(indicator.id as IndicatorId, value)}
          />
        ))}
        <button className="submit-button" type="submit" disabled={busy} aria-busy={busy}>
          {busy ? "Analyzing…" : "Analyze"}
        </button>
      </form>
      <div aria-live="polite" aria-atomic="true">
        {result?.kind === "ok" && <ResultCard levelId={result.level} />}
      </div>
    </main>
  );
}

export default App;
