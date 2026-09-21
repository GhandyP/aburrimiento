import { useState } from "react";
import { capture, type CaptureResult } from "../api";
import { API_BASE_URL } from "../config";
import { INDICATOR_IDS, LEVELS, type IndicatorId, type LevelId } from "../generated/schema";
import { Banner } from "./Banner";

type CapturePanelProps = {
  values: Record<IndicatorId, number>;
  onValidationErrors: (errors: Record<string, string>) => void;
};

export function CapturePanel({ values, onValidationErrors }: CapturePanelProps) {
  const [observedLevel, setObservedLevel] = useState<LevelId | "">("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CaptureResult | null>(null);

  const save = async () => {
    setBusy(true);
    setResult(null);
    const response = await capture(values, observedLevel || undefined);
    setResult(response);
    if (response.kind === "validation_error") onValidationErrors(response.fieldErrors);
    setBusy(false);
  };

  return (
    <section className="capture-panel" aria-labelledby="capture-heading">
      <h2 id="capture-heading">Keep this observation</h2>
      <p>This stores {INDICATOR_IDS.length} values, the predicted level, and the optional observed level; it does not store identity.</p>
      <label htmlFor="observed-level">What was the group actually like?</label>
      <select
        id="observed-level"
        value={observedLevel}
        onChange={(event) => setObservedLevel(event.target.value as LevelId | "")}
        disabled={busy}
      >
        <option value="">prefiero no decirlo</option>
        {LEVELS.map((level) => <option key={level.id} value={level.id}>{level.labels.es}</option>)}
      </select>
      <button className="capture-button" type="button" onClick={() => void save()} disabled={busy} aria-busy={busy}>
        {busy ? "Saving…" : "Save observation"}
      </button>
      {result?.kind === "network_error" && <Banner tone="error">{result.message} Configured API: {API_BASE_URL}.</Banner>}
      {result?.kind === "validation_error" && Object.keys(result.fieldErrors).length === 0 && <Banner tone="error">{result.message}</Banner>}
      <div aria-live="polite" aria-atomic="true">
        {result?.kind === "ok" && <p className="capture-confirmation">Observation saved with id {result.id}; server predicted {result.level}.</p>}
      </div>
    </section>
  );
}
