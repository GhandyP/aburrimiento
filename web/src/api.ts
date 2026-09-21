import { API_BASE_URL } from "./config";
import type { AnalyzeResponse, IndicatorId, LevelId } from "./generated/schema";

export type AnalysisResult =
  | { kind: "ok"; level: LevelId }
  | { kind: "validation_error"; fieldErrors: Record<string, string>; message: string }
  | { kind: "network_error"; message: string };

export type CaptureResult =
  | { kind: "ok"; id: number; level: LevelId }
  | { kind: "validation_error"; fieldErrors: Record<string, string>; message: string }
  | { kind: "network_error"; message: string };

type ValidationDetail = {
  loc?: unknown;
  msg?: unknown;
};

type ErrorPayload = { detail?: unknown };

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as ErrorPayload).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => (typeof item === "object" && item !== null && "msg" in item ? String((item as ValidationDetail).msg) : String(item))).join("; ");
    }
  }
  return fallback;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

function validationErrors(payload: unknown): Record<string, string> {
  const fieldErrors: Record<string, string> = {};
  const details = typeof payload === "object" && payload !== null && "detail" in payload
    ? (payload as ErrorPayload).detail
    : undefined;
  if (Array.isArray(details)) {
    for (const detail of details as ValidationDetail[]) {
      const location = Array.isArray(detail.loc) ? detail.loc : [];
      const field = location.length >= 3 && location[0] === "body" && typeof location[2] === "string"
        ? location[2]
        : undefined;
      if (field && typeof detail.msg === "string") fieldErrors[field] = detail.msg;
    }
  }
  return fieldErrors;
}

export async function analyze(
  values: Record<IndicatorId, number>,
): Promise<AnalysisResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ datos: values }),
    });
    const payload = await readJson(response);
    if (response.ok) {
      return { kind: "ok", level: (payload as AnalyzeResponse).nivel };
    }
    if (response.status === 422) {
      return { kind: "validation_error", fieldErrors: validationErrors(payload), message: errorMessage(payload, "The submitted values are invalid.") };
    }
    return { kind: "network_error", message: errorMessage(payload, `API returned HTTP ${response.status}.`) };
  } catch {
    return { kind: "network_error", message: `Unable to reach the API at ${API_BASE_URL}.` };
  }
}

export async function capture(
  values: Record<IndicatorId, number>,
  observedLevel?: LevelId,
): Promise<CaptureResult> {
  const body: { datos: Record<IndicatorId, number>; nivel_observado?: LevelId } = { datos: values };
  if (observedLevel !== undefined) body.nivel_observado = observedLevel;
  try {
    const response = await fetch(`${API_BASE_URL}/samples`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await readJson(response);
    if (response.ok) {
      const sample = payload as { id: number; nivel: LevelId };
      return { kind: "ok", id: sample.id, level: sample.nivel };
    }
    if (response.status === 422) {
      return { kind: "validation_error", fieldErrors: validationErrors(payload), message: errorMessage(payload, "The submitted values are invalid.") };
    }
    return { kind: "network_error", message: errorMessage(payload, `API returned HTTP ${response.status}.`) };
  } catch {
    return { kind: "network_error", message: `Unable to reach the API at ${API_BASE_URL}.` };
  }
}

export async function checkHealth(): Promise<{ reachable: boolean; detail: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const payload = await readJson(response);
    return {
      reachable: response.ok,
      detail: response.ok ? "API is reachable." : errorMessage(payload, `API returned HTTP ${response.status}.`),
    };
  } catch {
    return { reachable: false, detail: `Unable to reach the API at ${API_BASE_URL}.` };
  }
}
