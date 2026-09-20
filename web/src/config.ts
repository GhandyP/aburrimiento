const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export type ApiBaseUrlResolutionReason = "configured" | "default" | "invalid";

export function resolveApiBaseUrl(raw?: string): string {
  const candidate = raw?.trim();
  if (!candidate) return DEFAULT_API_BASE_URL;

  try {
    const url = new URL(candidate);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return DEFAULT_API_BASE_URL;
    }
    return candidate.replace(/\/+$/, "");
  } catch {
    return DEFAULT_API_BASE_URL;
  }
}

export function resolveApiBaseUrlReason(
  raw?: string,
): ApiBaseUrlResolutionReason {
  const candidate = raw?.trim();
  if (!candidate) return "default";
  try {
    const url = new URL(candidate);
    return url.protocol === "http:" || url.protocol === "https:"
      ? "configured"
      : "invalid";
  } catch {
    return "invalid";
  }
}

const rawApiUrl = (import.meta as ImportMeta & { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL;
export const API_BASE_URL = resolveApiBaseUrl(rawApiUrl);
export const API_BASE_URL_REASON = resolveApiBaseUrlReason(rawApiUrl);
