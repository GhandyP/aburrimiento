import { beforeEach, describe, expect, it, vi } from "vitest";
import { analyze, capture } from "./api";
import { INDICATOR_IDS } from "./generated/schema";

const values = Object.fromEntries(INDICATOR_IDS.map((id, index) => [id, index / 14])) as Record<typeof INDICATOR_IDS[number], number>;

beforeEach(() => vi.restoreAllMocks());

describe("capture", () => {
  it("sends only datos when no observed level is supplied", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ id: 4, nivel: "medio" }), { status: 200 }));
    await capture(values);
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(Object.keys(body)).toEqual(["datos"]);
    expect(Object.keys(body.datos)).toHaveLength(14);
  });
  it("sends an observed level when supplied", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ id: 4, nivel: "medio" }), { status: 200 }));
    await capture(values, "alto");
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({ nivel_observado: "alto" });
  });
  it("returns the saved id and predicted level", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ id: 4, nivel: "medio" }), { status: 200 }));
    await expect(capture(values)).resolves.toEqual({ kind: "ok", id: 4, level: "medio" });
  });
  it("maps validation errors and keeps rejected fetches as network errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: [{ loc: ["body", "datos", "desenganche"], msg: "Invalid" }] }), { status: 422 }));
    await expect(capture(values)).resolves.toMatchObject({ kind: "validation_error", fieldErrors: { desenganche: "Invalid" } });
    vi.mocked(globalThis.fetch).mockRejectedValue(new Error("offline"));
    await expect(capture(values)).resolves.toMatchObject({ kind: "network_error" });
  });
});

describe("analyze", () => {
  it("returns the level and sends every typed value", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ nivel: "medio" }), { status: 200 }));
    await expect(analyze(values)).resolves.toEqual({ kind: "ok", level: "medio" });
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(Object.keys(body.datos)).toHaveLength(14);
    expect(body.datos).toEqual(values);
  });
  it("maps per-field validation details", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: [{ loc: ["body", "datos", "desenganche"], msg: "Input should be less than or equal to 1" }] }), { status: 422 }));
    await expect(analyze(values)).resolves.toMatchObject({ kind: "validation_error", fieldErrors: { desenganche: "Input should be less than or equal to 1" } });
  });
  it("keeps model-level validation in the message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: [{ loc: ["body", "datos"], msg: "Legacy field rejected" }] }), { status: 422 }));
    await expect(analyze(values)).resolves.toMatchObject({ kind: "validation_error", fieldErrors: {}, message: "Legacy field rejected" });
  });
  it("distinguishes rejected fetches from validation", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    await expect(analyze(values)).resolves.toMatchObject({ kind: "network_error" });
  });
});
