// @ts-expect-error The web project does not include Node typings; Vitest supplies this runtime module.
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { analyze } from "./api";
import {
  INDICATOR_IDS,
  LEVELS,
  REJECTED_FIELD_NAMES,
  type AnalyzeResponse,
  type IndicatorId,
} from "./generated/schema";

type ContractGolden = {
  frozenAt: string;
  why: string;
  indicatorIds: string[];
  levelIds: string[];
  request: { keys: string[]; fields: string[] };
  response: { keys: string[] };
};

const goldenUrl = new URL("./contract.golden.json", import.meta.url);
const goldenPath = goldenUrl.protocol === "file:" ? goldenUrl : "src/contract.golden.json";
const golden = JSON.parse(readFileSync(goldenPath, "utf8")) as ContractGolden;

export function compareContractLists(actual: readonly string[], expected: readonly string[], label: string): string | undefined {
  if (actual.length !== expected.length) {
    return `${label} differs: actual ${JSON.stringify(actual)} vs golden ${JSON.stringify(expected)}. The golden is an approval lock; update it deliberately.`;
  }
  const mismatch = actual.findIndex((value, index) => value !== expected[index]);
  if (mismatch !== -1) {
    return `${label} offending field ${JSON.stringify(actual[mismatch] ?? expected[mismatch])}: actual ${JSON.stringify(actual)} vs golden ${JSON.stringify(expected)}. The golden is an approval lock; update it deliberately.`;
  }
  return undefined;
}

function assertContractList(actual: readonly string[], expected: readonly string[], label: string): void {
  expect(compareContractLists(actual, expected, label)).toBeUndefined();
}

export function isLevelId(value: unknown): value is AnalyzeResponse["nivel"] {
  return typeof value === "string" && golden.levelIds.includes(value);
}

function buildAnalyzePayload(values: Record<IndicatorId, number>): { datos: Record<IndicatorId, number> } {
  return { datos: values };
}

describe("frozen frontend contract", () => {
  it("locks indicator order and level ids to the golden", () => {
    assertContractList(INDICATOR_IDS, golden.indicatorIds, "indicator ids");
    assertContractList(LEVELS.map(({ id }) => id), golden.levelIds, "level ids");
  });

  it("keeps rejected names mapped to canonical golden fields", () => {
    for (const canonical of Object.values(REJECTED_FIELD_NAMES)) {
      expect(golden.indicatorIds).toContain(canonical);
    }
  });

  it("locks the request envelope used by analyze", () => {
    const values = Object.fromEntries(golden.indicatorIds.map((field) => [field, 0.5])) as Record<IndicatorId, number>;
    const payload = buildAnalyzePayload(values);
    expect(Object.keys(payload)).toEqual(golden.request.keys);
    expect(Object.keys(payload.datos)).toEqual(golden.request.fields);
  });

  it("locks the response envelope and rejects unknown levels at runtime", () => {
    const response = { nivel: "medio" } satisfies AnalyzeResponse;
    expect(Object.keys(response)).toEqual(golden.response.keys);
    expect(isLevelId(response.nivel)).toBe(true);
    expect(isLevelId("unknown-level")).toBe(false);
  });

  it("reports the offending field when a contract list is renamed", () => {
    const renamed = [...golden.indicatorIds];
    renamed[0] = "renamed_indicator";
    const message = compareContractLists(renamed, golden.indicatorIds, "indicator ids");
    expect(message).toContain('offending field "renamed_indicator"');
    expect(message).toContain(JSON.stringify(renamed));
    expect(message).toContain(JSON.stringify(golden.indicatorIds));
    expect(message).toContain("approval lock");
  });

  it("sends analyze values inside the locked request envelope", async () => {
    const values = Object.fromEntries(golden.indicatorIds.map((field) => [field, 0.5])) as Record<IndicatorId, number>;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ nivel: "medio" }), { status: 200 }),
    );
    await analyze(values);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ body: JSON.stringify({ datos: values }) }),
    );
    fetchMock.mockRestore();
  });
});
