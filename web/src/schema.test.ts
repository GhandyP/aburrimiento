import { describe, expect, it } from "vitest";
import {
  GROUPS,
  INDICATOR_IDS,
  INDICATORS,
  LEVELS,
  VALUE_RANGE,
} from "./generated/schema";

describe("generated schema", () => {
  it("contains 14 unique indicator ids", () => {
    expect(INDICATOR_IDS).toHaveLength(14);
    expect(new Set(INDICATOR_IDS).size).toBe(INDICATOR_IDS.length);
  });

  it("keeps indicator ids and definitions in sync", () => {
    expect(new Set(INDICATORS.map((indicator) => indicator.id))).toEqual(
      new Set(INDICATOR_IDS),
    );
  });

  it("references existing groups", () => {
    const groupIds = new Set(GROUPS.map((group) => group.id));
    expect(INDICATORS.every((indicator) => groupIds.has(indicator.groupId))).toBe(
      true,
    );
  });

  it("defines the three expected levels", () => {
    expect(LEVELS).toHaveLength(3);
    expect(LEVELS.map((level) => level.id)).toEqual(["bajo", "medio", "alto"]);
  });

  it("uses the normalized value range", () => {
    expect(VALUE_RANGE).toEqual({ min: 0, max: 1 });
  });
});
