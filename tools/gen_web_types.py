"""Generate TypeScript schema types from the canonical runtime schema."""

from __future__ import annotations

import json
from pathlib import Path

from aburrimiento.schema import Schema, load_schema

OUTPUT_PATH = Path(__file__).parents[1] / "web" / "src" / "generated" / "schema.ts"
HEADER = """// Generated from assets/schema.json. Do not hand-edit this file.
// Regenerate with: make gen-types

"""


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def render_schema(schema: Schema | None = None) -> str:
    """Return deterministic TypeScript generated from ``schema``."""
    resolved = load_schema() if schema is None else schema
    groups = sorted(resolved.groups, key=lambda item: item.order)
    indicators = sorted(resolved.indicators, key=lambda item: item.order)
    levels = sorted(resolved.levels, key=lambda item: item.order)

    lines = [HEADER.rstrip("\n")]
    lines.extend(
        [
            f"export const SCHEMA_VERSION = {_json(resolved.version)} as const;",
            "export const VALUE_RANGE = {",
            f"  min: {_number(resolved.min_value)},",
            f"  max: {_number(resolved.max_value)},",
            "} as const;",
            "",
            "export interface Labels {",
            "  es: string;",
            "  en?: string;",
            "}",
            "",
            "export interface Group {",
            "  id: string;",
            "  order: number;",
            "  labels: Labels;",
            "}",
            "",
            "export interface Indicator {",
            "  id: string;",
            "  order: number;",
            "  labels: Labels;",
            "  groupId: string;",
            "  theoreticalBasis: string;",
            "}",
            "",
            "export interface Level {",
            "  id: string;",
            "  order: number;",
            "  labels: Labels;",
            "  color: string;",
            "  interpretation: string;",
            "  recommendedAction: string;",
            "  range: { min: number; max: number };",
            "}",
            "",
            "export const INDICATOR_IDS = [",
        ]
    )
    lines.extend(f"  {_json(indicator.id)}," for indicator in indicators)
    lines.extend(["] as const;", "export type IndicatorId = (typeof INDICATOR_IDS)[number];", ""])

    lines.extend(["export const GROUPS: readonly Group[] = ["])
    for group in groups:
        labels = {"es": group.labels.es}
        if group.labels.en is not None:
            labels["en"] = group.labels.en
        lines.append(
            f"  {{ id: {_json(group.id)}, order: {group.order}, labels: {_json(labels)} }},"
        )
    lines.extend(["] as const;", ""])

    lines.extend(["export const INDICATORS: readonly Indicator[] = ["])
    for indicator in indicators:
        labels = {"es": indicator.labels.es}
        if indicator.labels.en is not None:
            labels["en"] = indicator.labels.en
        lines.append(
            "  { "
            f"id: {_json(indicator.id)}, groupId: {_json(indicator.group_id)}, "
            f"order: {indicator.order}, labels: {_json(labels)}, "
            f"theoreticalBasis: {_json(indicator.theoretical_basis)} }},"
        )
    lines.extend(["] as const;", ""])

    lines.extend(["export const LEVELS: readonly Level[] = ["])
    for level in levels:
        lines.append(
            "  { "
            f"id: {_json(level.id)}, order: {level.order}, "
            f"labels: {_json({'es': level.labels.es})}, "
            f"color: {_json(level.color)}, interpretation: {_json(level.interpretation)}, "
            f"recommendedAction: {_json(level.recommended_action)}, "
            f"range: {{ min: {_number(level.min_value)}, max: {_number(level.max_value)} }} }},"
        )
    lines.extend(["] as const;", 'export type LevelId = (typeof LEVELS)[number]["id"];', ""])

    lines.extend(
        [
            "export const REJECTED_FIELD_NAMES = "
            f"{_json(dict(resolved.rejected_field_names))} as const;",
            "",
            "export interface AnalyzeRequest {",
            "  datos: Record<IndicatorId, number>;",
            "}",
            "",
            "export interface AnalyzeResponse {",
            "  nivel: LevelId;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_schema(), encoding="utf-8")


if __name__ == "__main__":
    main()
