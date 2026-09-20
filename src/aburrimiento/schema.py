"""Canonical runtime schema loader.

This is the only module allowed to read ``assets/schema.json``. Runtime
contracts and generated TypeScript derive from that canonical schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class SchemaError(Exception):
    """Raised when the canonical schema is malformed or violates an invariant."""


@dataclass(frozen=True, slots=True)
class Labels:
    es: str
    en: str | None


@dataclass(frozen=True, slots=True)
class Group:
    id: str
    order: int
    labels: Labels


@dataclass(frozen=True, slots=True)
class Indicator:
    id: str
    group_id: str
    order: int
    labels: Labels
    theoretical_basis: str


@dataclass(frozen=True, slots=True)
class Level:
    id: str
    order: int
    labels: Labels
    color: str
    interpretation: str
    recommended_action: str
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True)
class Schema:
    version: str
    min_value: float
    max_value: float
    groups: tuple[Group, ...]
    indicators: tuple[Indicator, ...]
    levels: tuple[Level, ...]
    rejected_field_names: Mapping[str, str]

    @property
    def indicator_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in sorted(self.indicators, key=lambda item: item.order))

    @property
    def level_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in sorted(self.levels, key=lambda item: item.order))

    def indicator_ids_for_group(self, group_id: str) -> tuple[str, ...]:
        return tuple(
            item.id
            for item in sorted(self.indicators, key=lambda item: item.order)
            if item.group_id == group_id
        )

    def level(self, level_id: str) -> Level:
        for item in self.levels:
            if item.id == level_id:
                return item
        raise SchemaError(f"Unknown level id: {level_id}")

    def level_for_value(self, value: float) -> Level | None:
        for item in sorted(self.levels, key=lambda item: item.order):
            if item.min_value <= value < item.max_value or (
                item.max_value == self.max_value and value == item.max_value
            ):
                return item
        return None


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "assets" / "schema.json"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be a number")
    return float(value)


def _labels(value: object, name: str) -> Labels:
    labels = _mapping(value, name)
    es = _text(labels.get("es"), f"{name}.es")
    en_value = labels.get("en")
    if en_value is not None and (not isinstance(en_value, str) or not en_value.strip()):
        raise SchemaError(f"{name}.en must be a non-empty string when present")
    return Labels(es=es, en=en_value)


def _ordered(values: list[object], name: str) -> list[tuple[Mapping[str, object], int]]:
    orders: list[int] = []
    mapped: list[tuple[Mapping[str, object], int]] = []
    for index, value in enumerate(values):
        item = _mapping(value, f"{name}[{index}]")
        raw_order = item.get("order")
        if isinstance(raw_order, bool) or not isinstance(raw_order, int):
            raise SchemaError(f"{name}[{index}].order must be an integer")
        if raw_order in orders:
            raise SchemaError(f"{name} has duplicate order {raw_order}")
        orders.append(raw_order)
        mapped.append((item, raw_order))
    expected = list(range(1, len(values) + 1))
    if sorted(orders) != expected:
        raise SchemaError(f"{name}.order must be contiguous starting at 1; got {orders}")
    return mapped


def parse_schema(payload: object) -> Schema:
    root = _mapping(payload, "schema")
    version = _text(root.get("schemaVersion"), "schemaVersion")
    value_range = _mapping(root.get("valueRange"), "valueRange")
    min_value = _number(value_range.get("min"), "valueRange.min")
    max_value = _number(value_range.get("max"), "valueRange.max")
    if min_value >= max_value:
        raise SchemaError("valueRange.min must be less than valueRange.max")

    raw_groups = root.get("groups")
    raw_indicators = root.get("indicators")
    raw_levels = root.get("levels")
    if (
        not isinstance(raw_groups, list)
        or not isinstance(raw_indicators, list)
        or not isinstance(raw_levels, list)
    ):
        raise SchemaError("groups, indicators, and levels must be arrays")
    if not raw_groups or not raw_indicators or not raw_levels:
        raise SchemaError("groups, indicators, and levels must not be empty")

    groups: list[Group] = []
    group_ids: set[str] = set()
    for item, order in _ordered(raw_groups, "groups"):
        group_id = _text(item.get("id"), "group.id")
        if group_id in group_ids:
            raise SchemaError(f"duplicate group id: {group_id}")
        group_ids.add(group_id)
        groups.append(
            Group(group_id, order, _labels(item.get("labels"), f"group {group_id}.labels"))
        )

    indicators: list[Indicator] = []
    indicator_ids: set[str] = set()
    for item, order in _ordered(raw_indicators, "indicators"):
        indicator_id = _text(item.get("id"), "indicator.id")
        if indicator_id in indicator_ids:
            raise SchemaError(f"duplicate indicator id: {indicator_id}")
        indicator_ids.add(indicator_id)
        group_id = _text(item.get("groupId"), f"indicator {indicator_id}.groupId")
        if group_id not in group_ids:
            raise SchemaError(f"unknown groupId {group_id} for indicator {indicator_id}")
        theoretical_basis = _text(
            item.get("theoreticalBasis"), f"indicator {indicator_id}.theoreticalBasis"
        )
        indicators.append(
            Indicator(
                indicator_id,
                group_id,
                order,
                _labels(item.get("labels"), f"indicator {indicator_id}.labels"),
                theoretical_basis,
            )
        )

    for group in groups:
        if not any(indicator.group_id == group.id for indicator in indicators):
            raise SchemaError(f"group {group.id} has no indicators")

    levels: list[Level] = []
    level_ids: set[str] = set()
    for item, order in _ordered(raw_levels, "levels"):
        level_id = _text(item.get("id"), "level.id")
        if level_id in level_ids:
            raise SchemaError(f"duplicate level id: {level_id}")
        level_ids.add(level_id)
        level_range = _mapping(item.get("range"), f"level {level_id}.range")
        level_min = _number(level_range.get("min"), f"level {level_id}.range.min")
        level_max = _number(level_range.get("max"), f"level {level_id}.range.max")
        if level_min >= level_max:
            raise SchemaError(f"level {level_id} has min >= max")
        levels.append(
            Level(
                level_id,
                order,
                _labels(item.get("labels"), f"level {level_id}.labels"),
                _text(item.get("color"), f"level {level_id}.color"),
                _text(item.get("interpretation"), f"level {level_id}.interpretation"),
                _text(item.get("recommendedAction"), f"level {level_id}.recommendedAction"),
                level_min,
                level_max,
            )
        )

    previous_max = min_value
    for level in sorted(levels, key=lambda item: item.order):
        if level.min_value != previous_max:
            raise SchemaError(f"level {level.id} range is not contiguous")
        if level.min_value < min_value or level.max_value > max_value:
            raise SchemaError(f"level {level.id} range is outside valueRange")
        previous_max = level.max_value
    if previous_max != max_value:
        raise SchemaError("levels do not cover valueRange")

    rejected = _mapping(root.get("rejectedFieldNames"), "rejectedFieldNames")
    rejected_names = _mapping(rejected.get("names"), "rejectedFieldNames.names")
    rejected_field_names: dict[str, str] = {}
    for name, replacement in rejected_names.items():
        if not isinstance(name, str) or not isinstance(replacement, str):
            raise SchemaError("rejectedFieldNames.names must map strings to strings")
        if name in indicator_ids:
            raise SchemaError(f"rejected name shadows declared id: {name}")
        if replacement not in indicator_ids:
            raise SchemaError(f"rejected name {name} resolves to unknown indicator {replacement}")
        rejected_field_names[name] = replacement

    return Schema(
        version,
        min_value,
        max_value,
        tuple(groups),
        tuple(indicators),
        tuple(levels),
        rejected_field_names,
    )


def load_schema(path: Path | None = None) -> Schema:
    schema_path = DEFAULT_SCHEMA_PATH if path is None else path
    try:
        with schema_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Canonical schema file does not exist: {schema_path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Malformed JSON in schema file {schema_path}: {exc.msg}") from exc
    return parse_schema(payload)
