import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from aburrimiento.schema import SchemaError, load_schema, parse_schema

SCHEMA_PATH = Path(__file__).parents[1] / "assets" / "schema.json"
EXPECTED_FEATURE_ORDER = [
    "reflejo_sistemas_culturales",
    "productividad_capitalista",
    "alienacion_neoliberal",
    "racismo_sistemico",
    "malestar_generalizado",
    "carencia_de_sentido",
    "restriccion_de_libertad",
    "frustracion_de_agencia",
    "desenganche",
    "alta_excitacion",
    "inatencion",
    "percepcion_tiempo_lenta",
    "estrategias_bloqueadas",
    "angustia_profunda",
]


def schema_payload() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def test_canonical_schema_invariants() -> None:
    payload = schema_payload()
    assert isinstance(payload["schemaVersion"], str) and payload["schemaVersion"]
    assert payload["valueRange"] == {"min": 0.0, "max": 1.0}
    indicators = payload["indicators"]
    assert len(indicators) == 14
    assert len({item["id"] for item in indicators}) == 14
    assert all(item["id"] and item["id"].islower() and " " not in item["id"] for item in indicators)
    assert [item["order"] for item in indicators] == list(range(1, 15))
    assert [item["order"] for item in indicators] == sorted(item["order"] for item in indicators)
    assert {item["groupId"] for item in indicators} == {
        "estructuras_sistemicas",
        "manifestaciones_grupales",
        "dimensiones_medicion",
    }
    assert [
        sum(item["groupId"] == group for item in indicators)
        for group in ("estructuras_sistemicas", "manifestaciones_grupales", "dimensiones_medicion")
    ] == [4, 4, 6]
    groups = payload["groups"]
    assert len({item["id"] for item in groups}) == len(groups)
    assert len({item["order"] for item in groups}) == len(groups)
    assert all(item["labels"].get("es") for item in groups)
    assert all(item["labels"].get("es") for item in indicators)
    assert all(item["labels"].get("es") for item in payload["levels"])
    assert all(
        "en" not in item["labels"] or item["labels"]["en"]
        for item in indicators + groups + payload["levels"]
    )
    assert [item["id"] for item in payload["levels"]] == ["bajo", "medio", "alto"]
    assert all(
        item["color"] and item["interpretation"] and item["recommendedAction"]
        for item in payload["levels"]
    )
    assert [(item["range"]["min"], item["range"]["max"]) for item in payload["levels"]] == [
        (0.0, 0.4),
        (0.4, 0.7),
        (0.7, 1.0),
    ]
    assert all(
        name not in {item["id"] for item in indicators}
        for name in payload["rejectedFieldNames"]["names"]
    )
    assert all(
        replacement in {item["id"] for item in indicators}
        for replacement in payload["rejectedFieldNames"]["names"].values()
    )
    assert [
        item["id"] for item in sorted(indicators, key=lambda item: item["order"])
    ] == EXPECTED_FEATURE_ORDER


def test_legacy_names_resolve_to_canonical_ids() -> None:
    rejected = schema_payload()["rejectedFieldNames"]["names"]
    assert rejected == {
        "carencia_sentido": "carencia_de_sentido",
        "restriccion_libertad": "restriccion_de_libertad",
        "frustracion_agencia": "frustracion_de_agencia",
    }


def test_schema_to_payload_round_trips_canonical_json() -> None:
    assert load_schema(SCHEMA_PATH).to_payload() == schema_payload()


def test_schema_loader_exposes_ordered_contract() -> None:
    schema = load_schema(SCHEMA_PATH)
    assert schema.indicator_ids == tuple(EXPECTED_FEATURE_ORDER)
    assert schema.level_ids == ("bajo", "medio", "alto")
    assert schema.indicator_ids_for_group("estructuras_sistemicas") == tuple(
        EXPECTED_FEATURE_ORDER[:4]
    )
    level = schema.level_for_value(0.4)
    assert level is not None and level.id == "medio"
    level = schema.level_for_value(1.0)
    assert level is not None and level.id == "alto"


def test_load_schema_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_schema(Path("does-not-exist/schema.json"))


def malformed_payload() -> dict[str, Any]:
    return copy.deepcopy(schema_payload())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["indicators"][1].__setitem__("id", p["indicators"][0]["id"]),
            "duplicate indicator id",
        ),
        (lambda p: p["indicators"][0].__setitem__("groupId", "missing-group"), "missing-group"),
        (lambda p: p["indicators"][1].__setitem__("order", 99), "indicators.order"),
        (
            lambda p: p.__setitem__("indicators", p["indicators"][:4]),
            "group manifestaciones_grupales has no indicators",
        ),
        (
            lambda p: p["indicators"][0]["labels"].pop("es"),
            "indicator reflejo_sistemas_culturales.labels.es",
        ),
        (lambda p: p["valueRange"].__setitem__("max", 0.0), "valueRange.min"),
        (
            lambda p: p["rejectedFieldNames"]["names"].__setitem__("old_name", "missing-indicator"),
            "missing-indicator",
        ),
    ],
)
def test_parse_schema_rejects_invalid_payloads(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    payload = malformed_payload()
    mutate(payload)
    with pytest.raises(SchemaError, match=message):
        parse_schema(payload)


def test_load_schema_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaError, match="Malformed JSON"):
        load_schema(path)
