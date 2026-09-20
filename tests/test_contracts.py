from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from aburrimiento.contracts import (
    build_analysis_data_model,
    build_analyze_response_model,
)
from aburrimiento.schema import load_schema


def test_data_model_matches_schema_in_order() -> None:
    schema = load_schema()
    model = build_analysis_data_model(schema)
    assert len(model.model_fields) == 14
    assert tuple(model.model_fields) == schema.indicator_ids
    json_schema = model.model_json_schema()
    assert json_schema["additionalProperties"] is False
    for name in schema.indicator_ids:
        assert json_schema["properties"][name]["minimum"] == 0
        assert json_schema["properties"][name]["maximum"] == 1


def test_data_model_accepts_full_valid_payload() -> None:
    model = build_analysis_data_model()
    values = {name: index / 13 for index, name in enumerate(load_schema().indicator_ids)}
    assert model.model_validate(values).model_dump() == values


@pytest.mark.parametrize(
    ("change", "expected_field"),
    [
        (lambda values: values.pop(next(iter(values))), None),
        (lambda values: values.__setitem__(next(iter(values)), 1.5), None),
        (lambda values: values.__setitem__(next(iter(values)), -0.1), None),
        (lambda values: values.__setitem__("unknown", 0.5), "unknown"),
    ],
)
def test_data_model_rejects_invalid_payload(
    change: Callable[[dict[str, float]], object], expected_field: str | None
) -> None:
    model = build_analysis_data_model()
    values = dict.fromkeys(load_schema().indicator_ids, 0.5)
    first_field = next(iter(values))
    change(values)
    with pytest.raises(ValidationError) as error:
        model.model_validate(values)
    assert (expected_field or first_field) in str(error.value)


def test_legacy_names_explain_canonical_replacement() -> None:
    model = build_analysis_data_model()
    values = dict.fromkeys(load_schema().indicator_ids, 0.5)
    for rejected, replacement in load_schema().rejected_field_names.items():
        candidate = {**values, rejected: 0.5}
        with pytest.raises(ValidationError, match=f"{rejected}.*{replacement}"):
            model.model_validate(candidate)


def test_response_accepts_only_declared_levels() -> None:
    model = build_analyze_response_model()
    for level in load_schema().level_ids:
        assert model.model_validate({"nivel": level}).model_dump()["nivel"] == level
    with pytest.raises(ValidationError):
        model.model_validate({"nivel": "extremo"})
