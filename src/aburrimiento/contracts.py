"""Pydantic HTTP contracts derived from the canonical schema."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .schema import Schema, load_schema


def build_analysis_data_model(schema: Schema | None = None) -> type[BaseModel]:
    """Build the strict indicator payload model from the canonical schema."""
    resolved = load_schema() if schema is None else schema
    fields: dict[str, tuple[Any, Any]] = {
        indicator.id: (
            float,
            Field(
                ge=resolved.min_value,
                le=resolved.max_value,
                description=f"{indicator.labels.es} — {indicator.theoretical_basis}",
            ),
        )
        for indicator in sorted(resolved.indicators, key=lambda item: item.order)
    }

    @model_validator(mode="before")
    def reject_legacy_names(cls: type[BaseModel], value: object) -> object:
        if isinstance(value, dict):
            for rejected, replacement in resolved.rejected_field_names.items():
                if rejected in value:
                    raise ValueError(f"{rejected} is a rejected field name; use {replacement}")
        return value

    # Pydantic's dynamic-model overload cannot infer the decorated validator.
    validators = cast(dict[str, Callable[..., Any]], {"reject_legacy_names": reject_legacy_names})
    return cast(
        type[BaseModel],
        create_model(
            "AnalysisData",
            __config__=ConfigDict(extra="forbid"),
            __validators__=validators,
            **cast(dict[str, Any], fields),
        ),
    )


def build_analyze_request_model(schema: Schema | None = None) -> type[BaseModel]:
    """Build the analyze request envelope from the canonical schema."""
    data_model = build_analysis_data_model(schema)
    return create_model("AnalyzeRequest", datos=(data_model, ...))


def build_analyze_response_model(schema: Schema | None = None) -> type[BaseModel]:
    """Build the analyze response with only declared level ids."""
    resolved = load_schema() if schema is None else schema
    level_type = cast(Any, Literal)[tuple(resolved.level_ids)]
    return create_model("AnalyzeResponse", nivel=(level_type, ...))
