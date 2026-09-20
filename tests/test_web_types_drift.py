from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from aburrimiento.schema import load_schema

GENERATED = Path(__file__).parents[1] / "web" / "src" / "generated" / "schema.ts"
GENERATOR = Path(__file__).parents[1] / "tools" / "gen_web_types.py"


def generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_web_types", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_types_match_canonical_schema() -> None:
    generated = GENERATED.read_text(encoding="utf-8")
    expected = generator_module().render_schema(load_schema())
    assert generated == expected, "Generated web schema is stale; run make gen-types"


def test_generated_types_contain_all_ids() -> None:
    generated = GENERATED.read_text(encoding="utf-8")
    schema = load_schema()
    for identifier in (*schema.indicator_ids, *schema.level_ids):
        assert f'"{identifier}"' in generated
