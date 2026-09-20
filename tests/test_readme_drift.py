from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from aburrimiento.schema import load_schema

README = Path(__file__).parents[1] / "README.md"
GENERATOR = Path(__file__).parents[1] / "tools" / "gen_readme_table.py"


def generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_readme_table", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_indicator_table_matches_schema() -> None:
    module = generator_module()
    readme = README.read_text(encoding="utf-8")
    expected = module.regenerate_readme(readme, load_schema())
    assert readme == expected, "README indicator table is stale; run make gen-docs"


def test_generated_indicator_table_contains_each_schema_id_once() -> None:
    module = generator_module()
    readme = README.read_text(encoding="utf-8")
    begin = readme.index(module.BEGIN_MARKER)
    end = readme.index(module.END_MARKER)
    table = readme[begin:end]
    for identifier in load_schema().indicator_ids:
        assert table.count(f"`{identifier}`") == 1, (
            f"README indicator table must contain {identifier!r} exactly once"
        )
