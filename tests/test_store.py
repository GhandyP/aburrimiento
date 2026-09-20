import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from aburrimiento.schema import load_schema
from aburrimiento.store import SampleStore, StoreError


def values() -> dict[str, float]:
    schema = load_schema()
    return dict.fromkeys(schema.indicator_ids, (schema.min_value + schema.max_value) / 2)


def test_initialize_is_idempotent_and_derives_columns(tmp_path: Path) -> None:
    store = SampleStore(tmp_path / "nested" / "samples.db")
    store.initialize()
    store.initialize()
    with sqlite3.connect(store.db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(samples)")]
    assert columns == [
        "id",
        "created_at",
        *load_schema().indicator_ids,
        "predicted_level",
        "observed_level",
    ]
    store.close()


def test_add_and_stats_include_optional_observed_level(tmp_path: Path) -> None:
    store = SampleStore(tmp_path / "samples.db")
    store.initialize()
    row_id = store.add(values(), "medio", "alto")
    assert row_id == 1
    stats = store.stats()
    assert stats.total == 1
    assert stats.by_predicted_level["medio"] == 1
    assert stats.by_observed_level["alto"] == 1
    with sqlite3.connect(store.db_path) as connection:
        created_at, observed = connection.execute(
            "SELECT created_at, observed_level FROM samples"
        ).fetchone()
    datetime.fromisoformat(created_at)
    assert observed == "alto"
    store.close()


@pytest.mark.parametrize("bad_level", ["unknown"])
def test_add_rejects_undeclared_level(tmp_path: Path, bad_level: str) -> None:
    store = SampleStore(tmp_path / "samples.db")
    store.initialize()
    with pytest.raises(StoreError, match=r"unknown.*level"):
        store.add(values(), bad_level)
    store.close()


def test_add_rejects_unknown_key_and_out_of_range_value(tmp_path: Path) -> None:
    store = SampleStore(tmp_path / "samples.db")
    store.initialize()
    unknown = values()
    unknown["unknown"] = 0.5
    with pytest.raises(StoreError, match="unknown"):
        store.add(unknown, "medio")
    out_of_range = values()
    key = load_schema().indicator_ids[0]
    out_of_range[key] = 1.5
    with pytest.raises(StoreError, match=key):
        store.add(out_of_range, "medio")
    store.close()


def test_stores_on_separate_paths_do_not_interfere(tmp_path: Path) -> None:
    first = SampleStore(tmp_path / "first.db")
    second = SampleStore(tmp_path / "second.db")
    first.initialize()
    second.initialize()
    first.add(values(), "bajo")
    assert first.stats().total == 1
    assert second.stats().total == 0
    first.close()
    second.close()
