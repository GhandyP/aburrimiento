"""SQLite storage for real observations.

This store captures only the 14 numeric indicator values, the model's predicted
level, an optionally supplied observed level, and a UTC timestamp. It deliberately
stores no identifiers, names, free text, IP addresses, or device information.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import Schema, load_schema


class StoreError(Exception):
    """Raised when an observation cannot be stored."""


@dataclass(frozen=True, slots=True)
class SampleStats:
    total: int
    by_predicted_level: Mapping[str, int]
    by_observed_level: Mapping[str, int]


class SampleStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.schema: Schema = load_schema()
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        indicators = ", ".join(
            f'"{identifier}" REAL NOT NULL' for identifier in self.schema.indicator_ids
        )
        levels = ", ".join(f"'{level}'" for level in self.schema.level_ids)
        self._connection.execute(
            f"""CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                {indicators},
                predicted_level TEXT NOT NULL CHECK (predicted_level IN ({levels})),
                observed_level TEXT CHECK (observed_level IS NULL OR observed_level IN ({levels}))
            )"""
        )
        self._connection.commit()

    def _connection_or_error(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StoreError("SampleStore is not initialized")
        return self._connection

    def add(
        self,
        values: Mapping[str, float],
        predicted_level: str,
        observed_level: str | None = None,
    ) -> int:
        connection = self._connection_or_error()
        expected = set(self.schema.indicator_ids)
        actual = set(values)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            detail = ", ".join(sorted(missing or extra))
            raise StoreError(f"indicator keys must match schema; offending key: {detail}")
        for key in self.schema.indicator_ids:
            value = values[key]
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise StoreError(f"invalid value for {key}: {value!r}") from exc
            if not numeric == numeric or not (
                self.schema.min_value <= numeric <= self.schema.max_value
            ):
                raise StoreError(f"value for {key} is outside range: {value!r}")
        if predicted_level not in self.schema.level_ids:
            raise StoreError(f"unknown predicted level: {predicted_level}")
        if observed_level is not None and observed_level not in self.schema.level_ids:
            raise StoreError(f"unknown observed level: {observed_level}")

        columns = ("created_at", *self.schema.indicator_ids, "predicted_level", "observed_level")
        placeholders = ", ".join("?" for _ in columns)
        values_to_insert: tuple[Any, ...] = (
            datetime.now(UTC).isoformat(),
            *tuple(float(values[key]) for key in self.schema.indicator_ids),
            predicted_level,
            observed_level,
        )
        cursor = connection.execute(
            f"INSERT INTO samples ({', '.join(columns)}) VALUES ({placeholders})",
            values_to_insert,
        )
        connection.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def stats(self) -> SampleStats:
        connection = self._connection_or_error()
        total_result = connection.execute("SELECT COUNT(*) FROM samples").fetchone()
        assert total_result is not None
        total = int(total_result[0])
        predicted = dict.fromkeys(self.schema.level_ids, 0)
        observed = dict.fromkeys(self.schema.level_ids, 0)
        for level, count in connection.execute(
            "SELECT predicted_level, COUNT(*) FROM samples GROUP BY predicted_level"
        ):
            predicted[str(level)] = int(count)
        for level, count in connection.execute(
            "SELECT observed_level, COUNT(*) FROM samples "
            "WHERE observed_level IS NOT NULL GROUP BY observed_level"
        ):
            observed[str(level)] = int(count)
        return SampleStats(total, predicted, observed)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
