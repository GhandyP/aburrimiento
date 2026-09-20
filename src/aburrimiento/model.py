from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .schema import load_schema
from .synthetic import FEATURE_MODIFIERS, DatasetBundle, SyntheticDataGenerator


class BoredomModel:
    """Pipeline compatibility facade; canonical schema owns data resolution from T6 onward."""

    def __init__(self, schema_path: Path | None = None, seed: int = 42) -> None:
        schema = load_schema(schema_path)
        self.feature_names = list(schema.indicator_ids)
        self.levels = list(schema.level_ids)
        self.seed = seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=5,
            random_state=seed,
            class_weight="balanced",
        )

    def generate(self, n: int) -> DatasetBundle:
        generator = SyntheticDataGenerator(self.feature_names, self.levels, self.seed)
        return generator.generate(n)

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        encoded = self.label_encoder.fit_transform(labels)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, encoded)

    def predict(self, features: pd.DataFrame) -> list[str]:
        scaled = self.scaler.transform(features)
        encoded = self.model.predict(scaled)
        return cast(list[str], self.label_encoder.inverse_transform(encoded).tolist())

    def example_for(self, nivel: str = "alto") -> pd.DataFrame:
        nivel = nivel.lower()
        base_lookup = {"bajo": 0.3, "medio": 0.5, "alto": 0.8}
        base = base_lookup.get(nivel, 0.7)
        row = {
            feature: float(np.clip(base * FEATURE_MODIFIERS.get(feature, 1.0), 0, 1))
            for feature in self.feature_names
        }
        return pd.DataFrame([row])

    def features_from_payload(self, payload: dict[str, float]) -> pd.DataFrame:
        row = {feature: float(payload[feature]) for feature in self.feature_names}
        return pd.DataFrame([row])
