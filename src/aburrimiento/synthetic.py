from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_MODIFIERS: dict[str, float] = {
    "racismo_sistemico": 0.7,
    "alta_excitacion": 0.6,
    "angustia_profunda": 0.9,
}


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


class SyntheticDataGenerator:
    def __init__(self, feature_names: list[str], levels: list[str], seed: int = 42) -> None:
        self.feature_names = feature_names
        self.levels = levels
        self.seed = seed

    def generate(self, n: int) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        modifiers = FEATURE_MODIFIERS
        feature_rows: list[dict[str, float]] = []
        labels: list[str] = []

        for _ in range(n):
            nivel = rng.choice(self.levels)
            if nivel == "bajo":
                base, spread = 0.3, 0.2
            elif nivel == "medio":
                base, spread = 0.5, 0.15
            else:
                base, spread = 0.8, 0.15

            row: dict[str, float] = {}
            for feature in self.feature_names:
                factor = modifiers.get(feature, 1.0)
                value = rng.normal(base * factor, spread)
                row[feature] = float(np.clip(value, 0, 1))

            feature_rows.append(row)
            labels.append(nivel)

        features_df = pd.DataFrame(feature_rows)
        labels_series = pd.Series(labels, name="etiqueta")
        return DatasetBundle(features=features_df, labels=labels_series)
