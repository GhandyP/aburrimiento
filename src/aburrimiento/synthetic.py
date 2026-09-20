"""Synthetic data generation for the boredom classifier.

The labels are drawn from the same class-conditional distributions the model
then learns, so any accuracy measured on this data is a measure of
self-consistency of the generator, not evidence about boredom in the world.
There is no external validation anywhere in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClassProfile:
    level_id: str
    base: float
    spread: float


DEFAULT_SEED = 42
VALUE_MIN = 0.0
VALUE_MAX = 1.0
DEFAULT_CLASS_PROFILES: tuple[ClassProfile, ...] = (
    ClassProfile("bajo", 0.3, 0.2),
    ClassProfile("medio", 0.5, 0.15),
    ClassProfile("alto", 0.8, 0.15),
)
DEFAULT_FEATURE_MODIFIERS: dict[str, float] = {
    "racismo_sistemico": 0.7,
    "alta_excitacion": 0.6,
    "angustia_profunda": 0.9,
}
# Compatibility alias retained for model.py and callers of the original API.
FEATURE_MODIFIERS = DEFAULT_FEATURE_MODIFIERS


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


class SyntheticDataGenerator:
    """Generate labels and features from explicit class profiles.

    Accuracy on generated data measures self-consistency of the generator, not
    evidence about boredom in the world.
    """

    def __init__(
        self,
        feature_names: list[str],
        levels: list[str],
        seed: int = DEFAULT_SEED,
        class_profiles: tuple[ClassProfile, ...] = DEFAULT_CLASS_PROFILES,
        modifiers: dict[str, float] | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.levels = levels
        self.seed = seed
        self.class_profiles = class_profiles
        self.modifiers = DEFAULT_FEATURE_MODIFIERS if modifiers is None else modifiers

    def generate(self, n: int) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        profile_by_level = {profile.level_id: profile for profile in self.class_profiles}
        modifiers = self.modifiers
        feature_rows: list[dict[str, float]] = []
        labels: list[str] = []

        for _ in range(n):
            nivel = rng.choice(self.levels)
            try:
                profile = profile_by_level[nivel]
            except KeyError as exc:
                raise ValueError(f"No class profile configured for level {nivel!r}") from exc

            row: dict[str, float] = {}
            for feature in self.feature_names:
                factor = modifiers.get(feature, 1.0)
                value = rng.normal(profile.base * factor, profile.spread)
                row[feature] = float(np.clip(value, VALUE_MIN, VALUE_MAX))

            feature_rows.append(row)
            labels.append(nivel)

        features_df = pd.DataFrame(feature_rows)
        labels_series = pd.Series(labels, name="etiqueta")
        return DatasetBundle(features=features_df, labels=labels_series)
