from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .synthetic import FEATURE_MODIFIERS, DatasetBundle, SyntheticDataGenerator

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "assets"


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return cleaned


def _require_data_dir(data_dir: Path) -> Path:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    missing = [
        name
        for name in ("estructura_indicadores.csv", "niveles_clasificacion.csv")
        if not (data_dir / name).is_file()
    ]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Data directory {data_dir} is missing required CSVs: {names}")
    return data_dir


def load_indicator_names(data_dir: Path) -> list[str]:
    indicators_path = _require_data_dir(data_dir) / "estructura_indicadores.csv"
    df = pd.read_csv(indicators_path)
    return [_slugify(name) for name in df["Indicador"].tolist()]


def load_levels(data_dir: Path) -> list[str]:
    niveles_path = _require_data_dir(data_dir) / "niveles_clasificacion.csv"
    df = pd.read_csv(niveles_path)
    return [str(value).strip().lower() for value in df["Nivel"].tolist()]


class BoredomModel:
    """Pipeline compatibility facade; canonical schema owns data resolution from T6 onward."""

    def __init__(self, data_dir: Path | None = None, seed: int = 42) -> None:
        resolved_data_dir = DEFAULT_DATA_DIR if data_dir is None else data_dir
        self.data_dir = _require_data_dir(resolved_data_dir)
        self.feature_names = load_indicator_names(self.data_dir)
        self.levels = load_levels(self.data_dir)
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
