from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return cleaned


def load_indicator_names(data_dir: Path) -> list[str]:
    indicators_path = data_dir / "estructura_indicadores.csv"
    df = pd.read_csv(indicators_path)
    return [_slugify(name) for name in df["Indicador"].tolist()]


def load_niveles(data_dir: Path) -> list[str]:
    niveles_path = data_dir / "niveles_clasificacion.csv"
    df = pd.read_csv(niveles_path)
    return [str(value).strip().lower() for value in df["Nivel"].tolist()]


def _feature_modifiers() -> dict[str, float]:
    return {
        "racismo_sistemico": 0.7,
        "alta_excitacion": 0.6,
        "angustia_profunda": 0.9,
    }


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


class AnalizadorAburrimiento:
    def __init__(self, data_dir: Path | None = None, seed: int = 42) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or (base_dir / "data")
        self.feature_names = load_indicator_names(self.data_dir)
        self.niveles = load_niveles(self.data_dir)
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

    def generar_datos(self, n_muestras: int) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        modifiers = _feature_modifiers()
        feature_rows: list[dict[str, float]] = []
        labels: list[str] = []

        for _ in range(n_muestras):
            nivel = rng.choice(self.niveles)
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

    def entrenar(self, features: pd.DataFrame, labels: pd.Series) -> None:
        encoded = self.label_encoder.fit_transform(labels)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, encoded)

    def predecir(self, features: pd.DataFrame) -> list[str]:
        scaled = self.scaler.transform(features)
        encoded = self.model.predict(scaled)
        return self.label_encoder.inverse_transform(encoded).tolist()

    def construir_ejemplo(self, nivel: str = "alto") -> pd.DataFrame:
        nivel = nivel.lower()
        base_lookup = {"bajo": 0.3, "medio": 0.5, "alto": 0.8}
        base = base_lookup.get(nivel, 0.7)
        modifiers = _feature_modifiers()
        row = {
            feature: float(np.clip(base * modifiers.get(feature, 1.0), 0, 1))
            for feature in self.feature_names
        }
        return pd.DataFrame([row])

    def features_from_payload(self, payload: dict[str, float]) -> pd.DataFrame:
        row = {feature: float(payload[feature]) for feature in self.feature_names}
        return pd.DataFrame([row])
