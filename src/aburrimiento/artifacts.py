"""Persistence for trained boredom models and their schema identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .schema import DEFAULT_SCHEMA_PATH, load_schema


class ArtifactError(Exception):
    """Raised when a model artifact is invalid or incompatible with this schema."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    schema_version: str
    schema_sha256: str
    trained_at: str
    n_samples: int
    seed: int
    sklearn_version: str
    accuracy: float | None
    kappa: float | None
    macro_f1: float | None


@dataclass(slots=True)
class TrainedModel:
    estimator: Any
    scaler: Any
    label_encoder: Any
    feature_names: list[str]
    levels: list[str]
    metadata: ModelMetadata

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> TrainedModel:
        if not path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {path}. Run make train first.")
        try:
            artifact = joblib.load(path)
        except Exception as exc:
            raise ArtifactError(f"Could not deserialize model artifact {path}: {exc}") from exc
        if not isinstance(artifact, cls):
            raise ArtifactError(f"Model artifact {path} has an unexpected format")
        current_digest = schema_sha256()
        if artifact.metadata.schema_sha256 != current_digest:
            raise ArtifactError(
                "Model artifact schema does not match assets/schema.json; run make train."
            )
        return artifact


def schema_sha256(path: Path | None = None) -> str:
    """Return the SHA-256 digest of the canonical schema bytes."""
    schema_path = DEFAULT_SCHEMA_PATH if path is None else path
    return hashlib.sha256(schema_path.read_bytes()).hexdigest()


def train_artifact(path: Path, n_samples: int, seed: int, report_path: Path) -> None:
    from .model import BoredomModel

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["tree"]["scaled"]
    model = BoredomModel(seed=seed)
    dataset = model.generate(n_samples)
    model.train(dataset.features, dataset.labels)
    model.save_artifact(
        path,
        new_metadata(
            n_samples=n_samples,
            seed=seed,
            accuracy=float(metrics["accuracy"]),
            kappa=float(metrics["kappa"]),
            macro_f1=float(metrics["macro_f1"]),
        ),
    )


def new_metadata(
    *,
    n_samples: int,
    seed: int,
    accuracy: float | None = None,
    kappa: float | None = None,
    macro_f1: float | None = None,
) -> ModelMetadata:
    schema = load_schema()
    import sklearn

    return ModelMetadata(
        schema_version=schema.version,
        schema_sha256=schema_sha256(),
        trained_at=datetime.now(UTC).isoformat(),
        n_samples=n_samples,
        seed=seed,
        sklearn_version=sklearn.__version__,
        accuracy=accuracy,
        kappa=kappa,
        macro_f1=macro_f1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the boredom model artifact.")
    parser.add_argument("--path", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    train_artifact(args.path, args.samples, args.seed, args.report)
    print(f"Saved model artifact to {args.path}")


if __name__ == "__main__":
    main()
