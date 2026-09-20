from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import joblib
import pytest

from aburrimiento.artifacts import ArtifactError, TrainedModel, schema_sha256
from aburrimiento.model import BoredomModel


@pytest.fixture
def trained_model() -> BoredomModel:
    model = BoredomModel()
    dataset = model.generate(300)
    model.train(dataset.features, dataset.labels)
    return model


def test_save_load_predictions_match_for_all_levels(
    trained_model: BoredomModel, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "model.joblib"
    trained_model.save_artifact(path)
    loaded = BoredomModel.from_artifact(path)

    for level in ("bajo", "medio", "alto"):
        assert loaded.predict(loaded.example_for(level)) == trained_model.predict(
            trained_model.example_for(level)
        )


def test_artifact_records_real_schema_digest(trained_model: BoredomModel, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    trained_model.save_artifact(path)
    artifact = TrainedModel.load(path)

    assert artifact.metadata.schema_sha256 == schema_sha256(Path("assets/schema.json"))


def test_schema_mismatch_requires_retraining(trained_model: BoredomModel, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    trained_model.save_artifact(path)
    artifact = TrainedModel.load(path)
    joblib.dump(replace(artifact, metadata=replace(artifact.metadata, schema_sha256="wrong")), path)

    with pytest.raises(ArtifactError, match="make train"):
        TrainedModel.load(path)


def test_missing_artifact_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        TrainedModel.load(tmp_path / "missing.joblib")


def test_round_trip_stays_in_tmp_path(trained_model: BoredomModel, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    repository_artifact = Path("models/model.joblib")
    existed_before = repository_artifact.exists()
    trained_model.save_artifact(path)
    assert path.exists()
    assert repository_artifact.exists() is existed_before


def test_metadata_is_populated_and_iso8601(trained_model: BoredomModel, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    trained_model.save_artifact(path)
    metadata = TrainedModel.load(path).metadata

    assert metadata.schema_version == "1.0.0"
    assert metadata.schema_sha256 == schema_sha256()
    assert metadata.n_samples == 0
    assert metadata.seed == 42
    assert metadata.sklearn_version
    assert metadata.accuracy is None
    datetime.fromisoformat(metadata.trained_at)
