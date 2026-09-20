import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from aburrimiento.api import create_app
from aburrimiento.artifacts import new_metadata, schema_sha256
from aburrimiento.model import BoredomModel
from aburrimiento.schema import load_schema

SCHEMA_PATH = Path(__file__).parents[1] / "assets" / "schema.json"


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("api") / "model.joblib"
    model = BoredomModel(seed=load_schema().levels[0].order)
    dataset = model.generate(60)
    model.train(dataset.features, dataset.labels)
    model.save_artifact(path, new_metadata(n_samples=len(dataset.features), seed=model.seed))
    return path


@pytest.fixture(scope="module")
def client(artifact_path: Path) -> Generator[TestClient]:
    with TestClient(create_app(artifact_path=artifact_path)) as test_client:
        yield test_client


def valid_payload() -> dict[str, dict[str, float]]:
    schema = load_schema()
    midpoint = (schema.min_value + schema.max_value) / 2
    return {"datos": dict.fromkeys(schema.indicator_ids, midpoint)}


def schema_payload() -> dict[str, object]:
    return cast(dict[str, object], json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_health_reports_artifact_identity(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    metadata = cast(Any, cast(Any, client).app.state.analizador).metadata
    assert body["status"] == "ok"
    assert body["schema_version"] == metadata.schema_version
    assert body["schema_sha256"] == schema_sha256(SCHEMA_PATH)
    assert body["trained_at"] == metadata.trained_at
    assert body["n_samples"] == metadata.n_samples
    assert body["seed"] == metadata.seed
    assert body["sklearn_version"] == metadata.sklearn_version
    assert body["metrics"] == {
        "accuracy": metadata.accuracy,
        "kappa": metadata.kappa,
        "macro_f1": metadata.macro_f1,
    }


def test_schema_endpoint_returns_canonical_payload(client: TestClient) -> None:
    response = client.get("/schema")
    assert response.status_code == 200
    assert response.json() == schema_payload()


def test_analyze_accepts_valid_payload(client: TestClient) -> None:
    response = client.post("/analyze", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["nivel"] in load_schema().level_ids


def test_analyze_is_deterministic(client: TestClient) -> None:
    payload = valid_payload()
    first = client.post("/analyze", json=payload)
    second = client.post("/analyze", json=payload)
    assert first.json() == second.json()


@pytest.mark.parametrize("field", load_schema().indicator_ids)
def test_analyze_requires_each_canonical_field(client: TestClient, field: str) -> None:
    payload = valid_payload()
    del payload["datos"][field]
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert field in response.text


@pytest.mark.parametrize("value", (1.5, -0.1))
def test_analyze_rejects_out_of_range_values(client: TestClient, value: float) -> None:
    field = load_schema().indicator_ids[0]
    payload = valid_payload()
    payload["datos"][field] = value
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert field in response.text


def test_analyze_rejects_unknown_extra_field(client: TestClient) -> None:
    payload = valid_payload()
    payload["datos"]["unknown"] = (load_schema().min_value + load_schema().max_value) / 2
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert "unknown" in response.text


@pytest.mark.parametrize("legacy, replacement", load_schema().rejected_field_names.items())
def test_analyze_rejects_legacy_names(client: TestClient, legacy: str, replacement: str) -> None:
    payload = valid_payload()
    payload["datos"][legacy] = payload["datos"][replacement]
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    assert replacement in response.text


def test_cors_allows_configured_dev_origin(client: TestClient) -> None:
    response = client.options(
        "/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_disallowed_origin(client: TestClient) -> None:
    response = client.options(
        "/analyze",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_missing_artifact_explains_training_command(tmp_path: Path) -> None:
    with (
        pytest.raises(FileNotFoundError, match="make train"),
        TestClient(create_app(artifact_path=tmp_path / "missing.joblib")),
    ):
        pass
