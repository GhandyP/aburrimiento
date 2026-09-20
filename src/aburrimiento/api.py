import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .contracts import build_analyze_request_model, build_analyze_response_model
from .model import BoredomModel
from .schema import load_schema

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = REPOSITORY_ROOT / "models" / "model.joblib"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _configured_artifact_path(path: Path | None) -> Path:
    configured_value = os.environ.get("ABURRIMIENTO_MODEL_PATH")
    configured = path if path is not None else Path(configured_value) if configured_value else None
    artifact_path = configured or DEFAULT_ARTIFACT_PATH
    return artifact_path if artifact_path.is_absolute() else REPOSITORY_ROOT / artifact_path


def _configured_cors_origins(origins: Sequence[str] | None) -> list[str]:
    if origins is not None:
        return list(origins)
    configured = os.environ.get("ABURRIMIENTO_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return list(DEFAULT_CORS_ORIGINS)


def create_app(
    artifact_path: Path | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    schema = load_schema()
    analyze_request = build_analyze_request_model(schema)
    analyze_response = build_analyze_response_model(schema)
    example = dict.fromkeys(schema.indicator_ids, 0.5)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.analizador = BoredomModel.from_artifact(
            _configured_artifact_path(artifact_path)
        )
        yield

    application = FastAPI(
        title="Aburrimiento Analyzer API",
        description=(
            "Analyze group boredom indicators. The numbers come from a synthetic generator "
            "and are not evidence about boredom."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_configured_cors_origins(cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        model: BoredomModel = application.state.analizador
        metadata = model.metadata
        assert metadata is not None
        return {
            "status": "ok",
            "schema_version": metadata.schema_version,
            "schema_sha256": metadata.schema_sha256,
            "trained_at": metadata.trained_at,
            "n_samples": metadata.n_samples,
            "seed": metadata.seed,
            "sklearn_version": metadata.sklearn_version,
            "metrics": {
                "accuracy": metadata.accuracy,
                "kappa": metadata.kappa,
                "macro_f1": metadata.macro_f1,
            },
        }

    @application.get("/schema", tags=["system"])
    def schema_endpoint() -> dict[str, object]:
        return load_schema().to_payload()

    @application.post(
        "/analyze",
        response_model=analyze_response,
        tags=["analysis"],
        openapi_extra={
            "requestBody": {"content": {"application/json": {"example": {"datos": example}}}}
        },
    )
    def analyze(request: analyze_request) -> analyze_response:  # type: ignore[valid-type]
        model: BoredomModel = application.state.analizador
        features = model.features_from_payload(request.datos.model_dump())  # type: ignore[attr-defined]
        return analyze_response(nivel=model.predict(features)[0])

    return application


app = create_app()
