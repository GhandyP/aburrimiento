from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from aburrimiento.contracts import build_analyze_request_model, build_analyze_response_model
from aburrimiento.model import BoredomModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = REPOSITORY_ROOT / "models" / "model.joblib"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configured_path = os.environ.get("ABURRIMIENTO_MODEL_PATH")
    artifact_path = Path(configured_path) if configured_path else DEFAULT_ARTIFACT_PATH
    if not artifact_path.is_absolute():
        artifact_path = REPOSITORY_ROOT / artifact_path
    app.state.analizador = BoredomModel.from_artifact(artifact_path)
    yield


app = FastAPI(title="Analizador Aburrimiento API", lifespan=lifespan)
AnalyzeRequest = build_analyze_request_model()
AnalyzeResponse = build_analyze_response_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    analizador: BoredomModel = app.state.analizador
    features = analizador.features_from_payload(request.datos.model_dump())
    nivel = analizador.predict(features)[0]
    return AnalyzeResponse(nivel=nivel)
