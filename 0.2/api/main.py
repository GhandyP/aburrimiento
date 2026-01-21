from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = BASE_DIR / "python"
sys.path.insert(0, str(PYTHON_DIR))

from analizador import AnalizadorAburrimiento

app = FastAPI(title="Analizador Aburrimiento API")


class AnalyzeRequest(BaseModel):
    datos: dict[str, float]


class AnalyzeResponse(BaseModel):
    nivel: str


@app.on_event("startup")
def startup() -> None:
    analizador = AnalizadorAburrimiento()
    dataset = analizador.generar_datos(300)
    analizador.entrenar(dataset.features, dataset.labels)
    app.state.analizador = analizador


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    analizador: AnalizadorAburrimiento = app.state.analizador
    payload = _normalize_payload(request.datos, analizador.feature_names)
    missing = [name for name in analizador.feature_names if name not in payload]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {', '.join(missing)}",
        )

    features = analizador.features_from_payload(payload)
    nivel = analizador.predecir(features)[0]
    return AnalyzeResponse(nivel=nivel)


def _normalize_payload(
    payload: dict[str, float],
    expected_features: list[str],
) -> dict[str, float]:
    normalized = dict(payload)
    alias_map = {
        "carencia_sentido": "carencia_de_sentido",
        "restriccion_libertad": "restriccion_de_libertad",
        "frustracion_agencia": "frustracion_de_agencia",
    }

    for alias, target in alias_map.items():
        if alias in normalized and target not in normalized:
            normalized[target] = normalized[alias]

    missing = [name for name in expected_features if name not in normalized]
    if missing:
        return normalized

    return normalized
