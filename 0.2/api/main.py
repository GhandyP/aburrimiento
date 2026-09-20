from __future__ import annotations

from fastapi import FastAPI

from aburrimiento.contracts import build_analyze_request_model, build_analyze_response_model
from aburrimiento.model import BoredomModel

app = FastAPI(title="Analizador Aburrimiento API")
AnalyzeRequest = build_analyze_request_model()
AnalyzeResponse = build_analyze_response_model()


@app.on_event("startup")
def startup() -> None:
    analizador = BoredomModel()
    dataset = analizador.generate(300)
    analizador.train(dataset.features, dataset.labels)
    app.state.analizador = analizador


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    analizador: BoredomModel = app.state.analizador
    features = analizador.features_from_payload(request.datos.model_dump())
    nivel = analizador.predict(features)[0]
    return AnalyzeResponse(nivel=nivel)
