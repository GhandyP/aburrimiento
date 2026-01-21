# Component Plan: API Service

## Purpose
Expose a minimal REST API for boredom analysis using the Python core pipeline.

## Scope
- FastAPI app with two endpoints: `/health` and `/analyze`.
- Load model at startup using synthetic data.
- Accept 14 indicator values and return predicted level.

## Inputs
- `0.2/python/analizador.py`
- `0.2/data/estructura_indicadores.csv`

## Outputs
- `0.2/api/main.py`
- `0.2/api/requirements.txt`

## Interface
- `GET /health` returns `{"status": "ok"}`.
- `POST /analyze` accepts JSON with 14 features and returns `{ "nivel": "alto" }`.

## Tasks
1. Create `0.2/api/` directory.
2. Implement `main.py` with FastAPI app, startup training, and endpoints.
3. Add `requirements.txt` with FastAPI and Uvicorn.

## Validation
- Run `uvicorn main:app --reload --port 8000` and check `/health`.
- POST sample data to `/analyze` and confirm response.
