# Master Plan: 0.2 Minimal Working Version

## Purpose
Deliver a minimal, working 0.2 version that exposes a Python core pipeline and a small API service using the existing 0.1 data assets.

## Architecture
- Data assets live in `0.2/data` and define indicators and classification levels.
- Python core pipeline in `0.2/python` loads assets, generates synthetic samples, trains a model, and predicts a boredom level.
- API service in `0.2/api` loads the Python core, trains at startup with synthetic data, and exposes `/health` and `/analyze`.

## Components (Dependency Order)
1. Data Assets
   - Copy data files from 0.1 to `0.2/data`.
   - Validation: confirm files exist in `0.2/data`.
2. Python Core Pipeline
   - Implement `analizador.py`, `cli.py`, `requirements.txt` in `0.2/python`.
   - Validation: run `python 0.2/python/cli.py --muestras 100` (in venv).
3. API Service
   - Implement `main.py` and `requirements.txt` in `0.2/api`.
   - Validation: run `uvicorn` and call `/health` and `/analyze`.

## Status
- Data Assets: Completed
- Python Core Pipeline: Completed
- API Service: In progress (validation pending final /analyze response capture)

## Next Actions
1. Re-run API validation and capture `/analyze` response body.
2. Mark API Service as Completed if validation passes.
