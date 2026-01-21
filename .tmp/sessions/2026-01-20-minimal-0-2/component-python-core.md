# Component Plan: Python Core Pipeline

## Purpose
Provide a minimal, functional Python pipeline for generating data, training a model, and predicting boredom levels using the assets in `0.2/data`.

## Scope
- Minimal dependency footprint.
- Deterministic behavior with seeded data generation.
- CLI entry point for demo predictions.

## Inputs
- `0.2/data/estructura_aburrimiento.json`
- `0.2/data/estructura_indicadores.csv`
- `0.2/data/niveles_clasificacion.csv`

## Outputs
- `0.2/python/analizador.py`
- `0.2/python/cli.py`
- `0.2/python/requirements.txt`

## Interface
- `AnalizadorAburrimiento` class with:
  - `generar_datos(n_muestras: int) -> tuple[pd.DataFrame, pd.Series]`
  - `entrenar(X: pd.DataFrame, y: pd.Series) -> None`
  - `predecir(X: pd.DataFrame) -> list[str]`
- CLI usage: `python cli.py --muestras 200`

## Tasks
1. Create `0.2/python/` directory.
2. Implement `analizador.py` with data loading, generation, model training, prediction.
3. Implement `cli.py` to run a demo pipeline and print a sample prediction.
4. Add `requirements.txt` with minimal dependencies.

## Validation
- Run `python 0.2/python/cli.py --muestras 100` and confirm output.
