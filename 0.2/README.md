# Minimal Boredom Analyzer 0.2

Minimal working version with Python ML pipeline, FastAPI service, and Flutter UI.

## Structure
- `data/`: CSV and JSON assets
- `python/`: ML pipeline + CLI demo
- `api/`: FastAPI service
- `(frontend)`: see the Frontend status note at the end of this file.

## Python CLI
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r 0.2/python/requirements.txt
python 0.2/python/cli.py --muestras 100
```

## API
```bash
. .venv/bin/activate
pip install -r 0.2/api/requirements.txt
uvicorn main:app --app-dir 0.2/api --host 127.0.0.1 --port 8000
```

### Example Request
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"datos": {"reflejo_sistemas_culturales": 0.8, "productividad_capitalista": 0.75, "alienacion_neoliberal": 0.85, "racismo_sistemico": 0.6, "malestar_generalizado": 0.9, "carencia_de_sentido": 0.85, "restriccion_de_libertad": 0.8, "frustracion_de_agencia": 0.9, "desenganche": 0.85, "alta_excitacion": 0.5, "inatencion": 0.8, "percepcion_tiempo_lenta": 0.9, "estrategias_bloqueadas": 0.85, "angustia_profunda": 0.8}}'
```

## Frontend status

The Flutter client that used to live in `0.2/flutter/` was archived under `legacy/flutter/` and is no
longer maintained: it hardcoded `127.0.0.1:8000` and its only test was Flutter's default counter
stub. A React + Vite + TypeScript frontend replaces it under `web/`.

Until that frontend lands, the only supported interfaces are the Python CLI and the HTTP API, and
the `curl` example above is the reference client.

Note that this iteration's `python/` and `api/` directories are themselves being migrated into the
`src/aburrimiento/` package; see `odd/tasks/aburrimiento-v1.md`.

Field names above come from the canonical schema at `assets/schema.json`. The short forms used by the
archived Dart clients (`carencia_sentido`, `restriccion_libertad`, `frustracion_agencia`) are rejected
with a 422 that names the canonical replacement.
