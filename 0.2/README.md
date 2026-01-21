# Minimal Boredom Analyzer 0.2

Minimal working version with Python ML pipeline, FastAPI service, and Flutter UI.

## Structure
- `data/`: CSV and JSON assets
- `python/`: ML pipeline + CLI demo
- `api/`: FastAPI service
- `flutter/`: Flutter UI stub

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
  -d '{"datos": {"reflejo_sistemas_culturales": 0.8, "productividad_capitalista": 0.75, "alienacion_neoliberal": 0.85, "racismo_sistemico": 0.6, "malestar_generalizado": 0.9, "carencia_sentido": 0.85, "restriccion_libertad": 0.8, "frustracion_agencia": 0.9, "desenganche": 0.85, "alta_excitacion": 0.5, "inatencion": 0.8, "percepcion_tiempo_lenta": 0.9, "estrategias_bloqueadas": 0.85, "angustia_profunda": 0.8}}'
```

## Flutter
```bash
cd 0.2/flutter
flutter pub get
flutter run
```

### API Base URL
- Desktop: `http://127.0.0.1:8000`
- Android emulator: `http://10.0.2.2:8000`
- iOS simulator: `http://127.0.0.1:8000`
- Physical device: use your machine IP on the same network
