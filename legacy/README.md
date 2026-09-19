# legacy/ — archived iterations

**Nothing in this directory is built, tested, linted or imported.** It is kept for historical
reference only: to answer "what did the earlier attempts do, and why were they abandoned?" without
digging through git history.

The active project lives in `src/`, `web/`, `assets/` and `tests/`. See the root `README.md`.

---

## Contents

| Path | What it was | Why it was abandoned |
|---|---|---|
| `legacy/0.1/` | The original monolith: `analizador_aburrimiento_ml.py` (Random Forest + a 128→64→32 neural network + PyCM evaluation), `flutter_main.dart`, `DOCUMENTACION_SISTEMA.md`, CSV/JSON assets, and `script_1.py` … `script_4.py` plus `chart_script.py` — scripts that **generate** the project's own source files and documentation. | The Dart UI never called the backend: it computed three averages locally and applied thresholds. The neural network added a heavier dependency without a measured comparison against the tree model. The accuracy figures published in `comparacion_modelos.csv` (Random Forest 88–92 %, neural network 85–90 %) were never measured against a held-out set. Generating source files from scripts made every edit a two-file change. |
| `legacy/flutter/` | The Flutter frontend of the `0.2` iteration (a single screen collecting 14 values, posting to `/analyze`). | Hardcoded API base URL (`127.0.0.1:8000`) that contradicted the platform URLs documented in the very same iteration's README. Its widget test was Flutter's default counter stub, asserting `MyApp` and `Icons.add` while the app defined `BoredomApp` and had no counter — so the only test could never pass. Replaced by the React + Vite + TypeScript app in `web/`. |
| `legacy/flutter/` + `legacy/0.1/flutter_main.dart` | The Dart clients used short field names (`carencia_sentido`, `restriccion_libertad`, `frustracion_agencia`). | Those names were a second schema. The `0.2` CSV slugified the same concepts to `carencia_de_sentido`, `restriccion_de_libertad`, `frustracion_de_agencia`, and an `alias_map` in the API papered over the difference. Result: four sources of truth for one schema (conceptual JSON, runtime CSV, Dart literals, API aliases). Replaced by the single canonical `assets/schema.json`. |
| `legacy/sessions/` | Agent session transcripts and planning artifacts: two `master-plan.md` files, five component plans, and ~200 KB of session logs. | Process scratch, not project artifacts. They recorded the `0.2` build and its final state: `API Service: Blocked (validation inconclusive)`. Preserved because they document *why* the iteration stopped where it did. |

## What was not archived

Generated artifacts were deleted rather than archived, because a command can regenerate them and
they carry machine-specific absolute paths: Flutter `.dart_tool/`, `build/`, `.flutter-plugins-dependencies`,
and Python `__pycache__/`. See the root `.gitignore`.

## Ground rules for this directory

1. Do not import, build, test or lint anything here.
2. Do not fix bugs here. If something is worth fixing, it is worth writing properly in `src/`.
3. Do not copy a claim from here into the active tree. Every number in the active project must be
   reproducible by a committed command.
