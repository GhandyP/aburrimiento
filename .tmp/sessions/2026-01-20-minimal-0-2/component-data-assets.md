# Component Plan: Data Assets

## Purpose
Provide the minimal CSV/JSON assets required by the Python pipeline and API for a working demo in `0.2/`.

## Scope
- Copy or trim essential assets from `0.1/`.
- Ensure the Python pipeline can run using these assets without additional external data.

## Inputs
- `0.1/estructura_aburrimiento.json`
- `0.1/estructura_indicadores.csv`
- `0.1/niveles_clasificacion.csv`

## Outputs
- `0.2/data/estructura_aburrimiento.json`
- `0.2/data/estructura_indicadores.csv`
- `0.2/data/niveles_clasificacion.csv`

## Interface
- File paths in `0.2/data/` used by Python core.
- No runtime API changes in this component.

## Tasks
1. Create `0.2/data/` directory.
2. Copy selected JSON/CSV assets from `0.1/` to `0.2/data/`.
3. Verify file presence and readable paths.

## Validation
- List `0.2/data/` and confirm three files exist.
- Ensure filenames match expected paths.
