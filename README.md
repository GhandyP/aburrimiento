# aburrimiento

A small analyzer that classifies group-boredom indicators into three levels.

## Status

Today the repository contains a trained tree model served over HTTP, a React frontend, 97 existing
application tests plus README drift coverage under `make check`, and a capture path for real
observations. `assets/schema.json` is the single
source of truth for the indicators, groups, levels, and canonical field names.

## What this is NOT

- **Not a validated instrument.** Every published number comes from a synthetic generator whose
  labels are drawn from the same distributions the model learns. Accuracy therefore measures the
  self-consistency of the generator and nothing about boredom in the world.
- **Not production software.** There is no authentication, multi-tenancy, deployment target, or
  service level objective. It runs on one machine for one user.
- **Not a diagnostic tool.** Nothing here should inform a decision about a person or a group.

## Quickstart

From a fresh clone, run these four commands:

```bash
make setup
make train
make web-setup
make run
```

In a separate terminal, run the frontend:

```bash
make web-dev
```

`make help` lists every available target. The API listens on the configured host and port (by
default `0.0.0.0:8000`), and the Vite frontend uses its development default.

## Architecture

The canonical schema drives both runtime and generated clients. Python validates requests from it;
the TypeScript schema in `web/` is generated from the same file. Python checks the generated web
artifact for drift, while the frontend contract golden is frozen and tested deliberately.

```text
                    assets/schema.json
                    (single source of truth)
                       │             │
                       ▼             ▼
              src/aburrimiento/     tools/gen_web_types.py
              schema.py              │
              runtime validation    ▼
                       │      web/src/generated/schema.ts
                       ▼             │
                  FastAPI API   React frontend

       tests/test_readme_drift.py       web/src/contract.golden.json
       checks this README table           freezes the frontend contract
```

## The indicators

The table below is generated from `assets/schema.json`. Do not edit the generated region by hand;
run `make gen-docs` after changing the schema.

<!-- BEGIN GENERATED INDICATORS -->
| Group | Indicator ID | Spanish label | Theoretical basis |
|---|---|---|---|
| Estructuras Sistémicas | `reflejo_sistemas_culturales` | Reflejo sistemas culturales | Crítica artística al Fordismo |
| Estructuras Sistémicas | `productividad_capitalista` | Productividad capitalista | Lógica capitalista racional |
| Estructuras Sistémicas | `alienacion_neoliberal` | Alienación neoliberal | Individualización neoliberal |
| Estructuras Sistémicas | `racismo_sistemico` | Racismo sistémico | Desigualdad racial sistémica |
| Manifestaciones Grupales | `malestar_generalizado` | Malestar generalizado | Padecimiento colectivo crónico |
| Manifestaciones Grupales | `carencia_de_sentido` | Carencia de sentido | Meaninglessness social |
| Manifestaciones Grupales | `restriccion_de_libertad` | Restricción de libertad | Opresión sistémica |
| Manifestaciones Grupales | `frustracion_de_agencia` | Frustración de agencia | Falta de agencia efectiva |
| Dimensiones Medición | `desenganche` | Desenganche | Boredom Proneness Scale |
| Dimensiones Medición | `alta_excitacion` | Alta excitación | MSBS - Alta arousal |
| Dimensiones Medición | `inatencion` | Inatención | MSBS - Atención |
| Dimensiones Medición | `percepcion_tiempo_lenta` | Percepción tiempo lenta | MSBS - Percepción temporal |
| Dimensiones Medición | `estrategias_bloqueadas` | Estrategias bloqueadas | Respuestas bloqueadas |
| Dimensiones Medición | `angustia_profunda` | Angustia profunda | Potencial revolucionario |
<!-- END GENERATED INDICATORS -->

## Measured results

The committed [evaluation report](reports/evaluation.md) uses 3,000 synthetic samples with seed
42. Its measured test results are:

| Model | Test accuracy | Cohen's kappa | Macro F1 |
|---|---:|---:|---:|
| Random forest (120 trees) | 0.978333 | 0.967497 | 0.978493 |
| Logistic regression | 0.980000 | 0.969997 | 0.980146 |
| Dummy (most frequent) | 0.338333 | 0.000000 | 0.168535 |

Across twelve seeds, logistic regression matches the 120-tree forest: its mean advantage is
0.222 percentage points with a 0.416-point standard deviation, with logistic regression winning 6
seeds, the forest winning 1, and 5 ties. The models are statistically indistinguishable on this
generator; the supported finding is that the linear model matches 120 trees, not that it wins. This
means the task is largely linearly separable inside the generator; it says nothing about boredom in
the world.

## API contract

| Route | Method | Returns |
|---|---|---|
| `/health` | GET | Model status and artifact metadata, including schema version, training rows, seed, and measured metrics |
| `/schema` | GET | The canonical schema payload |
| `/analyze` | POST | `{"nivel": "bajo"|"medio"|"alto"}` for a valid request |
| `/samples` | POST | `{"id": <row id>, "nivel": <predicted level>}` after capturing an observation |
| `/samples/stats` | GET | Total rows and predicted/observed level distributions |

`/analyze` and `/samples` accept the request envelope `{"datos": {<14 canonical indicator ids>}}`;
`/samples` also accepts the optional `nivel_observado` field. Indicator values are floats in `[0,1]`.
Unknown fields are rejected with HTTP 422 and the offending field is named. The three archived short
names `carencia_sentido`, `restriccion_libertad`, and `frustracion_agencia` are also rejected with
HTTP 422, naming the offending field and its canonical replacement.

## Repository layout

| Path | Contents |
|---|---|
| `assets/` | Canonical schema and supporting conceptual assets |
| `src/aburrimiento/` | Python schema, synthetic generator, model, evaluation, storage, API, and CLI |
| `tests/` | Python contract, model, storage, web-drift, and README-drift tests |
| `tools/` | Generators for web types and the README indicator table |
| `web/` | React + Vite + TypeScript frontend and its generated schema |
| `reports/` | Committed synthetic evaluation reports |
| `models/` | Locally generated trained model artifacts |
| `data/` | Locally generated SQLite capture data |
| `legacy/` | Archived and inert iterations; nothing here is built, tested, linted, or imported |
| `odd/tasks/` | The feature plan and its verification record |
| `Makefile` | The project command entry point |

## Verification

```bash
make fmt
make lint
make typecheck
uv run pytest
npm test --prefix web
make check
```

`make check` is the single verification gate and runs the Python and frontend checks together.

## Project plan and prior iterations

- [Project plan](odd/tasks/aburrimiento-v1.md)
- [Prior iterations and why they were abandoned](legacy/README.md)
