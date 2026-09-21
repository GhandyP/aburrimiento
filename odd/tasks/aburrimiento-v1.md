# ODD Feature Plan: aburrimiento-v1

**Feature branch:** `feat/aburrimiento-v1` (create from `main` before the first commit)
**Status:** planned — no source writes yet
**Created:** 2026-09-19
**Owner decisions locked:** see *Locked Decisions* below

---

## 1. Why this feature exists

Read-only reconnaissance of 2026-09-19 produced three verified defects and one
blocking architectural problem. This plan fixes all four and lifts the project
from "two half-finished prototypes" to one coherent, locally verifiable system.

### 1.1 Verified findings this plan must close

| # | Finding | Evidence | Closed by |
|---|---|---|---|
| F1 | 15 generated artifacts are committed (Flutter `.dart_tool`, `build/`, `__pycache__/*.pyc`). Root `.gitignore` ignores only `.atl/` and is itself untracked. | `git ls-files \| grep -cE '__pycache__\|\.dart_tool\|/build/'` → 15 | T1 |
| F2 | Four sources of truth for one schema: CSV (`carencia_de_sentido`), conceptual JSON, Flutter string literals (`carencia_sentido`), API `alias_map`. Client/backend names differ and agree only by accident. | `0.2/api/main.py:61-69`; `main.dart:437`; `estructura_indicadores.csv` | T5–T8 |
| F3 | API validation is weak and partly dead code: `dict[str, float]` accepts any float (no `[0,1]` range), extras are silently ignored, and `_normalize_payload` never removes aliases while its internal `missing` check is computed and discarded on both return paths. | `0.2/api/main.py:18-20, 56-74` | T7, T14 |
| F4 | Flutter is unmaintainable for this project: hardcoded `apiBaseUrl` (`main.dart:7`) contradicts documented platform URLs in `0.2/README.md`; widget test is the default counter stub asserting `MyApp`/`Icons.add` while the app defines `BoredomApp`. | `0.2/flutter/test/widget_test.dart` | T2, T17–T21 |

### 1.2 Verified engineering debt (fixed as part of the work, not as a separate epic)

| # | Debt | Evidence | Closed by |
|---|---|---|---|
| D1 | Model retrains on every API start via deprecated `@app.on_event("startup")` | `0.2/api/main.py:26-31` | T12, T13 |
| D2 | `sys.path.insert` import hack couples API to a sibling directory layout | `0.2/api/main.py:13-15` | T3, T4 |
| D3 | `AnalizadorAburrimiento` resolves its data dir by counting `parents[1]`, so the module only works from its exact build location | `0.2/python/analizador.py:44` | T4 |
| D4 | Dependencies are unpinned (`numpy`, `pandas`, `scikit-learn`) — no reproducible environment | `0.2/python/requirements.txt` | T3 |
| D5 | `StandardScaler` is applied before a Random Forest, which is scale-invariant; it adds a persisted state object with no effect on results | `0.2/python/analizador.py:47,94-99` | T10, T12 |
| D6 | Accuracy claims (RF 88–92%, NN 85–90%) were never measured against a held-out set | `0.1/comparacion_modelos.csv`; `0.1/DOCUMENTACION_SISTEMA.md` | T10, T11 |
| D7 | Synthetic labels are generated from the same rule the model learns, so reported accuracy cannot establish real-world validity | `0.2/python/analizador.py:64-90` | T9, T11 |
| D8 | No tests, no linter, no type checker, no single verification entry point | repo-wide | T3, T14, T16, T21 |

### 1.3 Locked decisions (owner, 2026-09-19)

1. **Flutter is replaced by React + Vite + TypeScript.** Not a second frontend — Flutter is removed.
2. **Scope ceiling: locally verifiable excellence.** No Docker, no Kubernetes, no cloud, no CI
   provider. Everything in this plan must be executable and checkable on this machine.
3. **ML: honest synthetic + a real path.** Keep synthetic data, but split it properly, report
   measured metrics only, and ship the capture path for real samples.
4. **`0.1/` is archived, not deleted.** Moved under `legacy/` with an explanatory README.

---

## 2. Non-goals (explicit)

- No model serving infrastructure (Triton), GPU inference, ONNX/quantization.
- No Kubernetes, Helm, Redis, S3/MinIO, database servers, or message queues.
- No cloud CI/CD, no container registry, no deploy pipeline.
- No authentication, no multi-tenancy, no user accounts.
- No deep learning. The removed neural network is **not** reinstated: this feature buys rigor,
  not model count. A second model is added only as an honest comparison baseline (T11).
- No migration of `blueplan.md` content into code. It is reclassified as aspirational (T24).
- No real-world accuracy claims. Only metrics measured on a documented held-out split (T10).

---

## 3. Target architecture

```
aburrimiento/
├── README.md                      # living doc: what it is, what it is NOT, quickstart
├── Makefile                       # single entry point: setup / check / run / train
├── .gitignore                     # real ignore rules
├── .editorconfig
├── pyproject.toml                 # package + ruff + mypy + pytest + pinned deps
├── uv.lock                        # reproducible resolution
├── assets/
│   └── schema.json                # ★ SINGLE SOURCE OF TRUTH (14 indicators, 3 levels)
├── src/aburrimiento/
│   ├── __init__.py
│   ├── schema.py                  # loads + validates schema.json, derives runtime models
│   ├── synthetic.py               # explicit, documented synthetic generator
│   ├── model.py                   # train / persist / load / predict
│   ├── evaluate.py                # stratified splits, CV, honest metrics
│   ├── store.py                   # SQLite capture of real samples
│   ├── api.py                     # FastAPI app (lifespan-managed)
│   └── cli.py                     # CLI
├── tools/
│   └── gen_web_types.py           # schema.json → web/src/generated/schema.ts
├── tests/                         # pytest
├── web/                           # React + Vite + TypeScript
│   └── src/
│       ├── generated/schema.ts    # ★ generated — never hand-edited
│       ├── api.ts                 # typed client
│       └── components/
├── docs/
│   ├── architecture.md
│   ├── decisions/                 # ADRs
│   └── blueplan-aspiracional.md   # reclassified, with a map to reality
├── reports/                       # committed, reproducible evaluation output
├── legacy/                        # 0.1/, Flutter, transcripts, with a README
└── odd/tasks/aburrimiento-v1.md   # this plan
```

### 3.1 The schema flow (closes F2)

```
                    assets/schema.json
                    (indicators, groups, order,
                     ranges, level definitions)
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  schema.py              tools/gen_         README indicator
  (runtime validation,   web_types.py      table (checked
   Pydantic models)           │             by test)
        │                     ▼
        ▼              web/src/generated/schema.ts
  FastAPI contract            │
  (422 on violation)          ▼
                        typed React client
```

Drift is caught by `tests/test_web_types_drift.py` (fails if the generated file does not match
the schema) and by `npm run typecheck` (fails if the client uses a field that no longer exists).

### 3.2 API contract (target)

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + model artifact identity (version, trained-at, training rows) |
| `/schema` | GET | The canonical schema, so any client can adapt without reading code |
| `/analyze` | POST | Strict analysis. 14 required floats in `[0,1]`, `extra="forbid"`, structured 422 |
| `/samples` | POST | Capture a real observation (no PII, timestamped) — the path to real data |
| `/samples/stats` | GET | Row count and label distribution of captured real samples |

Legacy field names (`carencia_sentido`, `restriccion_libertad`, `frustracion_agencia`) are
**rejected with a 422 that names the canonical field**. The `alias_map` is deleted: with Flutter
gone, the only consumer was the client being replaced.

---

## 4. Tasks

Every task closes with **at least one work-unit commit** on `feat/aburrimiento-v1`, with tests and
docs alongside the behavior, using a Conventional Commit message. A task is checked off only after
its closure criterion was observed and the commit identity was recorded in this file.

### Phase 0 — Foundation and hygiene

**T1. Create the branch and fix repository hygiene**
- Deliverable: `feat/aburrimiento-v1` branch; rewritten `.gitignore` (Python caches, `__pycache__/`,
  `*.pyc`, `.venv/`, `node_modules/`, `web/dist/`, `.dart_tool/`, `build/`, `.env`, `*.db`,
  `.atl/`, `.codegraph/`); `git rm --cached` for all 15 tracked generated artifacts; the root
  `.gitignore` itself tracked.
- Closure: `git ls-files | grep -cE '__pycache__|\.dart_tool|/build/'` → `0`; working tree clean
  after running the pipeline and a Flutter-free `make check`.
- Commit: `chore(repo): add real gitignore and untrack generated artifacts`

**T2. Archive the legacy iteration**
- Deliverable: `legacy/0.1/` (whole tree), `legacy/flutter/` (former `0.2/flutter`),
  `legacy/flutter-monolith/` (`0.1/flutter_main.dart`, `0.1/pubspec.*`), `legacy/sessions/`
  (transcripts + `.tmp/sessions`), plus `legacy/README.md` stating what each artifact was, why it
  was abandoned, and that nothing here is built or tested. Generator scripts (`script_1..4.py`)
  move with the legacy code they produced.
- Closure: no build, test, or lint path references `legacy/`; `make check` green; `legacy/README.md`
  reviewed against the finding list in §1.
- Commit: `chore(legacy): archive 0.1 prototype and Flutter frontend`

**T3. Establish the Python project skeleton and toolchain**
- Deliverable: `pyproject.toml` (package `aburrimiento`, `src/` layout, pinned dependency ranges,
  ruff + mypy + pytest config), `uv.lock`, `Makefile` with `setup`, `fmt`, `lint`, `typecheck`,
  `test`, `check`, `clean`, `.editorconfig`, plus a root `README.md` placeholder (required by the
  build and referenced by `legacy/README.md`; T22 replaces it). Target Python 3.13.
- Closure: `uv sync` from clean; `make check` runs ruff, mypy, pytest and exits 0 (with an empty
  test suite at this point, reported honestly).
- Commit: `build(python): add uv project skeleton with ruff, mypy and pytest`

**T4. Migrate the pipeline into the package, behavior preserved**
- Deliverable: `src/aburrimiento/{model,synthetic,cli}.py` derived from `0.2/python/`, the data
  assets moved from `0.2/data` to `assets/`, `0.2/python/` deleted, and `0.2/api/main.py` importing
  the installed package so the tree stays runnable.
- **Pure move:** the `StandardScaler`, the hyperparameters, the RNG call order and the distributions
  are untouched. D5 (removing the scale-invariant scaler) moves to T10/T12, where the effect can be
  measured instead of assumed. `schema.py` arrives in T6, not here.
- Closure: a baseline test captured from the **legacy module recovered from git** asserts identical
  dataset digests, feature importances and `predict_proba`, plus the predicted labels; `make check`
  green; the guard is mutation-tested (one modifier changed → fails; reverted → passes).
- Commit: `refactor(core): migrate pipeline into the aburrimiento package`
- *Risk:* the highest-risk task for silent behavior change. **Resolved:** the digests matched
  exactly, so the move is proven, not assumed.

### Phase 1 — Canonical schema (closes F2, F3)

**T5. Author `assets/schema.json`**
- Deliverable: one file declaring the 14 indicators (canonical id, localized label, group,
  display order, range) and the 3 classification levels (id, range, meaning, recommended action).
  Canonical ids are the existing slugs; the three previously aliased names are declared as
  **rejected legacy names with the canonical replacement**, so the rejection message is data-driven.
- Closure: `tests/test_schema.py` asserts 14 unique ids, 4/4/6 group distribution, `[0,1]` range,
  contiguous display order, 3 levels, and that every legacy name maps to a canonical id.
- Commit: `feat(schema): add canonical indicator and level schema`

**T6. Load and validate the schema at runtime**
- Deliverable: `schema.py` as the only module that reads `schema.json`, exposing typed accessors;
  a `Indicators` value object the rest of the code depends on.
- Closure: tests for a missing file, malformed JSON, duplicate ids, out-of-range bounds, and an
  empty group — each failing loudly with an actionable message.
- Commit: `feat(schema): load and validate canonical schema`

**T7. Enforce the contract at the API boundary**
- Deliverable: Pydantic request model with the 14 canonical fields, `ge=0`, `le=1`,
  `extra="forbid"`, and structured 422 errors naming each offending field; `alias_map` and dead
  `_normalize_payload` deleted; legacy names rejected with a message pointing at the canonical id
  and at `/schema`.
- Closure: tests for a valid payload, a missing field, a value above 1, a value below 0, an unknown
  extra field, and each rejected legacy name; every case asserts the status code **and** the
  structured body.
- Commit: `feat(api): enforce strict typed request contract`
- *Note:* this intentionally breaks the current contract. Because the only client is being replaced
  in the same feature, no compatibility window is needed.

**T8. Generate and drift-check the TypeScript schema**
- Deliverable: `tools/gen_web_types.py` producing `web/src/generated/schema.ts` (field ids as a
  union type, groups, labels, level ids), plus a header banning hand edits.
- Closure: `tests/test_web_types_drift.py` fails when the generated file is stale; `make gen-types`
  regenerates it; generated output is committed.
- Commit: `feat(tooling): generate web types from canonical schema`

### Phase 2 — Honest machine learning (closes D6, D7)

**T9. Make the synthetic generator explicit and self-aware**
- Deliverable: `synthetic.py` with the class-conditional distributions, modifiers, seed and sample
  count as named parameters and documented defaults; the module docstring states plainly that
  labels are derived from the same distributions the model learns, so accuracy measures
  self-consistency, not real-world validity.
- Closure: tests for determinism under a fixed seed, distribution bounds within `[0,1]`, class
  balance within tolerance, and a snapshot of the documented default parameters.
- Commit: `refactor(ml): extract explicit synthetic generator`

**T10. Evaluate honestly: real splits, measured metrics**
- Deliverable: `evaluate.py` with a stratified train/validation/test split, cross-validation on the
  training partition, and metrics reported as measured values: accuracy, Cohen's kappa, macro F1,
  per-class precision/recall/F1, and the confusion matrix. Output written to `reports/` as JSON plus
  a human-readable summary, committed together with the exact command that reproduces it.
- Closure: `make train` regenerates `reports/` byte-identically for a fixed seed; a test asserts the
  report file matches the model's own metrics (no hand-written numbers).
- Commit: `feat(ml): add honest evaluation with held-out splits`
- *Note:* `0.1/comparacion_modelos.csv` keeps its claims only inside `legacy/`; nothing in the new
  tree may state a number that a committed report cannot reproduce.

**T11. Replace claimed model comparisons with measured ones**
- Deliverable: the tree model compared against a simple, declared baseline (multinomial logistic
  regression) under identical splits and CV folds; the comparison table is generated, not authored.
- Closure: the comparison is reproducible via `make train`; the report states the baseline's
  measured performance and explicitly notes that a small gap indicates the task is largely
  separable by a linear rule — i.e. evidence about the *generator*, not about boredom.
- Commit: `feat(ml): compare tree model against declared linear baseline`

**T12. Persist the trained model artifact**
- Deliverable: `make train` writes a versioned artifact (model + label encoder + schema hash +
  training metadata); the API loads the artifact through lifespan instead of training at startup
  (D1); a missing or schema-mismatched artifact fails loudly rather than silently retraining.
- Closure: API starts in under a second with a prebuilt artifact; a test proves a schema-hash
  mismatch is rejected with an actionable error; `/health` reports the artifact identity.
- Commit: `feat(ml): persist and load versioned model artifact`

### Phase 3 — API surface

**T13. Build the FastAPI application**
- Deliverable: `api.py` with lifespan management, CORS restricted to the Vite dev origin, structured
  error responses, `/health`, `/schema`, `/analyze`, and OpenAPI metadata (description, tags,
  examples for the 14-field payload).
- Closure: `uvicorn aburrimiento.api:app --reload` serves `/docs` showing all routes with the
  request schema rendered from canonical ids.
- Commit: `feat(api): serve health, schema and strict analyze endpoints`
- *Note:* this task also deletes `0.2/` entirely (its API, README and any remaining trace). `0.2/`
  is currently excluded from ruff precisely because it is scheduled for removal here; leaving that
  exclusion in place after this task would be a silent hole in verification.

**T14. Cover the API with a real test suite**
- Deliverable: `tests/test_api.py` using `TestClient` covering the happy path, each missing field,
  out-of-range values at both ends, unknown extras, rejected legacy names, `/schema` consistency
  with `assets/schema.json`, and response determinism for a fixed payload.
- Closure: suite green; a mutation of any canonical field name in the schema fails at least one
  test (verifies the tests are actually wired to the contract).
- Commit: `test(api): cover analyze contract and error paths`

**T15. Capture real samples (the path to real data)**
- Deliverable: `store.py` over SQLite with an explicit table schema, timestamps, no PII, and a
  migration-safe initialization; `POST /samples` and `GET /samples/stats`. The DB file is ignored by
  git; the schema is created on demand.
- Closure: tests for insert, duplicate handling, stats aggregation, and that a sample cannot be
  stored with a field outside the canonical schema; `docs/decisions` records what is captured and
  what deliberately is not (no identifiers, no free text).
- Commit: `feat(store): capture real observations in sqlite`

**T16. One command decides the truth**
- Deliverable: `make check` orchestrating ruff format check, ruff lint, mypy, pytest, and the web
  pipeline (`lint`, `typecheck`, `test`, `build`).
- Closure: green on a clean checkout after `make setup`; intentionally breaking one Python file and
  one TypeScript file each makes `make check` fail with a readable message.
- Commit: `build: add single verification entry point`

### Phase 4 — Web frontend (React + Vite + TypeScript)

**T17. Scaffold the web application**
- Deliverable: `web/` with Vite + React + TypeScript, strict `tsconfig`, ESLint, Vitest + Testing
  Library, and `npm run` scripts wired into the Makefile. `node_modules` and `dist` ignored.
- Closure: `npm run typecheck` and `npm test` pass on the scaffold; `npm run build` produces `dist/`.
- Commit: `feat(web): scaffold vite react typescript app`

**T18. Typed API client over the generated schema**
- Deliverable: `api.ts` consuming `generated/schema.ts`, with a discriminated result type
  (`ok` / `validation_error` / `network_error`) that surfaces backend field errors per input.
- Closure: unit tests for each result branch using a mocked fetch, including the structured 422 body
  that names offending fields.
- Commit: `feat(web): add typed api client with error mapping`

**T19. Build the analysis UI**
- Deliverable: the 14 indicators rendered from the generated schema in three groups, `[0,1]`
  constrained inputs, client-side validation mirroring the server, a submit flow with loading and
  disabled states, and a result card showing the level plus its meaning from the schema. Accessible
  (labeled controls, keyboard operable, `aria-live` for the result) and usable on a phone screen over
  LAN.
- Closure: component tests assert a group renders exactly its own indicators, out-of-range input is
  blocked before submission, and the result card shows the level's meaning.
- Commit: `feat(web): add grouping analysis form and result card`

**T20. Make the API base URL configurable per environment**
- Deliverable: `VITE_API_URL` with a documented default, `.env.example` committed, `README` notes for
  desktop / LAN / emulator, and a startup warning when the configured host is unreachable. The
  hardcoded localhost is gone (F4).
- Closure: the app points at a non-default URL purely through `.env`, verified by a test that reads
  the configured value; documented LAN steps work without touching source.
- Commit: `feat(web): configure api base url per environment`

**T21. Contract test between frontend and backend**
- Deliverable: a test that fails if the web client references a field absent from
  `generated/schema.ts`, plus an integration-style test posting a full 14-field payload to a mocked
  backend shaped exactly like the FastAPI contract.
- Closure: renaming a field in `assets/schema.json` and regenerating makes the web suite fail; the
  failure message names the offending field.
- Commit: `test(web): assert frontend contract matches canonical schema`

### Phase 5 — Documentation and closure

**T22. Write the living README**
- Deliverable: what the project is, an explicit "what it is NOT" section (synthetic data, no
  real-world validity claim, local-only, single user), a four-command quickstart, the architecture
  diagram from §3, the indicator table generated from the schema and verified by test, the API
  contract table from §3.2, and the verification commands.
- Closure: a fresh clone reaches a working analysis by following only the README; the indicator table
  test fails if it diverges from the schema.
- Commit: `docs: write living readme with honest scope`

**T23. Record the decisions as ADRs**
- Deliverable: `docs/decisions/` with four short ADRs: dropping Flutter; `schema.json` as single
  source of truth; honest synthetic data and its explicit ceiling; local-verifiable scope versus the
  aspirational blueprint.
- Closure: each ADR states context, decision, consequences and what would change the decision.
- Commit: `docs: add architecture decision records`

**T24. Reclassify the aspirational blueprint**
- Deliverable: `blueplan.md` moved to `docs/blueplan-aspiracional.md`, prefixed with a status banner
  (aspirational, unimplemented, not a roadmap) and a short map from its major sections to what
  actually exists.
- Closure: no file outside `docs/` references it as a plan; `README` links it as background reading.
- Commit: `docs: reclassify blueprint as aspirational reference`

**T25. Close the feature with observed evidence**
- Deliverable: `make check` output, a real end-to-end run (train → serve → analyze → capture a
  sample) with verbatim command output pasted into the closure notes, the measured metrics from
  `reports/`, and an honest list of what remains undone.
- Closure: every task above marked done with its commit hash recorded; every skipped or failed check
  reported rather than omitted.
- Commit: `docs: record v1 verification evidence`
- Then: native review of the resulting candidate under the user-owned review switch. Delivery
  (push, PR, merge) remains the user's decision.

---

## 5. Task-to-correction traceability

| Owner's request | Findings closed | Tasks |
|---|---|---|
| Fix repository hygiene | F1 | T1, T2 |
| Fix the quadruplicated schema | F2 | T5, T6, T7, T8, T21 |
| Close the blocked API validation, with real tests | F3, D1, D2, D3, D4, D8 | T3, T4, T13, T14, T16 |
| Replace Flutter | F4, and the hardcoded-URL defect | T2, T17, T18, T19, T20 |
| Correctness of the ML claims | D6, D7 | T9, T10, T11, T12 |
| Path to real data | D7 | T15 |
| Documentation truth | — | T22, T23, T24, T25 |

## 6. Definition of done

1. `make check` green on a clean checkout: ruff, mypy, pytest, web lint, typecheck, tests, build.
2. `make setup && make train && make run && make web` reaches a working analysis from scratch.
3. No artifact under version control that a command can regenerate.
4. One schema source; any divergence fails a test in Python **and** in TypeScript. *(Corrected during
   Phase 4: measurement showed the two halves catch different failures — see the note in §9.)*
5. Every number in the README is reproducible by a committed command.
6. No claim of real-world validity anywhere in the repository.
7. `legacy/` is inert: referenced by nothing that builds.
8. Every task has a commit hash recorded in this file.

## 7. Risks and assumptions

| Risk | Mitigation |
|---|---|
| T4 silently changes predictions while "just moving files" | Baseline test captured against the 0.2 implementation before the move; kept as regression guard |
| React frontend balloons in scope | UI limited to T19's described surface: 14 inputs, three groups, one result card, no routing, no state library |
| mypy strictness fights scikit-learn's typing | Per-module overrides for ML calls, documented in `pyproject.toml`; no blanket `ignore_errors` |
| Breaking the API contract breaks a consumer | The only consumer is the Flutter client removed in the same feature; verified by repo search before T7 |
| Synthetic ceiling mistaken for results | Stated in module docstring, README, ADR, and the generated report header |
| LAN/CORS misconfiguration | CORS origin from configuration, not hardcoded; T20 documents and tests the non-default URL path |
| `uv.lock` churn on shared machines | Committed lockfile is the contract; `make check` reads it, never rewrites it |

## 8. Assumption required before T7 — RESOLVED (2026-09-19)

Repo-wide search confirmed **no external consumer** of the legacy field names. Verified by
`grep -rn` across `*.py`, `*.dart`, `*.md`, `*.json`, `*.csv`, `*.js`, `*.ts`, plus a search of
`/home/vit` for Python files importing `analizador`.

Findings:

- Every occurrence of `carencia_sentido` / `restriccion_libertad` / `frustracion_agencia` lives in
  `0.1/` (the archived prototype, including its generator scripts) or in session transcripts.
- The only live documents using the legacy short names are `0.2/README.md` (replaced by T22) and the
  Flutter client (removed by T2).
- The `alias_map` in `0.2/api/main.py` therefore existed to bridge **0.1 and Flutter against the 0.2
  CSV**, which is exactly the quadruplicated-schema defect of F2.
- No file outside `/home/vit/aburrimiento` references the package.

Conclusion: T7 may break the contract cleanly, with no deprecation window and no compatibility
shim. Because `0.1/` uses the short names, the canonical ids must be declared explicitly in
`assets/schema.json` (T5) rather than derived by slugifying CSV labels — the previous derivation is
what created the split in the first place.

---

## 9. Execution log

### Phase 0 — Foundation and hygiene — COMPLETE

| Task | Commit | Closure evidence observed |
|---|---|---|
| Plan | `f38d7f3` | This document committed before any source write |
| T1 | `8b51397` | 17 generated paths untracked; `git ls-files` matches 0 of them; `.gitignore` tracked |
| T2 | `72bad94` | 158 files under `legacy/`; 0 references to `legacy/` from live code; no live `pubspec.yaml` |
| T3 | `024eb34` | `make check` green (ruff, mypy strict, 2 tests); `uv.lock` pinned |
| T4 | `5c5aa27` | Identical dataset digests, feature importances and `predict_proba`; guard mutation-tested |

**Deviations from this plan, and why:**

- **T3** additionally created a root `README.md` placeholder. Not in the task, but `pyproject.toml`
  requires a readme for the build and `legacy/README.md` referenced a root README that did not exist.
  T22 replaces it with the full version.
- **T3** excluded `0.2/` from ruff and limited mypy to `src` and `tests`: `0.2/` is removed in T13,
  and `tools/` does not exist until T8.
- **T4** was executed as a pure move. D5 moved to T10/T12 because the scaler's removal is a behavior
  change that a move-preservation test cannot validate.
- **T4** renamed the analyzer to `BoredomModel` and deleted Spanish method aliases introduced during
  the migration: the intermediate state exposed two public names for every method.
- The T4 verification uses the strongest evidence available instead of a smoke test. The legacy
  module was recovered with `git show HEAD:0.2/python/analizador.py`, executed against the migrated
  package, and compared on dataset digests, the full feature-importances vector and `predict_proba`
  for all three levels. All matched exactly.
- A first regression test was rejected as insufficient: it pinned only `example_for()` (pure
  arithmetic, no RNG involvement) and three widely separated labels, so it would have passed even if
  the RNG call order had changed. It was replaced by the digest-based guard above.

**Environment resolved at T3:** Python 3.13.5, uv 0.11.8, numpy 2.5.3, pandas 3.0.6,
scikit-learn 1.9.1, fastapi 0.141.1, pydantic 2.13.5; ruff and mypy pinned through `uv.lock`.

**Observed signal for T10/T11 (not yet a finding):** the three strongest features by importance are
`reflejo_sistemas_culturales` (0.128), `malestar_generalizado` (0.117) and `desenganche` (0.106),
while the three features carrying a generator modifier — `racismo_sistemico` (0.020),
`alta_excitacion` (0.022) and `angustia_profunda` (0.042) — are the weakest. That is consistent with
the modifiers applying to every class equally, so they cannot discriminate. T10 must state whether
this holds on a held-out split.

---

### Phase 1 — Canonical schema — COMPLETE

| Task | Commit | Closure evidence observed |
|---|---|---|
| T5 | `cab17e7` | `assets/schema.json` verified faithful to both CSVs and to the model's pinned feature order before it was committed |
| T6 | `cab17e7` | `schema.py` is the only module that opens the file; both CSVs retired; baseline digests unchanged |
| T7 | `d95e97e` | 14 fields derived at runtime, `additionalProperties: false`, 0–1 bounds; legacy names rejected with their replacement |
| T8 | `ff1bba3` | Generator deterministic across runs (identical SHA-256); drift test fails on a stale file |

**Deviations from this plan, and why:**

- **T5 and T6 landed in one commit** (`cab17e7`). `tests/test_schema.py` carries both the data
  invariants and the loader tests; separating them would have required an intermediate commit with a
  half-tested loader. The shared test file is the reason, not the size.
- **T7 also corrected `0.2/README.md`**, which documented a `curl` request using the rejected legacy
  names. The strict contract would have answered that documented example with a 422: the
  documentation contradicted the code it documented.
- **`cli.py` changed its injectable flag** from `--data-dir` to `--schema-path`, because the
  injectable path now points at the schema rather than at a data directory.
- **`assets/estructura_aburrimiento.json` was renamed** to `assets/conceptual-framework.json`, content
  untouched. It is a theoretical taxonomy, and the old name invited it to be read as the runtime
  schema.

**Verified contract behavior — production evidence, not a claim:**

```
valid payload        -> 200 {"nivel":"alto"}
carencia_sentido     -> 422 "carencia_sentido is a rejected field name; use carencia_de_sentido"
desenganche = 1.5    -> 422 less_than_equal at body.datos.desenganche
inatencion missing   -> 422 missing at body.datos.inatencion
```

**Zero-duplication check:** the 14 indicator ids now appear in exactly one hand-written place, the
schema file itself. `src/aburrimiento/` derives them at runtime, and `web/src/generated/schema.ts` is
produced from the same source by a generator whose output is drift-checked.

**Test count:** 19 after T6, 29 after T8.

**Open item carried into Phase 4:** §6 condition 4 requires drift to fail a test in Python *and* in
TypeScript. The Python half exists (pytest drift test). The TypeScript half — a typecheck that breaks
when a field is renamed — cannot exist until the web app is scaffolded in T17.

---

### Phase 2 — Honest machine learning — COMPLETE

| Task | Commit | Closure evidence observed |
|---|---|---|
| T9 | `7f2a1c1` | Distributions, modifiers, seed and bounds are named data; RNG consumption unchanged, so the digests still match |
| T10+T11 | `535544b` | Measured on 3000 samples; `reports/` regenerated by `make evaluate`; no hand-written number anywhere |
| T12 | `d9cb26c` | The API loads the artifact in its lifespan; a schema mismatch refuses to serve; startup no longer trains |

**Measured results — 3000 samples, seed 42, stratified 1800/600/600, 5-fold CV:**

| Model | Test accuracy | Cohen's kappa | Macro F1 |
|---|---|---|---|
| Random forest (120 trees) | 0.9783 | 0.9675 | 0.9785 |
| Logistic regression | **0.9800** | **0.9700** | **0.9801** |
| Dummy (most frequent) | 0.3383 | 0.0000 | 0.1685 |

**The linear baseline matches the forest.** The committed twelve-seed sweep finds a mean difference
(LogReg - forest) of +0.222 percentage points with a 0.416-point standard deviation; logistic regression
wins 6 seeds, the forest 1, and 5 tie. The models are statistically indistinguishable here, so 120 trees
add no measured accuracy over a linear model. That is a statement about the generator and nothing about
boredom, which is what the report's own header says.

**D5 is closed by measurement, not by a cosmetic edit.** The scaled and unscaled runs produce
identical accuracy, kappa, macro F1 and confusion matrix: the `StandardScaler` is inert before a tree
model. It is nonetheless kept, because it is part of the trained artifact and of the byte-level
regression guard; removing it would shift `predict_proba` decimals and invalidate that guard for no
observable gain. An inert object with a measured justification beats an unverified deletion.

**A false signal from Phase 0, corrected.** With 300 samples, T4 observed
`reflejo_sistemas_culturales` as the most important feature (0.128). With 3000 samples it ranks ninth
(0.072). That was small-sample noise. It was recorded as a signal "not yet a finding" precisely so it
could be revised here rather than promoted.

**What the importances do show:** the two features carrying the most aggressive modifiers are the
least used — `racismo_sistemico` 0.0142 and `alta_excitacion` 0.0137, against
`frustracion_de_agencia` 0.1056. A modifier applied to every class equally compresses a feature
without separating classes, so the tree barely uses it. Same explanation, now with n large enough to
trust.

**Deviations, and defects found during execution:**

- **T10 and T11 landed in one commit** (`535544b`): both live in `evaluate.py` and produce a single
  report, so splitting them would have meant two commits editing the same file with one report.
- The first generated report serialized `notes` as a list of single characters. The Markdown renderer
  hid it, so the human-readable artifact looked right while the machine-readable one was wrong. Fixed,
  with a regression test that asserts entry shape — the assertion that would have caught it.
- The Markdown report was also too thin: it carried accuracy summaries only. It now renders per-class
  metrics, the confusion matrix with its label order, the cross-validation folds, and the feature
  importances sorted descending.
- `make train` regenerates `reports/`, so the artifact and the published numbers always describe the
  same run. The report timestamp therefore changes on every training run; the numeric content does
  not, and a test asserts that the report is reproducible apart from that timestamp.
- The artifact is ~2.3 MB and stays out of version control; `reports/` is committed because it is
  reviewable text.

---

### Phase 3 — API — COMPLETE (T16 moved to Phase 4)

| Task | Commit | Closure evidence observed |
|---|---|---|
| T13+T14 | `36484c9` | 70 tests; `/health` reports the served model's identity; CORS allows Vite and refuses an unrelated origin |
| T15 | `15f8b36` | 75 tests; capture verified against a real server and a real database file, then removed |

**Plan correction — T16 moves to Phase 4.** The original order put `make check` covering the web
pipeline at the end of Phase 3, but `web/` does not exist until T17. A verification target cannot be
completed before the thing it verifies. T16 now runs after T17, where web lint, typecheck, test and
build can actually be wired in. This was a dependency error in the plan, found by executing it.

**The `0.2/` tree is gone.** Its API was replaced, its README described a layout that no longer
existed, and keeping it would have preserved a second unverified copy of the contract. Removing it also
removed its ruff exclusion, which is what turns a scheduled deletion into an actual one.

**What that removal exposed:** ruff formats the Python code inside `blueplan.md`'s fenced blocks, so an
aspirational document was gating `make check` the moment `0.2` was no longer excluded beside it.
Confirmed directly — `ruff format --check blueplan.md` reports "1 file would be reformatted". The
exclusion is now explicit, documented next to the rule, and written as a glob so T24 can move the file
without silently breaking verification again.

**What `/health` reports, verbatim from a real server:**

```json
{
  "status": "ok",
  "schema_version": "1.0.0",
  "schema_sha256": "704953ce41cfab640dc709eb83714fae37f6a47e5e0ffdf5710783dc9cb49323",
  "trained_at": "2026-09-20T02:52:30.723973+00:00",
  "n_samples": 3000,
  "seed": 42,
  "sklearn_version": "1.9.1",
  "metrics": {"accuracy": 0.9783, "kappa": 0.9675, "macro_f1": 0.9785}
}
```

A caller can tell which model is being served, and what it scored, without opening a file.

**CORS is verified rather than asserted:** a preflight from `http://localhost:5173` returns
`access-control-allow-origin: http://localhost:5173`, while a preflight from an unrelated origin
returns no allow-origin header at all. `allow_origins=["*"]` would have made that test impossible to
write.

**The capture path exists.** `POST /samples` stores the 14 values, the predicted level, an optional
observed level and a UTC timestamp; `GET /samples/stats` aggregates them. The table's 18 columns are
derived from the schema at runtime, and `store.py`'s docstring declares what is deliberately not
stored: no identifiers, names, free text, IP addresses or device information.

**Deviations:**

- T13 and T14 landed in one commit (`36484c9`): the application and its contract suite are one
  artifact, and a commit carrying an untested app would have been worse than a larger commit.
- A worker reported `make check` green while it was red (RUF043, an unescaped regex in a test's
  `match=`). The parent re-ran the check and caught it. Reported outcomes are re-verified here, not
  trusted.
- A commit message for T15 was amended once to remove literal backslash-escapes from a code sample;
  the recorded hash is the post-amend one.

---

## 10. Test policy — resolved 2026-09-20

A worker correctly refused to start without a declared TDD mode, source and runner. The gap was in the
delegation, not in the worker: no task brief in this plan had ever declared them. Resolved here once, so
no future task has to ask.

| Field | Value |
|---|---|
| TDD mode | `disabled` — nothing in this repository enables strict TDD |
| Source | none — no configuration file, no CLI setting, nothing to read |
| Python runner | `uv run pytest` |
| Web runner | `npm test` (from T17 onward) |
| Single verification gate | `make check` |
| Test requirement | **independent of the mode** |

The mode being `disabled` means tests are not written before the code. It never means tests are
optional: **every task closes with tests, and a task whose tests do not exercise its own claim is not
closed.** The distinction matters because a disabled mode is easy to misread as permission.

Evidence for the `disabled` value rather than an assumption: `~/.config/gentle-ai` does not exist on
this machine, and `gentle-ai help` lists `install`, `uninstall`, `sync`, `skill-registry`, the `sdd-*`
commands and `review`, with no TDD setting among them. Nothing in the repository declares one either.

**Consequence for every remaining delegation:** the task brief states the mode, the runner and the gate
explicitly. A worker should never have to ask and never have to guess.

**Consequence for review:** a task whose closure criterion is a test that would pass without its
feature is a defect in the criterion, not a passing task. Two such cases were caught and replaced
during execution: the T4 regression guard, which initially pinned only arithmetic and three widely
separated labels, and the T10 report serializer, whose `notes` field was emitted as a list of single
characters while the Markdown renderer hid it.

---

### Phase 4 — Web frontend — COMPLETE (T16 absorbed here)

| Task | Commit | Closure evidence observed |
|---|---|---|
| T17 | `1b84740` | lint, typecheck, 5 tests and a production build pass; no indicator id typed by hand outside the generated file |
| T18+T19+T20 | `e143874` | 16 tests; client typed `Record<IndicatorId, number>`; zero host literals outside `config.ts` |
| T21+T16 | `62c15a4` | 97 tests; the contract lock fails on a renamed field; `make check` runs both halves |

#### §6 condition 4 — corrected by measurement

The definition of done claimed that "any divergence fails a test in Python **and** in TypeScript".
Measured, the two halves divide the work differently than assumed:

- **Renaming an id inside the generated file does NOT fail the TypeScript typecheck.** No hand-written
  code names an indicator: the UI derives its groups, fields, labels, range and level meanings from
  `generated/schema.ts` and adapts by itself. That is the intended outcome of a single schema, not a
  gap. The earlier expectation that the typecheck would catch such a rename was simply wrong.
- **What Python catches:** a stale generated artifact. `tests/test_web_types_drift.py` fails with two
  assertions naming the missing id when the file on disk is not what the generator would emit.
- **What TypeScript catches:** a hand-written id or a wrong field type. Verified by injecting
  `const probe: IndicatorId = "indicador_que_no_existe"` — the typecheck rejects it and lists all
  fourteen valid ids in the error message.
- **What neither catches, and the contract lock does:** a contract change that regenerates cleanly.
  `web/src/contract.golden.json` is a hand-maintained approval lock, deliberately not generated, so a
  genuine change requires updating it on purpose. Verified by renaming a field in
  `assets/schema.json`: the web suite fails naming the offending field, and restoring the schema
  returns the generated file to the identical sha256.

Corrected condition: **one schema source; the generated artifact cannot go stale (Python), the code
cannot name a field that does not exist (TypeScript), and a deliberate contract change must update
the approval lock (the golden).**

#### The Flutter defect is closed

The archived client hardcoded `127.0.0.1:8000` while its own README documented `10.0.2.2` for the
Android emulator. The replacement reads `VITE_API_URL`, validates it, falls back with a recorded
reason instead of blanking the page, and documents the LAN case including the `HOST=0.0.0.0`
requirement. A test asserts the default still resolves when the variable is absent.

#### Two defects found by verifying rather than trusting

- **Vitest's `setupFiles` pointed at the test file** instead of the setup module, so Vitest imported
  the suite as setup and then ran it again as a suite. The reported "10 tests passed" was five tests
  executed twice, and the run took 27 seconds instead of 5.2. The inflated count *was* the bug. The
  repaired suite reports the true number in a fifth of the time.
- **The harness safety policy refuses to write dot-prefixed env paths**, so `.env.example` was
  impossible even for the parent. Escalated instead of worked around: the user chose `web/env.example`.
  The purpose is unchanged — a human copies it, Vite never reads it, and a test proves the default
  still resolves without it.

#### Deviations

- T18, T19 and T20 landed in one commit (`e143874`): the client, its configuration and the UI are one
  flow, and the configuration is what the client reads.
- T21 and T16 landed together (`62c15a4`): the contract lock and the gate are one closing gesture —
  a lock is only enforced if a single command runs it.
- `web/src/test-setup.ts` was created by a worker outside its declared surfaces, to satisfy a setup
  file the task asked for but failed to list. It was reported rather than hidden, and it is kept.

#### Totals at the end of Phase 4

97 tests — 75 Python, 22 web — green under one command. 242 files changed across the branch,
9,929 insertions and 2,309 deletions, most of the churn being the 158 archived files and the deleted
`0.2/` tree.

---

## 11. Native review: what it covered and what it could not — 2026-09-20

The review switch is enabled, so the preflight was run. The outcome is recorded here because the
limit belongs to the tooling, not to the work.

### What was reviewed

The pending change to this plan document was submitted as a workspace candidate. It closed
**approved**: risk tier `low`, single reason `non_executable_only`, zero lenses required, no consent
required, and its authority was acknowledged and burned
(`burn_evidence: gentle-ai.review-acknowledged/v1`). The mechanism works on candidates that fit.

### What could not be reviewed

The committed range `main..HEAD` — 22 commits, 9,929 insertions — was offered by the provider as a
`base-diff` candidate and refused with:

```
code: lens_context_budget_exceeded
phase: preflight
mutation_outcome: not_started
authority_applicability: not_evaluated
```

No authority was created, so nothing was abandoned or repaired, and the failure is retry-safe. The
provider's own explanation: the complete reviewer evidence exceeds the native context budget, that
evidence is **never truncated**, and retrying the identical candidate cannot succeed. Its suggested
remedy is to split the work into a chained sequence of smaller reviewable commits.

**Why slicing by phase was not an option either.** The committed projection always spans
`baseRef → HEAD`; there is no way to fix the upper bound. An earlier proposal in this plan's
conversation — review phase by phase — was never implementable, and offering it was a mistake. What
actually inflates the candidate is not the new code but the 158 files archived out of `main` plus the
deleted `0.2/` tree spanning five platform directories: hundreds of file-level entries that are
moves and removals, not work.

### Decision taken

Review the candidates that fit from here on — the Phase 5 documents are new, small, self-contained
files — and leave the historical range covered by the verification actually performed per task
rather than by a lens review that cannot hold it:

- the migration in T4, proven by byte-identical dataset digests, feature importances and probabilities
  against the legacy module recovered from git, with the guard mutation-tested;
- the canonical schema swap in T5/T6, proven by those same digests staying identical;
- the strict contract in T7, proven by four verbatim 422 responses each naming the offending field;
- the honest evaluation in T10/T11, where every published number comes from a committed command;
- the artifact loading in T12, proven against a real server with a schema digest that refuses a
  mismatched model;
- the frontend contract in T21, proven by a rename that fails the suite and a restore that returns the
  exact sha256.

That is **not** equivalent to an independent four-lens review, and it is not recorded as if it were.
It is the evidence that exists, stated at the strength it actually has.

### Consequence for the remaining tasks

Each remaining task is implemented and then submitted as its own workspace candidate **before** it is
committed, so the review sees a small diff instead of an accumulating branch.

---

### Phase 5 — Documentation and closure — COMPLETE

| Task | Commit | Closure evidence observed |
|---|---|---|
| T22 | `40924e4` | README table generated, drift test fails on a stale row, reviewed by all four lenses |
| T23 | `9a6b9d2` | Four ADRs carrying the verified numbers; index links resolve |
| T24 | `924df2b` | 2,652-line body untouched; the lint exclusion's cause pinned down by removing it |
| T25 | this commit | The end-to-end run below, pasted verbatim |

**All 25 tasks are closed**, each with a commit hash and an observed closure criterion.

#### The end-to-end run, verbatim

```
$ make train
uv run python -c 'from aburrimiento.artifacts import main; main()' --samples 3000 --seed 42
Saved model artifact to models/model.joblib          # 2,291,589 bytes

$ uv run uvicorn aburrimiento.api:app --host 127.0.0.1 --port 8199

$ curl -s http://127.0.0.1:8199/health
{
    "status": "ok",
    "schema_version": "1.0.0",
    "schema_sha256": "704953ce41cfab640dc709eb83714fae37f6a47e5e0ffdf5710783dc9cb49323",
    "trained_at": "2026-09-20T16:08:53.511063+00:00",
    "n_samples": 3000,
    "seed": 42,
    "sklearn_version": "1.9.1",
    "metrics": {
        "accuracy": 0.9783333333333334,
        "kappa": 0.96749715600115,
        "macro_f1": 0.9784925048082943
    }
}

$ curl -s -X POST http://127.0.0.1:8199/analyze -H 'Content-Type: application/json' \
    -d '{"datos":{"reflejo_sistemas_culturales":0.9,"productividad_capitalista":0.85,"alienacion_neoliberal":0.9,"racismo_sistemico":0.7,"malestar_generalizado":0.9,"carencia_de_sentido":0.85,"restriccion_de_libertad":0.8,"frustracion_de_agencia":0.9,"desenganche":0.9,"alta_excitacion":0.6,"inatencion":0.85,"percepcion_tiempo_lenta":0.9,"estrategias_bloqueadas":0.9,"angustia_profunda":0.85}}'
{"nivel":"alto"}

$ curl -s -X POST http://127.0.0.1:8199/samples -H 'Content-Type: application/json' \
    -d '{"datos":{"reflejo_sistemas_culturales":0.2,"productividad_capitalista":0.25,"alienacion_neoliberal":0.2,"racismo_sistemico":0.3,"malestar_generalizado":0.25,"carencia_de_sentido":0.2,"restriccion_de_libertad":0.3,"frustracion_de_agencia":0.25,"desenganche":0.2,"alta_excitacion":0.3,"inatencion":0.25,"percepcion_tiempo_lenta":0.2,"estrategias_bloqueadas":0.25,"angustia_profunda":0.3},"nivel_observado":"bajo"}'
{"id":1,"nivel":"bajo"}

$ curl -s http://127.0.0.1:8199/samples/stats
{
    "total": 1,
    "by_predicted_level": {"bajo": 1, "medio": 0, "alto": 0},
    "by_observed_level": {"bajo": 1, "medio": 0, "alto": 0}
}

$ curl -s -X POST http://127.0.0.1:8199/analyze -H 'Content-Type: application/json' \
    -d '{"datos":{"reflejo_sistemas_culturales":0.5,"carencia_sentido":0.5}}'
{"detail":[{"type":"value_error","loc":["body","datos"],
  "msg":"Value error, carencia_sentido is a rejected field name; use carencia_de_sentido"}]}
```

What the run demonstrates, in order: the artifact is **loaded rather than trained**, and
`/health` reports which model is being served together with what it measured; a full 14-field
analysis returns a level; the capture path stores a real observation with a human-supplied label; the
aggregate reflects both the prediction and the label; and a rejected legacy name is answered with the
canonical replacement rather than silently accepted.

`make check` at the end of the branch: **99 tests** — 77 Python and 22 web — covering ruff format and
lint, mypy in strict mode, pytest, and the web lint, typecheck, tests and production build.

---

## 12. Remaining debt, stated plainly

Everything below is known, deliberate, and not hidden behind a passing suite.

| Item | Why it stands |
|---|---|
| The comparison claim was corrected | The committed twelve-seed sweep finds the models statistically indistinguishable: logistic regression wins 6 seeds, the forest 1, and 5 tie, with a +0.222-point mean difference and 0.416-point standard deviation. The linear model matches 120 trees, so there is no measured reason to change the served model. |
| The historical range was never lens-reviewed | `main..HEAD` exceeds the native reviewer's context budget. Phases 0–4 are covered by per-task verification — digests, mutation-tested guards, verbatim 422s, a frozen contract lock — and by nothing else. §11 states this at exactly the strength it has. Only T22 passed four lenses. |
| `StandardScaler` is kept although measured inert | Scaled and unscaled runs produce identical accuracy, kappa, macro F1 and confusion matrix. It survives because it is part of the artifact and of the byte-level guard; removing it would shift `predict_proba` decimals and invalidate that guard for no observable gain. |
| The capture path has no interface | `POST /samples` is implemented, tested and verified against a real database, but nothing in the frontend calls it. A route exists; a workflow does not. |
| `httpx`/`starlette` deprecation warning | Emitted by the installed stack during `TestClient` use, twice per run. Not from this repository's code. |
| `legacy/` holds 158 files | Archived by decision, not by accident. It is inert: nothing builds, imports or lints it, and `legacy/README.md` says why each iteration was abandoned. |
| No real-world validity | The generator's labels come from the same distributions the model learns. The README, the report header, the ADR and the generator's docstring each say so. Nothing here is evidence about people. |

---

## 13. Post-closure work — 2026-09-20

Two tasks were added after the plan closed, both taken from the debt table in §12. They are numbered
T26 and T27 so the log stays ordered; the count is now **27 tasks**, not 25.

| Task | Commit | What it closed |
|---|---|---|
| T26 | `f502352` | A published claim the evidence did not support |
| T27 | `b23ba6b` | The capture path having a route but no workflow |

### T26 — correcting an overclaim, including my own

The README, ADR 0003 and §9 of this plan all stated that the linear baseline **"beats"** the 120-tree
forest. That rested on a single seed where the margin was **one sample out of 600**: 587 against 588
correct, with the two confusion matrices differing in exactly one cell.

Re-measured across twelve seeds with the same splits, now reproducibly through `make evaluate`:

| Metric | Value |
|---|---|
| Mean difference (logistic − forest) | +0.222 percentage points |
| Standard deviation | 0.416 percentage points |
| Seeds won by logistic regression | 6 |
| Seeds won by the forest | 1 |
| Ties | 5 |

The mean is smaller than its own dispersion, and both models win on some seeds. The supported
statement is that the two are **statistically indistinguishable on this generator**. What survives is
the useful part: a linear model *matches* 120 trees, so the forest buys no accuracy for its
complexity.

This matters more than the correction itself. The project's premise is that every published number is
reproducible and honestly read — and the same process that removed the previous iteration's invented
88–92 % produced this overclaim from a one-sample margin. It was caught by re-measuring rather than by
review, and the sweep is now part of the evaluator so the next reader does not have to trust a single
run. A test encodes the rule with a name:
`test_seed_sweep_does_not_claim_winner_when_mean_is_smaller_than_std`.

The §12 row calling the served model "the worse of the two measured" is therefore gone. With the two
indistinguishable there is no measured reason to change it, so that debt is closed rather than
carried.

### T27 — the capture path gets a workflow

`POST /samples` was implemented, tested and verified against a real database, but nothing in the
frontend called it. The analysis screen now offers to keep an observation: the three levels read from
the generated schema, an explicit "prefer not to say" that sends no `nivel_observado` rather than an
empty string, a confirmation announced through an `aria-live` region, and one sentence on screen
stating what is stored and what is not. The panel does not appear before an analysis has run, and
re-running the analysis clears the previous confirmation.

**Test count is now 108: 79 Python, 29 web.**
