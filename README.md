# aburrimiento — group boredom analyzer

**Status: under construction.** The project is mid-migration from two half-finished prototypes
(`0.1` and `0.2`, now archived under `legacy/`) to a single, locally verifiable application. This
README is a placeholder; the complete version arrives with the documentation phase of the plan.

## What this is

A small analyzer that maps 14 sociological indicators — grouped into systemic structures, group
manifestations and measurement dimensions — onto a three-level classification: low, medium, high.

## What this is NOT

- **Not a validated instrument.** The model trains on synthetic data whose labels are drawn from the
  same distributions the model learns. Reported accuracy measures the self-consistency of that
  generator, not real-world boredom. No claim about people, groups or reality is supported here.
- **Not production software.** No authentication, no multi-tenancy, no deployment target, no service
  level objective. It runs on one machine, for one user.
- **Not a diagnostic tool.** Nothing here should inform a decision about a person or a group.

## Project plan

The authoritative plan is `odd/tasks/aburrimiento-v1.md`: 25 tasks in 6 phases, each with a
verification criterion. Why the earlier iterations were abandoned is documented in `legacy/README.md`.

## Target layout

Some of these paths do not exist yet; the plan's phases create them.

| Path | Contents |
|---|---|
| `src/aburrimiento/` | Python package: schema, synthetic data, model, evaluation, storage, API, CLI |
| `tests/` | Test suite |
| `assets/` | The canonical schema — single source of truth for indicators and levels |
| `web/` | React + Vite + TypeScript frontend (replaces the archived Flutter client) |
| `tools/` | Code generators |
| `docs/` | Architecture notes and decision records |
| `odd/tasks/` | The feature plan |
| `legacy/` | Archived iterations — nothing here is built, linted or tested |

## Development

```bash
make setup   # install Python dependencies from the lockfile
make check   # lint, type-check and test
make help    # list every target
```
