# 0002 — Make `assets/schema.json` the only hand-written definition of the schema

## Context

The same 14 indicators existed in four places, and they disagreed. `estructura_indicadores.csv` held
Spanish labels that the code slugified into field names such as `carencia_de_sentido`; the archived Dart
clients sent the shorter `carencia_sentido`; the `0.1` prototype used the short form throughout; and
`0.2/api/main.py` carried an `alias_map` that copied each short name onto its canonical counterpart so
that all four survived. That helper never removed the alias it copied, and it computed a missing-field
check that it then discarded on both return paths — dead code whose only purpose was to keep four
diverging copies working.

## Decision

`assets/schema.json` is the single hand-written definition: the 14 indicators with their canonical ids,
groups, display order, labels, theoretical basis, the three levels, and the names that are deliberately
rejected. `src/aburrimiento/schema.py` is the only module that opens it. The HTTP contract derives its
fields from it at runtime with `pydantic.create_model`, and `tools/gen_web_types.py` generates
`web/src/generated/schema.ts` from it. The three archived short names are rejected with a 422 that
names the canonical replacement.

## Consequences

- The 14 indicator ids appear in exactly one hand-written place. Adding an indicator to the JSON adds
  the field, its validation, its OpenAPI entry and its TypeScript type with no code edit.
- A generated artifact that goes stale fails `tests/test_web_types_drift.py`; a hand-written id that
  does not exist fails the TypeScript type check; and a deliberate contract change must update the
  frozen approval lock in `web/src/contract.golden.json`.
- Removing the alias map broke the old field names on purpose. The only consumers were the archived
  clients, verified before the change.

## What would change this decision

A second consumer that genuinely cannot read the schema, for example a third-party integration with a
fixed wire format. It would get a documented compatibility layer, not a second hand-written schema.
