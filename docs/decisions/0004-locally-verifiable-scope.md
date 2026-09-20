# 0004 — Cap the scope at what can be executed and verified on one machine

## Context

`blueplan.md` is a 2,652-line production blueprint: Kubernetes, Triton inference server, Redis, S3,
a model registry, GPU optimization, Prometheus and a full MLOps pipeline. None of it was implemented,
and most of it cannot be executed or checked on a single development machine. It was the largest
document in the repository and the least connected to anything that ran — at one point it was even
gating `make check`, because ruff formats the Python inside its fenced code blocks.

## Decision

Scope is capped at what can be executed and verified locally. No containers, no Kubernetes, no cloud
services, no CI provider. Every command in the README must be runnable, and every number it quotes must
be reproducible by a committed command. `blueplan.md` is reclassified as aspirational reference rather
than a roadmap.

## Consequences

- One command, `make check`, decides the truth for the whole repository: ruff format check, ruff lint,
  mypy in strict mode, pytest, and the web lint, typecheck, tests and production build.
- The blueprint stays in the repository as background reading, with a status banner and a map from its
  sections to what actually exists. It is excluded from linting by an explicit glob rather than by
  accident.
- Work that cannot be verified here is not claimed here. The trade-off is accepted deliberately: a
  smaller, checkable system over a larger, unverifiable description of one.

## What would change this decision

Real users and a real deployment target. That decision would bring its own requirements — availability,
authentication, data protection — and its own verification, neither of which this scope pretends to
provide.
