# Code Quality Standards

These standards apply to all code changes in this repository.

## Core Principles
- Prefer clarity over cleverness; optimize for maintainability.
- Keep functions small, focused, and single-purpose.
- Avoid premature abstraction; extract only when reuse is clear.
- Use descriptive names; avoid single-letter variables except for small loops.
- Minimize side effects; favor pure functions where practical.

## Structure
- Keep related logic together; avoid cross-module entanglement.
- Prefer composition over inheritance.
- Isolate IO at module edges; keep core logic deterministic.

## Types and Interfaces
- Use strong typing where supported; avoid `Any` or overly broad types.
- Define explicit interfaces for external boundaries (CLI, API, file IO).
- Validate inputs at boundaries; return explicit errors for invalid data.

## Error Handling
- Fail fast with clear error messages; avoid silent failures.
- Do not swallow exceptions; handle or rethrow with context.
- Keep error messages user-readable and actionable.

## Testing
- Write tests for critical logic paths and boundary conditions.
- Prefer deterministic tests; avoid randomness without fixed seeds.
- Keep tests fast and scoped; avoid integration tests when unit tests suffice.

## Performance and Dependencies
- Avoid unnecessary dependencies; justify each new dependency.
- Keep algorithms simple; measure before optimizing.

## Formatting and Style
- Follow existing file conventions for formatting and naming.
- Keep comments minimal and high-signal; code should be self-explanatory.
- Avoid non-ASCII characters unless the file already uses them.
