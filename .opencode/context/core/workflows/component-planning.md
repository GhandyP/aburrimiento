# Component Planning Workflow

Use this template and process when planning components.

## When to Create a Component Plan
- The work touches multiple files or modules.
- The work adds a new capability or API surface.
- The work introduces new dependencies or data flows.

## Component Plan Template
Create a file named `component-<name>.md` with the following sections.

```
# Component Plan: <Name>

## Purpose
<Why this component exists and the problem it solves.>

## Scope
<What is in scope and explicitly out of scope.>

## Inputs
<Key inputs, data sources, or dependencies.>

## Outputs
<Files, artifacts, APIs, or data produced.>

## Interface
<Public classes, functions, endpoints, or CLI usage.>

## Tasks
1. <Atomic task>
2. <Atomic task>

## Validation
- <Command or check to validate behavior.>
```

## Rules
- Keep tasks atomic and independently verifiable.
- Order tasks by dependencies.
- Avoid implementation details in Purpose and Scope.
- Include at least one validation step.
