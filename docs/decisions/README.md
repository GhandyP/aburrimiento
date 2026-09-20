# Architecture decision records

Short records of decisions that shaped this project, written so a future reader can tell what was
decided, what it cost, and what would reverse it. Each one records a decision that was actually taken
and is visible in the code.

| Record | Decision |
|---|---|
| [0001](0001-drop-flutter-for-react.md) | Replace the Flutter client with React + Vite + TypeScript |
| [0002](0002-one-schema-source.md) | Make `assets/schema.json` the only hand-written definition of the schema |
| [0003](0003-honest-synthetic-data.md) | Keep synthetic data, measure it honestly, and open a path to real observations |
| [0004](0004-locally-verifiable-scope.md) | Cap the scope at what can be executed and verified on one machine |

The plan that produced this state, including its deviations and the evidence for each task, is
`odd/tasks/aburrimiento-v1.md`.
