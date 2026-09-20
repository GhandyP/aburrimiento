# 0001 — Replace the Flutter client with React + Vite + TypeScript

## Context

The project carried two abandoned Dart clients. The `0.1` one never called the backend at all: it
computed three averages locally and applied thresholds. The `0.2` Flutter app did call `/analyze`, but
it hardcoded `const String apiBaseUrl = 'http://127.0.0.1:8000'` while its own README documented
`http://10.0.2.2:8000` for the Android emulator and a LAN address for physical devices — so the
documented workflows could not work without editing source. Its only test was Flutter's default counter
stub: it asserted `MyApp` and `Icons.add`, while the application defines `BoredomApp` and has no
counter. The one test that existed could never pass.

## Decision

Delete both Dart clients, archive them under `legacy/`, and build the frontend as a React application
with Vite and TypeScript under `web/`. The archived code is kept, with the reasons, in
`legacy/README.md`.

## Consequences

- Node and npm enter the repository, along with a build step and a lockfile.
- The frontend now derives its types from the generated schema, so a schema change is caught by the
  TypeScript type checker when hand-written code names a field that no longer exists.
- The API base URL is configuration (`VITE_API_URL`) with a validated default, and the LAN case is
  documented including the `HOST=0.0.0.0` requirement the API needs to accept outside connections.
- The web suite runs under `make check` alongside the Python suite, so one command decides the truth.

## What would change this decision

A genuine requirement to ship native mobile or desktop binaries. Nothing in the current scope needs
one: the documented workflow is a browser, sometimes a phone on the same LAN.
