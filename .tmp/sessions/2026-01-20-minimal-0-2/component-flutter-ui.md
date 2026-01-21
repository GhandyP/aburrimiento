# Component Plan: Flutter UI Stub

## Purpose
Provide a minimal Flutter app with a single screen to collect indicator inputs, call the API, and display the prediction.

## Scope
- Single screen with 14 numeric inputs.
- HTTP POST to `/analyze` endpoint.
- Display result text.

## Inputs
- API base URL (default localhost).
- 14 indicator labels from `0.2/data/estructura_indicadores.csv`.

## Outputs
- `0.2/flutter/lib/main.dart`
- `0.2/flutter/pubspec.yaml`

## Interface
- Button triggers API call.
- Minimal error message on failed request.

## Tasks
1. Create `0.2/flutter/` project skeleton (minimal `pubspec.yaml`).
2. Implement `lib/main.dart` with inputs and request logic.
3. Provide a simple UI layout that works on mobile and desktop.

## Validation
- `flutter pub get` completes.
- `flutter run` launches (manual validation).
