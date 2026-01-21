# Master Plan: Minimal 0.2 Working Version

## Goal
Create a new `0.2/` folder with a minimal, working version of the boredom analyzer that includes Python analysis, a simple API, a Flutter UI stub, and required data assets. The system should run end-to-end with synthetic data and a basic user flow.

## Scope
- Keep the implementation minimal but functional.
- Reuse compatible assets from `0.1/` when possible.
- Provide a simple CLI or script to run the Python pipeline.
- Provide a minimal REST API endpoint to analyze a request.
- Provide a minimal Flutter screen that collects inputs and calls the API.

## Architecture Overview
- **Data Assets**: JSON/CSV definitions and example datasets.
- **Python Core**: Data generation, preprocessing, model training, prediction, and results formatting.
- **API Layer**: Minimal FastAPI app exposing a prediction endpoint that uses the Python core.
- **Flutter UI**: Single-screen input form posting to the API and showing the response.
- **Docs**: Minimal README with run steps.

## Components (Dependency Order)
1. **Data Assets**
   - Collect/clean minimal CSV/JSON inputs needed by the pipeline.
   - Ensure assets are referenced by the Python core.

2. **Python Core Pipeline**
   - Minimal module for loading assets, generating sample data, training a model, and predicting.
   - Provide a CLI entry point to run a sample analysis.

3. **API Service**
   - FastAPI app with `/health` and `/analyze` endpoints.
   - Uses Python core pipeline for inference.

4. **Flutter UI Stub**
   - Single screen for entering 14 indicator values.
   - HTTP call to `/analyze` and display of prediction.

5. **Project Wiring & Docs**
   - Minimal `README.md` with steps to run backend and Flutter.
   - Confirm file paths, environment assumptions, and example requests.

## Validation
- Run Python CLI pipeline and verify prediction output.
- Start API service and validate `/health` and `/analyze`.
- Run Flutter app (or provide run steps if not executed).

## Open Questions
- Confirm whether to bundle Python dependencies in `requirements.txt` within `0.2/`.
- Confirm preferred API base URL for the Flutter app.
