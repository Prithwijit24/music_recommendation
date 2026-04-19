# Quickstart

## 1. Create the environment

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## 2. Train the pipeline

```bash
uv run music-rec train
```

Artifacts are written into `artifacts/models` and `artifacts/reports`.

## 3. Launch the UI

```bash
uv run streamlit run streamlit_app.py
```

## 4. Optional interaction data

To enable collaborative filtering, hybrid ranking, and realistic deep recommendation workflows, add a CSV under `data/` with these columns:

```text
user_id,track_id,rating
```

Then set `interactions_csv` in `conf/config.yaml`.
