from __future__ import annotations

import json

import typer

from music_recommendation.config import get_config
from music_recommendation.pipelines.training import run_training_pipeline

app = typer.Typer(help="Music recommendation system CLI.")


@app.command()
def train() -> None:
    """Run the training pipeline and persist artifacts."""
    results = run_training_pipeline(get_config())
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
