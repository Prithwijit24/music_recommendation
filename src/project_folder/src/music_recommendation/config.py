from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class PathsConfig(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "artifacts"
    models_dir: Path = artifacts_dir / "models"
    reports_dir: Path = artifacts_dir / "reports"
    processed_dir: Path = artifacts_dir / "processed"


class DatasetConfig(BaseModel):
    content_csv: str = "YearPredictionMSD.csv"
    interactions_csv: str | None = None
    sample_size: int = 15000
    random_state: int = 42
    top_k: int = 10


class TrainConfig(BaseModel):
    test_size: float = 0.2
    knn_neighbors: int = 10
    latent_factors: int = 32
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def get_config() -> AppConfig:
    config_path = PathsConfig().project_root / "conf" / "config.yaml"
    if not config_path.exists():
        return AppConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        dataset=DatasetConfig(**payload.get("dataset", {})),
        train=TrainConfig(**payload.get("train", {})),
    )
