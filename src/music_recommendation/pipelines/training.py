from __future__ import annotations

from dataclasses import asdict

from music_recommendation.config import AppConfig, get_config
from music_recommendation.data.loaders import load_content_data, load_interaction_data
from music_recommendation.features.preprocessing import build_content_features
from music_recommendation.models.collaborative import CollaborativeFilteringRecommender
from music_recommendation.models.content_based import ContentBasedRecommender
from music_recommendation.models.deep_models import DeepContentTrainer
from music_recommendation.models.hybrid import HybridRecommender
from music_recommendation.models.ml_models import YearRegressionBaseline
from music_recommendation.services.evaluation import summarize_recommendations
from music_recommendation.utils.io import ensure_dir, save_joblib, save_json


def run_training_pipeline(config: AppConfig | None = None) -> dict:
    config = config or get_config()
    ensure_dir(config.paths.artifacts_dir)
    ensure_dir(config.paths.models_dir)
    ensure_dir(config.paths.reports_dir)

    content_dataset = load_content_data(config)
    feature_bundle = build_content_features(content_dataset, latent_dims=config.train.latent_factors)

    content_model = ContentBasedRecommender(top_k=config.dataset.top_k).fit(
        content_dataset, feature_bundle
    )
    seed_item = content_dataset.frame.iloc[0]["track_id"]
    content_recs = content_model.recommend(seed_item)

    interaction_dataset = load_interaction_data(config)
    collaborative_status = {"available": interaction_dataset is not None}
    collaborative_model = None
    if interaction_dataset is not None:
        collaborative_model = CollaborativeFilteringRecommender(
            latent_factors=config.train.latent_factors,
            top_k=config.dataset.top_k,
        ).fit(interaction_dataset)

    hybrid_model = HybridRecommender(content_model, collaborative_model, top_k=config.dataset.top_k).fit()
    hybrid_recs = hybrid_model.recommend(seed_item)

    ml_result = YearRegressionBaseline().fit_and_evaluate(content_dataset)
    dl_result = DeepContentTrainer(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
    ).fit(content_dataset)

    save_joblib(config.paths.models_dir / "content_model.joblib", content_model)
    save_joblib(config.paths.models_dir / "feature_bundle.joblib", feature_bundle)
    if collaborative_model is not None:
        save_joblib(config.paths.models_dir / "collaborative_model.joblib", collaborative_model)
    save_json(
        config.paths.reports_dir / "training_summary.json",
        {
            "content_summary": summarize_recommendations(content_recs),
            "hybrid_summary": summarize_recommendations(hybrid_recs),
            "collaborative": collaborative_status,
            "ml_metrics": asdict(ml_result),
            "dl_metrics": asdict(dl_result),
        },
    )
    return {
        "seed_item": seed_item,
        "content_preview": content_recs.head(5).to_dict(orient="records"),
        "hybrid_preview": hybrid_recs.head(5).to_dict(orient="records"),
        "collaborative": collaborative_status,
        "ml_metrics": asdict(ml_result),
        "dl_metrics": asdict(dl_result),
    }
