from music_recommendation.config import get_config
from music_recommendation.data.loaders import load_content_data
from music_recommendation.features.preprocessing import build_content_features
from music_recommendation.models.content_based import ContentBasedRecommender


def test_content_data_loading() -> None:
    config = get_config()
    dataset = load_content_data(config)
    assert "track_id" in dataset.frame.columns
    assert "year" in dataset.frame.columns
    assert len(dataset.feature_columns) > 0


def test_content_recommender_returns_results() -> None:
    config = get_config()
    dataset = load_content_data(config)
    bundle = build_content_features(dataset, latent_dims=8)
    model = ContentBasedRecommender(top_k=5).fit(dataset, bundle)
    seed_item = dataset.frame.iloc[0]["track_id"]
    recommendations = model.recommend(seed_item)
    assert not recommendations.empty
    assert seed_item not in recommendations["track_id"].tolist()
