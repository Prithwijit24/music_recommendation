from __future__ import annotations

import json

import streamlit as st

from music_recommendation.config import get_config
from music_recommendation.data.loaders import load_content_data, load_interaction_data
from music_recommendation.features.preprocessing import build_content_features
from music_recommendation.models.collaborative import CollaborativeFilteringRecommender
from music_recommendation.models.content_based import ContentBasedRecommender
from music_recommendation.models.hybrid import HybridRecommender
from music_recommendation.models.ml_models import YearRegressionBaseline


st.set_page_config(page_title="PulseMix Recommender", page_icon="🎵", layout="wide")

config = get_config()
content_dataset = load_content_data(config)
feature_bundle = build_content_features(content_dataset, latent_dims=config.train.latent_factors)
content_model = ContentBasedRecommender(top_k=config.dataset.top_k).fit(content_dataset, feature_bundle)
interaction_dataset = load_interaction_data(config)
collaborative_model = (
    CollaborativeFilteringRecommender(
        latent_factors=config.train.latent_factors,
        top_k=config.dataset.top_k,
    ).fit(interaction_dataset)
    if interaction_dataset is not None
    else None
)
hybrid_model = HybridRecommender(content_model, collaborative_model, top_k=config.dataset.top_k).fit()

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 133, 81, 0.25), transparent 22%),
            radial-gradient(circle at top right, rgba(58, 134, 255, 0.18), transparent 30%),
            linear-gradient(160deg, #0c1821 0%, #1b2a41 50%, #324a5f 100%);
        color: #f6f1e9;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: rgba(12, 24, 33, 0.68);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>PulseMix</h1>
        <p>Production-style music recommendation demo spanning content, collaborative, hybrid, ML, and deep-learning layers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 1.8])
with left:
    item_id = st.selectbox("Anchor track", options=content_dataset.frame["track_id"].head(300).tolist())
    top_k = st.slider("Recommendations", min_value=5, max_value=20, value=10)
    if interaction_dataset is not None:
        user_id = st.selectbox("User", options=interaction_dataset.frame["user_id"].astype(str).unique().tolist())
    else:
        user_id = None
        st.info("Collaborative and realistic hybrid recommendations activate when interaction data is added.")

with right:
    rec_type = st.radio("Engine", ["Content-Based", "Hybrid"], horizontal=True)
    recommendations = (
        content_model.recommend(item_id, top_k=top_k)
        if rec_type == "Content-Based"
        else hybrid_model.recommend(item_id=item_id, user_id=user_id, top_k=top_k)
    )
    st.dataframe(recommendations, width="stretch", hide_index=True)

metric_1, metric_2, metric_3 = st.columns(3)
with metric_1:
    st.markdown('<div class="metric-card">Dataset rows<br><h3>{:,}</h3></div>'.format(len(content_dataset.frame)), unsafe_allow_html=True)
with metric_2:
    st.markdown('<div class="metric-card">Feature count<br><h3>{}</h3></div>'.format(len(content_dataset.feature_columns)), unsafe_allow_html=True)
with metric_3:
    st.markdown('<div class="metric-card">Interaction data<br><h3>{}</h3></div>'.format("available" if interaction_dataset is not None else "missing"), unsafe_allow_html=True)

st.subheader("Model snapshot")
ml_metrics = YearRegressionBaseline().fit_and_evaluate(content_dataset)
st.code(
    json.dumps(
        {
            "ml_baseline": {
                "model": ml_metrics.model_name,
                "mae": round(ml_metrics.mae, 4),
                "rmse": round(ml_metrics.rmse, 4),
            },
            "content_similarity": content_model.similarity_report(item_id),
        },
        indent=2,
    ),
    language="json",
)
