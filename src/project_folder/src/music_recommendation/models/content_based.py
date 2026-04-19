from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from music_recommendation.features.preprocessing import FeatureBundle
from music_recommendation.models.base import Recommender
from music_recommendation.schemas import ContentDataset


class ContentBasedRecommender(Recommender):
    name = "content_based"

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self.dataset: ContentDataset | None = None
        self.bundle: FeatureBundle | None = None
        self.nn = NearestNeighbors(metric="cosine")

    def fit(self, dataset: ContentDataset, bundle: FeatureBundle) -> "ContentBasedRecommender":
        self.dataset = dataset
        self.bundle = bundle
        self.nn.fit(bundle.embedding_matrix)
        return self

    def recommend(self, item_id: str, top_k: int | None = None) -> pd.DataFrame:
        if self.dataset is None or self.bundle is None:
            raise RuntimeError("Model must be fitted before calling recommend().")
        top_k = top_k or self.top_k
        frame = self.dataset.frame
        if item_id not in set(frame["track_id"]):
            raise KeyError(f"Unknown item_id: {item_id}")
        row_index = frame.index[frame["track_id"] == item_id][0]
        distances, indices = self.nn.kneighbors(
            self.bundle.embedding_matrix[row_index].reshape(1, -1),
            n_neighbors=min(top_k + 1, len(frame)),
        )
        recommendations = frame.iloc[indices[0]].copy()
        recommendations["score"] = 1 - distances[0]
        recommendations = recommendations[recommendations["track_id"] != item_id]
        return recommendations[["track_id", "year", "score"]].head(top_k).reset_index(drop=True)

    def similarity_report(self, item_id: str) -> dict[str, float]:
        if self.dataset is None or self.bundle is None:
            raise RuntimeError("Model must be fitted before calling similarity_report().")
        frame = self.dataset.frame
        row_index = frame.index[frame["track_id"] == item_id][0]
        similarities = cosine_similarity(
            self.bundle.embedding_matrix[row_index].reshape(1, -1),
            self.bundle.embedding_matrix,
        )[0]
        return {
            "mean_similarity": float(similarities.mean()),
            "max_similarity": float(similarities.max()),
            "min_similarity": float(similarities.min()),
        }
