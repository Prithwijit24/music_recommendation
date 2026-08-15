from __future__ import annotations

import pandas as pd

from music_recommendation.models.base import Recommender
from music_recommendation.models.collaborative import CollaborativeFilteringRecommender
from music_recommendation.models.content_based import ContentBasedRecommender


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeFilteringRecommender | None = None,
        alpha: float = 0.6,
        top_k: int = 10,
    ) -> None:
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha
        self.top_k = top_k

    def fit(self) -> "HybridRecommender":
        return self

    def recommend(self, item_id: str, user_id: str | None = None, top_k: int | None = None) -> pd.DataFrame:
        top_k = top_k or self.top_k
        content_scores = self.content_model.recommend(item_id, top_k=top_k * 2).rename(
            columns={"score": "content_score"}
        )
        if self.collaborative_model is None or user_id is None:
            content_scores["hybrid_score"] = content_scores["content_score"]
            return content_scores.sort_values("hybrid_score", ascending=False).head(top_k)
        collaborative_scores = self.collaborative_model.recommend(user_id, top_k=top_k * 2).rename(
            columns={"score": "collaborative_score"}
        )
        merged = content_scores.merge(collaborative_scores, on="track_id", how="outer")
        merged["content_score"] = merged["content_score"].fillna(0.0)
        merged["collaborative_score"] = merged["collaborative_score"].fillna(0.0)
        merged["hybrid_score"] = (
            self.alpha * merged["content_score"] + (1 - self.alpha) * merged["collaborative_score"]
        )
        return merged.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
