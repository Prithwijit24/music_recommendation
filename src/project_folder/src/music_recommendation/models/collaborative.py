from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from music_recommendation.models.base import Recommender
from music_recommendation.schemas import InteractionDataset


class CollaborativeFilteringRecommender(Recommender):
    name = "collaborative_filtering"

    def __init__(self, latent_factors: int = 32, top_k: int = 10) -> None:
        self.latent_factors = latent_factors
        self.top_k = top_k
        self.dataset: InteractionDataset | None = None
        self.user_item_matrix: pd.DataFrame | None = None
        self.svd: TruncatedSVD | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, dataset: InteractionDataset | None) -> "CollaborativeFilteringRecommender":
        if dataset is None:
            raise ValueError("Collaborative filtering requires interaction data.")
        self.dataset = dataset
        matrix = dataset.frame.pivot_table(
            index=dataset.user_column,
            columns=dataset.item_column,
            values=dataset.target_column,
            aggfunc="mean",
            fill_value=0.0,
        )
        self.user_item_matrix = matrix
        self.svd = TruncatedSVD(
            n_components=min(self.latent_factors, min(matrix.shape) - 1),
            random_state=42,
        )
        self.item_factors = self.svd.fit_transform(matrix.T)
        return self

    def recommend(self, user_id: str, top_k: int | None = None) -> pd.DataFrame:
        if self.dataset is None or self.user_item_matrix is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend().")
        top_k = top_k or self.top_k
        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"Unknown user_id: {user_id}")
        user_vector = self.user_item_matrix.loc[user_id].to_numpy()
        scores = self.item_factors @ self.item_factors.T @ user_vector
        scored = pd.DataFrame(
            {"track_id": self.user_item_matrix.columns, "score": scores},
        )
        seen = set(
            self.dataset.frame.loc[self.dataset.frame["user_id"] == user_id, "track_id"].tolist()
        )
        scored = scored[~scored["track_id"].isin(seen)]
        return scored.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
