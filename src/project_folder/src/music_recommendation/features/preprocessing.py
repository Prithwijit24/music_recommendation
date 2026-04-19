from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from music_recommendation.schemas import ContentDataset


@dataclass(slots=True)
class FeatureBundle:
    scaled_matrix: np.ndarray
    embedding_matrix: np.ndarray
    scaler: StandardScaler
    reducer: TruncatedSVD


def build_content_features(dataset: ContentDataset, latent_dims: int = 32) -> FeatureBundle:
    features = dataset.frame[dataset.feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    reducer = TruncatedSVD(n_components=min(latent_dims, scaled.shape[1] - 1), random_state=42)
    embedding = reducer.fit_transform(scaled)
    return FeatureBundle(
        scaled_matrix=scaled,
        embedding_matrix=embedding,
        scaler=scaler,
        reducer=reducer,
    )
