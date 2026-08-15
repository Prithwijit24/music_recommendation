from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(self, *args, **kwargs) -> "Recommender":
        raise NotImplementedError

    @abstractmethod
    def recommend(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
