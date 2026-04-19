from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from music_recommendation.schemas import ContentDataset


@dataclass(slots=True)
class MLTrainingResult:
    model_name: str
    mae: float
    rmse: float


class YearRegressionBaseline:
    """Uses the available dataset to learn item representations useful for ranking and retrieval."""

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        )

    def fit_and_evaluate(self, dataset: ContentDataset) -> MLTrainingResult:
        frame = dataset.frame
        X = frame[dataset.feature_columns]
        y = frame[dataset.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return MLTrainingResult(
            model_name="random_forest_year_regressor",
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(np.sqrt(mse)),
        )
