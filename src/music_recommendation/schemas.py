from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ContentDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str = "year"
    item_column: str = "track_id"


@dataclass(slots=True)
class InteractionDataset:
    frame: pd.DataFrame
    user_column: str = "user_id"
    item_column: str = "track_id"
    target_column: str = "rating"
