from __future__ import annotations

import pandas as pd

from music_recommendation.schemas import InteractionDataset


def build_user_product_matrix(dataset: InteractionDataset) -> pd.DataFrame:
    return dataset.frame.pivot_table(
        index=dataset.user_column,
        columns=dataset.item_column,
        values=dataset.target_column,
        aggfunc="mean",
        fill_value=0.0,
    )
