from __future__ import annotations

import pandas as pd


def summarize_recommendations(frame: pd.DataFrame) -> dict[str, float]:
    score_column = "hybrid_score" if "hybrid_score" in frame.columns else "score"
    return {
        "recommendation_count": float(len(frame)),
        "avg_score": float(frame[score_column].mean()),
        "max_score": float(frame[score_column].max()),
    }
