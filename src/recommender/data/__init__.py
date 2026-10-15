"""Data loading, implicit-feedback construction and leakage-safe splitting."""

from recommender.data.loader import (
    InteractionData,
    build_implicit_feedback,
    load_anime_metadata,
    load_ratings,
)
from recommender.data.splitting import Split, leave_last_one_out

__all__ = [
    "InteractionData",
    "Split",
    "build_implicit_feedback",
    "leave_last_one_out",
    "load_anime_metadata",
    "load_ratings",
]
