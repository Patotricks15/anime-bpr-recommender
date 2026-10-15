"""Point-in-time feature transformations.

Every transform here receives only the interactions it is allowed to see. The
caller is responsible for passing the training partition; the functions
additionally accept an explicit ``cutoff_order`` so the point-in-time rule can
be asserted directly in tests using a strict ``<`` comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATING_WATCHED_UNSCORED = -1


def _apply_cutoff(frame: pd.DataFrame, cutoff_order: int | None) -> pd.DataFrame:
    """Keep only interactions strictly before the cut-off position."""
    if cutoff_order is None:
        return frame
    return frame.loc[frame["interaction_order"] < cutoff_order]


def _normalize_user_index(series: pd.Series) -> pd.Series:
    """Force a stable int64 index so outputs are comparable across callers.

    Pandas promotes integer dtypes depending on how a frame was assembled, so
    without this the same logical feature could carry an int32 index in one
    call and int64 in another, breaking equality checks downstream.
    """
    series.index = series.index.astype(np.int64)
    series.index.name = "user_index"
    return series


def user_activity_count(
    interactions: pd.DataFrame, cutoff_order: int | None = None
) -> pd.Series:
    """Count interactions per user, observable before the cut-off."""
    frame = _apply_cutoff(interactions, cutoff_order)
    counts = frame.groupby("user_index").size().astype(np.float32)
    counts.name = "user_activity_count"
    return _normalize_user_index(counts)


def user_rating_mean(
    interactions: pd.DataFrame,
    cutoff_order: int | None = None,
    unrated_sentinel: int = RATING_WATCHED_UNSCORED,
) -> pd.Series:
    """Mean explicit score per user, ignoring watched-but-unscored rows.

    Users who only have unscored rows produce NaN, which the pipeline fills
    with the global mean rather than with zero, since zero is not on the
    1-10 rating scale and would distort the feature.
    """
    frame = _apply_cutoff(interactions, cutoff_order)
    scored = frame.loc[frame["rating"] != unrated_sentinel]
    means = scored.groupby("user_index")["rating"].mean().astype(np.float32)
    means.name = "user_rating_mean"
    return _normalize_user_index(means)


def item_popularity_rank(
    interactions: pd.DataFrame,
    n_items: int,
    cutoff_order: int | None = None,
    normalize: bool = True,
) -> pd.Series:
    """Rank items by training interaction count; rank 0 is the most popular.

    Items unseen during training receive the worst rank, which makes the
    cold-start ordering explicit instead of accidental.
    """
    frame = _apply_cutoff(interactions, cutoff_order)
    counts = frame.groupby("item_index").size()

    full = pd.Series(0, index=pd.RangeIndex(n_items), dtype=np.int64)
    full.loc[counts.index] = counts.to_numpy()

    # Descending popularity; ties broken by index for determinism.
    order = np.lexsort((full.index.to_numpy(), -full.to_numpy()))
    ranks = np.empty(n_items, dtype=np.float32)
    ranks[order] = np.arange(n_items, dtype=np.float32)

    if normalize and n_items > 1:
        ranks = ranks / (n_items - 1)

    series = pd.Series(ranks, index=pd.RangeIndex(n_items), dtype=np.float32)
    series.name = "item_popularity_rank"
    return series


def item_member_count(
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
    log_transform: bool = True,
) -> pd.Series:
    """Catalogue audience size aligned to internal item indices.

    This is a catalogue attribute, not an interaction aggregate, so it is
    available for items that never appear in training.
    """
    lookup = catalogue.set_index("anime_id")["members"]
    values = lookup.reindex(item_ids).to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)

    if log_transform:
        values = np.log1p(values)

    series = pd.Series(values.astype(np.float32), index=pd.RangeIndex(len(item_ids)))
    series.name = "item_member_count"
    return series


def item_genre_tokens(
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
    separator: str = ", ",
) -> pd.Series:
    """Normalized genre token strings aligned to internal item indices."""
    lookup = catalogue.set_index("anime_id")["genre"]
    raw = lookup.reindex(item_ids).fillna("")

    tokens = [
        "|".join(sorted(t.strip().lower() for t in str(value).split(separator) if t.strip()))
        for value in raw
    ]
    series = pd.Series(tokens, index=pd.RangeIndex(len(item_ids)), dtype=object)
    series.name = "item_genre_tokens"
    return series


def user_genre_affinity(
    interactions: pd.DataFrame,
    genre_tokens: pd.Series,
    cutoff_order: int | None = None,
    decay_lambda: float = 0.08,
    top_k: int = 5,
) -> pd.Series:
    """Time-decayed genre preference profile per user.

    Each past interaction contributes ``exp(-lambda * age)`` to the genres of
    the watched title, where ``age`` is measured in positions back from the
    user's most recent training interaction. Recent taste therefore dominates
    stale history.

    Args:
        interactions: Training interactions only.
        genre_tokens: Pipe-separated genres indexed by ``item_index``.
        cutoff_order: Strict point-in-time boundary.
        decay_lambda: Decay rate; larger forgets faster.
        top_k: Number of genres retained per user.

    Returns:
        Pipe-separated top genres per ``user_index``.
    """
    frame = _apply_cutoff(interactions, cutoff_order)
    if frame.empty:
        return pd.Series(dtype=object, name="user_genre_affinity")

    last_order = frame.groupby("user_index")["interaction_order"].transform("max")
    age = (last_order - frame["interaction_order"]).to_numpy(dtype=np.float64)
    weight = np.exp(-decay_lambda * age)

    genres = genre_tokens.reindex(frame["item_index"].to_numpy()).to_numpy()

    scores: dict[int, dict[str, float]] = {}
    for user, genre_blob, w in zip(frame["user_index"].to_numpy(), genres, weight, strict=True):
        if not genre_blob:
            continue
        bucket = scores.setdefault(int(user), {})
        for token in str(genre_blob).split("|"):
            if token:
                bucket[token] = bucket.get(token, 0.0) + float(w)

    profiles = {
        user: "|".join(
            name for name, _ in sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        )
        for user, bucket in scores.items()
    }

    series = pd.Series(profiles, dtype=object)
    series.index = series.index.astype(np.int64)
    series.index.name = "user_index"
    series.name = "user_genre_affinity"
    return series
