"""Loading and implicit-feedback construction for the Anime Recommendations Database.

The raw Kaggle dataset ships two files:

``rating.csv``
    ``user_id,anime_id,rating`` where ``rating == -1`` means the title was
    watched but not scored.

``anime.csv``
    ``anime_id,name,genre,type,episodes,rating,members``.

Timestamps are **not** present in the raw dataset. Because a recommender must
still avoid using the future to predict the past, this module derives an
``interaction_order`` column: the position of each row inside its user block,
which reflects the append order of the export. It is a weak proxy, and the
limitation is stated explicitly in the paper. The synthetic fixture produced by
``scripts/download_data.py`` carries a true ``timestamp`` column, so the
point-in-time machinery is exercised against real ordering in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RATING_WATCHED_UNSCORED = -1
MIN_POSITIVE_RATING = 6


@dataclass(frozen=True)
class InteractionData:
    """Implicit-feedback interactions with contiguous internal indices.

    Attributes:
        frame: Interactions with ``user_index``, ``item_index`` and ordering.
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original item ids, positionally aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Read the raw ratings file and validate its contract.

    Args:
        path: Path to ``rating.csv`` or the synthetic equivalent.

    Returns:
        The raw frame, with ``timestamp`` preserved when present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path)
    required = {"user_id", "anime_id", "rating"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return frame


def load_anime_metadata(path: str | Path) -> pd.DataFrame:
    """Read the item catalogue used for content and popularity features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path)
    required = {"anime_id", "name", "genre", "members"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["genre"] = frame["genre"].fillna("")
    frame["members"] = pd.to_numeric(frame["members"], errors="coerce").fillna(0)
    return frame


def build_implicit_feedback(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> InteractionData:
    """Convert raw ratings into implicit feedback with contiguous indices.

    A row is a positive when the title was watched, regardless of score:
    ``rating == -1`` (watched, unscored) and ``rating >= MIN_POSITIVE_RATING``
    are both positives. Scores below the threshold are dropped rather than used
    as negatives, because an unobserved item is not evidence of dislike -- the
    assumption BPR is built on.

    Iterative filtering removes users and items below the interaction floor;
    it loops because removing items can push a user below the threshold.

    Args:
        ratings: Raw frame from :func:`load_ratings`.
        min_user_interactions: Minimum positives required to keep a user.
        min_item_interactions: Minimum positives required to keep an item.

    Returns:
        Indexed :class:`InteractionData`.

    Raises:
        ValueError: If filtering removes every interaction.
    """
    frame = ratings.copy()
    frame = frame.drop_duplicates(subset=["user_id", "anime_id"], keep="first")

    is_watched_unscored = frame["rating"] == RATING_WATCHED_UNSCORED
    is_liked = frame["rating"] >= MIN_POSITIVE_RATING
    frame = frame.loc[is_watched_unscored | is_liked].copy()

    if frame.empty:
        raise ValueError("No positive interactions remain after implicit-feedback filtering.")

    # Iterative core filtering: dropping items can invalidate users and vice versa.
    while True:
        user_counts = frame["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        item_counts = frame["anime_id"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = frame.loc[
            frame["user_id"].isin(keep_users) & frame["anime_id"].isin(keep_items)
        ]
        if filtered.shape[0] == frame.shape[0]:
            frame = filtered
            break
        frame = filtered
        if frame.empty:
            raise ValueError(
                "Core filtering removed every interaction; lower the minimum thresholds."
            )

    if "timestamp" not in frame.columns:
        # Raw Kaggle export has no timestamps; use within-user append order.
        frame = frame.reset_index(drop=True)
        frame["interaction_order"] = frame.groupby("user_id").cumcount()
    else:
        frame = frame.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
            drop=True
        )
        frame["interaction_order"] = frame.groupby("user_id").cumcount()

    user_ids = np.sort(frame["user_id"].unique())
    item_ids = np.sort(frame["anime_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    frame["user_index"] = frame["user_id"].map(user_map).astype(np.int32)
    frame["item_index"] = frame["anime_id"].map(item_map).astype(np.int32)

    return InteractionData(frame=frame, user_ids=user_ids, item_ids=item_ids)
