"""Leave-last-one-out splitting with a strict point-in-time guarantee."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData


@dataclass(frozen=True)
class Split:
    """A leakage-safe partition of the interaction data.

    ``train`` holds every interaction strictly before each user's validation
    position, so any aggregate computed on ``train`` is automatically
    point-in-time correct with respect to both evaluation partitions.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def train_matrix_entries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(user_index, item_index)`` arrays of the train split."""
        return (
            self.train["user_index"].to_numpy(dtype=np.int64),
            self.train["item_index"].to_numpy(dtype=np.int64),
        )

    def user_positives(self) -> dict[int, set[int]]:
        """Map each user to the items seen during training, for negative sampling."""
        grouped = self.train.groupby("user_index")["item_index"]
        return {int(user): set(map(int, items)) for user, items in grouped}


def leave_last_one_out(data: InteractionData, min_train_interactions: int = 3) -> Split:
    """Split per user: last interaction to test, second-to-last to validation.

    Users without enough history to leave at least ``min_train_interactions``
    rows in training contribute all of their interactions to training only.
    They are therefore never evaluated, which avoids inflating metrics with
    users whose "history" is a single item.

    Args:
        data: Indexed interactions carrying ``interaction_order``.
        min_train_interactions: Rows a user must retain to be evaluated.

    Returns:
        A :class:`Split` whose partitions are disjoint by construction.

    Raises:
        ValueError: If no user qualifies for evaluation.
    """
    frame = data.frame.sort_values(["user_index", "interaction_order"], kind="mergesort")

    position_from_end = frame.groupby("user_index").cumcount(ascending=False)
    user_sizes = frame.groupby("user_index")["item_index"].transform("size")

    evaluable = user_sizes >= (min_train_interactions + 2)

    is_test = evaluable & (position_from_end == 0)
    is_validation = evaluable & (position_from_end == 1)
    is_train = ~(is_test | is_validation)

    test = frame.loc[is_test].copy()
    validation = frame.loc[is_validation].copy()
    train = frame.loc[is_train].copy()

    if test.empty:
        raise ValueError(
            "No user has enough interactions to be evaluated; "
            "lower min_train_interactions or relax the core filter."
        )

    return Split(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users=data.n_users,
        n_items=data.n_items,
    )
