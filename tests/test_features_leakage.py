"""Leakage tests.

These are the most important tests in the repository. A recommender that leaks
future information reports excellent offline metrics and then fails in
production, so each leakage class gets an explicit, failing-by-default check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data import build_implicit_feedback, leave_last_one_out
from recommender.features import transforms


def test_future_interaction_does_not_change_past_features(split) -> None:
    """Injecting an event after the cut-off must not alter any feature value.

    This is the direct operationalization of the point-in-time rule: the
    feature computed at cut-off C must be identical whether or not events at
    positions >= C exist in the frame.
    """
    train = split.train
    cutoff = int(train["interaction_order"].max())

    before = transforms.user_activity_count(train, cutoff_order=cutoff)

    # Fabricate an interaction strictly after the cut-off for an existing user.
    victim = int(train["user_index"].iloc[0])
    future_row = train.iloc[[0]].copy()
    future_row["user_index"] = victim
    future_row["interaction_order"] = cutoff + 50
    polluted = pd.concat([train, future_row], ignore_index=True)

    after = transforms.user_activity_count(polluted, cutoff_order=cutoff)

    pd.testing.assert_series_equal(
        before.sort_index(), after.sort_index(), check_exact=True
    )


def test_future_interaction_does_not_change_rating_mean(split) -> None:
    train = split.train
    cutoff = int(train["interaction_order"].max())

    before = transforms.user_rating_mean(train, cutoff_order=cutoff)

    future_row = train.iloc[[0]].copy()
    future_row["interaction_order"] = cutoff + 10
    future_row["rating"] = 10  # an extreme score that would move any mean
    polluted = pd.concat([train, future_row], ignore_index=True)

    after = transforms.user_rating_mean(polluted, cutoff_order=cutoff)

    pd.testing.assert_series_equal(
        before.sort_index(), after.sort_index(), check_exact=True
    )


def test_cutoff_boundary_is_strictly_exclusive(split) -> None:
    """An event exactly at the cut-off belongs to the future, not the past."""
    train = split.train.copy()
    user = int(train["user_index"].iloc[0])
    user_rows = train.loc[train["user_index"] == user]
    cutoff = int(user_rows["interaction_order"].max())

    counts = transforms.user_activity_count(train, cutoff_order=cutoff)
    expected = int((user_rows["interaction_order"] < cutoff).sum())

    assert int(counts.get(user, 0)) == expected


def test_popularity_is_fitted_on_training_partition_only(split, interactions) -> None:
    """Popularity computed on train must differ from popularity on all data.

    If they were identical, the training statistic would already encode the
    held-out interactions and the evaluation would be optimistic.
    """
    train_pop = transforms.item_popularity_rank(split.train, n_items=split.n_items)
    full_pop = transforms.item_popularity_rank(
        interactions.frame, n_items=split.n_items
    )

    assert not np.array_equal(train_pop.to_numpy(), full_pop.to_numpy()), (
        "train-only popularity matches full-data popularity; the fit scope is not enforced"
    )


def test_split_partitions_are_disjoint(split) -> None:
    def pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return set(
            zip(
                frame["user_index"].to_numpy().tolist(),
                frame["item_index"].to_numpy().tolist(),
                strict=True,
            )
        )

    train, validation, test = pairs(split.train), pairs(split.validation), pairs(split.test)

    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_test_interaction_is_the_users_latest(split) -> None:
    """Leave-last-one-out must take the most recent interaction, not a random one."""
    merged = split.test.merge(
        split.train.groupby("user_index")["interaction_order"].max().rename("train_max"),
        on="user_index",
        how="inner",
    )
    assert (merged["interaction_order"] > merged["train_max"]).all()


def test_validation_precedes_test(split) -> None:
    merged = split.validation.merge(
        split.test[["user_index", "interaction_order"]].rename(
            columns={"interaction_order": "test_order"}
        ),
        on="user_index",
        how="inner",
    )
    assert (merged["interaction_order"] < merged["test_order"]).all()


def test_no_duplicate_interactions_before_splitting(interactions) -> None:
    """Duplicates would let the same event land on both sides of the split."""
    frame = interactions.frame
    assert not frame.duplicated(subset=["user_index", "item_index"]).any()


def test_shuffled_labels_destroy_ranking_quality(split, interactions) -> None:
    """A model trained on shuffled item assignments must not beat random.

    This catches leakage that survives the structural checks: if performance
    stayed high after destroying the user-item association, a feature would be
    carrying the answer.
    """
    from recommender.evaluation import ndcg_at_k
    from recommender.models import BPRRecommender

    rng = np.random.default_rng(0)
    shuffled = split.train.copy()
    shuffled["item_index"] = rng.permutation(shuffled["item_index"].to_numpy())

    broken_split = type(split)(
        train=shuffled,
        validation=split.validation,
        test=split.test,
        n_users=split.n_users,
        n_items=split.n_items,
    )

    model = BPRRecommender(n_factors=8, n_epochs=3, random_state=0)
    model.fit(broken_split)

    score = ndcg_at_k(model, broken_split, k=10)
    random_baseline = 10 / split.n_items

    assert score < max(0.05, random_baseline * 6), (
        f"NDCG@10 stayed at {score:.4f} after shuffling labels; a feature is leaking"
    )


@pytest.mark.parametrize("min_train", [3, 5])
def test_users_without_history_are_not_evaluated(interactions, min_train: int) -> None:
    """Evaluating users with no training history would measure nothing."""
    split = leave_last_one_out(interactions, min_train_interactions=min_train)
    train_users = set(split.train["user_index"].unique().tolist())
    test_users = set(split.test["user_index"].unique().tolist())

    assert test_users <= train_users


def test_core_filter_rejects_impossible_thresholds(raw_ratings) -> None:
    with pytest.raises(ValueError):
        build_implicit_feedback(
            raw_ratings, min_user_interactions=10_000, min_item_interactions=10_000
        )
