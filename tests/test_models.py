"""Behavioural tests for the models and the evaluation protocol.

These assert properties that must hold for any correct implementation
(determinism, history exclusion, learning signal) rather than pinning exact
metric values, which would turn routine tuning into a test failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation import evaluate
from recommender.models import (
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)

FAST_BPR = {"n_factors": 32, "n_epochs": 60, "learning_rate": 0.2, "batch_size": 1024}


@pytest.fixture(scope="module")
def fitted_bpr(split):
    return BPRRecommender(reg_user=0.05, reg_item=0.05, random_state=42, **FAST_BPR).fit(split)


def test_bpr_decreases_its_own_loss(fitted_bpr) -> None:
    """The BPR objective must actually be optimized."""
    history = fitted_bpr.loss_history
    assert history[-1] < history[0] * 0.8, f"loss barely moved: {history[0]} -> {history[-1]}"


def test_bpr_is_deterministic_given_a_seed(split) -> None:
    a = BPRRecommender(random_state=7, n_factors=8, n_epochs=5).fit(split)
    b = BPRRecommender(random_state=7, n_factors=8, n_epochs=5).fit(split)
    np.testing.assert_allclose(a.user_factors, b.user_factors)
    np.testing.assert_allclose(a.item_bias, b.item_bias)


def test_different_seeds_give_different_factors(split) -> None:
    a = BPRRecommender(random_state=1, n_factors=8, n_epochs=5).fit(split)
    b = BPRRecommender(random_state=2, n_factors=8, n_epochs=5).fit(split)
    assert not np.allclose(a.user_factors, b.user_factors)


def test_negative_sampler_never_returns_a_training_positive(split) -> None:
    """A sampled 'negative' that the user actually consumed corrupts the gradient."""
    model = BPRRecommender(n_factors=8, n_epochs=1, random_state=0).fit(split)
    positives = split.user_positives()
    users, _ = split.train_matrix_entries()
    rng = np.random.default_rng(123)

    for _ in range(10):
        negatives = model._sample_negatives(rng, users)
        collisions = [
            (int(u), int(j))
            for u, j in zip(users, negatives, strict=True)
            if int(j) in positives.get(int(u), set())
        ]
        assert not collisions, f"sampled false negatives: {collisions[:5]}"


@pytest.mark.parametrize(
    "model",
    [
        BPRRecommender(n_factors=8, n_epochs=5, random_state=0),
        PopularityRecommender(),
        ItemKNNRecommender(),
        RandomRecommender(),
    ],
    ids=["bpr", "popularity", "itemknn", "random"],
)
def test_recommend_contract(model, split) -> None:
    """Every model returns k distinct, in-range items and honours `exclude`."""
    model.fit(split)
    excluded = {0, 1, 2}
    ranked = model.recommend(user_index=0, k=10, exclude=excluded)

    assert len(ranked) == 10
    assert len(set(ranked.tolist())) == 10, "duplicate items in a single ranking"
    assert all(0 <= int(i) < split.n_items for i in ranked)
    assert not excluded.intersection(int(i) for i in ranked)


def test_bpr_beats_untrained_baselines(fitted_bpr, split) -> None:
    """BPR must extract personalised signal beyond popularity and chance.

    Note: this repository's headline finding is that a *tuned* item-KNN still
    edges out BPR on this dataset, so the assertion deliberately does not claim
    BPR is the best model -- only that it learns something real.
    """
    bpr = evaluate(fitted_bpr, split, k=10, partition="validation").ndcg
    popularity = evaluate(
        PopularityRecommender().fit(split), split, k=10, partition="validation"
    ).ndcg
    random_model = evaluate(
        RandomRecommender().fit(split), split, k=10, partition="validation"
    ).ndcg

    assert bpr > popularity > random_model, (
        f"expected bpr > popularity > random, got {bpr:.4f} / {popularity:.4f} / {random_model:.4f}"
    )


def test_bpr_is_more_personalised_than_popularity(fitted_bpr, split) -> None:
    """Popularity gives everyone the same list; BPR must not."""
    lists = {tuple(fitted_bpr.recommend(u, k=10).tolist()) for u in range(min(50, split.n_users))}
    assert len(lists) > 1, "BPR collapsed to a single global ranking"


def test_coverage_is_higher_than_popularity(fitted_bpr, split) -> None:
    """Catalogue coverage is the business reason to move off a popularity list."""
    bpr = evaluate(fitted_bpr, split, k=10, partition="validation").coverage
    pop = evaluate(PopularityRecommender().fit(split), split, k=10, partition="validation").coverage
    assert bpr > pop


def test_evaluation_excludes_training_history(split) -> None:
    """A model that only re-recommends the history must score zero.

    This is the guard against the classic evaluation bug where seen items are
    left in the candidate set and inflate every metric.
    """

    class HistoryParrot:
        def recommend(self, user_index, k=10, exclude=None):
            seen = sorted(exclude or {0})
            padded = (seen * k)[:k]
            return np.array(padded, dtype=int)

    report = evaluate(HistoryParrot(), split, k=10, partition="validation")
    assert report.ndcg == 0.0
    assert report.hit_rate == 0.0


def test_unknown_partition_is_rejected(fitted_bpr, split) -> None:
    with pytest.raises(ValueError):
        evaluate(fitted_bpr, split, partition="train_oops")


def test_fit_rejects_empty_training_data(split) -> None:
    empty = type(split)(
        train=split.train.iloc[0:0],
        validation=split.validation,
        test=split.test,
        n_users=split.n_users,
        n_items=split.n_items,
    )
    with pytest.raises(ValueError):
        BPRRecommender().fit(empty)


def test_scoring_an_unfitted_model_raises() -> None:
    with pytest.raises(RuntimeError):
        BPRRecommender().score_user(0)
