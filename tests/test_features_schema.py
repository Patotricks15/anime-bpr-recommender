"""Schema contract tests for the declarative feature specification."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.features import load_feature_specs, validate_registry
from recommender.features.spec import VALID_FAMILIES, FeatureSpec


def test_registry_matches_declaration() -> None:
    """Every declared feature is implemented and vice versa."""
    validate_registry(load_feature_specs())


def test_every_feature_declares_a_known_family() -> None:
    for spec in load_feature_specs():
        assert spec.family in VALID_FAMILIES


def test_every_feature_documents_a_rationale() -> None:
    """A feature without a stated reason cannot be reviewed or ablated."""
    for spec in load_feature_specs():
        assert len(spec.rationale.strip()) > 20, f"{spec.name} has a thin rationale"


def test_interaction_features_are_point_in_time() -> None:
    """Aggregates over user history must declare the point-in-time contract."""
    for spec in load_feature_specs():
        if spec.family == "interaction":
            assert spec.point_in_time, f"{spec.name} aggregates history but is not point-in-time"
            assert spec.fit_on == "train_only", f"{spec.name} must fit on train only"


def test_catalogue_features_are_not_train_scoped() -> None:
    """Item metadata exists before any interaction, so it is cold-start safe."""
    for spec in load_feature_specs():
        if spec.fit_on == "catalogue":
            assert not spec.point_in_time


def test_invalid_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec(
            name="bad", family="nonsense", dtype="float32", transform="t",
            params={}, fit_on="train_only", point_in_time=True, fill={},
            rationale="long enough rationale for the validator to accept it",
        )


def test_missing_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="rationale"):
        FeatureSpec(
            name="bad", family="item", dtype="float32", transform="t",
            params={}, fit_on="catalogue", point_in_time=False, fill={},
            rationale="   ",
        )


def test_feature_store_has_no_nulls_and_correct_dtypes(feature_store) -> None:
    """Declared fill policies must leave no missing values downstream."""
    frames = feature_store.to_frames()

    assert frames["users"].isna().sum().sum() == 0
    assert frames["items"].isna().sum().sum() == 0

    assert len(frames["users"]) == feature_store.n_users
    assert len(frames["items"]) == feature_store.n_items

    assert np.isfinite(frames["items"]["item_popularity_rank"]).all()
    assert np.isfinite(frames["items"]["item_member_count"]).all()


def test_popularity_rank_is_a_normalized_permutation(feature_store) -> None:
    ranks = feature_store.item_popularity_rank.to_numpy()
    assert ranks.min() == pytest.approx(0.0)
    assert ranks.max() == pytest.approx(1.0)
    assert len(np.unique(ranks)) == len(ranks), "ranks must be a strict ordering"


def test_unknown_entities_fall_back_to_declared_defaults(feature_store) -> None:
    """Cold entities must not raise; they must return the declared fill value."""
    ghost_user = feature_store.n_users + 999
    ghost_item = feature_store.n_items + 999

    assert feature_store.user_vector(ghost_user)["user_activity_count"] == 0.0
    assert feature_store.user_vector(ghost_user)["user_rating_mean"] == pytest.approx(
        feature_store.global_rating_mean
    )
    assert feature_store.item_vector(ghost_item)["item_popularity_rank"] == 1.0
