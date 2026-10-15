"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import transforms
from recommender.features.spec import FeatureSpec

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "user_activity_count": transforms.user_activity_count,
    "user_rating_mean": transforms.user_rating_mean,
    "item_popularity_rank": transforms.item_popularity_rank,
    "item_member_count": transforms.item_member_count,
    "item_genre_tokens": transforms.item_genre_tokens,
    "user_genre_affinity": transforms.user_genre_affinity,
}


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, or an
            implemented transform is never declared.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    undeclared = implemented - declared
    unimplemented = declared - implemented

    problems: list[str] = []
    if unimplemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
