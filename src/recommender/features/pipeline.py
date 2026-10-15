"""Feature pipeline: builds every declared feature from a training partition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData
from recommender.data.splitting import Split
from recommender.features import transforms
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs


@dataclass
class FeatureStore:
    """Materialized features, fitted on the training partition only.

    The same object is used by training and by the serving path, which is what
    makes offline/online parity testable rather than aspirational.
    """

    user_activity_count: pd.Series
    user_rating_mean: pd.Series
    item_popularity_rank: pd.Series
    item_member_count: pd.Series
    item_genre_tokens: pd.Series
    user_genre_affinity: pd.Series
    n_users: int
    n_items: int
    global_rating_mean: float
    specs: list[FeatureSpec] = field(default_factory=list)

    def user_vector(self, user_index: int) -> dict[str, float | str]:
        """Return the feature values for one user, applying declared fills."""
        return {
            "user_activity_count": float(
                self.user_activity_count.get(user_index, 0.0)
            ),
            "user_rating_mean": float(
                self.user_rating_mean.get(user_index, self.global_rating_mean)
            ),
            "user_genre_affinity": str(self.user_genre_affinity.get(user_index, "")),
        }

    def item_vector(self, item_index: int) -> dict[str, float | str]:
        """Return the feature values for one item, applying declared fills."""
        return {
            "item_popularity_rank": float(
                self.item_popularity_rank.get(item_index, 1.0)
            ),
            "item_member_count": float(self.item_member_count.get(item_index, 0.0)),
            "item_genre_tokens": str(self.item_genre_tokens.get(item_index, "")),
        }

    def popularity_order(self) -> np.ndarray:
        """Item indices ordered from most to least popular."""
        return np.argsort(self.item_popularity_rank.to_numpy(), kind="stable")

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Export user-level and item-level features as tidy frames."""
        users = pd.DataFrame(
            {
                "user_index": np.arange(self.n_users),
                "user_activity_count": self.user_activity_count.reindex(
                    range(self.n_users)
                ).fillna(0.0).to_numpy(),
                "user_rating_mean": self.user_rating_mean.reindex(
                    range(self.n_users)
                ).fillna(self.global_rating_mean).to_numpy(),
                "user_genre_affinity": self.user_genre_affinity.reindex(
                    range(self.n_users)
                ).fillna("").to_numpy(),
            }
        )
        items = pd.DataFrame(
            {
                "item_index": np.arange(self.n_items),
                "item_popularity_rank": self.item_popularity_rank.to_numpy(),
                "item_member_count": self.item_member_count.to_numpy(),
                "item_genre_tokens": self.item_genre_tokens.to_numpy(),
            }
        )
        return {"users": users, "items": items}


def build_features(
    split: Split,
    data: InteractionData,
    catalogue: pd.DataFrame,
    spec_path: str | Path | None = None,
) -> FeatureStore:
    """Materialize every declared feature from the training partition.

    Only ``split.train`` is passed to the interaction transforms, which is what
    enforces the ``fit_on: train_only`` contract. Catalogue features read the
    item metadata directly because they are known before any interaction.

    Args:
        split: Leakage-safe partition.
        data: Indexed interactions, used for the item id mapping.
        catalogue: Raw anime metadata.
        spec_path: Optional override of ``conf/features.yaml``.

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs(spec_path)
    validate_registry(specs)

    train = split.train
    params = {spec.transform: spec.params for spec in specs}

    genre_tokens = transforms.item_genre_tokens(
        catalogue,
        data.item_ids,
        separator=params["item_genre_tokens"].get("separator", ", "),
    )

    scored = train.loc[train["rating"] != -1, "rating"]
    global_rating_mean = float(scored.mean()) if not scored.empty else 0.0

    return FeatureStore(
        user_activity_count=transforms.user_activity_count(train),
        user_rating_mean=transforms.user_rating_mean(
            train,
            unrated_sentinel=params["user_rating_mean"].get("unrated_sentinel", -1),
        ),
        item_popularity_rank=transforms.item_popularity_rank(
            train,
            n_items=split.n_items,
            normalize=params["item_popularity_rank"].get("normalize", True),
        ),
        item_member_count=transforms.item_member_count(
            catalogue,
            data.item_ids,
            log_transform=params["item_member_count"].get("log_transform", True),
        ),
        item_genre_tokens=genre_tokens,
        user_genre_affinity=transforms.user_genre_affinity(
            train,
            genre_tokens,
            decay_lambda=params["user_genre_affinity"].get("decay_lambda", 0.08),
            top_k=params["user_genre_affinity"].get("top_k", 5),
        ),
        n_users=split.n_users,
        n_items=split.n_items,
        global_rating_mean=global_rating_mean,
        specs=specs,
    )
