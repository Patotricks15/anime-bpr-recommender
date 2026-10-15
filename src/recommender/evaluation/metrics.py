"""Ranking metrics evaluated on held-out interactions.

All metrics follow the same protocol: for each evaluated user, rank the full
catalogue excluding the training history, then measure where the held-out item
lands. Excluding the history matters -- without it, a model is rewarded for
re-recommending what the user already consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from recommender.data.splitting import Split


class Recommender(Protocol):
    """Structural interface shared by BPR and the baselines."""

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics for one model on one partition."""

    model: str
    k: int
    ndcg: float
    recall: float
    precision: float
    mrr: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def _history_by_user(split: Split) -> dict[int, set[int]]:
    grouped = split.train.groupby("user_index")["item_index"]
    return {int(user): set(map(int, items)) for user, items in grouped}


def evaluate(
    model: Recommender,
    split: Split,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Compute ranking metrics for a model on a held-out partition.

    Because the split is leave-last-one-out, each user has exactly one relevant
    item. Recall@K therefore equals hit rate, and NDCG@K reduces to
    ``1 / log2(rank + 2)`` at the hit position; both are still reported so the
    table stays comparable with multi-positive projects in the portfolio.

    Args:
        model: Any object exposing ``recommend``.
        split: The partition container.
        k: Cut-off.
        partition: ``"test"`` or ``"validation"``.
        model_name: Label used in the results table.

    Returns:
        A populated :class:`EvaluationReport`.

    Raises:
        ValueError: If the partition name is unknown or empty.
    """
    if partition not in {"train", "validation", "test"}:
        raise ValueError(
            f"Unknown partition '{partition}'; expected 'validation' or 'test'."
        )
    frame: pd.DataFrame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    history = _history_by_user(split)

    ndcg_total = 0.0
    recall_total = 0.0
    rr_total = 0.0
    hits = 0
    recommended_items: set[int] = set()

    users = frame["user_index"].to_numpy()
    targets = frame["item_index"].to_numpy()

    for user, target in zip(users, targets, strict=True):
        ranked = model.recommend(int(user), k=k, exclude=history.get(int(user), set()))
        recommended_items.update(int(i) for i in ranked)

        positions = np.flatnonzero(ranked == target)
        if positions.size:
            rank = int(positions[0])
            ndcg_total += 1.0 / np.log2(rank + 2)
            rr_total += 1.0 / (rank + 1)
            recall_total += 1.0
            hits += 1

    n_users = len(users)
    return EvaluationReport(
        model=model_name,
        k=k,
        ndcg=ndcg_total / n_users,
        recall=recall_total / n_users,
        precision=recall_total / (n_users * k),
        mrr=rr_total / n_users,
        hit_rate=hits / n_users,
        coverage=len(recommended_items) / split.n_items,
        n_users=n_users,
    )


def ndcg_at_k(model: Recommender, split: Split, k: int = 10, partition: str = "test") -> float:
    """Convenience wrapper returning only NDCG@K."""
    return evaluate(model, split, k=k, partition=partition).ndcg
