"""Baselines that the main method must beat to justify its complexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


@dataclass
class PopularityRecommender:
    """Recommends the globally most consumed items to everyone.

    This is the baseline that naive collaborative filtering notebooks often
    fail to beat, so it is reported explicitly rather than omitted.
    """

    item_scores: np.ndarray | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> PopularityRecommender:
        self.n_users = split.n_users
        self.n_items = split.n_items

        counts = np.zeros(self.n_items, dtype=np.float64)
        _, items = split.train_matrix_entries()
        np.add.at(counts, items, 1.0)

        self.item_scores = counts
        return self

    def score_user(self, user_index: int) -> np.ndarray:  # noqa: ARG002
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return self.item_scores

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


@dataclass
class ItemKNNRecommender:
    """Item-item cosine collaborative filtering.

    This is the method most Kaggle notebooks for this dataset actually use, so
    beating it is the bar that justifies the BPR project.
    """

    top_k_neighbours: int = 50
    similarity: np.ndarray | None = None
    user_items: dict[int, list[int]] | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> ItemKNNRecommender:
        self.n_users = split.n_users
        self.n_items = split.n_items

        users, items = split.train_matrix_entries()
        matrix = np.zeros((self.n_items, self.n_users), dtype=np.float32)
        matrix[items, users] = 1.0

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, 0.0)

        # Keep only the strongest neighbours; dense similarity is noisy.
        if self.top_k_neighbours < self.n_items:
            cutoff_idx = self.n_items - self.top_k_neighbours
            thresholds = np.partition(similarity, cutoff_idx, axis=1)[:, cutoff_idx]
            similarity[similarity < thresholds[:, None]] = 0.0

        self.similarity = similarity
        self.user_items = {
            int(u): split.train.loc[
                split.train["user_index"] == u, "item_index"
            ].to_numpy().tolist()
            for u in np.unique(users)
        }
        return self

    def score_user(self, user_index: int) -> np.ndarray:
        if self.similarity is None or self.user_items is None:
            raise RuntimeError("Model is not fitted; call fit() first.")

        history = self.user_items.get(int(user_index))
        if not history:
            return np.zeros(self.n_items)
        return self.similarity[history].sum(axis=0)

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


@dataclass
class RandomRecommender:
    """Uniform random ordering; the floor every other model must clear."""

    random_state: int = 0
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> RandomRecommender:
        self.n_users = split.n_users
        self.n_items = split.n_items
        return self

    def score_user(self, user_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state + int(user_index))
        return rng.random(self.n_items)

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]
