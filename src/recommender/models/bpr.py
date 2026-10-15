"""Bayesian Personalized Ranking matrix factorization.

BPR optimizes a ranking criterion rather than a reconstruction error. For each
observed interaction ``(u, i)`` it samples an unobserved item ``j`` and
maximizes the probability that ``u`` prefers ``i`` over ``j``:

    L = sum log sigmoid(x_ui - x_uj) - lambda * ||Theta||^2

with the matrix factorization score

    x_ui = p_u . q_i + b_i

The implementation is plain NumPy with mini-batch SGD. Updates are accumulated
with ``np.add.at`` so that a user or item appearing several times inside one
batch receives every one of its gradients, which a plain fancy-indexed ``+=``
would silently drop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
                    np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))))


@dataclass
class BPRRecommender:
    """BPR matrix factorization trained with uniform negative sampling.

    Args:
        n_factors: Latent dimensionality ``d``.
        n_epochs: Passes over the training interactions.
        learning_rate: SGD step size.
        reg_user: L2 penalty on user factors.
        reg_item: L2 penalty on positive/negative item factors.
        reg_bias: L2 penalty on item biases.
        init_scale: Standard deviation of the Gaussian initialization.
        batch_size: Number of triplets per vectorized update.
        lr_decay: Multiplicative decay applied to the step size each epoch.
        random_state: Seed controlling initialization and sampling.
    """

    n_factors: int = 32
    n_epochs: int = 30
    learning_rate: float = 0.05
    reg_user: float = 0.01
    reg_item: float = 0.01
    reg_bias: float = 0.001
    init_scale: float = 0.1
    batch_size: int = 4096
    lr_decay: float = 0.95
    random_state: int = 42

    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    n_users: int = 0
    n_items: int = 0
    loss_history: list[float] | None = None
    _seen_keys: np.ndarray | None = None

    def fit(self, split: Split, verbose: bool = False) -> BPRRecommender:
        """Train on the training partition only.

        Args:
            split: Leakage-safe partition; only ``split.train`` is read.
            verbose: Print mean loss per epoch.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If the training partition is empty.
        """
        users, items = split.train_matrix_entries()
        if users.size == 0:
            raise ValueError("Cannot fit BPR on an empty training partition.")

        self.n_users = split.n_users
        self.n_items = split.n_items

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, self.init_scale, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, self.init_scale, (self.n_items, self.n_factors))
        self.item_bias = np.zeros(self.n_items)

        # Sorted encoded (user, item) keys: O(nnz) memory and vectorized lookup,
        # unlike a dense mask which would be n_users * n_items booleans.
        self._seen_keys = np.sort(users.astype(np.int64) * self.n_items + items.astype(np.int64))
        self.loss_history = []

        n_samples = users.shape[0]
        lr = self.learning_rate
        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, self.batch_size):
                batch = order[start : start + self.batch_size]
                u = users[batch]
                i = items[batch]
                j = self._sample_negatives(rng, u)

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                qj = self.item_factors[j]

                x_uij = np.einsum("bf,bf->b", pu, qi - qj) + self.item_bias[i] - self.item_bias[j]
                # d/dx log sigmoid(x) = sigmoid(-x)
                grad = _sigmoid(-x_uij)[:, None]
                epoch_loss += float(-np.log(np.maximum(_sigmoid(x_uij), 1e-12)).sum())

                grad_u = grad * (qi - qj) - self.reg_user * pu
                grad_i = grad * pu - self.reg_item * qi
                grad_j = -grad * pu - self.reg_item * qj
                grad_bi = grad[:, 0] - self.reg_bias * self.item_bias[i]
                grad_bj = -grad[:, 0] - self.reg_bias * self.item_bias[j]

                # np.add.at accumulates duplicate indices instead of overwriting.
                np.add.at(self.user_factors, u, lr * grad_u)
                np.add.at(self.item_factors, i, lr * grad_i)
                np.add.at(self.item_factors, j, lr * grad_j)
                np.add.at(self.item_bias, i, lr * grad_bi)
                np.add.at(self.item_bias, j, lr * grad_bj)

            mean_loss = epoch_loss / n_samples
            self.loss_history.append(mean_loss)
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{self.n_epochs}  loss={mean_loss:.5f}  lr={lr:.4f}")
            lr *= self.lr_decay

        return self

    def _sample_negatives(self, rng: np.random.Generator, users: np.ndarray) -> np.ndarray:
        """Vectorized uniform sampling of items each user has not interacted with.

        Resamples only the still-colliding entries, for a bounded number of
        rounds. The residual collision rate after the loop is negligible because
        no user has consumed a meaningful fraction of the catalogue; leaving a
        few false negatives in is standard practice and far cheaper than an
        exact per-user complement.
        """
        candidates = rng.integers(0, self.n_items, size=users.shape[0])
        for _ in range(16):
            keys = users.astype(np.int64) * self.n_items + candidates
            pos = np.searchsorted(self._seen_keys, keys)
            pos = np.minimum(pos, self._seen_keys.shape[0] - 1)
            collides = self._seen_keys[pos] == keys
            if not collides.any():
                break
            candidates[collides] = rng.integers(0, self.n_items, size=int(collides.sum()))
        return candidates

    def score_user(self, user_index: int) -> np.ndarray:
        """Return scores over the full catalogue for one user."""
        self._check_fitted()
        if not 0 <= user_index < self.n_users:
            # Cold user: fall back to the bias term, i.e. global popularity.
            return self.item_bias.copy()
        return self.item_factors @ self.user_factors[user_index] + self.item_bias

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        """Return the top-``k`` item indices for a user.

        Args:
            user_index: Internal user index.
            k: List length.
            exclude: Items to filter out, normally the training history.

        Returns:
            Item indices ordered by descending score.
        """
        scores = self.score_user(user_index)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]

    def _check_fitted(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
