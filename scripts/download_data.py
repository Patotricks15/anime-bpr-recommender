#!/usr/bin/env python3
"""Fetch the Anime Recommendations Database or generate a synthetic fixture.

The synthetic generator is the default so the repository is runnable from a
clean checkout with no Kaggle account. It reproduces the real schema,
including the ``-1`` watched-but-unscored convention and a long-tailed
popularity distribution, and adds a true ``timestamp`` column so the
point-in-time tests exercise real ordering.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_DATASET = "CooperUnion/anime-recommendations-database"

GENRE_POOL = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
    "Mystery", "Psychological", "Romance", "Sci-Fi", "Slice of Life",
    "Sports", "Supernatural", "Thriller",
]
ANIME_TYPES = ["TV", "Movie", "OVA", "Special", "ONA"]


def generate_synthetic(
    n_users: int = 900,
    n_items: int = 320,
    seed: int = 20260115,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixture with realistic sparsity, popularity skew and taste clusters.

    Users are assigned to latent taste clusters and items to genre clusters, so
    a collaborative model can actually recover structure. Without this, every
    model would score at the random baseline and the evaluation would be
    meaningless.

    Args:
        n_users: Number of synthetic users.
        n_items: Number of synthetic anime titles.
        seed: Seed controlling the whole generation.

    Returns:
        ``(ratings, anime)`` frames matching the Kaggle schema.
    """
    rng = np.random.default_rng(seed)

    n_clusters = 6
    item_cluster = rng.integers(0, n_clusters, size=n_items)
    user_cluster = rng.integers(0, n_clusters, size=n_users)

    # Long-tailed popularity: a few titles dominate, as in the real catalogue.
    popularity = rng.pareto(1.4, size=n_items) + 1.0
    popularity = popularity / popularity.sum()

    anime = pd.DataFrame(
        {
            "anime_id": np.arange(1, n_items + 1),
            "name": [f"Synthetic Anime {i:04d}" for i in range(1, n_items + 1)],
            "genre": [
                ", ".join(
                    sorted(
                        rng.choice(
                            GENRE_POOL,
                            size=int(rng.integers(1, 4)),
                            replace=False,
                        )
                    )
                )
                for _ in range(n_items)
            ],
            "type": rng.choice(ANIME_TYPES, size=n_items, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            "episodes": rng.integers(1, 64, size=n_items),
            "rating": np.round(rng.normal(7.0, 0.9, size=n_items).clip(2.0, 10.0), 2),
            "members": (popularity * 4_000_000).astype(int) + rng.integers(50, 900, n_items),
        }
    )

    records: list[dict[str, object]] = []
    for user in range(1, n_users + 1):
        cluster = user_cluster[user - 1]
        # Preference mixes cluster affinity with global popularity.
        affinity = np.where(item_cluster == cluster, 3.0, 0.35)
        weights = affinity * popularity
        weights = weights / weights.sum()

        n_interactions = int(rng.integers(6, 42))
        n_interactions = min(n_interactions, n_items)
        chosen = rng.choice(n_items, size=n_interactions, replace=False, p=weights)

        # Timestamps increase within a user so ordering is meaningful.
        base_time = int(rng.integers(1_300_000_000, 1_400_000_000))
        offsets = np.sort(rng.integers(0, 60_000_000, size=n_interactions))

        for position, item_idx in enumerate(chosen):
            in_cluster = item_cluster[item_idx] == cluster
            if rng.random() < 0.22:
                rating = -1  # watched, not scored
            else:
                centre = 8.4 if in_cluster else 6.1
                rating = int(np.clip(round(rng.normal(centre, 1.3)), 1, 10))

            records.append(
                {
                    "user_id": user,
                    "anime_id": int(anime.loc[item_idx, "anime_id"]),
                    "rating": rating,
                    "timestamp": base_time + int(offsets[position]),
                }
            )

    ratings = pd.DataFrame.from_records(records)
    return ratings, anime


def download_from_kaggle(target: Path) -> None:
    """Download the real dataset using credentials from the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.\n"
            "Credentials are never read from or written to this repository."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", str(target), "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("The 'kaggle' package is not installed: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true",
        help="Generate the offline fixture (default).",
    )
    source.add_argument(
        "--kaggle", action="store_true",
        help="Download the real dataset using KAGGLE_USERNAME/KAGGLE_KEY.",
    )
    parser.add_argument("--users", type=int, default=900)
    parser.add_argument("--items", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260115)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_from_kaggle(DATA_DIR)
        print(f"Downloaded {KAGGLE_DATASET} into {DATA_DIR}")
        return 0

    ratings, anime = generate_synthetic(args.users, args.items, args.seed)
    ratings.to_csv(DATA_DIR / "rating.csv", index=False)
    anime.to_csv(DATA_DIR / "anime.csv", index=False)

    density = len(ratings) / (args.users * args.items)
    print(
        f"Synthetic fixture written to {DATA_DIR}\n"
        f"  users={args.users} items={args.items} interactions={len(ratings)}\n"
        f"  density={density:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
