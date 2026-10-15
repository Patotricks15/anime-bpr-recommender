"""Shared pytest fixtures built from the synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recommender.data import (
    build_implicit_feedback,
    leave_last_one_out,
    load_anime_metadata,
    load_ratings,
)
from recommender.features import build_features

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _ensure_dataset() -> None:
    """Generate the synthetic fixture on demand so tests are self-contained."""
    if (DATA_DIR / "rating.csv").exists() and (DATA_DIR / "anime.csv").exists():
        return

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from download_data import generate_synthetic

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ratings, anime = generate_synthetic(n_users=300, n_items=140, seed=7)
    ratings.to_csv(DATA_DIR / "rating.csv", index=False)
    anime.to_csv(DATA_DIR / "anime.csv", index=False)


@pytest.fixture(scope="session")
def raw_ratings() -> pd.DataFrame:
    _ensure_dataset()
    return load_ratings(DATA_DIR / "rating.csv")


@pytest.fixture(scope="session")
def catalogue() -> pd.DataFrame:
    _ensure_dataset()
    return load_anime_metadata(DATA_DIR / "anime.csv")


@pytest.fixture(scope="session")
def interactions(raw_ratings: pd.DataFrame):
    return build_implicit_feedback(raw_ratings)


@pytest.fixture(scope="session")
def split(interactions):
    return leave_last_one_out(interactions)


@pytest.fixture(scope="session")
def feature_store(split, interactions, catalogue):
    return build_features(split, interactions, catalogue)
