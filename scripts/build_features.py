#!/usr/bin/env python3
"""Build and persist the feature store from the training partition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recommender.data import (  # noqa: E402
    build_implicit_feedback,
    leave_last_one_out,
    load_anime_metadata,
    load_ratings,
)
from recommender.features import build_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir / "rating.csv")
    catalogue = load_anime_metadata(args.data_dir / "anime.csv")

    data = build_implicit_feedback(ratings)
    split = leave_last_one_out(data)
    store = build_features(split, data, catalogue)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = store.to_frames()
    frames["users"].to_parquet(args.out_dir / "features_users.parquet", index=False)
    frames["items"].to_parquet(args.out_dir / "features_items.parquet", index=False)

    print(
        f"Features written to {args.out_dir}\n"
        f"  users={store.n_users} items={store.n_items}\n"
        f"  train={len(split.train)} validation={len(split.validation)} test={len(split.test)}\n"
        f"  declared features={len(store.specs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
