"""Command line interface: train, evaluate and serve recommendations.

Hyperparameter defaults are the values selected by a grid search on the
*validation* partition (see ``paper/`` for the sweep). The test partition is
touched only by ``evaluate``, and only once, which is what keeps the reported
numbers honest.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from recommender.data.loader import build_implicit_feedback, load_anime_metadata, load_ratings
from recommender.data.splitting import leave_last_one_out
from recommender.evaluation import evaluate
from recommender.models import (
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_RATINGS = PROJECT_ROOT / "data" / "rating.csv"
DEFAULT_ANIME = PROJECT_ROOT / "data" / "anime.csv"

# Selected on the validation partition; see `scripts/tune.py`.
BEST_BPR = {
    "n_factors": 32,
    "n_epochs": 150,
    "learning_rate": 0.2,
    "reg_user": 0.05,
    "reg_item": 0.05,
    "batch_size": 1024,
}
BEST_KNN = {"top_k_neighbours": 50}


def _load_split(ratings_path: Path):
    data = build_implicit_feedback(load_ratings(ratings_path))
    return data, leave_last_one_out(data)


def _build_models(seed: int) -> dict:
    return {
        "Random": RandomRecommender(random_state=seed),
        "Popularity": PopularityRecommender(),
        "ItemKNN": ItemKNNRecommender(**BEST_KNN),
        "BPR": BPRRecommender(random_state=seed, **BEST_BPR),
    }


def cmd_train(args: argparse.Namespace) -> int:
    data, split = _load_split(args.ratings)
    print(f"users={split.n_users}  items={split.n_items}  train={len(split.train)}")

    model = BPRRecommender(random_state=args.seed, **BEST_BPR)
    model.fit(split, verbose=args.verbose)

    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "bpr.pkl").open("wb") as handle:
        pickle.dump({"model": model, "item_ids": data.item_ids}, handle)

    loss = model.loss_history
    print(f"BPR trained: loss {loss[0]:.4f} -> {loss[-1]:.4f}")
    print(f"saved -> {ARTIFACTS / 'bpr.pkl'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _, split = _load_split(args.ratings)

    rows = []
    for name, model in _build_models(args.seed).items():
        model.fit(split)
        report = evaluate(model, split, k=args.k, partition=args.partition, model_name=name)
        rows.append(report.as_row())
        print(
            f"{name:12} ndcg@{args.k}={report.ndcg:.4f}  recall={report.recall:.4f}  "
            f"mrr={report.mrr:.4f}  coverage={report.coverage:.3f}"
        )

    ARTIFACTS.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("ndcg", ascending=False)
    frame.to_csv(ARTIFACTS / "results.csv", index=False)
    print(f"\nsaved -> {ARTIFACTS / 'results.csv'}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    data, split = _load_split(args.ratings)

    user_lookup = {int(uid): idx for idx, uid in enumerate(data.user_ids)}
    if args.user_id not in user_lookup:
        raise SystemExit(f"user_id {args.user_id} is not in the filtered dataset.")
    user_index = user_lookup[args.user_id]

    model = BPRRecommender(random_state=args.seed, **BEST_BPR).fit(split)
    history = set(split.train.loc[split.train["user_index"] == user_index, "item_index"])
    ranked = model.recommend(user_index, k=args.k, exclude=history)

    titles: dict[int, str] = {}
    if args.anime.exists():
        meta = load_anime_metadata(args.anime)
        titles = dict(zip(meta["anime_id"], meta["name"], strict=False))

    payload = []
    for rank, item_index in enumerate(ranked, start=1):
        anime_id = int(data.item_ids[int(item_index)])
        payload.append(
            {"rank": rank, "anime_id": anime_id, "title": titles.get(anime_id, "?")}
        )

    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anime-bpr", description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit BPR and persist the artifact.")
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Benchmark BPR against the baselines.")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--partition", choices=["validation", "test"], default="test")
    ev.set_defaults(func=cmd_evaluate)

    rec = sub.add_parser("recommend", help="Top-k list for one user.")
    rec.add_argument("--user-id", type=int, required=True)
    rec.add_argument("--k", type=int, default=10)
    rec.add_argument("--anime", type=Path, default=DEFAULT_ANIME)
    rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args(argv)
    if not args.ratings.exists():
        raise SystemExit(
            f"{args.ratings} not found. Run: python scripts/download_data.py --synthetic"
        )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
