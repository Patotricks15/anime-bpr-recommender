"""Grid-search BPR hyperparameters on the validation partition.

The test partition is deliberately never read here. Selecting on test is the
single easiest way to produce a portfolio number that does not reproduce.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender.cli import _load_split  # noqa: E402
from recommender.evaluation import evaluate  # noqa: E402
from recommender.models import BPRRecommender, ItemKNNRecommender  # noqa: E402

GRID = {
    "n_factors": [16, 32, 64],
    "learning_rate": [0.1, 0.2],
    "reg": [0.02, 0.05, 0.1],
    "n_epochs": [150],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=PROJECT_ROOT / "data" / "rating.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    _, split = _load_split(args.ratings)
    keys = list(GRID)
    results = []

    for values in itertools.product(*(GRID[key] for key in keys)):
        params = dict(zip(keys, values, strict=True))
        reg = params.pop("reg")
        model = BPRRecommender(
            reg_user=reg, reg_item=reg, batch_size=1024, random_state=42, **params
        ).fit(split)
        score = evaluate(model, split, k=args.k, partition="validation").ndcg
        results.append({**params, "reg": reg, "val_ndcg": round(float(score), 4)})
        print(f"{params} reg={reg} -> {score:.4f}", flush=True)

    results.sort(key=lambda row: row["val_ndcg"], reverse=True)

    knn = [
        {
            "top_k_neighbours": n,
            "val_ndcg": round(
                float(
                    evaluate(
                        ItemKNNRecommender(top_k_neighbours=n).fit(split),
                        split,
                        k=args.k,
                        partition="validation",
                    ).ndcg
                ),
                4,
            ),
        }
        for n in (20, 50, 100, 200)
    ]
    knn.sort(key=lambda row: row["val_ndcg"], reverse=True)

    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "tuning.json").write_text(json.dumps({"bpr": results, "itemknn": knn}, indent=2))

    print(f"\nbest BPR:     {results[0]}")
    print(f"best ItemKNN: {knn[0]}")
    print(f"wrote {artifacts / 'tuning.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
