"""Training/serving parity tests.

The failure mode these guard against is skew: the offline evaluation scores
one ranking while production serves a different one, so good offline numbers
never materialize. Every assertion compares the served artifact against the
exact objects the evaluation used.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from recommender.evaluation import evaluate
from recommender.models import BPRRecommender
from recommender.serving import build_topn_payloads, parse_request
from recommender.serving.handler import _DynamoEncoder, lambda_handler


@pytest.fixture(scope="module")
def model(split):
    return BPRRecommender(n_factors=16, n_epochs=20, random_state=5).fit(split)


@pytest.fixture(scope="module")
def payloads(model, split, interactions):
    return build_topn_payloads(
        model, split, item_ids=interactions.item_ids, user_ids=interactions.user_ids, k=10
    )


def test_every_user_gets_a_payload(payloads, split) -> None:
    assert len(payloads) == split.n_users


def test_served_ranking_matches_offline_ranking(payloads, model, split, interactions) -> None:
    """The stored list must equal what `recommend` produced offline."""
    history = {
        int(u): set(map(int, i)) for u, i in split.train.groupby("user_index")["item_index"]
    }

    for user_index in range(min(25, split.n_users)):
        expected = model.recommend(user_index, k=10, exclude=history.get(user_index, set()))
        expected_ids = [str(int(interactions.item_ids[int(i)])) for i in expected]
        served_ids = [r["anime_id"] for r in payloads[user_index]["recommendations"]]
        assert served_ids == expected_ids, f"serving skew for user_index {user_index}"


def test_served_lists_never_contain_watch_history(payloads, split, interactions) -> None:
    """Re-recommending an already-watched title is the most visible failure."""
    id_lookup = {int(v): k for k, v in enumerate(interactions.item_ids)}

    for user_index, payload in enumerate(payloads):
        seen = set(split.train.loc[split.train["user_index"] == user_index, "item_index"])
        served = {id_lookup[int(r["anime_id"])] for r in payload["recommendations"]}
        assert not served & seen, f"user_index {user_index} was served items from its history"


def test_payload_is_json_serialisable(payloads) -> None:
    """DynamoDB rejects numpy scalars; this catches them before deployment."""
    json.dumps(payloads[:5])


def test_ranks_are_contiguous_and_ordered(payloads) -> None:
    for payload in payloads[:25]:
        ranks = [r["rank"] for r in payload["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))


def test_build_topn_rejects_misaligned_ids(model, split, interactions) -> None:
    with pytest.raises(ValueError, match="user_ids"):
        build_topn_payloads(
            model, split, item_ids=interactions.item_ids, user_ids=interactions.user_ids[:-1]
        )


def test_build_topn_rejects_non_positive_k(model, split, interactions) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        build_topn_payloads(
            model, split, item_ids=interactions.item_ids, user_ids=interactions.user_ids, k=0
        )


def test_offline_metrics_are_reproducible(model, split) -> None:
    """Evaluating twice must give identical numbers, or tuning is meaningless."""
    first = evaluate(model, split, k=10, partition="validation")
    second = evaluate(model, split, k=10, partition="validation")
    assert first == second


# --- Request validation (no AWS required) ----------------------------------


@pytest.mark.parametrize(
    "event",
    [
        {"queryStringParameters": {}},
        {"queryStringParameters": {"user_id": ""}},
        {"queryStringParameters": None},
    ],
    ids=["missing", "empty", "null-params"],
)
def test_missing_user_id_is_rejected(event) -> None:
    with pytest.raises(ValueError, match="user_id"):
        parse_request(event)


@pytest.mark.parametrize("bad_k", ["0", "-1", "101", "abc"])
def test_invalid_k_is_rejected(bad_k) -> None:
    with pytest.raises(ValueError, match="k"):
        parse_request({"queryStringParameters": {"user_id": "7", "k": bad_k}})


def test_valid_request_is_parsed() -> None:
    assert parse_request({"queryStringParameters": {"user_id": "7", "k": "5"}}) == ("7", 5)


def test_handler_returns_400_without_touching_dynamodb() -> None:
    """A malformed request must fail fast, before any network call."""
    response = lambda_handler({"queryStringParameters": {}})
    assert response["statusCode"] == 400
    assert "user_id" in json.loads(response["body"])["error"]


def test_response_serialises_dynamodb_decimals() -> None:
    """DynamoDB returns every number as Decimal, which json.dumps rejects.

    Regression test: this only reproduces on the *read* path, so the
    pre-write payload checks above cannot catch it. It was found by invoking
    the deployed Lambda against LocalStack, not by the unit suite.
    """
    body = {
        "user_id": "1",
        "k": Decimal("3"),
        "recommendations": [{"rank": Decimal("1"), "anime_id": "275"}],
        "score": Decimal("0.5"),
    }
    decoded = json.loads(json.dumps(body, cls=_DynamoEncoder))

    assert decoded["k"] == 3
    assert isinstance(decoded["k"], int)
    assert decoded["recommendations"][0]["rank"] == 1
    assert decoded["score"] == 0.5


def test_encoder_still_rejects_genuinely_unserialisable_types() -> None:
    """The encoder must not become a silent catch-all."""
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, cls=_DynamoEncoder)
