#!/usr/bin/env python3
"""
Vanilla LightGCN-style recommender for the phone review dataset.

The script uses the processed interaction file with real user_id/product_id,
builds a user-item graph, trains a lightweight BPR recommender, applies
LightGCN-style neighbor propagation, and exports Top-K recommendations.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from utils import ensure_dirs, setup_logging


logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
SENTIMENT_VECTOR_FILE = PROJECT_ROOT / "data" / "processed" / "sentiment_vectors.csv"
IMPLICIT_SENTIMENT_FILE = PROJECT_ROOT / "data" / "processed" / "implicit_sentiment_full.csv"
RECOMMEND_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommend_result.csv"
ENHANCED_RECOMMEND_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommend_result_sentiment.csv"
METRICS_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_metrics.txt"
ALPHA_SWEEP_OUTPUT = PROJECT_ROOT / "results" / "reports" / "sentiment_alpha_sweep.csv"
QUALITY_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_data_quality.txt"

TOP_K = 10
EMBED_DIM = 16
EPOCHS = 40
LEARNING_RATE = 0.035
REG = 0.0005
RANDOM_STATE = 42
MIN_USER_ITEMS = 2
POSITIVE_RATING = 4.0
MAX_TRAIN_USERS = 6000
SENTIMENT_ALPHAS = [1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_SENTIMENT_ALPHA = 3.0


Vector = List[float]
Interaction = Tuple[str, str, float, str, float]
Recommendation = Dict[str, List[Tuple[str, float]]]
EnhancedRecommendation = Dict[str, List[Tuple[str, float, float, float]]]


def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


def sigmoid(value: float) -> float:
    if value > 20:
        return 1.0
    if value < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def init_embedding(count: int, dim: int, rng: random.Random) -> List[Vector]:
    return [[rng.uniform(-0.05, 0.05) for _ in range(dim)] for _ in range(count)]


def normalize_key_part(value: str | None) -> str:
    return (value or "").strip()


def interaction_key(content: str | None, score: str | None, time_value: str | None) -> Tuple[str, str, str]:
    return (normalize_key_part(content), normalize_key_part(score), normalize_key_part(time_value))


def load_implicit_scores(input_file: Path) -> Dict[Tuple[str, str, str], float]:
    if not input_file.exists():
        logger.warning("Implicit sentiment file not found: %s", input_file)
        return {}

    scores: Dict[Tuple[str, str, str], float] = {}
    with input_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"content", "score", "time", "implicit_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_file}: {sorted(missing)}")

        for row in reader:
            raw_score = safe_float(row.get("implicit_score"), -1.0)
            # -1 means this row has explicit sentiment words; keep it neutral for implicit-only signal.
            implicit_score = raw_score if 0.0 <= raw_score <= 1.0 else 0.5
            scores[interaction_key(row.get("content"), row.get("score"), row.get("time"))] = implicit_score
    return scores


def read_positive_interactions(input_file: Path) -> List[Interaction]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    implicit_scores = load_implicit_scores(IMPLICIT_SENTIMENT_FILE)
    interactions: List[Interaction] = []
    with input_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"user_id", "product_id", "score", "time"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_file}: {sorted(missing)}")

        for row in reader:
            user_id = (row.get("user_id") or "").strip()
            item_id = (row.get("product_id") or "").strip()
            rating = safe_float(row.get("score"), 0.0)
            if not user_id or not item_id or rating < POSITIVE_RATING:
                continue
            implicit_score = implicit_scores.get(
                interaction_key(row.get("content"), row.get("score"), row.get("time")),
                0.5,
            )
            interactions.append((user_id, item_id, rating, row.get("time") or "", implicit_score))
    return interactions


def parse_sentiment_vector(raw_value: str) -> Vector:
    return [safe_float(value, 0.5) for value in raw_value.split()]


def load_sentiment_vectors(vector_file: Path) -> Dict[Tuple[str, str], Vector]:
    if not vector_file.exists():
        logger.warning("Sentiment vector file not found: %s", vector_file)
        return {}

    vectors: Dict[Tuple[str, str], Vector] = {}
    with vector_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"user_id", "item_id", "vector"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {vector_file}: {sorted(missing)}")

        for row in reader:
            user_id = (row.get("user_id") or "").strip()
            item_id = (row.get("item_id") or "").strip()
            vector = parse_sentiment_vector(row.get("vector") or "")
            if user_id and item_id and vector:
                vectors[(user_id, item_id)] = vector
    return vectors


def filter_active_users(interactions: List[Interaction]) -> List[Interaction]:
    user_items: Dict[str, set[str]] = defaultdict(set)
    for user_id, item_id, _, _, _ in interactions:
        user_items[user_id].add(item_id)

    active_users = {user_id for user_id, items in user_items.items() if len(items) >= MIN_USER_ITEMS}
    return [row for row in interactions if row[0] in active_users]


def split_leave_one_out(interactions: List[Interaction]) -> Tuple[List[Interaction], List[Interaction]]:
    by_user: Dict[str, List[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row[0]].append(row)

    train_rows: List[Interaction] = []
    test_rows: List[Interaction] = []
    for rows in by_user.values():
        dedup: Dict[str, Interaction] = {}
        for row in sorted(rows, key=lambda value: value[3]):
            dedup[row[1]] = row
        user_rows = list(dedup.values())
        if len(user_rows) < 2:
            continue
        test_rows.append(user_rows[-1])
        train_rows.extend(user_rows[:-1])
    return train_rows, test_rows


def build_id_maps(rows: List[Interaction]) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    users = sorted({row[0] for row in rows})
    items = sorted({row[1] for row in rows})
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    return user_to_idx, item_to_idx, users, items


def average_vectors(vectors: List[Vector]) -> Vector:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dim)]


def build_sentiment_profiles(
    train_rows: List[Interaction],
    sentiment_vectors: Dict[Tuple[str, str], Vector],
) -> Tuple[Dict[str, Vector], Dict[str, Vector], Vector, Dict[str, float], Dict[str, float], float]:
    user_vectors: Dict[str, List[Vector]] = defaultdict(list)
    item_vectors: Dict[str, List[Vector]] = defaultdict(list)
    user_implicit_scores: Dict[str, List[float]] = defaultdict(list)
    item_implicit_scores: Dict[str, List[float]] = defaultdict(list)
    all_vectors: List[Vector] = []
    all_implicit_scores: List[float] = []

    for user_id, item_id, _, _, implicit_score in train_rows:
        vector = sentiment_vectors.get((user_id, item_id))
        if vector:
            user_vectors[user_id].append(vector)
            item_vectors[item_id].append(vector)
            all_vectors.append(vector)
        user_implicit_scores[user_id].append(implicit_score)
        item_implicit_scores[item_id].append(implicit_score)
        all_implicit_scores.append(implicit_score)

    user_profiles = {user_id: average_vectors(vectors) for user_id, vectors in user_vectors.items()}
    item_profiles = {item_id: average_vectors(vectors) for item_id, vectors in item_vectors.items()}
    global_profile = average_vectors(all_vectors)
    user_implicit = {
        user_id: sum(scores) / len(scores) for user_id, scores in user_implicit_scores.items()
    }
    item_implicit = {
        item_id: sum(scores) / len(scores) for item_id, scores in item_implicit_scores.items()
    }
    global_implicit = sum(all_implicit_scores) / len(all_implicit_scores) if all_implicit_scores else 0.5
    return user_profiles, item_profiles, global_profile, user_implicit, item_implicit, global_implicit


def centered(vector: Sequence[float]) -> Vector:
    return [value - 0.5 for value in vector]


def sentiment_match_score(user_profile: Vector, item_profile: Vector) -> float:
    if not user_profile or not item_profile:
        return 0.5

    centered_user = centered(user_profile)
    centered_item = centered(item_profile)
    similarity = (cosine(centered_user, centered_item) + 1.0) / 2.0
    item_sentiment = sum(item_profile) / len(item_profile)
    return 0.7 * similarity + 0.3 * item_sentiment


def implicit_match_score(user_score: float, item_score: float) -> float:
    # Higher when the user's implicit preference level is close to the item's implicit sentiment profile.
    return 1.0 - abs(user_score - item_score)


def encode_rows(
    rows: List[Interaction], user_to_idx: Dict[str, int], item_to_idx: Dict[str, int]
) -> List[Tuple[int, int]]:
    encoded = []
    for user_id, item_id, _, _, _ in rows:
        if user_id in user_to_idx and item_id in item_to_idx:
            encoded.append((user_to_idx[user_id], item_to_idx[item_id]))
    return encoded


def limit_users_for_training(
    train_pairs: List[Tuple[int, int]], max_users: int
) -> List[Tuple[int, int]]:
    allowed_users = sorted({user for user, _ in train_pairs})[:max_users]
    allowed = set(allowed_users)
    return [(user, item) for user, item in train_pairs if user in allowed]


def train_bpr_embeddings(
    train_pairs: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
    user_items: Dict[int, set[int]],
) -> Tuple[List[Vector], List[Vector]]:
    rng = random.Random(RANDOM_STATE)
    user_emb = init_embedding(num_users, EMBED_DIM, rng)
    item_emb = init_embedding(num_items, EMBED_DIM, rng)
    all_items = list(range(num_items))

    if not train_pairs:
        raise ValueError("No training pairs available for recommender training.")

    for epoch in range(EPOCHS):
        rng.shuffle(train_pairs)
        total_loss = 0.0
        updates = 0

        for user_idx, pos_item_idx in train_pairs:
            if len(user_items[user_idx]) >= num_items:
                continue

            neg_item_idx = rng.choice(all_items)
            while neg_item_idx in user_items[user_idx]:
                neg_item_idx = rng.choice(all_items)

            user_vec = user_emb[user_idx]
            pos_vec = item_emb[pos_item_idx]
            neg_vec = item_emb[neg_item_idx]
            score_diff = dot(user_vec, pos_vec) - dot(user_vec, neg_vec)
            grad = 1.0 - sigmoid(score_diff)

            old_user = user_vec[:]
            old_pos = pos_vec[:]
            old_neg = neg_vec[:]
            for dim in range(EMBED_DIM):
                user_vec[dim] += LEARNING_RATE * (
                    grad * (old_pos[dim] - old_neg[dim]) - REG * old_user[dim]
                )
                pos_vec[dim] += LEARNING_RATE * (grad * old_user[dim] - REG * old_pos[dim])
                neg_vec[dim] += LEARNING_RATE * (-grad * old_user[dim] - REG * old_neg[dim])

            total_loss += -math.log(max(sigmoid(score_diff), 1e-12))
            updates += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(updates, 1)
            logger.info("Epoch %03d/%03d | BPR loss %.4f", epoch + 1, EPOCHS, avg_loss)

    return user_emb, item_emb


def lightgcn_propagate(
    user_emb: List[Vector],
    item_emb: List[Vector],
    user_items: Dict[int, set[int]],
    item_users: Dict[int, set[int]],
) -> Tuple[List[Vector], List[Vector]]:
    propagated_users = [vec[:] for vec in user_emb]
    propagated_items = [vec[:] for vec in item_emb]

    for user_idx, items in user_items.items():
        if not items:
            continue
        for dim in range(EMBED_DIM):
            neighbor_avg = sum(item_emb[item_idx][dim] for item_idx in items) / len(items)
            propagated_users[user_idx][dim] = 0.5 * user_emb[user_idx][dim] + 0.5 * neighbor_avg

    for item_idx, users in item_users.items():
        if not users:
            continue
        for dim in range(EMBED_DIM):
            neighbor_avg = sum(user_emb[user_idx][dim] for user_idx in users) / len(users)
            propagated_items[item_idx][dim] = 0.5 * item_emb[item_idx][dim] + 0.5 * neighbor_avg

    return propagated_users, propagated_items


def recommend(
    user_emb: List[Vector],
    item_emb: List[Vector],
    users: List[str],
    items: List[str],
    seen_items: Dict[int, set[int]],
    top_k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    recommendations: Dict[str, List[Tuple[str, float]]] = {}
    for user_idx, user_id in enumerate(users):
        scores = []
        for item_idx, item_id in enumerate(items):
            if item_idx in seen_items.get(user_idx, set()):
                continue
            scores.append((item_id, dot(user_emb[user_idx], item_emb[item_idx])))
        scores.sort(key=lambda row: row[1], reverse=True)
        recommendations[user_id] = scores[:top_k]
    return recommendations


def recommend_with_sentiment(
    user_emb: List[Vector],
    item_emb: List[Vector],
    users: List[str],
    items: List[str],
    seen_items: Dict[int, set[int]],
    user_profiles: Dict[str, Vector],
    item_profiles: Dict[str, Vector],
    global_profile: Vector,
    user_implicit: Dict[str, float],
    item_implicit: Dict[str, float],
    global_implicit: float,
    alpha: float,
    top_k: int,
) -> EnhancedRecommendation:
    recommendations: EnhancedRecommendation = {}
    for user_idx, user_id in enumerate(users):
        user_profile = user_profiles.get(user_id, global_profile)
        scores = []
        for item_idx, item_id in enumerate(items):
            if item_idx in seen_items.get(user_idx, set()):
                continue
            lightgcn_score = dot(user_emb[user_idx], item_emb[item_idx])
            item_profile = item_profiles.get(item_id, global_profile)
            aspect_score = sentiment_match_score(user_profile, item_profile)
            implicit_score = implicit_match_score(
                user_implicit.get(user_id, global_implicit),
                item_implicit.get(item_id, global_implicit),
            )
            sentiment_score = 0.75 * aspect_score + 0.25 * implicit_score
            final_score = lightgcn_score + alpha * sentiment_score
            scores.append((item_id, final_score, lightgcn_score, sentiment_score))
        scores.sort(key=lambda row: row[1], reverse=True)
        recommendations[user_id] = scores[:top_k]
    return recommendations


def evaluate(
    recommendations: Dict[str, List[Tuple[str, float]]],
    test_rows: List[Interaction],
    top_k: int,
) -> Tuple[float, float, int]:
    hits = 0
    ndcg = 0.0
    evaluated = 0

    for user_id, target_item, _, _, _ in test_rows:
        ranked_items = [item_id for item_id, _ in recommendations.get(user_id, [])[:top_k]]
        if not ranked_items:
            continue
        evaluated += 1
        if target_item in ranked_items:
            rank = ranked_items.index(target_item) + 1
            hits += 1
            ndcg += 1.0 / math.log2(rank + 1)

    if evaluated == 0:
        return 0.0, 0.0, 0
    return hits / evaluated, ndcg / evaluated, evaluated


def evaluate_enhanced(
    recommendations: EnhancedRecommendation,
    test_rows: List[Interaction],
    top_k: int,
) -> Tuple[float, float, int]:
    compact_recommendations: Recommendation = {
        user_id: [(item_id, final_score) for item_id, final_score, _, _ in rows]
        for user_id, rows in recommendations.items()
    }
    return evaluate(compact_recommendations, test_rows, top_k)


def write_recommendations(recommendations: Dict[str, List[Tuple[str, float]]]) -> None:
    with RECOMMEND_OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "item_id", "rank", "score", "model"])
        writer.writeheader()
        for user_id in sorted(recommendations):
            for rank, (item_id, score) in enumerate(recommendations[user_id], start=1):
                writer.writerow(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rank": rank,
                        "score": f"{score:.6f}",
                        "model": "vanilla_lightgcn",
                    }
                )


def write_enhanced_recommendations(recommendations: EnhancedRecommendation) -> None:
    with ENHANCED_RECOMMEND_OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "user_id",
                "item_id",
                "rank",
                "score",
                "lightgcn_score",
                "sentiment_score",
                "model",
            ],
        )
        writer.writeheader()
        for user_id in sorted(recommendations):
            for rank, (item_id, final_score, lightgcn_score, sentiment_score) in enumerate(
                recommendations[user_id], start=1
            ):
                writer.writerow(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rank": rank,
                        "score": f"{final_score:.6f}",
                        "lightgcn_score": f"{lightgcn_score:.6f}",
                        "sentiment_score": f"{sentiment_score:.6f}",
                        "model": "sentiment_enhanced_lightgcn",
                    }
                )


def write_reports(
    raw_count: int,
    filtered_count: int,
    train_count: int,
    test_count: int,
    user_count: int,
    item_count: int,
    recall: float,
    ndcg: float,
    evaluated_users: int,
    sentiment_vector_count: int,
    alpha_results: List[Tuple[float, float, float, int]],
) -> None:
    QUALITY_OUTPUT.write_text(
        "\n".join(
            [
                "========== Recommendation Data Quality ==========",
                f"Input file: {INPUT_FILE}",
                f"Raw positive interactions (score >= {POSITIVE_RATING:.1f}): {raw_count}",
                f"Filtered interactions (users with >= {MIN_USER_ITEMS} items): {filtered_count}",
                f"Train interactions: {train_count}",
                f"Test interactions: {test_count}",
                f"Users used: {user_count}",
                f"Items used: {item_count}",
                f"Sentiment vectors loaded: {sentiment_vector_count}",
                "Note: product_id is used as item_id for the phone-category recommendation experiment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    METRICS_OUTPUT.write_text(
        "\n".join(
            [
                "========== Recommendation Metrics ==========",
                "Model: Vanilla LightGCN-style BPR",
                f"Recall@{TOP_K}: {recall:.6f}",
                f"NDCG@{TOP_K}: {ndcg:.6f}",
                f"Evaluated users: {evaluated_users}",
                f"Candidate items: {item_count}",
                "",
                "Model: Sentiment-enhanced LightGCN re-ranker",
                "Fusion: final_score = lightgcn_score + alpha * sentiment_score",
                "sentiment_score = 0.75 * aspect_vector_match + 0.25 * implicit_score_match",
                "Alpha sweep:",
            ]
            + [
                f"alpha={alpha:.2f} | Recall@{TOP_K}: {recall_value:.6f} | "
                f"NDCG@{TOP_K}: {ndcg_value:.6f} | Evaluated users: {evaluated_value}"
                for alpha, recall_value, ndcg_value, evaluated_value in alpha_results
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with ALPHA_SWEEP_OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["alpha", f"recall@{TOP_K}", f"ndcg@{TOP_K}", "evaluated_users"],
        )
        writer.writeheader()
        for alpha, recall_value, ndcg_value, evaluated_value in alpha_results:
            writer.writerow(
                {
                    "alpha": f"{alpha:.2f}",
                    f"recall@{TOP_K}": f"{recall_value:.6f}",
                    f"ndcg@{TOP_K}": f"{ndcg_value:.6f}",
                    "evaluated_users": evaluated_value,
                }
            )


def main() -> None:
    ensure_dirs()
    random.seed(RANDOM_STATE)

    raw_interactions = read_positive_interactions(INPUT_FILE)
    filtered_interactions = filter_active_users(raw_interactions)
    train_rows, test_rows = split_leave_one_out(filtered_interactions)
    sentiment_vectors = load_sentiment_vectors(SENTIMENT_VECTOR_FILE)

    user_to_idx, item_to_idx, users, items = build_id_maps(train_rows + test_rows)
    train_pairs = encode_rows(train_rows, user_to_idx, item_to_idx)
    train_pairs = limit_users_for_training(train_pairs, MAX_TRAIN_USERS)

    user_items: Dict[int, set[int]] = defaultdict(set)
    item_users: Dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in train_pairs:
        user_items[user_idx].add(item_idx)
        item_users[item_idx].add(user_idx)

    logger.info(
        "Training vanilla LightGCN-style recommender: users=%d items=%d train_pairs=%d",
        len(users),
        len(items),
        len(train_pairs),
    )
    user_emb, item_emb = train_bpr_embeddings(train_pairs, len(users), len(items), user_items)
    user_emb, item_emb = lightgcn_propagate(user_emb, item_emb, user_items, item_users)
    recommendations = recommend(user_emb, item_emb, users, items, user_items, TOP_K)
    recall, ndcg, evaluated_users = evaluate(recommendations, test_rows, TOP_K)
    (
        user_profiles,
        item_profiles,
        global_profile,
        user_implicit,
        item_implicit,
        global_implicit,
    ) = build_sentiment_profiles(train_rows, sentiment_vectors)

    alpha_results: List[Tuple[float, float, float, int]] = []
    enhanced_recommendations: EnhancedRecommendation = {}
    for alpha in SENTIMENT_ALPHAS:
        candidate_recommendations = recommend_with_sentiment(
            user_emb,
            item_emb,
            users,
            items,
            user_items,
            user_profiles,
            item_profiles,
            global_profile,
            user_implicit,
            item_implicit,
            global_implicit,
            alpha,
            TOP_K,
        )
        alpha_recall, alpha_ndcg, alpha_evaluated_users = evaluate_enhanced(
            candidate_recommendations, test_rows, TOP_K
        )
        alpha_results.append((alpha, alpha_recall, alpha_ndcg, alpha_evaluated_users))
        if alpha == DEFAULT_SENTIMENT_ALPHA:
            enhanced_recommendations = candidate_recommendations

    if not enhanced_recommendations:
        enhanced_recommendations = recommend_with_sentiment(
            user_emb,
            item_emb,
            users,
            items,
            user_items,
            user_profiles,
            item_profiles,
            global_profile,
            user_implicit,
            item_implicit,
            global_implicit,
            SENTIMENT_ALPHAS[0],
            TOP_K,
        )

    write_recommendations(recommendations)
    write_enhanced_recommendations(enhanced_recommendations)
    write_reports(
        raw_count=len(raw_interactions),
        filtered_count=len(filtered_interactions),
        train_count=len(train_pairs),
        test_count=len(test_rows),
        user_count=len(users),
        item_count=len(items),
        recall=recall,
        ndcg=ndcg,
        evaluated_users=evaluated_users,
        sentiment_vector_count=len(sentiment_vectors),
        alpha_results=alpha_results,
    )

    logger.info("Recommendation output: %s", RECOMMEND_OUTPUT)
    logger.info("Enhanced recommendation output: %s", ENHANCED_RECOMMEND_OUTPUT)
    logger.info("Metrics output: %s", METRICS_OUTPUT)
    logger.info("Alpha sweep output: %s", ALPHA_SWEEP_OUTPUT)
    logger.info("Data quality output: %s", QUALITY_OUTPUT)


if __name__ == "__main__":
    main()
