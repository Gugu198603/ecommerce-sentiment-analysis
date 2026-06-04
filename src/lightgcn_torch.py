#!/usr/bin/env python3
"""
Standard PyTorch LightGCN recommender with causal edge weighting.

This script trains a multi-layer LightGCN model with BPR loss on the processed
phone review interactions. It uses user_id as user nodes and product_id as item
nodes, estimates propensity scores for sentiment treatment, converts the causal
signal into edge/sample weights, and exports Top-K recommendations and
Recall/NDCG metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from causalml.inference.tree import CausalTreeRegressor
    HAS_CAUSALML = True
except ImportError:
    HAS_CAUSALML = False
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils import ensure_dirs, setup_logging


logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
IMPLICIT_SENTIMENT_FILE = PROJECT_ROOT / "data" / "processed" / "implicit_sentiment_full.csv"
RECOMMEND_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommend_result_torch.csv"
METRICS_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_metrics_torch.txt"
QUALITY_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_data_quality_torch.txt"

TOP_K = 10
POSITIVE_RATING = 4.0
MIN_USER_ITEMS = 2
RANDOM_STATE = 42
CALIPER = 0.05


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float
    time: str
    sentiment: float
    implicit_score: float
    overall_sentiment: float
    review_length: int


@dataclass(frozen=True)
class EncodedInteraction:
    user_idx: int
    item_idx: int
    weight: float


@dataclass
class CausalStats:
    treatment_count: int
    control_count: int
    propensity_mean: float
    propensity_min: float
    propensity_max: float
    matched_pairs: int
    psm_ate: float
    cate_mean: float
    cate_min: float
    cate_max: float
    weight_mean: float
    weight_min: float
    weight_max: float


@dataclass
class DataStats:
    raw_positive_count: int
    filtered_count: int
    train_count: int
    test_count: int
    user_count: int
    item_count: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
            scores[interaction_key(row.get("content"), row.get("score"), row.get("time"))] = safe_float(
                row.get("implicit_score"), -1.0
            )
    return scores


def read_positive_interactions(input_file: Path, positive_rating: float) -> List[Interaction]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    implicit_lookup = load_implicit_scores(IMPLICIT_SENTIMENT_FILE)
    interactions: List[Interaction] = []
    with input_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"user_id", "product_id", "score", "time", "sentiment", "content"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_file}: {sorted(missing)}")

        for row in reader:
            user_id = (row.get("user_id") or "").strip()
            item_id = (row.get("product_id") or "").strip()
            rating = safe_float(row.get("score"), 0.0)
            if not user_id or not item_id or rating < positive_rating:
                continue

            explicit_sentiment = safe_float(row.get("sentiment"), 0.5)
            raw_implicit = implicit_lookup.get(
                interaction_key(row.get("content"), row.get("score"), row.get("time")),
                explicit_sentiment,
            )
            # When implicit_score == -1, the row already has explicit sentiment. Use explicit sentiment
            # instead of passing -1 downstream.
            implicit_score = raw_implicit if 0.0 <= raw_implicit <= 1.0 else explicit_sentiment
            overall_sentiment = 0.6 * explicit_sentiment + 0.4 * implicit_score

            interactions.append(
                Interaction(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    time=row.get("time") or "",
                    sentiment=explicit_sentiment,
                    implicit_score=implicit_score,
                    overall_sentiment=overall_sentiment,
                    review_length=len((row.get("content") or "").strip()),
                )
            )
    return interactions


def filter_active_users(interactions: List[Interaction], min_items: int) -> List[Interaction]:
    user_items: Dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        user_items[row.user_id].add(row.item_id)

    active_users = {user_id for user_id, items in user_items.items() if len(items) >= min_items}
    return [row for row in interactions if row.user_id in active_users]


def split_leave_one_out(interactions: List[Interaction]) -> Tuple[List[Interaction], List[Interaction]]:
    by_user: Dict[str, List[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row.user_id].append(row)

    train_rows: List[Interaction] = []
    test_rows: List[Interaction] = []
    for rows in by_user.values():
        dedup: Dict[str, Interaction] = {}
        for row in sorted(rows, key=lambda value: value.time):
            dedup[row.item_id] = row
        user_rows = list(dedup.values())
        if len(user_rows) < 2:
            continue
        test_rows.append(user_rows[-1])
        train_rows.extend(user_rows[:-1])
    return train_rows, test_rows


def build_id_maps(rows: Sequence[Interaction]) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    users = sorted({row.user_id for row in rows})
    items = sorted({row.item_id for row in rows})
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    return user_to_idx, item_to_idx, users, items


def smd_score(group1: np.ndarray, group2: np.ndarray) -> float:
    diff = float(np.mean(group1) - np.mean(group2))
    pooled_std = math.sqrt(float((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2.0))
    if pooled_std < 1e-10:
        return 0.0
    return diff / pooled_std


def ps_matching(propensity_scores: np.ndarray, treatment: np.ndarray, caliper: float = CALIPER) -> Tuple[np.ndarray, np.ndarray]:
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    ps_treated = propensity_scores[treated_idx].reshape(-1, 1)
    ps_control = propensity_scores[control_idx].reshape(-1, 1)
    n_neighbors = min(10, len(control_idx))
    nn_match = NearestNeighbors(n_neighbors=n_neighbors)
    nn_match.fit(ps_control)
    distances, indices = nn_match.kneighbors(ps_treated)

    matched_t, matched_c = [], []
    control_used = set()
    order = np.argsort(distances[:, 0])
    for i in order:
        for j in range(distances.shape[1]):
            dist = distances[i, j]
            if dist > caliper:
                break
            c_rel = indices[i, j]
            c_abs = control_idx[c_rel]
            if c_abs not in control_used:
                matched_t.append(treated_idx[i])
                matched_c.append(c_abs)
                control_used.add(c_abs)
                break

    return np.array(matched_t, dtype=int), np.array(matched_c, dtype=int)


class SimpleCausalForest:
    def __init__(self, n_estimators: int = 30, max_depth: int = 6, random_state: int = RANDOM_STATE) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: List[CausalTreeRegressor] = []

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "SimpleCausalForest":
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            tree = CausalTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + i)
            tree.fit(X[idx], treatment[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            return np.zeros(X.shape[0], dtype=float)
        preds = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            preds += tree.predict(X)
        return preds / len(self.trees)


def normalize_signal(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values)
    centered = (values - values.mean()) / (values.std() + 1e-8)
    return np.tanh(centered)


def compute_causal_weights(
    interactions: Sequence[Interaction],
    causal_alpha: float,
    causal_mode: str,
) -> Tuple[Dict[Tuple[str, str, str], float], CausalStats]:
    frame = pd.DataFrame(
        {
            "user_id": [row.user_id for row in interactions],
            "item_id": [row.item_id for row in interactions],
            "rating": [row.rating for row in interactions],
            "time": [row.time for row in interactions],
            "overall_sentiment": [row.overall_sentiment for row in interactions],
            "review_length": [row.review_length for row in interactions],
        }
    )
    frame["treatment"] = (frame["overall_sentiment"] > 0.5).astype(int)
    treatment = frame["treatment"].to_numpy()
    treatment_count = int(treatment.sum())
    control_count = int(len(frame) - treatment_count)

    if treatment_count == 0 or control_count == 0:
        logger.warning("Treatment or control group is empty; causal weighting falls back to 1.0.")
        default_weights = {(row.user_id, row.item_id, row.time): 1.0 for row in interactions}
        return default_weights, CausalStats(
            treatment_count=treatment_count,
            control_count=control_count,
            propensity_mean=0.5,
            propensity_min=0.5,
            propensity_max=0.5,
            matched_pairs=0,
            psm_ate=0.0,
            cate_mean=0.0,
            cate_min=0.0,
            cate_max=0.0,
            weight_mean=1.0,
            weight_min=1.0,
            weight_max=1.0,
        )

    user_counts = frame["user_id"].value_counts()
    frame["user_activity"] = frame["user_id"].map(user_counts)
    item_encoder = LabelEncoder()
    frame["brand"] = item_encoder.fit_transform(frame["item_id"].astype(str))
    item_freq = frame["item_id"].value_counts()
    _, price_bins = pd.qcut(
        frame["item_id"].map(item_freq),
        q=4,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    frame["price_level"] = pd.cut(
        frame["item_id"].map(item_freq),
        bins=price_bins,
        labels=False,
        include_lowest=True,
    )
    frame["price_level"] = frame["price_level"].fillna(0)

    confounders = frame[["user_activity", "brand", "price_level", "review_length"]].to_numpy(dtype=float)
    outcome = frame["rating"].to_numpy(dtype=float)

    ps_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    ps_model.fit(confounders, treatment)
    propensity = ps_model.predict_proba(confounders)[:, 1]
    matched_t, matched_c = ps_matching(propensity, treatment, caliper=CALIPER)

    if len(matched_t) >= 2:
        psm_ate = float(np.mean(outcome[matched_t]) - np.mean(outcome[matched_c]))
    else:
        psm_ate = float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0]))

    # Residual sentiment after adjusting for confounders; positive means sentiment is stronger
    # than what confounders alone would predict.
    residual = frame["overall_sentiment"].to_numpy(dtype=float) - propensity
    cate = np.zeros(len(frame), dtype=float)
    if HAS_CAUSALML and len(matched_t) >= 50 and len(matched_c) >= 50:
        matched_idx = np.concatenate([matched_t, matched_c])
        forest = SimpleCausalForest(n_estimators=30, max_depth=6, random_state=RANDOM_STATE)
        forest.fit(confounders[matched_idx], treatment[matched_idx], outcome[matched_idx])
        cate = forest.predict(confounders)

    residual_signal = normalize_signal(residual)
    cate_signal = normalize_signal(cate)
    combined_signal = 0.6 * residual_signal + 0.4 * cate_signal
    causal_multiplier = max(psm_ate, 0.0) / 2.0

    if causal_mode == "rating_direct":
        rating_signal = (frame["rating"].to_numpy(dtype=float) - 3.0) / 2.0
        combined_signal = 0.5 * combined_signal + 0.5 * rating_signal

    weights = np.clip(1.0 + causal_alpha * causal_multiplier * combined_signal, 0.2, 3.0)

    weight_map = {
        (row.user_id, row.item_id, row.time): float(weight)
        for row, weight in zip(interactions, weights, strict=False)
    }
    return weight_map, CausalStats(
        treatment_count=treatment_count,
        control_count=control_count,
        propensity_mean=float(propensity.mean()),
        propensity_min=float(propensity.min()),
        propensity_max=float(propensity.max()),
        matched_pairs=int(len(matched_t)),
        psm_ate=psm_ate,
        cate_mean=float(cate.mean()) if len(cate) else 0.0,
        cate_min=float(cate.min()) if len(cate) else 0.0,
        cate_max=float(cate.max()) if len(cate) else 0.0,
        weight_mean=float(weights.mean()),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
    )


def encode_rows(
    rows: Iterable[Interaction],
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
    causal_weights: Dict[Tuple[str, str, str], float],
) -> List[EncodedInteraction]:
    pairs = []
    for row in rows:
        if row.user_id in user_to_idx and row.item_id in item_to_idx:
            pairs.append(
                EncodedInteraction(
                    user_idx=user_to_idx[row.user_id],
                    item_idx=item_to_idx[row.item_id],
                    weight=causal_weights.get((row.user_id, row.item_id, row.time), 1.0),
                )
            )
    return pairs


def build_user_items(pairs: Iterable[EncodedInteraction]) -> Dict[int, set[int]]:
    user_items: Dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        user_items[pair.user_idx].add(pair.item_idx)
    return user_items


class BPRDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[EncodedInteraction],
        user_items: Dict[int, set[int]],
        num_items: int,
        seed: int,
    ) -> None:
        self.pairs = list(pairs)
        self.user_items = user_items
        self.num_items = num_items
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[int, int, int, float]:
        pair = self.pairs[index]
        user_idx, pos_item_idx = pair.user_idx, pair.item_idx
        if len(self.user_items[user_idx]) >= self.num_items:
            neg_item_idx = pos_item_idx
        else:
            neg_item_idx = self.rng.randrange(self.num_items)
            while neg_item_idx in self.user_items[user_idx]:
                neg_item_idx = self.rng.randrange(self.num_items)
        return user_idx, pos_item_idx, neg_item_idx, pair.weight


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        normalized_adj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.normalized_adj = normalized_adj
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embeddings = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.normalized_adj, all_embeddings)
            layer_embeddings.append(all_embeddings)
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        sample_weight: torch.Tensor,
        reg_weight: float,
    ) -> torch.Tensor:
        user_embeddings, item_embeddings = self.propagate()
        user_vec = user_embeddings[users]
        pos_vec = item_embeddings[pos_items]
        neg_vec = item_embeddings[neg_items]

        pos_scores = torch.sum(user_vec * pos_vec, dim=1)
        neg_scores = torch.sum(user_vec * neg_vec, dim=1)
        ranking_terms = -F.logsigmoid(pos_scores - neg_scores)
        ranking_loss = (ranking_terms * sample_weight).mean()

        ego_user = self.user_embedding(users)
        ego_pos = self.item_embedding(pos_items)
        ego_neg = self.item_embedding(neg_items)
        reg_loss = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / (2.0 * users.shape[0])
        return ranking_loss + reg_weight * reg_loss


def build_normalized_adj(
    train_pairs: Sequence[EncodedInteraction],
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = num_users + num_items
    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []

    for pair in train_pairs:
        item_node = num_users + pair.item_idx
        rows.extend([pair.user_idx, item_node])
        cols.extend([item_node, pair.user_idx])
        weights.extend([pair.weight, pair.weight])

    if not rows:
        raise ValueError("No training pairs available for adjacency construction.")

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(weights, dtype=torch.float32)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.index_add_(0, indices[0], values)
    degree_inv_sqrt = degree.clamp(min=1.0).pow(-0.5)
    norm_values = degree_inv_sqrt[indices[0]] * values * degree_inv_sqrt[indices[1]]

    return torch.sparse_coo_tensor(
        indices.to(device),
        norm_values.to(device),
        (num_nodes, num_nodes),
        device=device,
    ).coalesce()


def train_model(
    model: LightGCN,
    train_pairs: Sequence[EncodedInteraction],
    user_items: Dict[int, set[int]],
    num_items: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = BPRDataset(train_pairs, user_items, num_items, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batches = 0
        for users, pos_items, neg_items, sample_weight in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            sample_weight = sample_weight.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items, sample_weight, args.reg)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            logger.info("Epoch %03d/%03d | BPR loss %.6f", epoch, args.epochs, total_loss / max(batches, 1))


@torch.no_grad()
def recommend_topk(
    model: LightGCN,
    train_user_items: Dict[int, set[int]],
    users: List[str],
    items: List[str],
    top_k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    model.eval()
    user_embeddings, item_embeddings = model.propagate()
    score_matrix = torch.matmul(user_embeddings, item_embeddings.t()).cpu()

    recommendations: Dict[str, List[Tuple[str, float]]] = {}
    for user_idx, user_id in enumerate(users):
        seen = train_user_items.get(user_idx, set())
        if seen:
            score_matrix[user_idx, list(seen)] = -float("inf")
        k = min(top_k, len(items) - len(seen))
        if k <= 0:
            recommendations[user_id] = []
            continue
        scores, item_indices = torch.topk(score_matrix[user_idx], k=k)
        recommendations[user_id] = [
            (items[int(item_idx)], float(score))
            for item_idx, score in zip(item_indices.tolist(), scores.tolist())
        ]
    return recommendations


def evaluate(
    recommendations: Dict[str, List[Tuple[str, float]]],
    test_rows: Sequence[Interaction],
    top_k: int,
) -> Tuple[float, float, int]:
    hits = 0
    ndcg = 0.0
    evaluated = 0

    for row in test_rows:
        ranked_items = [item_id for item_id, _ in recommendations.get(row.user_id, [])[:top_k]]
        if not ranked_items:
            continue
        evaluated += 1
        if row.item_id in ranked_items:
            rank = ranked_items.index(row.item_id) + 1
            hits += 1
            ndcg += 1.0 / math.log2(rank + 1)

    if evaluated == 0:
        return 0.0, 0.0, 0
    return hits / evaluated, ndcg / evaluated, evaluated


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
                        "model": "pytorch_lightgcn_causal",
                    }
                )


def write_reports(stats: DataStats, causal_stats: CausalStats, recall: float, ndcg: float, evaluated_users: int, args: argparse.Namespace) -> None:
    QUALITY_OUTPUT.write_text(
        "\n".join(
            [
                "========== PyTorch LightGCN Data Quality ==========" ,
                f"Input file: {INPUT_FILE}",
                f"Raw positive interactions (score >= {args.positive_rating:.1f}): {stats.raw_positive_count}",
                f"Filtered interactions (users with >= {args.min_user_items} items): {stats.filtered_count}",
                f"Train interactions: {stats.train_count}",
                f"Test interactions: {stats.test_count}",
                f"Users used: {stats.user_count}",
                f"Items used: {stats.item_count}",
                f"Treatment count: {causal_stats.treatment_count}",
                f"Control count: {causal_stats.control_count}",
                f"Matched pairs: {causal_stats.matched_pairs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    METRICS_OUTPUT.write_text(
        "\n".join(
            [
                "========== PyTorch LightGCN Metrics ==========" ,
                "Model: Standard PyTorch LightGCN with BPR Loss + causal edge weights",
                f"Embedding dim: {args.embedding_dim}",
                f"Graph layers: {args.layers}",
                f"Epochs: {args.epochs}",
                f"Batch size: {args.batch_size}",
                f"Learning rate: {args.lr}",
                f"Regularization: {args.reg}",
                f"Causal alpha: {args.causal_alpha}",
                f"Causal mode: {args.causal_mode}",
                f"Causal forest available: {HAS_CAUSALML}",
                "Implicit score fallback: if implicit_score == -1, use explicit sentiment instead",
                f"Propensity mean: {causal_stats.propensity_mean:.6f}",
                f"Propensity range: [{causal_stats.propensity_min:.6f}, {causal_stats.propensity_max:.6f}]",
                f"PSM matched pairs: {causal_stats.matched_pairs}",
                f"PSM ATE (rating diff): {causal_stats.psm_ate:.6f}",
                f"CATE mean: {causal_stats.cate_mean:.6f}",
                f"CATE range: [{causal_stats.cate_min:.6f}, {causal_stats.cate_max:.6f}]",
                f"Causal edge weight range: [{causal_stats.weight_min:.6f}, {causal_stats.weight_max:.6f}]",
                f"Causal edge weight mean: {causal_stats.weight_mean:.6f}",
                f"Recall@{args.top_k}: {recall:.6f}",
                f"NDCG@{args.top_k}: {ndcg:.6f}",
                f"Evaluated users: {evaluated_users}",
                f"Candidate items: {stats.item_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal-weighted PyTorch LightGCN recommender.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Processed interaction CSV file.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="User/item embedding size.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LightGCN graph propagation layers.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="BPR mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 regularization weight for BPR loss.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top-K recommendation size.")
    parser.add_argument("--positive-rating", type=float, default=POSITIVE_RATING, help="Minimum rating for positive feedback.")
    parser.add_argument("--min-user-items", type=int, default=MIN_USER_ITEMS, help="Minimum unique items per evaluated user.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs.")
    parser.add_argument("--causal-alpha", type=float, default=2.0, help="Strength of causal residual weighting.")
    parser.add_argument(
        "--causal-mode",
        choices=["residual", "rating_direct"],
        default="residual",
        help="How causal weights are injected into training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw_interactions = read_positive_interactions(args.input, args.positive_rating)
    filtered_interactions = filter_active_users(raw_interactions, args.min_user_items)
    train_rows, test_rows = split_leave_one_out(filtered_interactions)
    causal_weights, causal_stats = compute_causal_weights(train_rows, args.causal_alpha, args.causal_mode)

    user_to_idx, item_to_idx, users, items = build_id_maps(train_rows + test_rows)
    train_pairs = encode_rows(train_rows, user_to_idx, item_to_idx, causal_weights)
    train_user_items = build_user_items(train_pairs)

    if not train_pairs:
        raise ValueError("No train pairs available. Check positive rating and active user filters.")

    logger.info(
        "Training causal PyTorch LightGCN: users=%d items=%d train_pairs=%d device=%s mode=%s matched_pairs=%d psm_ate=%.4f",
        len(users),
        len(items),
        len(train_pairs),
        device,
        args.causal_mode,
        causal_stats.matched_pairs,
        causal_stats.psm_ate,
    )

    normalized_adj = build_normalized_adj(train_pairs, len(users), len(items), device)
    model = LightGCN(
        num_users=len(users),
        num_items=len(items),
        embedding_dim=args.embedding_dim,
        num_layers=args.layers,
        normalized_adj=normalized_adj,
    ).to(device)

    train_model(model, train_pairs, train_user_items, len(items), args, device)
    recommendations = recommend_topk(model, train_user_items, users, items, args.top_k)
    recall, ndcg, evaluated_users = evaluate(recommendations, test_rows, args.top_k)

    write_recommendations(recommendations)
    write_reports(
        DataStats(
            raw_positive_count=len(raw_interactions),
            filtered_count=len(filtered_interactions),
            train_count=len(train_pairs),
            test_count=len(test_rows),
            user_count=len(users),
            item_count=len(items),
        ),
        causal_stats,
        recall,
        ndcg,
        evaluated_users,
        args,
    )

    logger.info("Causal PyTorch LightGCN recommendation output: %s", RECOMMEND_OUTPUT)
    logger.info("Causal PyTorch LightGCN metrics output: %s", METRICS_OUTPUT)
    logger.info("Causal PyTorch LightGCN data quality output: %s", QUALITY_OUTPUT)


if __name__ == "__main__":
    main()
