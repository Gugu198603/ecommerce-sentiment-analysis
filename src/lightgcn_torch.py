#!/usr/bin/env python3
"""
Standard PyTorch LightGCN recommender.

This script trains a multi-layer LightGCN model with BPR loss on the processed
phone review interactions. It uses user_id as user nodes and product_id as item
nodes, then exports Top-K recommendations and Recall/NDCG metrics.
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
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils import ensure_dirs, setup_logging


logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
RECOMMEND_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommend_result_torch.csv"
METRICS_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_metrics_torch.txt"
QUALITY_OUTPUT = PROJECT_ROOT / "results" / "reports" / "recommendation_data_quality_torch.txt"

TOP_K = 10
POSITIVE_RATING = 4.0
MIN_USER_ITEMS = 2
RANDOM_STATE = 42


Interaction = Tuple[str, str, float, str]
EncodedPair = Tuple[int, int]


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


def read_positive_interactions(input_file: Path, positive_rating: float) -> List[Interaction]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

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
            if user_id and item_id and rating >= positive_rating:
                interactions.append((user_id, item_id, rating, row.get("time") or ""))
    return interactions


def filter_active_users(interactions: List[Interaction], min_items: int) -> List[Interaction]:
    user_items: Dict[str, set[str]] = defaultdict(set)
    for user_id, item_id, _, _ in interactions:
        user_items[user_id].add(item_id)

    active_users = {user_id for user_id, items in user_items.items() if len(items) >= min_items}
    return [row for row in interactions if row[0] in active_users]


def split_leave_one_out(interactions: List[Interaction]) -> Tuple[List[Interaction], List[Interaction]]:
    by_user: Dict[str, List[Interaction]] = defaultdict(list)
    for row in interactions:
        by_user[row[0]].append(row)

    train_rows: List[Interaction] = []
    test_rows: List[Interaction] = []
    for rows in by_user.values():
        # Deduplicate by item and keep the latest interaction for each item.
        dedup: Dict[str, Interaction] = {}
        for row in sorted(rows, key=lambda value: value[3]):
            dedup[row[1]] = row
        user_rows = list(dedup.values())
        if len(user_rows) < 2:
            continue
        test_rows.append(user_rows[-1])
        train_rows.extend(user_rows[:-1])
    return train_rows, test_rows


def build_id_maps(rows: Sequence[Interaction]) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    users = sorted({row[0] for row in rows})
    items = sorted({row[1] for row in rows})
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    return user_to_idx, item_to_idx, users, items


def encode_rows(
    rows: Iterable[Interaction],
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
) -> List[EncodedPair]:
    pairs = []
    for user_id, item_id, _, _ in rows:
        if user_id in user_to_idx and item_id in item_to_idx:
            pairs.append((user_to_idx[user_id], item_to_idx[item_id]))
    return pairs


def build_user_items(pairs: Iterable[EncodedPair]) -> Dict[int, set[int]]:
    user_items: Dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in pairs:
        user_items[user_idx].add(item_idx)
    return user_items


class BPRDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[EncodedPair],
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

    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        user_idx, pos_item_idx = self.pairs[index]
        if len(self.user_items[user_idx]) >= self.num_items:
            neg_item_idx = pos_item_idx
        else:
            neg_item_idx = self.rng.randrange(self.num_items)
            while neg_item_idx in self.user_items[user_idx]:
                neg_item_idx = self.rng.randrange(self.num_items)
        return user_idx, pos_item_idx, neg_item_idx


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
        user_embeddings, item_embeddings = torch.split(
            final_embeddings,
            [self.num_users, self.num_items],
            dim=0,
        )
        return user_embeddings, item_embeddings

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        reg_weight: float,
    ) -> torch.Tensor:
        user_embeddings, item_embeddings = self.propagate()
        user_vec = user_embeddings[users]
        pos_vec = item_embeddings[pos_items]
        neg_vec = item_embeddings[neg_items]

        pos_scores = torch.sum(user_vec * pos_vec, dim=1)
        neg_scores = torch.sum(user_vec * neg_vec, dim=1)
        ranking_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

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
    train_pairs: Sequence[EncodedPair],
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = num_users + num_items
    rows: List[int] = []
    cols: List[int] = []

    for user_idx, item_idx in train_pairs:
        item_node = num_users + item_idx
        rows.extend([user_idx, item_node])
        cols.extend([item_node, user_idx])

    if not rows:
        raise ValueError("No training pairs available for adjacency construction.")

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
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
    train_pairs: Sequence[EncodedPair],
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
        for users, pos_items, neg_items in loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items, args.reg)
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

    for user_id, target_item, _, _ in test_rows:
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
                        "model": "pytorch_lightgcn",
                    }
                )


@dataclass
class DataStats:
    raw_positive_count: int
    filtered_count: int
    train_count: int
    test_count: int
    user_count: int
    item_count: int


def write_reports(stats: DataStats, recall: float, ndcg: float, evaluated_users: int, args: argparse.Namespace) -> None:
    QUALITY_OUTPUT.write_text(
        "\n".join(
            [
                "========== PyTorch LightGCN Data Quality ==========",
                f"Input file: {INPUT_FILE}",
                f"Raw positive interactions (score >= {args.positive_rating:.1f}): {stats.raw_positive_count}",
                f"Filtered interactions (users with >= {args.min_user_items} items): {stats.filtered_count}",
                f"Train interactions: {stats.train_count}",
                f"Test interactions: {stats.test_count}",
                f"Users used: {stats.user_count}",
                f"Items used: {stats.item_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    METRICS_OUTPUT.write_text(
        "\n".join(
            [
                "========== PyTorch LightGCN Metrics ==========",
                "Model: Standard PyTorch LightGCN with BPR Loss",
                f"Embedding dim: {args.embedding_dim}",
                f"Graph layers: {args.layers}",
                f"Epochs: {args.epochs}",
                f"Batch size: {args.batch_size}",
                f"Learning rate: {args.lr}",
                f"Regularization: {args.reg}",
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
    parser = argparse.ArgumentParser(description="Train a standard PyTorch LightGCN recommender.")
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
    user_to_idx, item_to_idx, users, items = build_id_maps(train_rows + test_rows)
    train_pairs = encode_rows(train_rows, user_to_idx, item_to_idx)
    train_user_items = build_user_items(train_pairs)

    if not train_pairs:
        raise ValueError("No train pairs available. Check positive rating and active user filters.")

    logger.info(
        "Training PyTorch LightGCN: users=%d items=%d train_pairs=%d device=%s",
        len(users),
        len(items),
        len(train_pairs),
        device,
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
        recall,
        ndcg,
        evaluated_users,
        args,
    )

    logger.info("PyTorch LightGCN recommendation output: %s", RECOMMEND_OUTPUT)
    logger.info("PyTorch LightGCN metrics output: %s", METRICS_OUTPUT)
    logger.info("PyTorch LightGCN data quality output: %s", QUALITY_OUTPUT)


if __name__ == "__main__":
    main()
