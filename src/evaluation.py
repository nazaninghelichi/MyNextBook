"""
Offline evaluation for MyNextBook recommenders.

Strategy: Leave-One-Out on category groups.

How it works:
  1. Group the cleaned catalogue by primary category.
  2. For each category with enough books, iterate through each book as the
     held-out item (ground truth).
  3. Use N_QUERY same-category books as the "liked" query.
  4. Ask the model to recommend from the rest of the catalogue.
  5. Check whether the held-out book appears in the top-k results.

Why this is correct:
  - The model NEVER sees the held-out book during the query — no data leakage.
  - Ground truth is an actual specific book, not a fuzzy "same genre" proxy.
  - Averaging over hundreds of test cases gives stable estimates.

Metrics:
  - Hit Rate@k  : fraction of test cases where the held-out book is in top-k
  - NDCG@k      : rewards finding the held-out book higher in the ranked list
  - Diversity   : mean pairwise category distance of the recommendation list
"""

import random
import numpy as np


# ── Metrics ───────────────────────────────────────────────────────────────────

def _hit_rate(rec_ids: list[str], target_id: str, k: int) -> float:
    return float(target_id in rec_ids[:k])


def _ndcg(rec_ids: list[str], target_id: str, k: int) -> float:
    for i, bid in enumerate(rec_ids[:k]):
        if bid == target_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def diversity(recommendations: list[dict]) -> float:
    if len(recommendations) < 2:
        return 0.0
    dists = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            a = set(recommendations[i].get("categories", []))
            b = set(recommendations[j].get("categories", []))
            union = a | b
            dists.append(1.0 - len(a & b) / len(union) if union else 0.0)
    return float(np.mean(dists))


# ── Leave-one-out evaluation ──────────────────────────────────────────────────

def leave_one_out_eval(
    model,
    books: list[dict],
    n_query: int = 3,
    min_cat_size: int = 6,
    top_n: int = 10,
    k_values: tuple = (5, 10),
    max_cases: int = 300,
    seed: int = 42,
) -> dict:
    """
    Parameters
    ----------
    model         : fitted recommender with .recommend(liked, candidates, top_n)
    books         : full cleaned catalogue the model was fit on
    n_query       : number of same-category books used as the "liked" query
    min_cat_size  : skip categories smaller than this
    top_n         : recommendation list length (must be >= max(k_values))
    k_values      : which cut-offs to evaluate
    max_cases     : cap total test cases to keep runtime reasonable
    seed          : for reproducibility
    """
    rng = random.Random(seed)

    # Group by primary category
    by_cat: dict[str, list[dict]] = {}
    for book in books:
        cat = book["categories"][0] if book["categories"] else None
        if cat:
            by_cat.setdefault(cat, []).append(book)

    results   = {k: {"hit_rate": [], "ndcg": []} for k in k_values}
    diversities: list[float] = []
    n_cases = 0

    cats = [c for c, bs in by_cat.items() if len(bs) >= min_cat_size]
    rng.shuffle(cats)

    for cat in cats:
        if n_cases >= max_cases:
            break
        cat_books = by_cat[cat].copy()
        rng.shuffle(cat_books)

        for i, held_out in enumerate(cat_books):
            if n_cases >= max_cases:
                break
            # Query: n_query books from the same category (not the held-out)
            query_pool = [b for b in cat_books if b["id"] != held_out["id"]]
            if len(query_pool) < n_query:
                continue
            liked = query_pool[:n_query]

            # Candidates: everything except liked books
            liked_ids  = {b["id"] for b in liked}
            candidates = [b for b in books if b["id"] not in liked_ids]

            try:
                recs = model.recommend(liked, candidates, top_n=top_n)
            except Exception:
                continue

            rec_ids = [r["id"] for r in recs]

            for k in k_values:
                results[k]["hit_rate"].append(_hit_rate(rec_ids, held_out["id"], k))
                results[k]["ndcg"].append(_ndcg(rec_ids, held_out["id"], k))

            diversities.append(diversity(recs))
            n_cases += 1

    if n_cases == 0:
        return {"error": "No test cases — lower min_cat_size or n_query"}

    summary = {"n_test_cases": n_cases}
    for k in k_values:
        summary[f"hit_rate@{k}"] = round(np.mean(results[k]["hit_rate"]), 4)
        summary[f"ndcg@{k}"]     = round(np.mean(results[k]["ndcg"]), 4)
    summary["diversity"] = round(np.mean(diversities), 4)
    return summary
