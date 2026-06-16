import numpy as np
from .content_based import ContentBasedRecommender
from .matrix_factorization import SVDRecommender


class HybridRecommender:
    """Weighted combination of content-based TF-IDF and SVD latent factor scores."""

    def __init__(
        self,
        content_weight: float = 0.5,
        svd_components: int = 100,
    ):
        if not 0.0 <= content_weight <= 1.0:
            raise ValueError("content_weight must be in [0, 1]")
        self.content_weight = content_weight
        self.svd_weight = 1.0 - content_weight

        self._cb = ContentBasedRecommender()
        self._svd = SVDRecommender(n_components=svd_components)
        self._fitted = False

    def fit(self, books: list[dict]) -> "HybridRecommender":
        self._cb.fit(books)
        tfidf = self._cb.tfidf_matrix
        n_components = min(self._svd._svd.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_components > 0:
            self._svd._svd.n_components = n_components
            self._svd.fit(tfidf, books)
        self._fitted = True
        return self

    def recommend(
        self,
        liked_books: list[dict],
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        cb_results = self._cb.recommend(liked_books, candidates, top_n=len(candidates))
        svd_results = self._svd.recommend(liked_books, candidates, top_n=len(candidates))

        content_scores = {r["id"]: r["content_score"] for r in cb_results}
        svd_scores = {r["id"]: r["svd_score"] for r in svd_results}
        id_to_book = {b["id"]: b for b in candidates}

        all_ids = set(content_scores) | set(svd_scores)
        ranked = []
        for bid in all_ids:
            cs = content_scores.get(bid, 0.0)
            ss = svd_scores.get(bid, 0.0)
            hybrid = self.content_weight * cs + self.svd_weight * ss
            book = id_to_book.get(bid, {})
            ranked.append({
                **book,
                "content_score": round(cs, 4),
                "svd_score": round(ss, 4),
                "hybrid_score": round(hybrid, 4),
            })

        ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_n]
