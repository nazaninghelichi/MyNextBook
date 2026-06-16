"""
SVD-based latent factor recommender (Latent Semantic Analysis style).

We apply TruncatedSVD to the TF-IDF feature matrix to discover latent
topics/themes. Similarity in the reduced space captures semantic overlap
that raw TF-IDF cosine similarity may miss (e.g. synonyms, topic clusters).
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class SVDRecommender:
    def __init__(self, n_components: int = 100, random_state: int = 42):
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._latent_matrix: np.ndarray | None = None
        self._books: list[dict] = []

    def fit(self, tfidf_matrix: np.ndarray, books: list[dict]) -> "SVDRecommender":
        self._books = books
        reduced = self._svd.fit_transform(tfidf_matrix)
        self._latent_matrix = normalize(reduced)
        return self

    def _query_vector(self, liked_books: list[dict]) -> np.ndarray:
        book_index = {b["id"]: i for i, b in enumerate(self._books)}
        vecs = [
            self._latent_matrix[book_index[b["id"]]]
            for b in liked_books
            if b["id"] in book_index
        ]
        if not vecs:
            return np.zeros((1, self._latent_matrix.shape[1]))
        avg = np.mean(vecs, axis=0, keepdims=True)
        return normalize(avg)

    def recommend(
        self,
        liked_books: list[dict],
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        if self._latent_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        liked_ids = {b["id"] for b in liked_books}
        book_index = {b["id"]: i for i, b in enumerate(self._books)}
        query_vec = self._query_vector(liked_books)

        scores = {}
        for book in candidates:
            idx = book_index.get(book["id"])
            if idx is None or book["id"] in liked_ids:
                continue
            score = cosine_similarity(query_vec, self._latent_matrix[idx : idx + 1])[0, 0]
            scores[book["id"]] = score

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
        id_to_book = {b["id"]: b for b in candidates}
        return [
            {**id_to_book[bid], "svd_score": round(scores[bid], 4)}
            for bid in ranked_ids
            if bid in id_to_book
        ]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._svd.explained_variance_ratio_
