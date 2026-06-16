import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _build_feature_text(book: dict) -> str:
    parts = [
        book.get("description", ""),
        " ".join(book.get("categories", [])),
        " ".join(book.get("authors", [])),
        book.get("title", ""),
    ]
    return " ".join(p for p in parts if p).lower()


class ContentBasedRecommender:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self._matrix: np.ndarray | None = None
        self._books: list[dict] = []

    def fit(self, books: list[dict]) -> "ContentBasedRecommender":
        self._books = books
        texts = [_build_feature_text(b) for b in books]
        self._matrix = self._vectorizer.fit_transform(texts).toarray()
        return self

    def _query_vector(self, query_books: list[dict]) -> np.ndarray:
        texts = [_build_feature_text(b) for b in query_books]
        vecs = self._vectorizer.transform(texts).toarray()
        return vecs.mean(axis=0, keepdims=True)

    def recommend(
        self,
        liked_books: list[dict],
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        if self._matrix is None or not self._books:
            raise RuntimeError("Call fit() before recommend().")

        liked_ids = {b["id"] for b in liked_books}
        candidate_ids = [b["id"] for b in candidates]

        query_vec = self._query_vector(liked_books)
        book_index = {b["id"]: i for i, b in enumerate(self._books)}

        scores = {}
        for book in candidates:
            idx = book_index.get(book["id"])
            if idx is None or book["id"] in liked_ids:
                continue
            score = cosine_similarity(query_vec, self._matrix[idx : idx + 1])[0, 0]
            scores[book["id"]] = score

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
        id_to_book = {b["id"]: b for b in candidates}
        return [
            {**id_to_book[bid], "content_score": round(scores[bid], 4)}
            for bid in ranked_ids
            if bid in id_to_book
        ]

    @property
    def tfidf_matrix(self) -> np.ndarray | None:
        return self._matrix

    @property
    def books(self) -> list[dict]:
        return self._books
