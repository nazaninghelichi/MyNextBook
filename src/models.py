"""
All recommender models share the same interface:

    model.fit(books)                              # learn from the full catalogue
    model.recommend(liked, candidates, top_n)     # return ranked list
"""

import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ── Shared text feature builder ───────────────────────────────────────────────

def _book_text(book: dict) -> str:
    parts = [
        book.get("title", ""),
        " ".join(book.get("authors", [])),
        " ".join(book.get("categories", [])),
        book.get("description", ""),
    ]
    return " ".join(p for p in parts if p).lower()


# ── Base class ────────────────────────────────────────────────────────────────

class BaseRecommender:
    def fit(self, books: list[dict]) -> "BaseRecommender":
        self._books = books
        self._index = {b["id"]: i for i, b in enumerate(books)}
        return self

    def recommend(
        self, liked_books: list[dict], candidates: list[dict], top_n: int = 10
    ) -> list[dict]:
        raise NotImplementedError

    def _filter_candidates(self, liked_books, candidates):
        liked_ids = {b["id"] for b in liked_books}
        return [b for b in candidates if b["id"] not in liked_ids]


# ── 1. Random baseline ────────────────────────────────────────────────────────

class RandomRecommender(BaseRecommender):
    """Randomly shuffles candidates. The floor — every model should beat this."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    def fit(self, books: list[dict]) -> "RandomRecommender":
        super().fit(books)
        return self

    def recommend(self, liked_books, candidates, top_n=10):
        pool = self._filter_candidates(liked_books, candidates)
        rng = random.Random(self._seed)
        rng.shuffle(pool)
        return pool[:top_n]


# ── 2. TF-IDF content-based ───────────────────────────────────────────────────

class TFIDFRecommender(BaseRecommender):
    """TF-IDF on title + authors + categories + description → cosine similarity."""

    def __init__(self, max_features: int = 5000):
        self._vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )
        self._matrix = None

    def fit(self, books: list[dict]) -> "TFIDFRecommender":
        super().fit(books)
        texts = [_book_text(b) for b in books]
        self._matrix = self._vec.fit_transform(texts)
        return self

    def _query_vec(self, liked_books):
        texts = [_book_text(b) for b in liked_books]
        vecs = self._vec.transform(texts)
        return np.asarray(vecs.mean(axis=0))

    def recommend(self, liked_books, candidates, top_n=10):
        pool = self._filter_candidates(liked_books, candidates)
        query = self._query_vec(liked_books)
        candidate_indices = [self._index[b["id"]] for b in pool if b["id"] in self._index]
        candidate_matrix  = self._matrix[candidate_indices]
        scores = cosine_similarity(query, candidate_matrix)[0]
        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        return [b for b, _ in ranked[:top_n]]


# ── 3. Sentence Transformer ───────────────────────────────────────────────────

class SentenceTransformerRecommender(BaseRecommender):
    """
    Dense neural embeddings using all-MiniLM-L6-v2.
    Captures semantic meaning — 'sci-fi' and 'space opera' will be close
    even if they share no words.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._embeddings = None

    def fit(self, books: list[dict]) -> "SentenceTransformerRecommender":
        from sentence_transformers import SentenceTransformer
        super().fit(books)
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        texts = [_book_text(b) for b in books]
        self._embeddings = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return self

    def _query_embedding(self, liked_books):
        texts = [_book_text(b) for b in liked_books]
        vecs = self._model.encode(texts, normalize_embeddings=True)
        avg = vecs.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-9)

    def recommend(self, liked_books, candidates, top_n=10):
        pool = self._filter_candidates(liked_books, candidates)
        query = self._query_embedding(liked_books)
        candidate_embeddings = np.stack(
            [self._embeddings[self._index[b["id"]]] for b in pool if b["id"] in self._index]
        )
        scores = candidate_embeddings @ query
        ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        return [b for b, _ in ranked[:top_n]]


# ── 4. Hybrid (TF-IDF + Sentence Transformer) ────────────────────────────────

class HybridRecommender(BaseRecommender):
    """
    Weighted combination of TF-IDF and Sentence Transformer scores.
    alpha=0 → pure TF-IDF, alpha=1 → pure Sentence Transformer.
    Default alpha=0.6 leans on the neural model.
    """

    def __init__(self, alpha: float = 0.6, model_name: str = "all-MiniLM-L6-v2"):
        self.alpha = alpha
        self._tfidf = TFIDFRecommender()
        self._st    = SentenceTransformerRecommender(model_name)

    def fit(self, books: list[dict]) -> "HybridRecommender":
        super().fit(books)
        self._tfidf.fit(books)
        self._st.fit(books)
        return self

    def recommend(self, liked_books, candidates, top_n=10):
        pool = self._filter_candidates(liked_books, candidates)

        # TF-IDF scores
        query_tfidf = self._tfidf._query_vec(liked_books)
        tfidf_idx   = [self._tfidf._index[b["id"]] for b in pool if b["id"] in self._tfidf._index]
        tfidf_mat   = self._tfidf._matrix[tfidf_idx]
        tfidf_scores = cosine_similarity(query_tfidf, tfidf_mat)[0]

        # Sentence Transformer scores
        query_st   = self._st._query_embedding(liked_books)
        st_embeds  = np.stack([self._st._embeddings[self._st._index[b["id"]]] for b in pool if b["id"] in self._st._index])
        st_scores  = st_embeds @ query_st

        # Normalize both to [0,1] then blend
        def _minmax(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)

        tfidf_norm = _minmax(tfidf_scores)
        st_norm    = _minmax(st_scores)
        hybrid     = self.alpha * st_norm + (1 - self.alpha) * tfidf_norm
        ranked = sorted(
            zip(pool, hybrid, tfidf_norm, st_norm),
            key=lambda x: x[1], reverse=True,
        )
        return [
            {**b, "hybrid_score": round(float(h), 4), "tfidf_score": round(float(t), 4), "st_score": round(float(s), 4)}
            for b, h, t, s in ranked[:top_n]
        ]
