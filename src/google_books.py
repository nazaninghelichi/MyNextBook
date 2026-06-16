import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

def _get_api_key() -> str | None:
    key = os.getenv("GOOGLE_BOOKS_API_KEY")
    source = "env"
    if not key:
        try:
            import streamlit as st
            val = st.secrets.get("GOOGLE_BOOKS_API_KEY", "")
            if val and not val.startswith("PASTE"):
                key = val
                source = "secrets"
            else:
                source = f"secrets_raw={repr(val[:10]) if val else 'empty'}"
        except Exception as e:
            source = f"secrets_error={e}"

    try:
        import streamlit as st
        st.sidebar.caption(f"🔑 key source: {source} | prefix: {key[:8] if key else 'NONE'}")
    except Exception:
        pass
    return key

_BASE_URL = "https://www.googleapis.com/books/v1/volumes"


def _params(**kwargs):
    p = dict(kwargs)
    key = _get_api_key()
    if key:
        p["key"] = key
    return p


def _parse(item: dict) -> dict:
    info = item.get("volumeInfo", {})
    return {
        "id": item.get("id", ""),
        "title": info.get("title", "Unknown"),
        "authors": info.get("authors", []),
        "description": info.get("description", ""),
        "categories": info.get("categories", []),
        "average_rating": info.get("averageRating"),
        "ratings_count": info.get("ratingsCount", 0),
        "published_date": info.get("publishedDate", ""),
        "page_count": info.get("pageCount"),
        "thumbnail": info.get("imageLinks", {}).get("thumbnail", ""),
        "language": info.get("language", "en"),
    }


def search_books(
    query: str,
    max_results: int = 20,
    lang: str = "en",
    start_index: int = 0,
) -> list[dict]:
    params = _params(
        q=query,
        maxResults=min(max_results, 40),
        langRestrict=lang,
        startIndex=start_index,
    )
    resp = requests.get(_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return [_parse(item) for item in resp.json().get("items", [])]


def get_book(volume_id: str) -> dict:
    resp = requests.get(f"{_BASE_URL}/{volume_id}", params=_params(), timeout=10)
    resp.raise_for_status()
    return _parse(resp.json())


def search_by_genre(genre: str, max_results: int = 20) -> list[dict]:
    return search_books(f"subject:{genre}", max_results=max_results)


def search_by_author(author: str, max_results: int = 20) -> list[dict]:
    return search_books(f"inauthor:{author}", max_results=max_results)


def fetch_candidates(liked_books: list[dict], per_book: int = 10) -> list[dict]:
    """Fetch candidate books from the API based on genres and authors of liked books."""
    seen_ids = {b["id"] for b in liked_books}
    candidates = []

    for book in liked_books:
        queries = []
        if book.get("categories"):
            queries.append(f"subject:{book['categories'][0]}")
        if book.get("authors"):
            queries.append(f"inauthor:{book['authors'][0]}")

        for q in queries:
            try:
                results = search_books(q, max_results=per_book)
                for r in results:
                    if r["id"] not in seen_ids and r.get("description"):
                        seen_ids.add(r["id"])
                        candidates.append(r)
            except requests.RequestException:
                continue

    return candidates
