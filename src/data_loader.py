"""Load books from the pipeline CSV and optionally apply cleaning."""

import csv
from pathlib import Path

from .cleaning import clean_books

_DEFAULT_PATH = Path("data/raw/books.csv")


def _parse_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "authors": [a for a in row["authors"].split("|") if a],
        "description": row.get("description", ""),
        "categories": [c for c in row["categories"].split("|") if c],
        "average_rating": float(row["average_rating"]) if row["average_rating"] else None,
        "ratings_count": int(row["ratings_count"]) if row["ratings_count"] else 0,
        "published_date": row.get("published_date", ""),
        "page_count": int(row["page_count"]) if row["page_count"] else None,
        "thumbnail": row.get("thumbnail", ""),
        "language": row.get("language", "en"),
    }


def load_raw(path: str | Path = _DEFAULT_PATH) -> list[dict]:
    """Load books from CSV with no cleaning applied."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the pipeline first:\n"
            "  python -m src.pipeline"
        )
    with open(path, newline="", encoding="utf-8") as f:
        return [_parse_row(row) for row in csv.DictReader(f)]


def load_books(
    path: str | Path = _DEFAULT_PATH,
    clean: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Load books from CSV and (by default) apply the full cleaning pipeline."""
    books = load_raw(path)
    if clean:
        books, _ = clean_books(books, verbose=verbose)
    return books


def stats(path: str | Path = _DEFAULT_PATH) -> dict:
    raw = load_raw(path)
    cleaned, report = clean_books(raw, verbose=False)
    with_rating = [b for b in cleaned if b["average_rating"] is not None]
    with_cats = [b for b in cleaned if b["categories"]]
    return {
        "raw_books": report["raw"],
        "clean_books": report["final"],
        "removed_total": report["raw"] - report["final"],
        "with_rating": len(with_rating),
        "with_categories": len(with_cats),
        "avg_rating": round(
            sum(b["average_rating"] for b in with_rating) / len(with_rating), 2
        ) if with_rating else None,
    }
