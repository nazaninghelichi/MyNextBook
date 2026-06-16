"""
Data collection pipeline for MyNextBook.

Usage:
    python -m src.pipeline                         # full run (~8k-15k books)
    python -m src.pipeline --max-pages 3           # lighter run (~3k books)
    python -m src.pipeline --output data/raw/books.csv
    python -m src.pipeline --resume                # skip queries already in CSV

Strategy:
  80+ queries × up to 10 pages × 40 results = up to ~32,000 raw rows
  After deduplication by volume ID: typically 8,000-15,000 unique books.
"""

import argparse
import csv
import time
import random
import sys
from pathlib import Path

import requests

from .google_books import search_books

# ── Query corpus ──────────────────────────────────────────────────────────────
# Broad genres
_GENRE_QUERIES = [
    "subject:fiction",
    "subject:science fiction",
    "subject:fantasy",
    "subject:mystery",
    "subject:thriller",
    "subject:horror",
    "subject:romance",
    "subject:historical fiction",
    "subject:literary fiction",
    "subject:adventure",
    "subject:biography",
    "subject:autobiography",
    "subject:memoir",
    "subject:history",
    "subject:science",
    "subject:philosophy",
    "subject:psychology",
    "subject:self help",
    "subject:business",
    "subject:economics",
    "subject:politics",
    "subject:true crime",
    "subject:travel",
    "subject:cooking",
    "subject:art",
    "subject:music",
    "subject:sports",
    "subject:religion",
    "subject:spirituality",
    "subject:health",
    "subject:nature",
    "subject:technology",
    "subject:computers",
    "subject:mathematics",
    "subject:poetry",
    "subject:drama",
    "subject:graphic novel",
    "subject:children",
    "subject:young adult",
]

# Sub-genre / topic queries
_TOPIC_QUERIES = [
    "subject:dystopia",
    "subject:space opera",
    "subject:cyberpunk",
    "subject:steampunk",
    "subject:magical realism",
    "subject:urban fantasy",
    "subject:epic fantasy",
    "subject:dark fantasy",
    "subject:noir",
    "subject:cozy mystery",
    "subject:legal thriller",
    "subject:psychological thriller",
    "subject:spy fiction",
    "subject:war fiction",
    "subject:post apocalyptic",
    "subject:time travel",
    "subject:alternate history",
    "subject:climate fiction",
    "subject:artificial intelligence",
    "subject:neuroscience",
    "subject:behavioral economics",
    "subject:startup",
    "subject:leadership",
    "subject:personal finance",
    "subject:mindfulness",
    "subject:feminism",
    "subject:social justice",
    "subject:environmentalism",
    "subject:ancient history",
    "subject:world war",
    "subject:cold war",
    "subject:civil rights",
    "subject:mythology",
    "subject:folklore",
]

# Popular author searches (returns books by that author + similar)
_AUTHOR_QUERIES = [
    "inauthor:Stephen King",
    "inauthor:Agatha Christie",
    "inauthor:Isaac Asimov",
    "inauthor:Frank Herbert",
    "inauthor:J.R.R. Tolkien",
    "inauthor:George R.R. Martin",
    "inauthor:Ursula Le Guin",
    "inauthor:Philip K Dick",
    "inauthor:Cormac McCarthy",
    "inauthor:Toni Morrison",
    "inauthor:Gabriel Garcia Marquez",
    "inauthor:Haruki Murakami",
    "inauthor:Dostoevsky",
    "inauthor:Tolstoy",
    "inauthor:Jane Austen",
    "inauthor:Charles Dickens",
    "inauthor:Ernest Hemingway",
    "inauthor:Virginia Woolf",
    "inauthor:Malcolm Gladwell",
    "inauthor:Yuval Noah Harari",
    "inauthor:Michael Lewis",
    "inauthor:Walter Isaacson",
]

ALL_QUERIES = _GENRE_QUERIES + _TOPIC_QUERIES + _AUTHOR_QUERIES

# ── CSV schema ────────────────────────────────────────────────────────────────
_FIELDS = [
    "id", "title", "authors", "description", "categories",
    "average_rating", "ratings_count", "published_date",
    "page_count", "thumbnail", "language", "source_query",
]


def _serialize(book: dict, query: str) -> dict:
    return {
        "id": book["id"],
        "title": book["title"],
        "authors": "|".join(book.get("authors") or []),
        "description": book.get("description", ""),
        "categories": "|".join(book.get("categories") or []),
        "average_rating": book.get("average_rating", ""),
        "ratings_count": book.get("ratings_count", 0),
        "published_date": book.get("published_date", ""),
        "page_count": book.get("page_count", ""),
        "thumbnail": book.get("thumbnail", ""),
        "language": book.get("language", ""),
        "source_query": query,
    }


# ── Fetch with retry ──────────────────────────────────────────────────────────
def _fetch_page(query: str, page: int, page_size: int = 40, retries: int = 3) -> list[dict]:
    start_index = page * page_size
    for attempt in range(retries):
        try:
            return search_books(query, max_results=page_size, start_index=start_index)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"    [rate limit] waiting {wait:.1f}s…", flush=True)
                time.sleep(wait)
            else:
                print(f"    [HTTP error] {e} — skipping page {page}", flush=True)
                return []
        except requests.RequestException as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"    [request error] {e} — retry {attempt+1}/{retries}", flush=True)
            time.sleep(wait)
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(
    output: str = "data/raw/books.csv",
    max_pages: int = 10,
    page_size: int = 40,
    delay: float = 0.3,
    resume: bool = False,
    queries: list[str] | None = None,
) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = queries or ALL_QUERIES

    # Load already-seen IDs (for deduplication) and done queries (for resume)
    seen_ids: set[str] = set()
    done_queries: set[str] = set()
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    if resume and file_exists:
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_ids.add(row["id"])
                done_queries.add(row["source_query"])
        print(f"Resuming — {len(seen_ids):,} books already collected, "
              f"{len(done_queries)} queries done.\n")

    write_mode = "a" if (resume and file_exists) else "w"
    write_header = not (resume and file_exists)

    total_new = 0
    pending = [q for q in queries if q not in done_queries]
    print(f"Queries to run: {len(pending)}  |  max_pages: {max_pages}  "
          f"|  page_size: {page_size}  |  output: {output_path}\n")

    with open(output_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()

        for q_idx, query in enumerate(pending, 1):
            q_new = 0
            print(f"[{q_idx:>3}/{len(pending)}] {query}", end="", flush=True)

            for page in range(max_pages):
                books = _fetch_page(query, page, page_size)
                if not books:
                    break  # no more results for this query

                new_books = [b for b in books if b["id"] not in seen_ids and b.get("description")]
                for b in new_books:
                    seen_ids.add(b["id"])
                    writer.writerow(_serialize(b, query))
                    q_new += 1

                f.flush()
                time.sleep(delay + random.uniform(0, 0.2))

                if len(books) < page_size:
                    break  # last page was partial — no point fetching more

            total_new += q_new
            print(f"  +{q_new:>4} books  (total: {len(seen_ids):,})", flush=True)

    print(f"\nDone. {total_new:,} new books added → {len(seen_ids):,} total unique in {output_path}")
    return total_new


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Fetch books from Google Books API")
    parser.add_argument("--output", default="data/raw/books.csv")
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages per query (40 books/page). Default: 10 (~400 books/query)")
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between requests (default 0.3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries already present in the output CSV")
    args = parser.parse_args()
    run(
        output=args.output,
        max_pages=args.max_pages,
        page_size=args.page_size,
        delay=args.delay,
        resume=args.resume,
    )


if __name__ == "__main__":
    _cli()
