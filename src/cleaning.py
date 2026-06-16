"""
Data cleaning for Google Books API data.

Problems we fix:
  1. HTML tags in descriptions  (Google Books often injects <b>, <br> etc.)
  2. Near-duplicate editions    (same title + author, different volume IDs)
  3. Non-English books          (langRestrict=en is not perfectly enforced)
  4. Stub descriptions          (< 80 chars after stripping)
  5. Invalid ratings            (must be 1.0–5.0; API sometimes returns 0)
  6. Outlier page counts        (< 10 or > 6000 are almost always data errors)
  7. Messy categories           (hierarchical "Juvenile Fiction / Fantasy & Magic"
                                 → extract leaf; normalize casing)
  8. Published year extraction  ("2021-04-15" → 2021; keep None if unparseable)
"""

import re
import unicodedata
from html.parser import HTMLParser


# ── HTML stripping ─────────────────────────────────────────────────────────────
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    if "<" not in text:
        return text
    s = _HTMLStripper()
    s.feed(text)
    return re.sub(r"\s+", " ", s.get_text()).strip()


# ── Category normalisation ─────────────────────────────────────────────────────
_CATEGORY_MAP = {
    "science fiction": "Science Fiction",
    "sci-fi": "Science Fiction",
    "sf": "Science Fiction",
    "speculative fiction": "Science Fiction",
    "fantasy": "Fantasy",
    "epic fantasy": "Fantasy",
    "dark fantasy": "Fantasy",
    "urban fantasy": "Fantasy",
    "mystery": "Mystery",
    "mysteries": "Mystery",
    "detective": "Mystery",
    "thriller": "Thriller",
    "suspense": "Thriller",
    "horror": "Horror",
    "romance": "Romance",
    "historical fiction": "Historical Fiction",
    "history": "History",
    "biography": "Biography",
    "autobiography": "Biography",
    "memoir": "Biography",
    "self help": "Self-Help",
    "self-help": "Self-Help",
    "personal development": "Self-Help",
    "psychology": "Psychology",
    "philosophy": "Philosophy",
    "science": "Science",
    "technology": "Technology",
    "computers": "Technology",
    "business": "Business",
    "economics": "Economics",
    "politics": "Politics",
    "literary fiction": "Literary Fiction",
    "fiction": "Fiction",
    "young adult": "Young Adult",
    "juvenile fiction": "Young Adult",
    "children": "Children",
    "poetry": "Poetry",
    "drama": "Drama",
    "travel": "Travel",
    "cooking": "Cooking",
    "health": "Health",
    "sports": "Sports",
    "art": "Art",
    "music": "Music",
    "religion": "Religion",
    "spirituality": "Spirituality",
}


def _normalise_category(raw: str) -> str:
    """Extract leaf segment and map to canonical name."""
    # "Juvenile Fiction / Fantasy & Magic" → "Fantasy & Magic"
    leaf = raw.split("/")[-1].strip()
    # strip parenthetical notes
    leaf = re.sub(r"\(.*?\)", "", leaf).strip()
    key = leaf.lower().strip()
    return _CATEGORY_MAP.get(key, leaf.title())


def _normalise_categories(cats: list[str]) -> list[str]:
    seen, out = set(), []
    for c in cats:
        n = _normalise_category(c)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ── Published year ─────────────────────────────────────────────────────────────
def _extract_year(date_str: str) -> int | None:
    m = re.search(r"\b(1[5-9]\d{2}|20[012]\d)\b", date_str)
    return int(m.group()) if m else None


# ── Title / author key for deduplication ──────────────────────────────────────
def _dedup_key(book: dict) -> str:
    title = unicodedata.normalize("NFKD", book["title"]).lower()
    title = re.sub(r"[^a-z0-9 ]", "", title).strip()
    title = re.sub(r"\s+", " ", title)
    # drop common edition suffixes
    title = re.sub(r"\b(revised|updated|edition|ed|vol|volume|book|part)\b.*$", "", title).strip()
    author = book["authors"][0].split(",")[0].strip().lower() if book["authors"] else ""
    return f"{title}||{author}"


# ── Non-latin / language detection heuristic ──────────────────────────────────
_NON_LATIN_RE = re.compile(r"[^\x00-\x7FÀ-ɏḀ-ỿ]")

def _is_mostly_latin(text: str) -> bool:
    if not text:
        return True
    non_latin = len(_NON_LATIN_RE.findall(text))
    return (non_latin / len(text)) < 0.15


# ── Main clean function ────────────────────────────────────────────────────────
def clean_books(books: list[dict], verbose: bool = True) -> tuple[list[dict], dict]:
    """
    Apply all cleaning steps and return (cleaned_books, report).

    The report dict contains before/after counts for each step.
    """
    report: dict[str, int] = {"raw": len(books)}

    def _log(step: str, before: int, after: int):
        removed = before - after
        if verbose:
            print(f"  [{step}]  removed {removed:>5,}  →  {after:,} remaining")
        report[step] = removed

    # Step 0 — deduplicate by volume ID (pipeline may overlap)
    seen_ids: set[str] = set()
    books = [b for b in books if not (b["id"] in seen_ids or seen_ids.add(b["id"]))]
    _log("dedup_id", report["raw"], len(books))

    # Step 1 — language filter
    n = len(books)
    books = [
        b for b in books
        if b.get("language", "en") == "en"
        and _is_mostly_latin(b.get("description", "") + b.get("title", ""))
    ]
    _log("language", n, len(books))

    # Step 2 — strip HTML from descriptions, then filter short stubs
    n = len(books)
    for b in books:
        b["description"] = _strip_html(b["description"])
    books = [b for b in books if len(b["description"]) >= 80]
    _log("short_description", n, len(books))

    # Step 3 — invalid ratings
    n = len(books)
    for b in books:
        r = b.get("average_rating")
        if r is not None and not (1.0 <= r <= 5.0):
            b["average_rating"] = None
    # (we don't remove books with bad ratings — just null them out)
    report["nulled_bad_ratings"] = sum(
        1 for b in books if b.get("average_rating") is None
    )
    _log("bad_ratings_nulled", n, len(books))  # count stays same

    # Step 4 — page count outliers
    for b in books:
        pc = b.get("page_count")
        if pc is not None and not (10 <= pc <= 6000):
            b["page_count"] = None

    # Step 5 — normalise categories
    for b in books:
        b["categories"] = _normalise_categories(b.get("categories") or [])

    # Step 6 — extract publication year
    for b in books:
        b["pub_year"] = _extract_year(b.get("published_date", ""))

    # Step 7 — deduplicate near-duplicate editions
    n = len(books)
    seen_keys: set[str] = set()
    deduped = []
    for b in books:
        key = _dedup_key(b)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(b)
    books = deduped
    _log("dedup_editions", n, len(books))

    report["final"] = len(books)
    if verbose:
        print(f"\n  Raw: {report['raw']:,}  →  Clean: {report['final']:,}  "
              f"({report['raw'] - report['final']:,} removed total)")
    return books, report
