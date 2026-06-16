import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from pathlib import Path

from src.google_books import search_books
from src.evaluation import diversity as list_diversity

st.set_page_config(page_title="MyNextBook", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #111111; }
    section[data-testid="stSidebar"] { background-color: #f7f7f7; border-right: 1px solid #e0e0e0; }
    .stButton > button {
        background-color: #ffffff;
        color: #111111;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    .stButton > button:hover { background-color: #f0f0f0; border-color: #999; }
    hr { border-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

CSV_PATH = Path(__file__).parent.parent / "data" / "raw" / "books.csv"

GENRES = [
    "Science Fiction", "Fantasy", "Mystery", "Thriller", "Historical Fiction",
    "Biography", "History", "Psychology", "Philosophy", "Self-Help",
    "Business", "Science", "Romance", "Young Adult", "Technology",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_name: str, csv_path: str):
    from src.models import TFIDFRecommender, SentenceTransformerRecommender, HybridRecommender
    from src.data_loader import load_books

    catalogue = load_books(csv_path, clean=True, verbose=False) if Path(csv_path).exists() else []
    model = {
        "Hybrid":               HybridRecommender(alpha=0.6),
        "Sentence Transformer": SentenceTransformerRecommender(),
        "TF-IDF":               TFIDFRecommender(),
    }[model_name]
    if catalogue:
        model.fit(catalogue)
    return model, catalogue


def cached_search(query: str) -> list[dict]:
    return search_books(query, max_results=8)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 MyNextBook")

    st.subheader("Your preferences")
    liked_genres = st.multiselect("Favourite genres", GENRES, max_selections=3)

    st.subheader("Search for books you like")
    query = st.text_input("Title, author, or topic", placeholder="e.g. Dune, Tolkien…")

    if "liked_books" not in st.session_state:
        st.session_state.liked_books = []

    if query:
        try:
            results = cached_search(query)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            st.error(f"Search unavailable (HTTP {status}).")
            results = []
        for book in results:
            already = any(b["id"] == book["id"] for b in st.session_state.liked_books)
            label = f"{'✓ ' if already else ''}{book['title'][:40]}"
            if st.button(label, key=f"add_{book['id']}", disabled=already):
                st.session_state.liked_books.append(book)
                st.rerun()

    if st.session_state.liked_books:
        st.subheader(f"Your list ({len(st.session_state.liked_books)})")
        for book in list(st.session_state.liked_books):
            col1, col2 = st.columns([5, 1])
            col1.caption(book["title"][:35])
            if col2.button("✕", key=f"rm_{book['id']}"):
                st.session_state.liked_books = [
                    b for b in st.session_state.liked_books if b["id"] != book["id"]
                ]
                st.rerun()

    st.divider()
    model_name = st.selectbox("Model", ["Hybrid", "Sentence Transformer", "TF-IDF"])
    top_n = st.slider("Results", 5, 20, 10)


# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Your recommendations")

if len(st.session_state.liked_books) < 3:
    st.info(f"Add at least 3 books in the sidebar to get recommendations. "
            f"({len(st.session_state.liked_books)} added so far)")
    st.stop()

model, catalogue = load_model(model_name, str(CSV_PATH))

liked_ids    = {b["id"] for b in st.session_state.liked_books}
liked_titles = {b["title"].lower().strip() for b in st.session_state.liked_books}
candidates   = [b for b in catalogue if b["id"] not in liked_ids and b["title"].lower().strip() not in liked_titles]

# If any liked books are outside the catalogue, refit with them included
extra = [b for b in st.session_state.liked_books if b["id"] not in {c["id"] for c in catalogue}]
if extra:
    from src.models import TFIDFRecommender, SentenceTransformerRecommender, HybridRecommender
    fresh = {"Hybrid": HybridRecommender(0.6), "Sentence Transformer": SentenceTransformerRecommender(),
             "TF-IDF": TFIDFRecommender()}[model_name]
    with st.spinner("Fitting model…"):
        fresh.fit(catalogue + extra)
    use_model = fresh
else:
    use_model = model

with st.spinner("Finding your next books…"):
    if liked_genres:
        genre_candidates = [
            b for b in candidates
            if set(b.get("categories", [])) & set(liked_genres)
        ]
        pool = genre_candidates if len(genre_candidates) >= top_n else candidates
    else:
        pool = candidates
    recs = use_model.recommend(st.session_state.liked_books, pool, top_n=top_n)

div = list_diversity(recs)
c1, c2, c3 = st.columns(3)
c1.metric("Recommendations", len(recs))
c2.metric("Based on", f"{len(st.session_state.liked_books)} books")
c3.metric("Diversity", f"{div:.2f}")

st.divider()

cols = st.columns(5)
for i, book in enumerate(recs):
    with cols[i % 5]:
        if book.get("thumbnail"):
            st.image(book["thumbnail"], width=120)
        st.markdown(f"**{book['title']}**")
        if book.get("authors"):
            st.caption(", ".join(book["authors"][:2]))
        cats = book.get("categories", [])
        if cats:
            st.caption(" · ".join(cats[:2]))
        if book.get("average_rating"):
            st.caption(f"{'★' * round(book['average_rating'])} {book['average_rating']}")
        query = "+".join((book["title"] + " " + " ".join(book.get("authors", [])[:1])).split())
        st.markdown(f"[Buy on Amazon](https://www.amazon.com/s?k={query})")
