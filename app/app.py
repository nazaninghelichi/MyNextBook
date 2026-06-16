import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from pathlib import Path

from src.google_books import search_books
from src.evaluation import diversity as list_diversity

st.set_page_config(page_title="MyNextBook", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #fafafa; color: #1a1a1a; }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #ebebeb;
        padding-top: 2rem;
    }

    /* Search result buttons */
    .stButton > button {
        width: 100%;
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        font-size: 0.82rem;
        padding: 0.35rem 0.6rem;
        text-align: left;
        transition: background 0.15s;
    }
    .stButton > button:hover { background: #f5f5f5; border-color: #bbb; }
    .stButton > button:disabled { color: #aaa; background: #f9f9f9; }

    /* Remove top padding from main area */
    .block-container { padding-top: 2.5rem; max-width: 1200px; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #ebebeb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 500; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #888; }

    /* Book cards */
    .book-card {
        background: #ffffff;
        border: 1px solid #ebebeb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .book-title { font-size: 0.88rem; font-weight: 500; line-height: 1.3; margin: 0.5rem 0 0.2rem; }
    .book-author { font-size: 0.78rem; color: #666; margin-bottom: 0.2rem; }
    .book-meta { font-size: 0.72rem; color: #999; }
    .book-link a {
        font-size: 0.75rem;
        color: #1a1a1a;
        text-decoration: none;
        border-bottom: 1px solid #ccc;
        padding-bottom: 1px;
    }
    .book-link a:hover { border-color: #1a1a1a; }

    hr { border-color: #ebebeb; margin: 1.5rem 0; }
    h1 { font-weight: 400; font-size: 1.6rem; letter-spacing: -0.3px; }
    h3 { font-weight: 400; font-size: 1rem; color: #555; }
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
    st.markdown("### MyNextBook")
    st.markdown("<div style='color:#999;font-size:0.8rem;margin-bottom:1.5rem'>Find your next favourite read</div>", unsafe_allow_html=True)

    liked_genres = st.multiselect("Genres", GENRES, max_selections=3, placeholder="Pick up to 3…")

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    query = st.text_input("Books you like", placeholder="Search title or author…")

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
            label = f"{'✓  ' if already else ''}{book['title'][:38]}"
            if st.button(label, key=f"add_{book['id']}", disabled=already):
                st.session_state.liked_books.append(book)
                st.rerun()

    if st.session_state.liked_books:
        st.markdown(f"<div style='margin-top:1.2rem;font-size:0.75rem;color:#999;text-transform:uppercase;letter-spacing:0.05em'>Your list ({len(st.session_state.liked_books)})</div>", unsafe_allow_html=True)
        for book in list(st.session_state.liked_books):
            col1, col2 = st.columns([6, 1])
            col1.markdown(f"<div style='font-size:0.8rem;padding:0.2rem 0'>{book['title'][:32]}</div>", unsafe_allow_html=True)
            if col2.button("✕", key=f"rm_{book['id']}"):
                st.session_state.liked_books = [
                    b for b in st.session_state.liked_books if b["id"] != book["id"]
                ]
                st.rerun()

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.divider()
    model_name = st.selectbox("Model", ["Hybrid", "Sentence Transformer", "TF-IDF"])
    top_n = st.slider("Results", 5, 20, 10)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Recommendations")

if len(st.session_state.liked_books) < 3:
    needed = 3 - len(st.session_state.liked_books)
    st.markdown(f"<div style='color:#999;font-size:0.9rem'>Add {needed} more book{'s' if needed > 1 else ''} in the sidebar to get recommendations.</div>", unsafe_allow_html=True)
    st.stop()

model, catalogue = load_model(model_name, str(CSV_PATH))

liked_ids    = {b["id"] for b in st.session_state.liked_books}
liked_titles = {b["title"].lower().strip() for b in st.session_state.liked_books}
candidates   = [b for b in catalogue if b["id"] not in liked_ids and b["title"].lower().strip() not in liked_titles]

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
        genre_candidates = [b for b in candidates if set(b.get("categories", [])) & set(liked_genres)]
        pool = genre_candidates if len(genre_candidates) >= top_n else candidates
    else:
        pool = candidates
    recs = use_model.recommend(st.session_state.liked_books, pool, top_n=top_n)

div = list_diversity(recs)
c1, c2, c3 = st.columns(3)
c1.metric("Recommendations", len(recs))
c2.metric("Based on", f"{len(st.session_state.liked_books)} books")
c3.metric("Diversity score", f"{div:.2f}")

st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

cols = st.columns(5)
for i, book in enumerate(recs):
    with cols[i % 5]:
        amz = "+".join((book["title"] + " " + " ".join(book.get("authors", [])[:1])).split())
        card = f"""<div class="book-card">"""
        if book.get("thumbnail"):
            card += f'<img src="{book["thumbnail"]}" width="100%" style="border-radius:4px;display:block">'
        card += f'<div class="book-title">{book["title"]}</div>'
        if book.get("authors"):
            card += f'<div class="book-author">{", ".join(book["authors"][:2])}</div>'
        if book.get("average_rating"):
            stars = "★" * round(book["average_rating"]) + "☆" * (5 - round(book["average_rating"]))
            card += f'<div class="book-meta">{stars}</div>'
        card += f'<div class="book-link" style="margin-top:0.6rem"><a href="https://www.amazon.com/s?k={amz}" target="_blank">Buy on Amazon →</a></div>'
        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)
