"""Run this to regenerate all EDA plots: python notebooks/generate_eda.py"""
import sys, traceback
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_loader import load_raw
from src.cleaning import clean_books

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120})
out = Path('notebooks/eda_plots')
out.mkdir(parents=True, exist_ok=True)

raw_books = load_raw('data/raw/books.csv')
cleaned_books, report = clean_books(raw_books, verbose=True)
df_raw = pd.DataFrame(raw_books)
df = pd.DataFrame(cleaned_books)
df['desc_length']  = df['description'].str.len()
df['category_str'] = df['categories'].apply(lambda c: c[0] if c else 'Unknown')
all_cats = [cat for cats in df['categories'] for cat in cats]

print(f'\nGenerating plots for {len(df):,} clean books...\n')


def save(name):
    plt.savefig(out / name, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {name}')


# ── 1: Cleaning waterfall ─────────────────────────────────────────────────────
steps  = [k for k in report if k not in ('raw','final','nulled_bad_ratings','bad_ratings_nulled')]
values = [report[s] for s in steps]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(steps, values, color='steelblue', edgecolor='white', width=0.6)
ax.bar_label(bars, fmt='%d', padding=4)
ax.set_ylabel('Books removed')
ax.set_title(f'Cleaning Pipeline  |  Raw: {report["raw"]:,}  →  Clean: {report["final"]:,}')
plt.xticks(rotation=18, ha='right')
plt.tight_layout()
save('01_cleaning_waterfall.png')

# ── 2: Missing data ───────────────────────────────────────────────────────────
missing_pct = {}
for col in ['description','authors','categories','average_rating',
            'ratings_count','page_count','published_date','thumbnail']:
    null  = df_raw[col].isnull().sum()
    empty = df_raw[col].apply(lambda v: (isinstance(v, list) and len(v)==0) or v == '').sum()
    missing_pct[col] = (null + empty) / len(df_raw) * 100
s = pd.Series(missing_pct).sort_values()
fig, ax = plt.subplots(figsize=(9, 4))
s.plot(kind='barh', ax=ax, color=['tomato' if v > 10 else 'steelblue' for v in s])
ax.axvline(10, color='red', linestyle='--', linewidth=0.9, label='10% threshold')
ax.set_xlabel('% missing or empty')
ax.set_title('Missing / Empty Data per Field (raw data)')
ax.legend()
plt.tight_layout()
save('02_missing_data.png')

# ── 3: Distributions ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
rated = df.dropna(subset=['average_rating'])
axes[0,0].hist(rated['average_rating'], bins=20, color='steelblue', edgecolor='white')
axes[0,0].axvline(rated['average_rating'].mean(), color='red', linestyle='--',
                   label=f'mean={rated["average_rating"].mean():.2f}')
axes[0,0].set_title(f'Rating Distribution  (n={len(rated):,})')
axes[0,0].set_xlabel('Average rating (1–5)')
axes[0,0].legend()

rc = df[df['ratings_count'] > 0]
axes[0,1].hist(np.log10(rc['ratings_count']+1), bins=30, color='coral', edgecolor='white')
axes[0,1].set_title('Ratings Count  (log₁₀ scale)')
axes[0,1].set_xlabel('log₁₀(ratings count)')

paged = df.dropna(subset=['page_count'])
axes[1,0].hist(paged['page_count'].clip(upper=800), bins=30, color='mediumseagreen', edgecolor='white')
axes[1,0].set_title(f'Page Count  (clipped at 800, n={len(paged):,})')
axes[1,0].set_xlabel('Pages')

axes[1,1].hist(df['desc_length'].clip(upper=2500), bins=30, color='mediumpurple', edgecolor='white')
axes[1,1].set_title('Description Length  (chars, clipped at 2500)')
axes[1,1].set_xlabel('Characters')

plt.suptitle(f'Feature Distributions  ({len(df):,} clean books)', fontsize=13, y=1.01)
plt.tight_layout()
save('03_distributions.png')

# ── 4: Top 20 categories ──────────────────────────────────────────────────────
top_cats = Counter(all_cats).most_common(20)
cats_df  = pd.DataFrame(top_cats, columns=['Category','Count'])
fig, ax  = plt.subplots(figsize=(9, 7))
ax.barh(cats_df['Category'][::-1], cats_df['Count'][::-1], color='steelblue')
ax.set_title('Top 20 Normalised Categories')
ax.set_xlabel('Number of books')
plt.tight_layout()
save('04_top_categories.png')

# ── 5: Mean rating per genre ──────────────────────────────────────────────────
cat_ratings = []
for cat, _ in Counter(all_cats).most_common(15):
    rated_in = df[df['categories'].apply(lambda cs: cat in cs)].dropna(subset=['average_rating'])
    if len(rated_in) >= 3:
        cat_ratings.append({'category': cat,
                            'mean_rating': rated_in['average_rating'].mean(),
                            'n': len(rated_in)})
cat_rat_df = pd.DataFrame(cat_ratings).sort_values('mean_rating', ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(cat_rat_df['category'][::-1], cat_rat_df['mean_rating'][::-1],
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_rat_df)))[::-1])
ax.set_xlim(3.0, 5.0)
ax.set_xlabel('Mean rating')
ax.set_title('Mean Rating per Genre')
for i, row in enumerate(cat_rat_df[::-1].itertuples()):
    ax.text(row.mean_rating + 0.01, i, f'{row.mean_rating:.2f}  (n={row.n})', va='center', fontsize=8)
plt.tight_layout()
save('05_rating_per_genre.png')

# ── 6: Popularity vs quality ──────────────────────────────────────────────────
top6 = df['category_str'].value_counts().head(6).index
df_p = df[df['category_str'].isin(top6) & df['ratings_count'].gt(0)].dropna(subset=['average_rating'])
fig, ax = plt.subplots(figsize=(9, 5))
for cat, grp in df_p.groupby('category_str'):
    ax.scatter(np.log10(grp['ratings_count']+1), grp['average_rating'],
               alpha=0.45, label=cat, s=22)
ax.set_xlabel('log₁₀(ratings count)')
ax.set_ylabel('Average rating')
ax.set_title('Popularity vs Quality by Genre')
ax.legend(bbox_to_anchor=(1.01, 1))
plt.tight_layout()
save('06_popularity_vs_quality.png')

# ── 7: Top words per genre ────────────────────────────────────────────────────
focus = ['Science Fiction','Fantasy','Mystery','Biography','Self-Help','History']
vec = TfidfVectorizer(max_features=8000, stop_words='english')
vec.fit(df['description'])
feat = np.array(vec.get_feature_names_out())
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, genre in zip(axes.flatten(), focus):
    mask = df['categories'].apply(lambda cs: genre in cs)
    sub  = df.loc[mask, 'description']
    if len(sub) < 3:
        ax.text(0.5, 0.5, f'{genre}\nnot enough data', ha='center', va='center', transform=ax.transAxes)
        continue
    scores = vec.transform(sub).toarray().mean(axis=0)
    top_i  = scores.argsort()[-12:][::-1]
    ax.barh(feat[top_i][::-1], scores[top_i][::-1], color='steelblue')
    ax.set_title(f'{genre}  (n={mask.sum()})', fontsize=10)
    ax.set_xlabel('Mean TF-IDF', fontsize=8)
plt.suptitle('Top Words per Genre (TF-IDF on descriptions)', fontsize=13, y=1.01)
plt.tight_layout()
save('07_top_words_per_genre.png')

# ── 8: Temporal ───────────────────────────────────────────────────────────────
dated = df.dropna(subset=['pub_year']).copy()
dated = dated[(dated['pub_year'] >= 1900) & (dated['pub_year'] <= 2025)]
dated['decade'] = (dated['pub_year'] // 10 * 10).astype(int)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
dc = dated['decade'].value_counts().sort_index()
axes[0].bar(dc.index, dc.values, width=8, color='steelblue', edgecolor='white')
axes[0].set_title('Books per Decade')
axes[0].set_xlabel('Decade')
dr = dated.dropna(subset=['average_rating']).groupby('decade')['average_rating'].mean()
axes[1].plot(dr.index, dr.values, marker='o', color='coral', linewidth=2)
axes[1].set_title('Mean Rating by Decade')
axes[1].set_xlabel('Decade')
axes[1].set_ylim(3.0, 5.0)
plt.tight_layout()
save('08_temporal.png')

# ── 9: Language breakdown ─────────────────────────────────────────────────────
lang = df_raw['language'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(7, 4))
lang.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Language Distribution (raw data)')
ax.set_xlabel('Language code')
plt.xticks(rotation=0)
plt.tight_layout()
save('09_language.png')

print(f'\nAll plots saved to {out.resolve()}')
