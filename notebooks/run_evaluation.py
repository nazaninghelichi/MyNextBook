"""
Model comparison evaluation.
Run from the project root:  python notebooks/run_evaluation.py
"""
import sys, time
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.data_loader import load_books
from src.models import RandomRecommender, TFIDFRecommender, SentenceTransformerRecommender, HybridRecommender
from src.evaluation import leave_one_out_eval

out = Path('notebooks/eda_plots')
out.mkdir(exist_ok=True)

# ── 1. Load & clean data ──────────────────────────────────────────────────────
print('Loading data...')
books = load_books('data/raw/books.csv', clean=True, verbose=False)
print(f'  {len(books):,} clean books ready\n')

# ── 2. Fit all models on the full catalogue ───────────────────────────────────
# Correct: we fit on ALL books. The evaluation withholds individual books
# ONLY during the query phase — the model's knowledge of the catalogue
# is not the issue; the issue is that the query never includes the held-out book.

models = {
    'Random':              RandomRecommender(),
    'TF-IDF':             TFIDFRecommender(),
    'Sentence Transformer': SentenceTransformerRecommender(),
    'Hybrid':             HybridRecommender(alpha=0.6),
}

print('Fitting models...')
for name, model in models.items():
    t0 = time.time()
    model.fit(books)
    print(f'  ✓ {name:<25} ({time.time()-t0:.1f}s)')

# ── 3. Leave-one-out evaluation ───────────────────────────────────────────────
print('\nRunning leave-one-out evaluation (max 300 test cases per model)...')
results = {}
for name, model in models.items():
    t0 = time.time()
    metrics = leave_one_out_eval(
        model=model,
        books=books,
        n_query=3,
        min_cat_size=6,
        top_n=10,
        k_values=(5, 10),
        max_cases=300,
    )
    elapsed = time.time() - t0
    results[name] = metrics
    if 'error' in metrics:
        print(f'  {name:<25}  ERROR: {metrics["error"]}')
    else:
        print(f'  {name:<25}  hit@10={metrics["hit_rate@10"]:.4f}  '
              f'ndcg@10={metrics["ndcg@10"]:.4f}  '
              f'diversity={metrics["diversity"]:.4f}  '
              f'({elapsed:.1f}s)')

# ── 4. Results table ──────────────────────────────────────────────────────────
df = pd.DataFrame(results).T
display_cols = ['hit_rate@5','hit_rate@10','ndcg@5','ndcg@10','diversity','n_test_cases']
df = df[[c for c in display_cols if c in df.columns]]

print('\n' + '='*65)
print('RESULTS')
print('='*65)
print(df.to_string())
print('='*65)

# ── 5. Comparison bar chart ───────────────────────────────────────────────────
plot_metrics = ['hit_rate@10', 'ndcg@10', 'diversity']
plot_labels  = ['Hit Rate @ 10', 'NDCG @ 10', 'Diversity']
model_names  = list(results.keys())
colors       = ['#999999', '#4C72B0', '#DD8452', '#55A868']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, metric, label in zip(axes, plot_metrics, plot_labels):
    values = [results[m].get(metric, 0) for m in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='white', width=0.6)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 + 0.01)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    # Highlight random baseline
    ax.axhline(values[0], color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

plt.suptitle('Model Comparison — Leave-One-Out Evaluation', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(out / '10_model_comparison.png', bbox_inches='tight')
plt.close()
print(f'\nPlot saved → {out / "10_model_comparison.png"}')
