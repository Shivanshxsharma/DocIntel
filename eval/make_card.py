"""
Render a 1200x627 results card (LinkedIn / README image) from results/summary.json.

    python eval/make_card.py   →  eval/results/docintel_card.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
S = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

# Light chart chrome + the validated two-series palette used in the UI.
PAGE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE, DOC, GOOD = "#eb6834", "#2a78d6", "#006300"

for f in ("segoeui.ttf", "segoeuib.ttf", "georgia.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
SANS = "Segoe UI"
SERIF = "Georgia"

R = S["retrieval"]
base, doc = R["naive"]["metrics"], R["docintel"]["metrics"]
pf = S["parse_fidelity"]
d = S["dataset"]
miss_b, miss_d = 1 - base["hit@5"]["mean"], 1 - doc["hit@5"]["mean"]

fig = plt.figure(figsize=(12, 6.27), dpi=100, facecolor=PAGE)

# ── Header ────────────────────────────────────────────────────────────────
fig.text(0.045, 0.885, "DocIntel vs. standard RAG", fontfamily=SERIF, fontsize=30, color=INK)
fig.text(0.045, 0.825,
         f"QASPER benchmark · {d['papers']} arXiv papers · {d['answerable']} expert-annotated questions · "
         "same embeddings, top-5, LLM and prompt",
         fontfamily=SANS, fontsize=12.5, color=INK2)

# ── Hero stats (left column) ──────────────────────────────────────────────
stats = [
    (f"{100 * (1 - miss_d / miss_b):.0f}%", "fewer retrieval misses",
     f"evidence in top-5: {100 * base['hit@5']['mean']:.0f}% → {100 * doc['hit@5']['mean']:.0f}%"),
    (f"+{100 * (doc['mrr@5']['mean'] / base['mrr@5']['mean'] - 1):.0f}%", "mean reciprocal rank",
     f"{base['mrr@5']['mean']:.2f} → {doc['mrr@5']['mean']:.2f}"),
    (f"{100 * pf['docling']['chunk_intact']['mean']:.0f}%", "evidence paragraphs kept intact",
     f"vs {100 * pf['pypdf_1000']['chunk_intact']['mean']:.0f}% with PyPDF + fixed chunks"),
]
y = 0.66
for big, label, sub in stats:
    fig.text(0.045, y, big, fontfamily=SERIF, fontsize=40, color=INK, va="baseline")
    fig.text(0.205, y + 0.035, label, fontfamily=SANS, fontsize=13.5, color=INK, fontweight="bold")
    fig.text(0.205, y - 0.005, sub, fontfamily=SANS, fontsize=11.5, color=MUTED)
    y -= 0.19

# ── Paired bars (right column) ────────────────────────────────────────────
ax = fig.add_axes([0.6, 0.17, 0.35, 0.56], facecolor=PAGE)
metrics = [("hit@1", "Hit@1"), ("hit@5", "Hit@5"), ("recall@5", "Evidence recall@5"), ("ndcg@5", "nDCG@5 (×100)")]
h = 0.34
for i, (k, label) in enumerate(metrics):
    yb = len(metrics) - 1 - i
    for off, val, color in ((h / 2 + 0.02, base[k]["mean"], BASE), (-h / 2 - 0.02, doc[k]["mean"], DOC)):
        ax.barh(yb + off, val, height=h, color=color, edgecolor="none")
        ax.text(val + 0.015, yb + off, f"{val * 100:.0f}",
                va="center", fontfamily=SANS, fontsize=10.5, color=INK2)
    ax.text(-0.02, yb, label, ha="right", va="center", fontfamily=SANS, fontsize=11.5, color=INK)
ax.set_xlim(0, 1.08)
ax.set_ylim(-0.6, len(metrics) - 0.4)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "25", "50", "75", "100"], fontfamily=SANS, fontsize=9.5, color=MUTED)
ax.grid(axis="x", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0)
# legend
fig.patches.extend([
    plt.Rectangle((0.6, 0.775), 0.012, 0.022, color=BASE, transform=fig.transFigure, figure=fig),
    plt.Rectangle((0.735, 0.775), 0.012, 0.022, color=DOC, transform=fig.transFigure, figure=fig),
])
fig.text(0.617, 0.778, "Standard RAG", fontfamily=SANS, fontsize=11, color=INK2)
fig.text(0.752, 0.778, "DocIntel", fontfamily=SANS, fontsize=11, color=INK2)

# ── Footer ────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.045, 0.955], [0.085, 0.085], color=GRID, linewidth=1, transform=fig.transFigure))
fig.text(0.045, 0.04,
         "Docling layout parsing · structure-aware chunking · hybrid dense + sparse retrieval · cross-encoder rerank",
         fontfamily=SANS, fontsize=10.5, color=MUTED)
fig.text(0.955, 0.04, "all retrieval gains p < 0.01 (paired bootstrap)", ha="right",
         fontfamily=SANS, fontsize=10.5, color=MUTED)

out = RESULTS / "docintel_card.png"
fig.savefig(out, facecolor=PAGE, dpi=100)
print(f"wrote {out}")
