"""
plot_ldm_results.py  —  Visualise LDM marker-peak analysis results
===================================================================
Produces two publication-ready figures:
  1. Centroid distance heatmap  (centroid_distances.csv)
  2. Marker peak dot plot       (marker_peaks_top50_adj.csv)

Usage:
    python plot_ldm_results.py

Requirements:
    pip install pandas numpy matplotlib seaborn scipy

Output files (saved in the same folder as this script):
    fig1_centroid_heatmap.pdf   (also .png)
    fig2_marker_peak_dotplot.pdf (also .png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from pathlib import Path

# ── Paths — edit if your CSVs are somewhere else ────────────────────────────
HERE      = Path(__file__).parent
DIST_CSV  = HERE / "/Users/thomaskaergaard/scatac-ldm-project/ldm_marker_peaks/centroid_distances.csv"
PEAKS_CSV = HERE / "/Users/thomaskaergaard/scatac-ldm-project/ldm_marker_peaks/marker_peaks_top50_adj.csv"

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

# Short labels for cell types (used on both figures)
SHORT = {
    "B_Cells":             "B cells",
    "CD34_Progenitors":    "CD34+ prog.",
    "CD4_HelperT":         "CD4 helper T",
    "Dendritic_Cells":     "Dendritic",
    "Memory_CD4_T_Cells":  "Mem. CD4 T",
    "Memory_CD8_T_Cells":  "Mem. CD8 T",
    "Monocytes":           "Monocytes",
    "NK_Cells":            "NK cells",
    "Naive_CD4_T_Cells":   "Naive CD4 T",
    "Naive_CD8_T_Cells":   "Naive CD8 T",
    "Regulatory_T_Cells":  "Treg",
}

# ── Colours matched exactly to UMAP figure (matplotlib default tab10 cycle) ──
LINEAGE_COLOUR = {
    "B_Cells":             "#1f77b4",   # blue
    "CD34_Progenitors":    "#ff7f0e",   # orange
    "CD4_HelperT":         "#2ca02c",   # green
    "Dendritic_Cells":     "#d62728",   # red
    "Memory_CD4_T_Cells":  "#9467bd",   # purple
    "Memory_CD8_T_Cells":  "#8c564b",   # brown
    "Monocytes":           "#e377c2",   # pink
    "NK_Cells":            "#7f7f7f",   # grey
    "Naive_CD4_T_Cells":   "#bcbd22",   # yellow-green
    "Naive_CD8_T_Cells":   "#17becf",   # cyan
    "Regulatory_T_Cells":  "#aec7e8",   # light blue
}


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Centroid distance heatmap
# ════════════════════════════════════════════════════════════════════════════
def plot_heatmap():
    df = pd.read_csv(DIST_CSV, index_col=0)
    df.index   = [SHORT.get(x, x) for x in df.index]
    df.columns = [SHORT.get(x, x) for x in df.columns]

    # Hierarchical clustering to reorder rows/cols
    # Use condensed distance form to avoid the hollow-matrix warning
    from scipy.spatial.distance import squareform
    condensed = squareform(df.values, checks=False)
    link  = linkage(condensed, method="average")
    order = leaves_list(link)
    df    = df.iloc[order, order]

    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.eye(len(df), dtype=bool)          # hide diagonal (zeros)

    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd_r",
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Euclidean distance in latent space", "shrink": 0.75},
        vmin=0,
    )

    ax.set_title(
        "Pairwise centroid distances between cell-type clusters\n"
        "in the LDM latent space (d = 16)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # Annotate the diagonal with '—' so it doesn't look blank
    n = len(df)
    for i in range(n):
        ax.text(i + 0.5, i + 0.5, "—", ha="center", va="center",
                fontsize=8, color="gray")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig1_centroid_heatmap.{ext}",
                    bbox_inches="tight", dpi=200)
    print("Saved fig1_centroid_heatmap.pdf / .png")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Marker peak dot plot (top N peaks per cell type)
# ════════════════════════════════════════════════════════════════════════════
def plot_dotplot(top_n: int = 5):
    df = pd.read_csv(PEAKS_CSV)
    df = df[df["rank"] <= top_n].copy()

    # Short cell-type labels
    df["ct_short"] = df["cell_type"].map(SHORT)

    # Clean peak label: chr2_87012592_87013092 → chr2:87.0 Mb
    def peak_label(feat: str) -> str:
        parts = feat.rsplit("_", 2)
        if len(parts) == 3:
            chrom, start = parts[0], int(parts[1])
            mb = start / 1_000_000
            return f"{chrom}:{mb:.1f} Mb"
        return feat

    df["peak_label"] = df["peak_feature"].apply(peak_label)

    # Order cell types by lineage grouping (myeloid → lymphoid → T cells)
    ct_order = [
        "CD34_Progenitors", "Monocytes", "Dendritic_Cells",
        "NK_Cells", "B_Cells",
        "Naive_CD4_T_Cells", "CD4_HelperT", "Memory_CD4_T_Cells",
        "Regulatory_T_Cells", "Naive_CD8_T_Cells", "Memory_CD8_T_Cells",
    ]
    ct_order_short = [SHORT[c] for c in ct_order if c in SHORT]
    df["ct_short"] = pd.Categorical(df["ct_short"], categories=ct_order_short, ordered=True)
    df = df.sort_values(["ct_short", "rank"])

    # Build a unique y-axis: "cell type  |  peak label"
    df["y_label"] = df["ct_short"].astype(str) + "  ·  " + df["peak_label"]

    # Assign integer y positions: rank 1 at top of each group
    # Sort descending so the first cell type's rank-1 is at the highest y
    df = df.sort_values(["ct_short", "rank"], ascending=[False, False])
    df["y"] = range(len(df))

    # y_labels and colour list aligned to ascending y (bottom → top)
    df_asc   = df.sort_values("y")
    y_labels = df_asc["y_label"].tolist()
    y_colours = [LINEAGE_COLOUR.get(r["cell_type"], "#333333")
                 for _, r in df_asc.iterrows()]

    # Fixed rectangular size suitable for a report (A4 column width)
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        colour = LINEAGE_COLOUR.get(row["cell_type"], "#888888")
        specificity = max(0.0, -row["omega_j"])
        size = 20 + specificity * 60

        ax.scatter(
            row["adj_score"], row["y"],
            s=size,
            color=colour,
            alpha=0.85,
            zorder=3,
            linewidths=0.3,
            edgecolors="white",
        )

    # Cell-type boundary lines
    prev_ct = None
    for _, row in df.sort_values("y", ascending=False).iterrows():
        ct = row["ct_short"]
        if prev_ct is not None and ct != prev_ct:
            ax.axhline(row["y"] + 0.5, color="lightgray", linewidth=0.8, zorder=1)
        prev_ct = ct

    # Tick labels — built from the same sorted df so counts always match
    ax.set_yticks(df_asc["y"].tolist())
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick_label, colour in zip(ax.get_yticklabels(), y_colours):
        tick_label.set_color(colour)

    ax.set_xlabel("Adjusted score  (distance − ω_j)", fontsize=10)
    ax.set_title(
        f"Top {top_n} marker peaks per cell type ranked by adjusted centroid distance\n"
        "Dot size = peak specificity (−ω_j);  smaller score = stronger marker",
        fontsize=10, pad=10,
    )
    ax.grid(axis="x", color="lightgray", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend for dot size
    for spec, label in [(0.0, "low spec."), (1.0, "med."), (2.0, "high spec.")]:
        ax.scatter([], [], s=20 + spec * 60, color="gray", alpha=0.7,
                   label=label, edgecolors="white", linewidths=0.3)
    ax.legend(title="Peak specificity\n(−ω_j)", loc="lower right",
              fontsize=7, title_fontsize=7, framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig2_marker_peak_dotplot.{ext}",
                    bbox_inches="tight", dpi=200)
    print("Saved fig2_marker_peak_dotplot.pdf / .png")
    return fig


# ── Run both ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Plotting centroid heatmap …")
    plot_heatmap()

    print("Plotting marker peak dot plot …")
    plot_dotplot(top_n=5)

    plt.show()
    print("\nDone.")