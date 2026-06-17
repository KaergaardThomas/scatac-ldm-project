"""
annotate_peaks.py  —  Annotate LDM marker peaks with nearest hg19 genes
========================================================================
Downloads the hg19 RefSeq gene table from UCSC once, caches it locally,
then for each top-N peak per cell type finds:
  - the nearest gene
  - distance to its transcription start site (TSS)
  - whether the peak overlaps the promoter (<2kb from TSS), gene body, or is intergenic

Usage:
    python annotate_peaks.py

Requirements:
    pip install pandas numpy requests

Output:
    peak_gene_annotation.csv   — full annotation table
    peak_gene_annotation.txt   — pretty-printed summary per cell type
"""

import pandas as pd
import numpy as np
import requests
import gzip
import io
from pathlib import Path

HERE      = Path(__file__).parent
PEAKS_CSV = HERE / "marker_peaks_top50_adj.csv"
CACHE     = HERE / "hg19_refseq_genes.tsv.gz"
OUT_CSV   = HERE / "peak_gene_annotation.csv"
OUT_TXT   = HERE / "peak_gene_annotation.txt"

TOP_N       = 5
PROMOTER_KB = 2000   # bp upstream of TSS counted as promoter

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


# ── 1. Download / load hg19 RefSeq gene table ────────────────────────────────
def load_refseq() -> pd.DataFrame:
    if CACHE.exists():
        print("Loading cached hg19 RefSeq gene table …")
        genes = pd.read_csv(CACHE, sep="\t", compression="gzip")
    else:
        print("Downloading hg19 RefSeq gene table from UCSC (~10 MB) …")
        url = (
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        cols = [
            "bin", "name", "chrom", "strand", "txStart", "txEnd",
            "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
            "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames",
        ]
        genes = pd.read_csv(
            io.BytesIO(gzip.decompress(r.content)),
            sep="\t", header=None, names=cols,
        )
        # Keep one record per gene symbol (longest transcript)
        genes["tx_len"] = genes["txEnd"] - genes["txStart"]
        genes = (
            genes.sort_values("tx_len", ascending=False)
            .drop_duplicates(subset=["chrom", "name2"])
            .reset_index(drop=True)
        )
        genes = genes[["name2", "chrom", "strand", "txStart", "txEnd"]].copy()
        genes.columns = ["gene", "chrom", "strand", "txStart", "txEnd"]

        genes.to_csv(CACHE, sep="\t", index=False, compression="gzip")
        print(f"  Cached to {CACHE.name}")

    print(f"  {len(genes):,} gene records loaded")
    return genes


# ── 2. Parse peak coordinates from feature string ────────────────────────────
def parse_peak(feat: str):
    parts = feat.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    return None, None, None


# ── 3. Annotate one peak against gene table ──────────────────────────────────
def annotate_peak(chrom, start, end, genes: pd.DataFrame):
    mid = (start + end) // 2

    g = genes[genes["chrom"] == chrom].copy()
    if g.empty:
        return {"nearest_gene": ".", "tss": np.nan,
                "dist_to_tss": np.nan, "region": "intergenic"}

    # TSS depends on strand
    g = g.copy()
    g["tss"] = np.where(g["strand"] == "+", g["txStart"], g["txEnd"])
    g["dist_to_tss"] = (g["tss"] - mid).abs()

    nearest = g.loc[g["dist_to_tss"].idxmin()]
    dist    = int(nearest["dist_to_tss"])
    gene    = nearest["gene"]
    tss     = int(nearest["tss"])

    # Classify region
    if dist <= PROMOTER_KB:
        region = "promoter"
    elif start <= nearest["txEnd"] and end >= nearest["txStart"]:
        region = "gene body"
    else:
        region = "intergenic"

    return {
        "nearest_gene": gene,
        "tss":          tss,
        "dist_to_tss":  dist,
        "region":       region,
    }


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    genes = load_refseq()

    df = pd.read_csv(PEAKS_CSV)
    df = df[df["rank"] <= TOP_N].copy()

    # Parse coordinates
    coords = df["peak_feature"].apply(
        lambda f: pd.Series(parse_peak(f), index=["chrom", "start", "end"])
    )
    df = pd.concat([df, coords], axis=1)

    # Annotate each peak
    print(f"\nAnnotating {len(df)} peaks …")
    ann = df.apply(
        lambda r: pd.Series(annotate_peak(r["chrom"], r["start"], r["end"], genes)),
        axis=1,
    )
    df = pd.concat([df, ann], axis=1)

    # Readable peak label
    df["peak_label"] = df.apply(
        lambda r: f"{r['chrom']}:{int(r['start']):,}–{int(r['end']):,}", axis=1
    )
    df["dist_kb"] = (df["dist_to_tss"] / 1000).round(1)
    df["ct_short"] = df["cell_type"].map(SHORT)

    # Save full CSV
    out_cols = [
        "ct_short", "rank", "peak_label",
        "nearest_gene", "dist_kb", "region",
        "distance", "omega_j", "adj_score",
    ]
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV.name}")

    # ── Pretty-printed summary ────────────────────────────────────────────────
    lines = []
    lines.append("LDM marker peak annotation — top-5 peaks per cell type")
    lines.append("=" * 75)
    lines.append(
        f"{'Cell type':<18} {'Rank':>4}  {'Nearest gene':<14} "
        f"{'Dist to TSS (kb)':>16}  {'Region'}"
    )
    lines.append("-" * 75)

    ct_order = [
        "CD34_Progenitors", "Monocytes", "Dendritic_Cells",
        "NK_Cells", "B_Cells",
        "Naive_CD4_T_Cells", "CD4_HelperT", "Memory_CD4_T_Cells",
        "Regulatory_T_Cells", "Naive_CD8_T_Cells", "Memory_CD8_T_Cells",
    ]
    df["ct_order"] = pd.Categorical(
        df["cell_type"], categories=ct_order, ordered=True
    )
    df = df.sort_values(["ct_order", "rank"])

    prev_ct = None
    for _, row in df.iterrows():
        if row["ct_short"] != prev_ct:
            if prev_ct is not None:
                lines.append("")
            prev_ct = row["ct_short"]

        lines.append(
            f"{row['ct_short']:<18} {int(row['rank']):>4}  "
            f"{row['nearest_gene']:<14} "
            f"{row['dist_kb']:>16.1f}  "
            f"{row['region']}"
        )

    lines.append("")
    lines.append(
        "Region:  promoter = within 2 kb of TSS  |  "
        "gene body = within transcript  |  intergenic = neither"
    )

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_TXT.write_text(summary)
    print(f"\nSaved {OUT_TXT.name}")


if __name__ == "__main__":
    main()
