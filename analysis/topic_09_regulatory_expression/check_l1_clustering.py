#!/usr/bin/env python3
"""
check_l1_clustering.py — L1 Annotation Clustering Analysis

For each DRS read that passed L1 filter, count how many L1 annotation
elements in the genome overlap that read's alignment span. This tells us
whether reads land in isolated L1 elements or in regions with multiple
adjacent/overlapping L1 annotations (fragmented ancient elements).
"""

import os
import subprocess
import pandas as pd
import numpy as np
from scipy import stats

# === Paths ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS_GROUP = os.path.join(PROJECT, "results_group")
L1_ANNOTATION_BED = os.path.join(
    PROJECT,
    "analysis/01_exploration/topic_05_cellline/pus_enzyme_analysis/"
    "matched_filter_tmp/L1_TE_6col.bed"
)
PART3_CACHE = os.path.join(
    PROJECT,
    "analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache"
)
OUTDIR = os.path.join(
    PROJECT,
    "analysis/01_exploration/topic_09_regulatory_expression/l1_clustering"
)
BEDTOOLS = "/blaze/apps/envs/bedtools/2.31.0/bin/bedtools"
os.makedirs(OUTDIR, exist_ok=True)

BASE_GROUPS = [
    "A549_4", "A549_5", "A549_6",
    "H9_2", "H9_3", "H9_4",
    "Hct116_3", "Hct116_4",
    "Hek293_1",
    "Hek293T_3", "Hek293T_4",
    "HeLa_1", "HeLa_2", "HeLa_3",
    "HepG2_5", "HepG2_6",
    "HEYA8_1", "HEYA8_2", "HEYA8_3",
    "K562_4", "K562_5", "K562_6",
    "MCF7_2", "MCF7_3", "MCF7_4",
    "SHSY5Y_1", "SHSY5Y_2", "SHSY5Y_3",
    "THP1_1",
]
YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}


def load_l1_summaries():
    frames = []
    for grp in BASE_GROUPS:
        path = os.path.join(RESULTS_GROUP, grp, "g_summary", f"{grp}_L1_summary.tsv")
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df["group"] = grp
        df["cell_line"] = grp.rsplit("_", 1)[0]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} reads from {len(frames)} groups")
    return combined


def load_part3_cache():
    frames = []
    for grp in BASE_GROUPS:
        path = os.path.join(PART3_CACHE, f"{grp}_l1_per_read.tsv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        df["group"] = grp
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    print(f"Part3 cache: {len(combined):,} reads from {len(frames)} groups")
    return combined


def run_bedtools_intersect(reads_df):
    reads_bed_path = os.path.join(OUTDIR, "reads.bed")
    annot_intersect_path = os.path.join(OUTDIR, "reads_vs_l1_annotation.tsv")

    bed_df = reads_df[["chr", "start", "end", "read_id", "read_strand"]].copy()
    bed_df["score"] = "."
    bed_df = bed_df[["chr", "start", "end", "read_id", "score", "read_strand"]]
    bed_df = bed_df.dropna(subset=["chr", "start", "end"])
    bed_df["start"] = bed_df["start"].astype(int)
    bed_df["end"] = bed_df["end"].astype(int)
    bed_df = bed_df[bed_df["end"] > bed_df["start"]]
    bed_df.to_csv(reads_bed_path, sep="\t", header=False, index=False)
    print(f"Wrote {len(bed_df):,} reads to {reads_bed_path}")

    sorted_reads = reads_bed_path + ".sorted"
    sorted_annot = os.path.join(OUTDIR, "l1_annot.sorted.bed")
    subprocess.run(f"sort -k1,1 -k2,2n {reads_bed_path} > {sorted_reads}",
                   shell=True, check=True)
    subprocess.run(f"sort -k1,1 -k2,2n {L1_ANNOTATION_BED} > {sorted_annot}",
                   shell=True, check=True)

    cmd = [BEDTOOLS, "intersect", "-a", sorted_reads, "-b", sorted_annot, "-wo", "-sorted"]
    print("Running bedtools intersect...")
    with open(annot_intersect_path, "w") as fout:
        result = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"  bedtools stderr: {result.stderr[:500]}")
        cmd2 = [BEDTOOLS, "intersect", "-a", reads_bed_path, "-b", L1_ANNOTATION_BED, "-wo"]
        print("  Retrying without -sorted...")
        with open(annot_intersect_path, "w") as fout:
            subprocess.run(cmd2, stdout=fout, stderr=subprocess.PIPE, text=True, check=True)

    col_names = [
        "read_chr", "read_start", "read_end", "read_id", "score", "read_strand",
        "annot_chr", "annot_start", "annot_end", "annot_name", "annot_subfamily",
        "annot_strand", "overlap_bp"
    ]
    intersect_df = pd.read_csv(annot_intersect_path, sep="\t", header=None, names=col_names)
    print(f"Intersect output: {len(intersect_df):,} overlap records")

    per_read = intersect_df.groupby("read_id").agg(
        n_l1_overlaps=("annot_name", "nunique"),
        total_overlap_bp=("overlap_bp", "sum"),
        max_overlap_bp=("overlap_bp", "max"),
        annot_names=("annot_name", lambda x: ";".join(sorted(set(x)))),
        annot_subfamilies=("annot_subfamily", lambda x: ";".join(sorted(set(x)))),
    ).reset_index()

    best_annot = intersect_df.loc[
        intersect_df.groupby("read_id")["overlap_bp"].idxmax()
    ][["read_id", "annot_name", "annot_subfamily"]].rename(
        columns={"annot_name": "best_annot_name", "annot_subfamily": "best_annot_subfamily"}
    )
    per_read = per_read.merge(best_annot, on="read_id", how="left")

    return per_read, intersect_df


def analyze_clustering(merged_df, summary_path):
    lines = []
    def p(msg=""):
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("L1 ANNOTATION CLUSTERING ANALYSIS")
    p("How many L1 annotation elements overlap each DRS read?")
    p("=" * 80)

    total = len(merged_df)
    p(f"\nTotal reads: {total:,}")

    # Distribution
    p("\n--- Distribution of L1 element overlaps per read ---")
    overlap_counts = merged_df["n_l1_overlaps"].value_counts().sort_index()
    for n, count in overlap_counts.items():
        pct = count / total * 100
        bar = "#" * max(1, int(pct / 2))
        p(f"  {int(n):>3} L1 elements: {count:>6,} reads ({pct:5.1f}%) {bar}")

    merged_df["overlap_cat"] = merged_df["n_l1_overlaps"].apply(
        lambda x: "1" if x == 1 else "2" if x == 2 else "3" if x == 3 else "4+"
    )
    cat_order = ["1", "2", "3", "4+"]
    cat_counts = merged_df["overlap_cat"].value_counts()
    p(f"\n  Single L1:   {cat_counts.get('1', 0):>6,} ({cat_counts.get('1', 0)/total*100:.1f}%)")
    p(f"  2 L1:        {cat_counts.get('2', 0):>6,} ({cat_counts.get('2', 0)/total*100:.1f}%)")
    p(f"  3 L1:        {cat_counts.get('3', 0):>6,} ({cat_counts.get('3', 0)/total*100:.1f}%)")
    p(f"  4+ L1:       {cat_counts.get('4+', 0):>6,} ({cat_counts.get('4+', 0)/total*100:.1f}%)")
    multi = merged_df[merged_df["n_l1_overlaps"] >= 2]
    p(f"\n  Multi-L1 reads (>=2): {len(multi):,} ({len(multi)/total*100:.1f}%)")

    # Overlap bp stats
    p("\n--- Overlap statistics (bp) ---")
    for cat in cat_order:
        sub = merged_df[merged_df["overlap_cat"] == cat]
        if len(sub) == 0:
            continue
        p(f"  {cat} L1: total_overlap median={sub['total_overlap_bp'].median():.0f}, "
          f"max_overlap median={sub['max_overlap_bp'].median():.0f}, "
          f"read_length median={sub['read_length'].median():.0f}")

    # Read length
    p("\n--- Read length by overlap category ---")
    for cat in cat_order:
        sub = merged_df[merged_df["overlap_cat"] == cat]
        if len(sub) == 0:
            continue
        p(f"  {cat} L1: n={len(sub):,}, RL median={sub['read_length'].median():.0f}, "
          f"mean={sub['read_length'].mean():.0f}")
    groups_rl = [merged_df[merged_df["overlap_cat"] == c]["read_length"].dropna().values
                 for c in cat_order if (merged_df["overlap_cat"] == c).sum() > 0]
    if len(groups_rl) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups_rl)
        p(f"  Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.2e}")

    # Young vs Ancient
    p("\n--- Young vs Ancient L1 in overlap categories ---")
    if "gene_id" in merged_df.columns:
        merged_df["is_young"] = merged_df["gene_id"].isin(YOUNG_SUBFAMILIES)
    else:
        merged_df["is_young"] = False
    for cat in cat_order:
        sub = merged_df[merged_df["overlap_cat"] == cat]
        if len(sub) == 0:
            continue
        young_pct = sub["is_young"].mean() * 100
        p(f"  {cat} L1: young={young_pct:.1f}% (n={sub['is_young'].sum()}/{len(sub)})")

    # Poly(A)
    p("\n--- Poly(A) length by overlap category ---")
    polya_col = "polya_length" if "polya_length" in merged_df.columns else None
    if polya_col:
        valid_polya = merged_df[merged_df[polya_col].notna() & (merged_df[polya_col] > 0)]
        for cat in cat_order:
            sub = valid_polya[valid_polya["overlap_cat"] == cat]
            if len(sub) == 0:
                continue
            p(f"  {cat} L1: n={len(sub):,}, poly(A) median={sub[polya_col].median():.1f}, "
              f"mean={sub[polya_col].mean():.1f}")
        g1 = valid_polya[valid_polya["overlap_cat"] == "1"][polya_col]
        g2 = valid_polya[valid_polya["n_l1_overlaps"] >= 2][polya_col]
        if len(g1) > 0 and len(g2) > 0:
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p(f"  1-L1 vs multi-L1 poly(A): MWU p={u_p:.2e}, "
              f"delta={g2.median() - g1.median():.1f}nt")

    # m6A/kb
    p("\n--- m6A/kb by overlap category ---")
    if "m6a_per_kb" in merged_df.columns:
        valid_m6a = merged_df[merged_df["m6a_per_kb"].notna()]
        for cat in cat_order:
            sub = valid_m6a[valid_m6a["overlap_cat"] == cat]
            if len(sub) == 0:
                continue
            p(f"  {cat} L1: n={len(sub):,}, m6A/kb median={sub['m6a_per_kb'].median():.2f}, "
              f"mean={sub['m6a_per_kb'].mean():.2f}")
        g1 = valid_m6a[valid_m6a["overlap_cat"] == "1"]["m6a_per_kb"]
        g2 = valid_m6a[valid_m6a["n_l1_overlaps"] >= 2]["m6a_per_kb"]
        if len(g1) > 0 and len(g2) > 0:
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p(f"  1-L1 vs multi-L1 m6A/kb: MWU p={u_p:.2e}, "
              f"delta={g2.median() - g1.median():.2f}")

    # psi/kb
    p("\n--- psi/kb by overlap category ---")
    if "psi_per_kb" in merged_df.columns:
        valid_psi = merged_df[merged_df["psi_per_kb"].notna()]
        for cat in cat_order:
            sub = valid_psi[valid_psi["overlap_cat"] == cat]
            if len(sub) == 0:
                continue
            p(f"  {cat} L1: n={len(sub):,}, psi/kb median={sub['psi_per_kb'].median():.2f}, "
              f"mean={sub['psi_per_kb'].mean():.2f}")
        g1 = valid_psi[valid_psi["overlap_cat"] == "1"]["psi_per_kb"]
        g2 = valid_psi[valid_psi["n_l1_overlaps"] >= 2]["psi_per_kb"]
        if len(g1) > 0 and len(g2) > 0:
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p(f"  1-L1 vs multi-L1 psi/kb: MWU p={u_p:.2e}, "
              f"delta={g2.median() - g1.median():.2f}")

    # Read length-matched
    p("\n--- Read length-matched comparison (500-2000bp) ---")
    rl_matched = merged_df[(merged_df["read_length"] >= 500) & (merged_df["read_length"] <= 2000)]
    p(f"  Reads in 500-2000bp range: {len(rl_matched):,}")
    for cat in cat_order:
        sub = rl_matched[rl_matched["overlap_cat"] == cat]
        if len(sub) == 0:
            continue
        parts = [f"n={len(sub):,}", f"RL={sub['read_length'].median():.0f}"]
        if polya_col and sub[polya_col].notna().sum() > 0:
            parts.append(f"polyA={sub[polya_col].median():.1f}")
        if "m6a_per_kb" in sub.columns and sub["m6a_per_kb"].notna().sum() > 0:
            parts.append(f"m6A/kb={sub['m6a_per_kb'].median():.2f}")
        if "psi_per_kb" in sub.columns and sub["psi_per_kb"].notna().sum() > 0:
            parts.append(f"psi/kb={sub['psi_per_kb'].median():.2f}")
        p(f"  {cat} L1: {', '.join(parts)}")
    # RL-matched MWU
    if polya_col:
        g1 = rl_matched[rl_matched["overlap_cat"] == "1"][polya_col].dropna()
        g2 = rl_matched[rl_matched["n_l1_overlaps"] >= 2][polya_col].dropna()
        if len(g1) > 0 and len(g2) > 0:
            _, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p(f"  RL-matched 1-L1 vs multi-L1 poly(A): MWU p={u_p:.2e}, delta={g2.median()-g1.median():.1f}nt")
    if "m6a_per_kb" in merged_df.columns:
        g1 = rl_matched[rl_matched["overlap_cat"] == "1"]["m6a_per_kb"].dropna()
        g2 = rl_matched[rl_matched["n_l1_overlaps"] >= 2]["m6a_per_kb"].dropna()
        if len(g1) > 0 and len(g2) > 0:
            _, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p(f"  RL-matched 1-L1 vs multi-L1 m6A/kb: MWU p={u_p:.2e}, delta={g2.median()-g1.median():.2f}")

    # Coverage fraction
    p("\n--- Fraction of alignment covered by L1 annotation ---")
    merged_df["frac_l1_covered"] = (merged_df["total_overlap_bp"] / merged_df["read_length"]).clip(upper=1.0)
    for cat in cat_order:
        sub = merged_df[merged_df["overlap_cat"] == cat]
        if len(sub) == 0:
            continue
        p(f"  {cat} L1: coverage fraction median={sub['frac_l1_covered'].median():.3f}, "
          f"mean={sub['frac_l1_covered'].mean():.3f}")

    # Per-cell-line
    p("\n--- Per-cell-line multi-L1 fraction ---")
    if "cell_line" in merged_df.columns:
        cl_stats = merged_df.groupby("cell_line").agg(
            n_reads=("read_id", "count"),
            multi_pct=("n_l1_overlaps", lambda x: (x >= 2).mean() * 100),
            median_overlaps=("n_l1_overlaps", "median"),
        ).sort_values("multi_pct", ascending=False)
        for cl, row in cl_stats.iterrows():
            p(f"  {cl:>10}: n={row['n_reads']:>5,.0f}, "
              f"multi-L1={row['multi_pct']:.1f}%, "
              f"median_overlaps={row['median_overlaps']:.0f}")

    # Subfamily diversity
    p("\n--- Multi-L1 reads: annotation subfamily diversity ---")
    multi = merged_df[merged_df["n_l1_overlaps"] >= 2].copy()
    if len(multi) > 0:
        n_single_sf = 0
        n_multi_sf = 0
        for sfs_str in multi["annot_subfamilies"]:
            sfs = set(str(sfs_str).split(";"))
            if len(sfs) == 1:
                n_single_sf += 1
            else:
                n_multi_sf += 1
        p(f"  Same subfamily across all overlapping L1: {n_single_sf:,} ({n_single_sf/len(multi)*100:.1f}%)")
        p(f"  Mixed subfamilies: {n_multi_sf:,} ({n_multi_sf/len(multi)*100:.1f}%)")

    # Top multi-L1 reads
    p("\n--- Top 10 reads by number of L1 overlaps ---")
    top10 = merged_df.nlargest(10, "n_l1_overlaps")
    for _, row in top10.iterrows():
        polya_str = f", polyA={row[polya_col]:.0f}" if polya_col and pd.notna(row.get(polya_col)) else ""
        m6a_str = ""
        if "m6a_per_kb" in merged_df.columns and pd.notna(row.get("m6a_per_kb")):
            m6a_str = f", m6A/kb={row['m6a_per_kb']:.1f}"
        p(f"  {row['read_id'][:24]}... : {int(row['n_l1_overlaps'])} L1 elements, "
          f"RL={row['read_length']:.0f}bp{polya_str}{m6a_str}")
        p(f"    {row.get('chr','?')}:{row.get('start','?')}-{row.get('end','?')}")
        p(f"    Subfamilies: {row.get('annot_subfamilies','?')}")

    # --- Fragmented L1 analysis: reads with multiple SAME-NAME annotations ---
    p("\n--- Fragmented vs truly clustered L1 ---")
    p("  (Same annot_name = fragmented single L1; different = adjacent distinct elements)")
    multi = merged_df[merged_df["n_l1_overlaps"] >= 2].copy()
    if len(multi) > 0:
        # n_l1_overlaps counts unique annot_names, so multi means >= 2 distinct elements
        p(f"  All multi-L1 reads have >= 2 DISTINCT L1 element names")
        # Check how many have the same subfamily
        same_sf = multi["annot_subfamilies"].apply(lambda x: len(set(str(x).split(";"))) == 1).sum()
        p(f"  Same subfamily: {same_sf:,} ({same_sf/len(multi)*100:.1f}%)")
        p(f"  Different subfamilies: {len(multi)-same_sf:,} ({(len(multi)-same_sf)/len(multi)*100:.1f}%)")

    # Interpretation
    p("\n" + "=" * 80)
    p("INTERPRETATION")
    p("=" * 80)
    single_pct = cat_counts.get("1", 0) / total * 100
    multi_pct = len(merged_df[merged_df["n_l1_overlaps"] >= 2]) / total * 100
    p(f"\n{single_pct:.1f}% of L1 reads overlap exactly one L1 annotation element.")
    p(f"{multi_pct:.1f}% of reads span regions with 2+ distinct L1 annotations.")
    p(f"\nThis is expected for ancient L1 elements, which are often fragmented in the")
    p(f"genome: a single original insertion can appear as 2-3+ separate annotation")
    p(f"entries due to insertions of other TEs, deletions, or inversions over time.")
    p(f"A long DRS read spanning such a region will intersect multiple L1 annotations.")

    if "m6a_per_kb" in merged_df.columns:
        g1_m = merged_df[merged_df["overlap_cat"] == "1"]["m6a_per_kb"].dropna().median()
        g2_m = merged_df[merged_df["n_l1_overlaps"] >= 2]["m6a_per_kb"].dropna().median()
        if g1_m > 0:
            p(f"\nm6A/kb: single-L1={g1_m:.2f} vs multi-L1={g2_m:.2f} (ratio={g2_m/g1_m:.2f}x)")

    no_overlap_n = (merged_df["n_l1_overlaps"] == 0).sum()
    if no_overlap_n > 0:
        p(f"\n{no_overlap_n:,} reads had no L1 annotation overlap (may span L1 flanking regions).")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


def main():
    print("=" * 80)
    print("Loading L1 summary data (base groups only)...")
    l1_df = load_l1_summaries()

    print("\nLoading Part3 cache (m6A/psi per read)...")
    p3_df = load_part3_cache()

    # Check if intersect already done
    overlap_path = os.path.join(OUTDIR, "per_read_l1_overlaps.tsv")
    intersect_path = os.path.join(OUTDIR, "reads_vs_l1_annotation.tsv")
    if os.path.exists(overlap_path) and os.path.getsize(overlap_path) > 100:
        print(f"\nLoading cached per-read overlaps from {overlap_path}")
        per_read_overlaps = pd.read_csv(overlap_path, sep="\t")
    else:
        print("\nRunning bedtools intersect: reads vs L1 annotation...")
        per_read_overlaps, _ = run_bedtools_intersect(l1_df)
        per_read_overlaps.to_csv(overlap_path, sep="\t", index=False)
        print(f"Saved per-read overlaps: {overlap_path}")

    # Merge
    print("\nMerging overlap counts with L1 summary...")
    merge_cols = ["read_id", "n_l1_overlaps", "total_overlap_bp", "max_overlap_bp",
                  "best_annot_name", "best_annot_subfamily", "annot_names", "annot_subfamilies"]
    merge_cols = [c for c in merge_cols if c in per_read_overlaps.columns]
    merged = l1_df.merge(per_read_overlaps[merge_cols], on="read_id", how="left")

    no_overlap = merged["n_l1_overlaps"].isna().sum()
    if no_overlap > 0:
        print(f"  [WARN] {no_overlap:,} reads had no L1 overlap -- setting to 0")
        merged["n_l1_overlaps"] = merged["n_l1_overlaps"].fillna(0).astype(int)
        merged["total_overlap_bp"] = merged["total_overlap_bp"].fillna(0)
        merged["max_overlap_bp"] = merged["max_overlap_bp"].fillna(0)

    if len(p3_df) > 0:
        p3_df["m6a_per_kb"] = p3_df["m6a_sites_high"] / (p3_df["read_length"] / 1000)
        p3_df["psi_per_kb"] = p3_df["psi_sites_high"] / (p3_df["read_length"] / 1000)
        merged = merged.merge(
            p3_df[["read_id", "m6a_sites_high", "psi_sites_high", "m6a_per_kb", "psi_per_kb"]],
            on="read_id", how="left"
        )
        print(f"  Merged Part3 data: {merged['m6a_per_kb'].notna().sum():,} reads with m6A data")

    # Save merged
    merged_path = os.path.join(OUTDIR, "l1_clustering_merged.tsv")
    save_cols = ["read_id", "chr", "start", "end", "read_length", "gene_id",
                 "read_strand", "group", "cell_line", "n_l1_overlaps",
                 "total_overlap_bp", "max_overlap_bp", "best_annot_subfamily",
                 "annot_subfamilies", "polya_length",
                 "m6a_sites_high", "psi_sites_high", "m6a_per_kb", "psi_per_kb"]
    save_cols = [c for c in save_cols if c in merged.columns]
    merged[save_cols].to_csv(merged_path, sep="\t", index=False)
    print(f"Saved merged data: {merged_path}")

    summary_path = os.path.join(OUTDIR, "l1_clustering_summary.txt")
    analyze_clustering(merged, summary_path)


if __name__ == "__main__":
    main()
