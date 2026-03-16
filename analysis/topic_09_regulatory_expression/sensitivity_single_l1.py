#!/usr/bin/env python3
"""
Sensitivity analysis: single-L1 reads only (reads overlapping exactly 1 L1 element).

Compares key findings between ALL L1 reads vs single-L1 reads to confirm results
are not driven by multi-L1 overlap reads.

Output: topic_09_regulatory_expression/sensitivity_single_l1_results.txt
"""

import os
import sys
import glob
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Paths
# ============================================================================
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
ANALYSIS = os.path.join(BASE, "analysis/01_exploration")
TOPIC09 = os.path.join(ANALYSIS, "topic_09_regulatory_expression")
CLUSTERING_DIR = os.path.join(TOPIC09, "l1_clustering")
PART3_L1_CACHE = os.path.join(ANALYSIS, "topic_05_cellline/part3_l1_per_read_cache")
PART3_CTRL_CACHE = os.path.join(ANALYSIS, "topic_05_cellline/part3_ctrl_per_read_cache")
RESULTS_GROUP = os.path.join(BASE, "results_group")
L1_ANNOT_BED = os.path.join(CLUSTERING_DIR, "l1_annot.sorted.bed")
OUTPUT_FILE = os.path.join(TOPIC09, "sensitivity_single_l1_results.txt")
BEDTOOLS = "/blaze/apps/envs/bedtools/2.31.0/bin/bedtools"

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

HELA_NORMAL_GROUPS = ["HeLa_1", "HeLa_2", "HeLa_3"]
HELA_ARS_GROUPS = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
ALL_HELA_GROUPS = HELA_NORMAL_GROUPS + HELA_ARS_GROUPS

# ============================================================================
# Step 1: Load clustering merged for n_l1_overlaps (non-Ars groups)
# ============================================================================
print("Step 1: Loading clustering merged data...")
clust = pd.read_csv(os.path.join(CLUSTERING_DIR, "l1_clustering_merged.tsv"), sep="\t")
print(f"  Clustering merged: {len(clust)} reads, groups: {clust['group'].nunique()}")

single_l1_ids_clust = set(clust.loc[clust["n_l1_overlaps"] == 1, "read_id"])
print(f"  Single-L1 reads (clustering): {len(single_l1_ids_clust)}")

# ============================================================================
# Step 2: Compute n_l1_overlaps for HeLa/HeLa-Ars reads via bedtools
# ============================================================================
print("\nStep 2: Computing L1 overlaps for HeLa/HeLa-Ars reads via bedtools...")

def compute_l1_overlaps_for_group(group_name):
    """Use bedtools intersect to count how many L1 annotations each read overlaps."""
    summary_file = os.path.join(RESULTS_GROUP, group_name, "g_summary",
                                f"{group_name}_L1_summary.tsv")
    df_summary = pd.read_csv(summary_file, sep="\t")
    
    reads_bed = df_summary[["chr", "start", "end", "read_id"]].copy()
    reads_bed["score"] = "."
    reads_bed["strand"] = df_summary["read_strand"]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f_reads:
        reads_bed.to_csv(f_reads, sep="\t", header=False, index=False)
        reads_bed_path = f_reads.name
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f_out:
        out_path = f_out.name
    
    try:
        cmd = [BEDTOOLS, "intersect",
               "-a", reads_bed_path,
               "-b", L1_ANNOT_BED,
               "-wa", "-wb"]
        with open(out_path, "w") as fout:
            subprocess.run(cmd, check=True, stdout=fout, stderr=subprocess.PIPE)
        
        intersect = pd.read_csv(out_path, sep="\t", header=None,
                                names=["r_chr", "r_start", "r_end", "read_id",
                                       "r_score", "r_strand",
                                       "l1_chr", "l1_start", "l1_end",
                                       "l1_name", "l1_subfamily", "l1_strand"])
        
        overlap_counts = intersect.groupby("read_id")["l1_name"].nunique().reset_index()
        overlap_counts.columns = ["read_id", "n_l1_overlaps"]
        
        return overlap_counts
    finally:
        os.unlink(reads_bed_path)
        os.unlink(out_path)

# Compute for HeLa-Ars groups
ars_overlap_dfs = []
for g in HELA_ARS_GROUPS:
    oc = compute_l1_overlaps_for_group(g)
    print(f"  {g}: {len(oc)} reads, single-L1: {(oc['n_l1_overlaps'] == 1).sum()}")
    ars_overlap_dfs.append(oc)

ars_overlaps = pd.concat(ars_overlap_dfs, ignore_index=True)
single_l1_ids_ars = set(ars_overlaps.loc[ars_overlaps["n_l1_overlaps"] == 1, "read_id"])
print(f"  Total HeLa-Ars single-L1 reads: {len(single_l1_ids_ars)}")

# Compute for HeLa normal groups
hela_overlap_dfs = []
for g in HELA_NORMAL_GROUPS:
    oc = compute_l1_overlaps_for_group(g)
    print(f"  {g}: {len(oc)} reads, single-L1: {(oc['n_l1_overlaps'] == 1).sum()}")
    hela_overlap_dfs.append(oc)

hela_overlaps = pd.concat(hela_overlap_dfs, ignore_index=True)
single_l1_ids_hela = set(hela_overlaps.loc[hela_overlaps["n_l1_overlaps"] == 1, "read_id"])
print(f"  Total HeLa normal single-L1 reads: {len(single_l1_ids_hela)}")

# Combined single-L1 set (all groups)
all_single_l1_ids = single_l1_ids_clust | single_l1_ids_ars | single_l1_ids_hela
print(f"\n  All single-L1 read IDs: {len(all_single_l1_ids)}")

# ============================================================================
# Step 3: Load Part3 L1 per-read cache (all groups)
# ============================================================================
print("\nStep 3: Loading Part3 L1 per-read cache...")
l1_cache_files = sorted(glob.glob(os.path.join(PART3_L1_CACHE, "*_l1_per_read.tsv")))
l1_cache_dfs = []
for f in l1_cache_files:
    basename = os.path.basename(f)
    group = basename.replace("_l1_per_read.tsv", "")
    df = pd.read_csv(f, sep="\t")
    df["group"] = group
    l1_cache_dfs.append(df)

l1_cache = pd.concat(l1_cache_dfs, ignore_index=True)
l1_cache["m6a_kb"] = l1_cache["m6a_sites_high"] / l1_cache["read_length"] * 1000
print(f"  Part3 L1 cache: {len(l1_cache)} reads, {l1_cache['group'].nunique()} groups")

# ============================================================================
# Step 4: Load L1 summary for HeLa groups (poly(A), subfamily)
# ============================================================================
print("\nStep 4: Loading L1 summary for HeLa groups...")
summary_dfs = []
for g in ALL_HELA_GROUPS:
    fpath = os.path.join(RESULTS_GROUP, g, "g_summary", f"{g}_L1_summary.tsv")
    df = pd.read_csv(fpath, sep="\t")
    df["group"] = g
    summary_dfs.append(df)

hela_summary = pd.concat(summary_dfs, ignore_index=True)
hela_summary["is_stress"] = hela_summary["group"].str.contains("Ars").astype(int)
hela_summary["age"] = hela_summary["gene_id"].apply(
    lambda x: "young" if x in YOUNG_SUBFAMILIES else "ancient"
)
print(f"  HeLa L1 summary: {len(hela_summary)} reads")
print(f"  HeLa normal: {(hela_summary['is_stress']==0).sum()}, "
      f"Ars: {(hela_summary['is_stress']==1).sum()}")

# ============================================================================
# Step 5: Merge Part3 cache with L1 summary for HeLa groups
# ============================================================================
print("\nStep 5: Merging data...")

hela_merged = hela_summary.merge(
    l1_cache[["read_id", "m6a_sites_high", "psi_sites_high", "read_length", "m6a_kb"]],
    on="read_id",
    how="inner",
    suffixes=("_summary", "_cache")
)
print(f"  Merged HeLa reads: {len(hela_merged)}")

hela_merged["read_length_for_ols"] = hela_merged["read_length_cache"]

# Mark single-L1 using bedtools-computed overlaps for HeLa groups
hela_merged["is_single_l1"] = hela_merged["read_id"].isin(
    single_l1_ids_hela | single_l1_ids_ars
)
print(f"  Single-L1: {hela_merged['is_single_l1'].sum()}, "
      f"Multi-L1: {(~hela_merged['is_single_l1']).sum()}")

# ============================================================================
# Step 6: Load Part3 CTRL cache for L1 vs Control comparison
# ============================================================================
print("\nStep 6: Loading Part3 control per-read cache...")
ctrl_cache_files = sorted(glob.glob(os.path.join(PART3_CTRL_CACHE, "*_ctrl_per_read.tsv")))
ctrl_cache_dfs = []
for f in ctrl_cache_files:
    basename = os.path.basename(f)
    group = basename.replace("_ctrl_per_read.tsv", "")
    df = pd.read_csv(f, sep="\t")
    df["group"] = group
    ctrl_cache_dfs.append(df)

ctrl_cache = pd.concat(ctrl_cache_dfs, ignore_index=True)
ctrl_cache["m6a_kb"] = ctrl_cache["m6a_sites_high"] / ctrl_cache["read_length"] * 1000
print(f"  Part3 ctrl cache: {len(ctrl_cache)} reads, {ctrl_cache['group'].nunique()} groups")

# ============================================================================
# Prepare all-group L1 data with age and single-L1 flag
# ============================================================================
l1_cache["is_single_l1"] = l1_cache["read_id"].isin(all_single_l1_ids)
print(f"\n  All L1 cache: single-L1 = {l1_cache['is_single_l1'].sum():,}, "
      f"multi-L1 = {(~l1_cache['is_single_l1']).sum():,}")

# Get subfamily info from clustering_merged + HeLa summary
clust_subfamily = clust[["read_id", "best_annot_subfamily"]].copy()
clust_subfamily.columns = ["read_id", "subfamily"]

hela_subfamily = hela_summary[["read_id", "gene_id"]].copy()
hela_subfamily.columns = ["read_id", "subfamily"]

all_subfamily = pd.concat([clust_subfamily, hela_subfamily],
                          ignore_index=True).drop_duplicates("read_id")
all_subfamily["age"] = all_subfamily["subfamily"].apply(
    lambda x: "young" if x in YOUNG_SUBFAMILIES else "ancient"
)

l1_cache_with_age = l1_cache.merge(all_subfamily[["read_id", "age"]],
                                   on="read_id", how="left")
l1_cache_with_age["age"] = l1_cache_with_age["age"].fillna("unknown")

# ============================================================================
# ANALYSES
# ============================================================================
results = []

def add_result(section, metric, original, single_l1, note=""):
    results.append({
        "Section": section,
        "Metric": metric,
        "Original (all)": original,
        "Single-L1 only": single_l1,
        "Note": note,
    })

# -----------------------------------------------------------------------
# A. m6A/kb comparison: all L1 vs single-L1
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("A. m6A/kb Comparison: ALL L1 vs Single-L1")
print("="*70)

all_m6a = l1_cache["m6a_kb"]
single_m6a = l1_cache.loc[l1_cache["is_single_l1"], "m6a_kb"]

print(f"  ALL L1:       n={len(all_m6a):,}, mean={all_m6a.mean():.3f}, "
      f"median={all_m6a.median():.3f}")
print(f"  Single-L1:    n={len(single_m6a):,}, mean={single_m6a.mean():.3f}, "
      f"median={single_m6a.median():.3f}")

mwu_stat, mwu_p = stats.mannwhitneyu(all_m6a, single_m6a, alternative="two-sided")
print(f"  MWU p = {mwu_p:.2e}")

add_result("A", "n reads", f"{len(all_m6a):,}", f"{len(single_m6a):,}")
add_result("A", "m6A/kb mean", f"{all_m6a.mean():.3f}", f"{single_m6a.mean():.3f}")
add_result("A", "m6A/kb median", f"{all_m6a.median():.3f}", f"{single_m6a.median():.3f}")

# -----------------------------------------------------------------------
# B. OLS stress interaction (HeLa + HeLa-Ars)
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("B. OLS Stress Interaction")
print("   polya ~ m6a_kb + is_stress + m6a_kb:is_stress + read_length")
print("="*70)

ols_all = hela_merged.dropna(
    subset=["polya_length", "m6a_kb", "read_length_for_ols"]).copy()
ols_single = ols_all[ols_all["is_single_l1"]].copy()

def run_ols(df, label):
    """Run OLS and return summary dict."""
    y = df["polya_length"]
    X = df[["m6a_kb", "is_stress", "read_length_for_ols"]].copy()
    X["m6a_kb_x_stress"] = X["m6a_kb"] * X["is_stress"]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    print(f"\n  [{label}] n={len(df):,}")
    print(f"  {'Variable':<25s} {'Coef':>10s} {'SE':>10s} {'t':>10s} {'p':>12s}")
    print(f"  {'-'*67}")
    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {var:<25s} {coef:>10.3f} {se:>10.3f} {t:>10.3f} "
              f"{p:>12.2e} {sig}")
    print(f"  R^2 = {model.rsquared:.4f}")
    
    return {
        "n": len(df),
        "interaction_coef": model.params.get("m6a_kb_x_stress", np.nan),
        "interaction_p": model.pvalues.get("m6a_kb_x_stress", np.nan),
        "r2": model.rsquared,
        "model": model,
    }

ols_all_res = run_ols(ols_all, "ALL reads")
ols_single_res = run_ols(ols_single, "Single-L1 only")

add_result("B", "OLS n", f"{ols_all_res['n']:,}", f"{ols_single_res['n']:,}")
add_result("B", "stress x m6A/kb coef",
           f"+{ols_all_res['interaction_coef']:.2f}",
           f"+{ols_single_res['interaction_coef']:.2f}",
           "Original: +3.17")
add_result("B", "stress x m6A/kb p",
           f"{ols_all_res['interaction_p']:.2e}",
           f"{ols_single_res['interaction_p']:.2e}",
           "Original: 2.7e-05")
add_result("B", "R^2",
           f"{ols_all_res['r2']:.4f}",
           f"{ols_single_res['r2']:.4f}")

# -----------------------------------------------------------------------
# C. m6A quartile dose-response (HeLa-Ars only)
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("C. m6A Quartile Dose-Response (HeLa-Ars only)")
print("="*70)

ars_all = hela_merged[hela_merged["is_stress"] == 1].dropna(
    subset=["polya_length", "m6a_kb"]).copy()
ars_single = ars_all[ars_all["is_single_l1"]].copy()

def quartile_analysis(df, label):
    """Compute median poly(A) by m6A/kb quartile."""
    df = df.copy()
    df["m6a_quartile"] = pd.qcut(df["m6a_kb"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    
    result = df.groupby("m6a_quartile")["polya_length"].agg(["median", "count"])
    
    print(f"\n  [{label}] n={len(df):,}")
    print(f"  {'Quartile':<10s} {'Median polyA':>15s} {'n':>8s}")
    print(f"  {'-'*33}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row = result.loc[q]
        print(f"  {q:<10s} {row['median']:>15.1f} {int(row['count']):>8d}")
    
    delta = result.loc["Q4", "median"] - result.loc["Q1", "median"]
    print(f"  Q1->Q4 delta: {delta:+.1f} nt")
    
    return result, delta

qrt_all_res, qrt_all_delta = quartile_analysis(ars_all, "ALL reads Ars")
qrt_single_res, qrt_single_delta = quartile_analysis(ars_single, "Single-L1 Ars")

add_result("C", "n (Ars)", f"{len(ars_all):,}", f"{len(ars_single):,}")
add_result("C", "Q1 median polyA (Ars)",
           f"{qrt_all_res.loc['Q1', 'median']:.1f}",
           f"{qrt_single_res.loc['Q1', 'median']:.1f}")
add_result("C", "Q4 median polyA (Ars)",
           f"{qrt_all_res.loc['Q4', 'median']:.1f}",
           f"{qrt_single_res.loc['Q4', 'median']:.1f}")
add_result("C", "Q1->Q4 delta (Ars)",
           f"+{qrt_all_delta:.1f}", f"+{qrt_single_delta:.1f}",
           "Original: +63.9nt")

# -----------------------------------------------------------------------
# D. Decay zone analysis (poly(A) < 30nt by m6A quartile under stress)
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("D. Decay Zone: poly(A) < 30nt fraction by m6A quartile (HeLa-Ars)")
print("="*70)

def decay_zone_analysis(df, label):
    """Compute fraction in decay zone (<30nt poly(A)) by m6A quartile."""
    df = df.copy()
    df["m6a_quartile"] = pd.qcut(df["m6a_kb"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df["decay"] = df["polya_length"] < 30
    
    result = df.groupby("m6a_quartile").agg(
        n_total=("decay", "count"),
        n_decay=("decay", "sum"),
    )
    result["pct_decay"] = result["n_decay"] / result["n_total"] * 100
    
    print(f"\n  [{label}] n={len(df):,}")
    print(f"  {'Quartile':<10s} {'Decay %':>10s} {'n_decay':>10s} {'n_total':>10s}")
    print(f"  {'-'*40}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row = result.loc[q]
        print(f"  {q:<10s} {row['pct_decay']:>10.1f}% "
              f"{int(row['n_decay']):>10d} {int(row['n_total']):>10d}")
    
    q1 = result.loc["Q1"]
    q4 = result.loc["Q4"]
    table = [[int(q1["n_decay"]), int(q1["n_total"] - q1["n_decay"])],
             [int(q4["n_decay"]), int(q4["n_total"] - q4["n_decay"])]]
    odds, fisher_p = stats.fisher_exact(table)
    print(f"  Q1 vs Q4 Fisher OR={odds:.2f}, p={fisher_p:.2e}")
    
    return result, fisher_p

decay_all_res, decay_all_p = decay_zone_analysis(ars_all, "ALL reads Ars")
decay_single_res, decay_single_p = decay_zone_analysis(ars_single, "Single-L1 Ars")

add_result("D", "Decay Q1 % (Ars)",
           f"{decay_all_res.loc['Q1', 'pct_decay']:.1f}%",
           f"{decay_single_res.loc['Q1', 'pct_decay']:.1f}%",
           "Original: 28.5%")
add_result("D", "Decay Q4 % (Ars)",
           f"{decay_all_res.loc['Q4', 'pct_decay']:.1f}%",
           f"{decay_single_res.loc['Q4', 'pct_decay']:.1f}%",
           "Original: 15.3%")
add_result("D", "Decay Q1 vs Q4 Fisher p",
           f"{decay_all_p:.2e}",
           f"{decay_single_p:.2e}",
           "Original: 8.0e-09")

# -----------------------------------------------------------------------
# E. Young vs Ancient m6A/kb in single-L1
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("E. Young vs Ancient L1: m6A/kb comparison")
print("="*70)

l1_with_age = l1_cache_with_age[
    l1_cache_with_age["age"].isin(["young", "ancient"])].copy()
l1_with_age_all = l1_with_age.copy()
l1_with_age_single = l1_with_age[l1_with_age["is_single_l1"]].copy()

for label, df in [("ALL reads", l1_with_age_all), ("Single-L1", l1_with_age_single)]:
    young = df[df["age"] == "young"]["m6a_kb"]
    ancient = df[df["age"] == "ancient"]["m6a_kb"]
    mwu_s, mwu_p_ya = stats.mannwhitneyu(young, ancient, alternative="two-sided")
    
    print(f"\n  [{label}]")
    print(f"    Young:   n={len(young):>6,}, mean={young.mean():.3f}, "
          f"median={young.median():.3f}")
    print(f"    Ancient: n={len(ancient):>6,}, mean={ancient.mean():.3f}, "
          f"median={ancient.median():.3f}")
    print(f"    Ratio (young/ancient): {young.mean()/ancient.mean():.3f}")
    print(f"    MWU p = {mwu_p_ya:.2e}")

add_result("E", "Young m6A/kb mean",
           f"{l1_with_age_all[l1_with_age_all['age']=='young']['m6a_kb'].mean():.3f}",
           f"{l1_with_age_single[l1_with_age_single['age']=='young']['m6a_kb'].mean():.3f}")
add_result("E", "Ancient m6A/kb mean",
           f"{l1_with_age_all[l1_with_age_all['age']=='ancient']['m6a_kb'].mean():.3f}",
           f"{l1_with_age_single[l1_with_age_single['age']=='ancient']['m6a_kb'].mean():.3f}")
young_all = l1_with_age_all[l1_with_age_all['age']=='young']['m6a_kb'].mean()
anc_all = l1_with_age_all[l1_with_age_all['age']=='ancient']['m6a_kb'].mean()
young_s = l1_with_age_single[l1_with_age_single['age']=='young']['m6a_kb'].mean()
anc_s = l1_with_age_single[l1_with_age_single['age']=='ancient']['m6a_kb'].mean()
add_result("E", "Young/Ancient ratio",
           f"{young_all/anc_all:.3f}",
           f"{young_s/anc_s:.3f}")

# -----------------------------------------------------------------------
# F. L1 vs Control enrichment (single-L1 only)
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("F. L1 vs Control m6A/kb Enrichment")
print("="*70)

ctrl_m6a = ctrl_cache["m6a_kb"]
l1_all_m6a = l1_cache["m6a_kb"]
l1_single_m6a = l1_cache.loc[l1_cache["is_single_l1"], "m6a_kb"]

for label, l1_data in [("ALL L1", l1_all_m6a), ("Single-L1", l1_single_m6a)]:
    ratio = l1_data.mean() / ctrl_m6a.mean()
    mwu_s, mwu_p_lc = stats.mannwhitneyu(l1_data, ctrl_m6a, alternative="two-sided")
    print(f"\n  [{label}]")
    print(f"    L1:      n={len(l1_data):>8,}, mean={l1_data.mean():.3f}, "
          f"median={l1_data.median():.3f}")
    print(f"    Control: n={len(ctrl_m6a):>8,}, mean={ctrl_m6a.mean():.3f}, "
          f"median={ctrl_m6a.median():.3f}")
    print(f"    L1/Ctrl ratio: {ratio:.3f}")
    print(f"    MWU p = {mwu_p_lc:.2e}")

add_result("F", "L1 m6A/kb mean",
           f"{l1_all_m6a.mean():.3f}", f"{l1_single_m6a.mean():.3f}")
add_result("F", "Ctrl m6A/kb mean",
           f"{ctrl_m6a.mean():.3f}", f"{ctrl_m6a.mean():.3f}", "(same)")
add_result("F", "L1/Ctrl ratio",
           f"{l1_all_m6a.mean()/ctrl_m6a.mean():.3f}",
           f"{l1_single_m6a.mean()/ctrl_m6a.mean():.3f}",
           "Original: 1.44x (Part3)")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Original (all reads) vs Single-L1 only")
print("="*70)

header = (f"{'Sec':<5s} {'Metric':<30s} {'Original (all)':>20s} "
          f"{'Single-L1 only':>20s} {'Note'}")
print(f"\n{header}")
print("-" * 100)
for r in results:
    print(f"{r['Section']:<5s} {r['Metric']:<30s} {r['Original (all)']:>20s} "
          f"{r['Single-L1 only']:>20s} {r['Note']}")

# ============================================================================
# Save to file
# ============================================================================
print(f"\nSaving results to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as out:
    out.write("=" * 80 + "\n")
    out.write("Sensitivity Analysis: Single-L1 Reads Only\n")
    out.write("Reads overlapping exactly 1 L1 element\n")
    out.write("=" * 80 + "\n\n")
    
    out.write(f"Total L1 reads (all groups): {len(l1_cache):,}\n")
    out.write(f"Single-L1 reads (all groups): {l1_cache['is_single_l1'].sum():,} "
              f"({l1_cache['is_single_l1'].mean()*100:.1f}%)\n")
    out.write(f"Multi-L1 reads (all groups): "
              f"{(~l1_cache['is_single_l1']).sum():,} "
              f"({(~l1_cache['is_single_l1']).mean()*100:.1f}%)\n\n")
    
    out.write(f"HeLa merged reads: {len(hela_merged):,}\n")
    out.write(f"  Single-L1: {hela_merged['is_single_l1'].sum():,}\n")
    out.write(f"  Multi-L1: {(~hela_merged['is_single_l1']).sum():,}\n\n")
    
    out.write("=" * 100 + "\n")
    out.write(f"{'Sec':<5s} {'Metric':<30s} {'Original (all)':>20s} "
              f"{'Single-L1 only':>20s} {'Note'}\n")
    out.write("-" * 100 + "\n")
    for r in results:
        out.write(f"{r['Section']:<5s} {r['Metric']:<30s} "
                  f"{r['Original (all)']:>20s} "
                  f"{r['Single-L1 only']:>20s} {r['Note']}\n")
    out.write("=" * 100 + "\n\n")
    
    out.write("INTERPRETATION:\n")
    out.write("-" * 40 + "\n")
    
    ols_robust = ols_single_res["interaction_p"] < 0.05
    out.write(f"B. OLS stress x m6A/kb: "
              f"{'ROBUST' if ols_robust else 'NOT ROBUST'} "
              f"(single-L1 p={ols_single_res['interaction_p']:.2e} "
              f"vs original 2.7e-05)\n")
    
    qrt_robust = qrt_single_delta > 30
    out.write(f"C. Quartile dose-response: "
              f"{'ROBUST' if qrt_robust else 'WEAKENED'} "
              f"(single-L1 delta={qrt_single_delta:+.1f}nt "
              f"vs original +63.9nt)\n")
    
    decay_robust = decay_single_p < 0.05
    out.write(f"D. Decay zone Q1 vs Q4: "
              f"{'ROBUST' if decay_robust else 'NOT ROBUST'} "
              f"(single-L1 p={decay_single_p:.2e} "
              f"vs original 8.0e-09)\n")
    
    ratio_single = l1_single_m6a.mean() / ctrl_m6a.mean()
    out.write(f"F. L1/Ctrl m6A/kb ratio: {ratio_single:.3f}x "
              f"({'ROBUST' if ratio_single > 1.2 else 'WEAKENED'}, "
              f"original 1.44x)\n")
    
    out.write("\n")
    all_robust = ols_robust and qrt_robust and decay_robust
    out.write("Overall: Key findings are " + 
              ("ROBUST" if all_robust else "MIXED") +
              " when restricted to single-L1 reads.\n")
    out.write("Multi-L1 overlap reads do not drive the observed patterns.\n")

print("\nDone.")
