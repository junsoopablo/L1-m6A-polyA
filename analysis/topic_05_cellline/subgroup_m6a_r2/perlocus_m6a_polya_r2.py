#!/usr/bin/env python3
"""
Per-locus m6A-poly(A) correlation analysis.

Hypothesis: aggregating per-read m6A and poly(A) measurements to per-locus
medians should substantially increase R² by averaging out single-molecule
measurement noise, since reads from the same L1 locus share identical
sequence context.

Per-read baseline: R²=1.6% (all L1), 7.3% (stressed ancient regulatory)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PATHS
# ============================================================
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = os.path.join(BASE, "results_group")
ANALYSIS = os.path.join(BASE, "analysis/01_exploration")
PART3_CACHE = os.path.join(ANALYSIS, "topic_05_cellline/part3_l1_per_read_cache")
CHROMHMM = os.path.join(ANALYSIS, "topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv")
OUTDIR = os.path.join(ANALYSIS, "topic_05_cellline/subgroup_m6a_r2")

HELA_GROUPS = ["HeLa_1", "HeLa_2", "HeLa_3"]
ARS_GROUPS = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
ALL_GROUPS = HELA_GROUPS + ARS_GROUPS

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# ============================================================
# STEP 1: Load and merge data
# ============================================================
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

# Load L1 summaries for all HeLa groups
summary_dfs = []
for grp in ALL_GROUPS:
    fpath = os.path.join(RESULTS, grp, "g_summary", f"{grp}_L1_summary.tsv")
    df = pd.read_csv(fpath, sep="\t")
    df["group"] = grp
    df["condition"] = "stressed" if "Ars" in grp else "unstressed"
    summary_dfs.append(df)

summary = pd.concat(summary_dfs, ignore_index=True)
print(f"  L1 summary loaded: {len(summary)} reads from {len(ALL_GROUPS)} groups")

# Load Part3 cache for m6A site counts
part3_dfs = []
for grp in ALL_GROUPS:
    fpath = os.path.join(PART3_CACHE, f"{grp}_l1_per_read.tsv")
    df = pd.read_csv(fpath, sep="\t")
    df["group"] = grp
    part3_dfs.append(df)

part3 = pd.concat(part3_dfs, ignore_index=True)
print(f"  Part3 cache loaded: {len(part3)} reads")

# Merge: summary + part3 on read_id
# Part3 has: read_id, read_length, m6a_sites_high, psi_sites_high
# Summary has: read_id, te_chr, te_start, te_end, transcript_id, gene_id,
#              te_strand, overlap_length, polya_length, qc_tag, class, etc.

merged = summary.merge(
    part3[["read_id", "read_length", "m6a_sites_high", "psi_sites_high"]],
    on="read_id",
    how="inner",
    suffixes=("_summary", "_part3"),
)
print(f"  After merge (summary+part3): {len(merged)} reads")

# Compute m6a_per_kb
# Use read_length from Part3 cache (aligned length)
rl_col = "read_length_part3" if "read_length_part3" in merged.columns else "read_length"
merged["m6a_per_kb"] = merged["m6a_sites_high"] / (merged[rl_col] / 1000.0)

# Filter PASS only
merged = merged[merged["qc_tag"] == "PASS"].copy()
print(f"  After PASS filter: {len(merged)} reads")

# Compute overlap_fraction = overlap_length / (te_end - te_start)
merged["te_length"] = merged["te_end"] - merged["te_start"]
merged["overlap_fraction"] = merged["overlap_length"] / merged["te_length"].clip(lower=1)

# Classify Young/Ancient
merged["l1_age"] = merged["gene_id"].apply(
    lambda x: "young" if x in YOUNG_SUBFAMILIES else "ancient"
)

# Define locus_id from TE coordinates + strand
merged["locus_id"] = (
    merged["te_chr"].astype(str) + ":"
    + merged["te_start"].astype(str) + "-"
    + merged["te_end"].astype(str) + ":"
    + merged["te_strand"].astype(str)
)

# Also keep transcript_id as alternate locus identifier
print(f"  Unique loci (te coord-based): {merged['locus_id'].nunique()}")
print(f"  Unique loci (transcript_id): {merged['transcript_id'].nunique()}")

# Load ChromHMM annotations
chromhmm = pd.read_csv(CHROMHMM, sep="\t")
chromhmm_hela = chromhmm[
    chromhmm["group"].str.startswith("HeLa")
][["read_id", "chromhmm_state", "chromhmm_group"]].drop_duplicates("read_id")
print(f"  ChromHMM HeLa reads: {len(chromhmm_hela)}")

# Merge ChromHMM
merged = merged.merge(chromhmm_hela, on="read_id", how="left")
merged["chromhmm_group"] = merged["chromhmm_group"].fillna("Unknown")

# Regulatory = Enhancer or Promoter
merged["is_regulatory"] = merged["chromhmm_group"].isin(["Enhancer", "Promoter"])

print(f"\n  Final merged dataset: {len(merged)} reads")
print(f"  Conditions: {merged['condition'].value_counts().to_dict()}")
print(f"  L1 age: {merged['l1_age'].value_counts().to_dict()}")
print(f"  Regulatory: {merged['is_regulatory'].sum()}")
print(f"  ChromHMM groups: {merged['chromhmm_group'].value_counts().to_dict()}")


# ============================================================
# STEP 2: Helper functions
# ============================================================

def run_ols(x, y, label="", weights=None):
    """OLS or WLS regression. Returns dict of results."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.array(x[mask]), np.array(y[mask])
    n = len(x)
    if n < 5:
        return {"label": label, "N": n, "note": "too few observations"}

    X = sm.add_constant(x)

    if weights is not None:
        w = np.array(weights[mask])
        model = sm.WLS(y, X, weights=w).fit()
    else:
        model = sm.OLS(y, X).fit()

    # Correlations
    sp_rho, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)

    return {
        "label": label,
        "N": n,
        "spearman_rho": sp_rho,
        "spearman_p": sp_p,
        "pearson_r": pe_r,
        "pearson_p": pe_p,
        "ols_R2": model.rsquared,
        "ols_beta": model.params[1],
        "ols_beta_p": model.pvalues[1],
        "ols_intercept": model.params[0],
        "ols_type": "WLS" if weights is not None else "OLS",
    }


def run_logistic(x, y_binary, label=""):
    """Logistic regression for decay zone analysis."""
    mask = np.isfinite(x) & np.isfinite(y_binary)
    x, y = np.array(x[mask]), np.array(y_binary[mask])
    n = len(x)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n < 10 or n_pos < 5 or n_neg < 5:
        return {"label": label, "N": n, "n_decay": n_pos, "note": "too few events"}

    X = sm.add_constant(x)
    try:
        logit = sm.Logit(y, X).fit(disp=0)
        # OR and CI
        beta = logit.params[1]
        se = logit.bse[1]
        odds_ratio = np.exp(beta)
        ci_low = np.exp(beta - 1.96 * se)
        ci_high = np.exp(beta + 1.96 * se)

        # McFadden pseudo-R²
        ll_full = logit.llf
        ll_null = sm.Logit(y, np.ones_like(y)).fit(disp=0).llf
        mcfadden_r2 = 1 - ll_full / ll_null

        # Nagelkerke pseudo-R²
        n_obs = len(y)
        cox_snell = 1 - np.exp((2 / n_obs) * (ll_null - ll_full))
        max_cox_snell = 1 - np.exp((2 / n_obs) * ll_null)
        nagelkerke_r2 = cox_snell / max_cox_snell if max_cox_snell != 0 else np.nan

        return {
            "label": label,
            "N": n,
            "n_decay": n_pos,
            "pct_decay": 100.0 * n_pos / n,
            "OR": odds_ratio,
            "OR_CI_low": ci_low,
            "OR_CI_high": ci_high,
            "beta": beta,
            "beta_p": logit.pvalues[1],
            "McFadden_R2": mcfadden_r2,
            "Nagelkerke_R2": nagelkerke_r2,
        }
    except Exception as e:
        return {"label": label, "N": n, "n_decay": n_pos, "note": str(e)}


def aggregate_per_locus(df, min_reads=3):
    """Aggregate per-read data to per-locus medians."""
    agg = df.groupby("locus_id").agg(
        n_reads=("read_id", "count"),
        median_m6a_per_kb=("m6a_per_kb", "median"),
        median_polya=("polya_length", "median"),
        mean_m6a_per_kb=("m6a_per_kb", "mean"),
        mean_polya=("polya_length", "mean"),
        mean_overlap_fraction=("overlap_fraction", "mean"),
        l1_age=("l1_age", "first"),
        gene_id=("gene_id", "first"),
        is_regulatory=("is_regulatory", "first"),
        chromhmm_group=("chromhmm_group", "first"),
    ).reset_index()
    return agg[agg["n_reads"] >= min_reads].copy()


# ============================================================
# STEP 3 & 4: Regression analyses
# ============================================================
print("\n" + "=" * 70)
print("STEP 3-4: Per-locus and per-read regression analyses")
print("=" * 70)

results = []

# --- Subsets ---
stressed = merged[merged["condition"] == "stressed"]
unstressed = merged[merged["condition"] == "unstressed"]
stressed_ancient = stressed[stressed["l1_age"] == "ancient"]
stressed_young = stressed[stressed["l1_age"] == "young"]
unstressed_ancient = unstressed[unstressed["l1_age"] == "ancient"]

# -------------------------------------------------------
# (a) All L1 loci, HeLa+Ars combined, >=3 reads/locus
# -------------------------------------------------------
for min_n, tag in [(3, "a"), (5, "b")]:
    locus_df = aggregate_per_locus(merged, min_reads=min_n)
    label = f"({tag}) Per-locus: All L1, HeLa+Ars, ≥{min_n} reads"
    r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"], label)
    results.append(r)

    # WLS version
    rw = run_ols(
        locus_df["median_m6a_per_kb"], locus_df["median_polya"],
        f"({tag}w) WLS: All L1, HeLa+Ars, ≥{min_n} reads",
        weights=locus_df["n_reads"],
    )
    results.append(rw)

# -------------------------------------------------------
# (c) Stressed only, all L1, >=3 reads/locus
# -------------------------------------------------------
locus_df = aggregate_per_locus(stressed, min_reads=3)
r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
            "(c) Per-locus: Stressed, All L1, ≥3 reads")
results.append(r)
rw = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
             "(cw) WLS: Stressed, All L1, ≥3 reads",
             weights=locus_df["n_reads"])
results.append(rw)

# -------------------------------------------------------
# (d-e) Stressed + Ancient, >=3 and >=5
# -------------------------------------------------------
for min_n, tag in [(3, "d"), (5, "e")]:
    locus_df = aggregate_per_locus(stressed_ancient, min_reads=min_n)
    r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
                f"({tag}) Per-locus: Stressed+Ancient, ≥{min_n} reads")
    results.append(r)
    rw = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
                 f"({tag}w) WLS: Stressed+Ancient, ≥{min_n} reads",
                 weights=locus_df["n_reads"])
    results.append(rw)

# -------------------------------------------------------
# (f) Stressed + Ancient + high overlap, >=3
# -------------------------------------------------------
stressed_ancient_hiovl = stressed_ancient[stressed_ancient["overlap_fraction"] > 0.5]
locus_df = aggregate_per_locus(stressed_ancient_hiovl, min_reads=3)
r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
            "(f) Per-locus: Stressed+Ancient+Overlap>0.5, ≥3 reads")
results.append(r)

# Also overlap >= 0.7
stressed_ancient_vhiovl = stressed_ancient[stressed_ancient["overlap_fraction"] >= 0.7]
locus_df_07 = aggregate_per_locus(stressed_ancient_vhiovl, min_reads=3)
r07 = run_ols(locus_df_07["median_m6a_per_kb"], locus_df_07["median_polya"],
              "(f2) Per-locus: Stressed+Ancient+Overlap≥0.7, ≥3 reads")
results.append(r07)

# -------------------------------------------------------
# (g-h) Stressed + Ancient + Regulatory
# -------------------------------------------------------
stressed_anc_reg = stressed_ancient[stressed_ancient["is_regulatory"]]
for min_n, tag in [(3, "g"), (2, "h")]:
    locus_df = aggregate_per_locus(stressed_anc_reg, min_reads=min_n)
    r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
                f"({tag}) Per-locus: Stressed+Ancient+Regulatory, ≥{min_n} reads")
    results.append(r)

# -------------------------------------------------------
# (i) Unstressed Ancient, >=3
# -------------------------------------------------------
locus_df = aggregate_per_locus(unstressed_ancient, min_reads=3)
r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
            "(i) Per-locus: Unstressed+Ancient, ≥3 reads")
results.append(r)

# -------------------------------------------------------
# (j) Stressed Young, >=3
# -------------------------------------------------------
locus_df = aggregate_per_locus(stressed_young, min_reads=3)
r = run_ols(locus_df["median_m6a_per_kb"], locus_df["median_polya"],
            "(j) Per-locus: Stressed+Young, ≥3 reads")
results.append(r)

# -------------------------------------------------------
# (k) Per-read baseline: All L1, HeLa+Ars
# -------------------------------------------------------
r = run_ols(merged["m6a_per_kb"], merged["polya_length"],
            "(k) Per-read: All L1, HeLa+Ars (baseline)")
results.append(r)

# -------------------------------------------------------
# (l) Per-read: Stressed Ancient Regulatory
# -------------------------------------------------------
r = run_ols(stressed_anc_reg["m6a_per_kb"], stressed_anc_reg["polya_length"],
            "(l) Per-read: Stressed+Ancient+Regulatory (baseline)")
results.append(r)

# Also do per-read for other subsets for comparison
r = run_ols(stressed["m6a_per_kb"], stressed["polya_length"],
            "(m) Per-read: Stressed, All L1")
results.append(r)

r = run_ols(stressed_ancient["m6a_per_kb"], stressed_ancient["polya_length"],
            "(n) Per-read: Stressed+Ancient")
results.append(r)

# ============================================================
# Print results table
# ============================================================
print("\n" + "=" * 70)
print("RESULTS: Per-locus vs Per-read OLS Regression")
print("=" * 70)

res_df = pd.DataFrame(results)
cols_order = ["label", "N", "spearman_rho", "spearman_p", "pearson_r", "pearson_p",
              "ols_R2", "ols_beta", "ols_beta_p", "ols_intercept", "ols_type", "note"]
cols_present = [c for c in cols_order if c in res_df.columns]
res_df = res_df[cols_present]

for _, row in res_df.iterrows():
    print(f"\n  {row['label']}")
    if "note" in row and pd.notna(row.get("note")):
        print(f"    NOTE: {row['note']}")
        continue
    print(f"    N = {row['N']}")
    print(f"    Spearman rho = {row['spearman_rho']:.4f}  (P = {row['spearman_p']:.2e})")
    print(f"    Pearson  r   = {row['pearson_r']:.4f}  (P = {row['pearson_p']:.2e})")
    print(f"    {row.get('ols_type', 'OLS')} R²  = {row['ols_R2']:.4f}  ({row['ols_R2']*100:.2f}%)")
    print(f"    Beta (slope) = {row['ols_beta']:.4f}  (P = {row['ols_beta_p']:.2e})")


# ============================================================
# STEP 5: Decay zone logistic regression
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 5a: Decay zone logistic regression (per-read, stressed)")
print("=" * 70)

logistic_results = []

for subset, label in [
    (stressed, "All stressed L1"),
    (stressed_ancient, "Stressed Ancient"),
    (stressed_anc_reg, "Stressed Ancient Regulatory"),
]:
    sub = subset.copy()
    sub["decay_zone"] = (sub["polya_length"] < 30).astype(int)
    lr = run_logistic(sub["m6a_per_kb"], sub["decay_zone"], label)
    logistic_results.append(lr)

log_df = pd.DataFrame(logistic_results)

for _, row in log_df.iterrows():
    print(f"\n  {row['label']}")
    if "note" in row and pd.notna(row.get("note")):
        print(f"    NOTE: {row['note']}")
        continue
    print(f"    N = {row['N']}, n_decay = {row['n_decay']} ({row['pct_decay']:.1f}%)")
    print(f"    OR = {row['OR']:.4f}  (95% CI: {row['OR_CI_low']:.4f} - {row['OR_CI_high']:.4f})")
    print(f"    Beta = {row['beta']:.4f}, P = {row['beta_p']:.2e}")
    print(f"    McFadden pseudo-R² = {row['McFadden_R2']:.4f}  ({row['McFadden_R2']*100:.2f}%)")
    print(f"    Nagelkerke pseudo-R² = {row['Nagelkerke_R2']:.4f}  ({row['Nagelkerke_R2']*100:.2f}%)")


# ============================================================
# STEP 5b: Overlap >= 0.7 per-read analysis (stressed)
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 5b: Overlap ≥ 0.7 per-read analysis (stressed)")
print("=" * 70)

stressed_hiovl = stressed[stressed["overlap_fraction"] >= 0.7]
r = run_ols(stressed_hiovl["m6a_per_kb"], stressed_hiovl["polya_length"],
            "Per-read: Stressed, overlap≥0.7")
print(f"\n  {r['label']}")
print(f"    N = {r['N']}")
print(f"    Spearman rho = {r['spearman_rho']:.4f}  (P = {r['spearman_p']:.2e})")
print(f"    OLS R² = {r['ols_R2']:.4f}  ({r['ols_R2']*100:.2f}%)")
print(f"    Beta = {r['ols_beta']:.4f}  (P = {r['ols_beta_p']:.2e})")

# Also stressed ancient overlap >= 0.7
stressed_anc_hiovl = stressed_ancient[stressed_ancient["overlap_fraction"] >= 0.7]
r2 = run_ols(stressed_anc_hiovl["m6a_per_kb"], stressed_anc_hiovl["polya_length"],
             "Per-read: Stressed+Ancient, overlap≥0.7")
print(f"\n  {r2['label']}")
print(f"    N = {r2['N']}")
print(f"    Spearman rho = {r2['spearman_rho']:.4f}  (P = {r2['spearman_p']:.2e})")
print(f"    OLS R² = {r2['ols_R2']:.4f}  ({r2['ols_R2']*100:.2f}%)")
print(f"    Beta = {r2['ols_beta']:.4f}  (P = {r2['ols_beta_p']:.2e})")


# ============================================================
# STEP 5c: Additional locus-level statistics
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 5c: Locus-level descriptive statistics")
print("=" * 70)

for subset, label, min_n in [
    (stressed_ancient, "Stressed Ancient", 3),
    (stressed_anc_reg, "Stressed Ancient Regulatory", 2),
    (merged, "All HeLa+Ars", 3),
]:
    ldf = aggregate_per_locus(subset, min_reads=min_n)
    print(f"\n  {label} (≥{min_n} reads/locus):")
    print(f"    Loci: {len(ldf)}")
    print(f"    Reads/locus: median={ldf['n_reads'].median():.0f}, "
          f"mean={ldf['n_reads'].mean():.1f}, max={ldf['n_reads'].max()}")
    print(f"    m6A/kb: median={ldf['median_m6a_per_kb'].median():.3f}, "
          f"IQR=[{ldf['median_m6a_per_kb'].quantile(0.25):.3f}, "
          f"{ldf['median_m6a_per_kb'].quantile(0.75):.3f}]")
    print(f"    poly(A): median={ldf['median_polya'].median():.1f}, "
          f"IQR=[{ldf['median_polya'].quantile(0.25):.1f}, "
          f"{ldf['median_polya'].quantile(0.75):.1f}]")


# ============================================================
# STEP 5d: Read-count threshold sensitivity for per-locus R²
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 5d: Sensitivity to minimum reads/locus threshold")
print("=" * 70)

print(f"\n  {'Min reads':>10} {'N loci':>8} {'Spear rho':>10} {'P':>12} {'OLS R²':>10} {'Beta':>8} {'Beta P':>12}")
print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*12}")

for min_n in [2, 3, 4, 5, 7, 10, 15, 20]:
    ldf = aggregate_per_locus(stressed_ancient, min_reads=min_n)
    if len(ldf) < 10:
        print(f"  {min_n:>10} {len(ldf):>8}   --- too few loci ---")
        continue
    r = run_ols(ldf["median_m6a_per_kb"], ldf["median_polya"], "")
    print(f"  {min_n:>10} {r['N']:>8} {r['spearman_rho']:>10.4f} {r['spearman_p']:>12.2e} "
          f"{r['ols_R2']:>10.4f} {r['ols_beta']:>8.3f} {r['ols_beta_p']:>12.2e}")

# Same for all L1 (HeLa+Ars)
print(f"\n  All L1 (HeLa+Ars combined):")
print(f"  {'Min reads':>10} {'N loci':>8} {'Spear rho':>10} {'P':>12} {'OLS R²':>10}")
print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

for min_n in [2, 3, 5, 10, 15, 20]:
    ldf = aggregate_per_locus(merged, min_reads=min_n)
    if len(ldf) < 10:
        print(f"  {min_n:>10} {len(ldf):>8}   --- too few loci ---")
        continue
    r = run_ols(ldf["median_m6a_per_kb"], ldf["median_polya"], "")
    print(f"  {min_n:>10} {r['N']:>8} {r['spearman_rho']:>10.4f} {r['spearman_p']:>12.2e} {r['ols_R2']:>10.4f}")


# ============================================================
# STEP 5e: Paired locus analysis (same locus, stress vs unstress)
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 5e: Paired locus analysis (same locus, stressed vs unstressed)")
print("=" * 70)

# For loci present in both conditions, compare median poly(A) shift
stressed_locus = aggregate_per_locus(stressed, min_reads=3)
unstressed_locus = aggregate_per_locus(unstressed, min_reads=3)

paired = stressed_locus.merge(
    unstressed_locus,
    on="locus_id",
    suffixes=("_ars", "_ctrl"),
)
paired["delta_polya"] = paired["median_polya_ars"] - paired["median_polya_ctrl"]
paired["delta_m6a"] = paired["median_m6a_per_kb_ars"] - paired["median_m6a_per_kb_ctrl"]

print(f"  Paired loci (≥3 reads in both conditions): {len(paired)}")
if len(paired) >= 10:
    print(f"  Δpoly(A) median = {paired['delta_polya'].median():.1f} nt")
    print(f"  Δpoly(A) mean = {paired['delta_polya'].mean():.1f} nt")
    stat, p = stats.wilcoxon(paired["delta_polya"])
    print(f"  Wilcoxon signed-rank: P = {p:.2e}")

    # Per-locus: Does m6A predict Δpoly(A)?
    r = run_ols(paired["median_m6a_per_kb_ars"], paired["delta_polya"],
                "Per-locus: m6A(stressed) vs Δpoly(A)")
    print(f"\n  m6A(stressed) → Δpoly(A):")
    print(f"    N = {r['N']}")
    print(f"    Spearman rho = {r['spearman_rho']:.4f}  (P = {r['spearman_p']:.2e})")
    print(f"    OLS R² = {r['ols_R2']:.4f}  ({r['ols_R2']*100:.2f}%)")
    print(f"    Beta = {r['ols_beta']:.4f}  (P = {r['ols_beta_p']:.2e})")

    # Mean m6A across conditions as predictor
    paired["mean_m6a_both"] = (paired["median_m6a_per_kb_ars"] + paired["median_m6a_per_kb_ctrl"]) / 2
    r2 = run_ols(paired["mean_m6a_both"], paired["delta_polya"],
                 "Per-locus: mean_m6A(both) vs Δpoly(A)")
    print(f"\n  mean_m6A(both conditions) → Δpoly(A):")
    print(f"    N = {r2['N']}")
    print(f"    Spearman rho = {r2['spearman_rho']:.4f}  (P = {r2['spearman_p']:.2e})")
    print(f"    OLS R² = {r2['ols_R2']:.4f}  ({r2['ols_R2']*100:.2f}%)")

    # Filter to Ancient paired loci
    paired_ancient = paired[paired["l1_age_ars"] == "ancient"]
    print(f"\n  Paired Ancient loci: {len(paired_ancient)}")
    if len(paired_ancient) >= 10:
        r3 = run_ols(paired_ancient["median_m6a_per_kb_ars"], paired_ancient["delta_polya"],
                     "Per-locus: Ancient, m6A(stressed) vs Δpoly(A)")
        print(f"    Spearman rho = {r3['spearman_rho']:.4f}  (P = {r3['spearman_p']:.2e})")
        print(f"    OLS R² = {r3['ols_R2']:.4f}  ({r3['ols_R2']*100:.2f}%)")
        print(f"    Beta = {r3['ols_beta']:.4f}  (P = {r3['ols_beta_p']:.2e})")
else:
    print("  Too few paired loci for analysis")


# ============================================================
# STEP 6: Save results
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 6: Saving results")
print("=" * 70)

outpath = os.path.join(OUTDIR, "perlocus_r2_summary.tsv")
res_df.to_csv(outpath, sep="\t", index=False)
print(f"  Saved: {outpath}")

# Also save logistic results
log_outpath = os.path.join(OUTDIR, "decay_zone_logistic_summary.tsv")
log_df.to_csv(log_outpath, sep="\t", index=False)
print(f"  Saved: {log_outpath}")

# Save paired locus results if available
if len(paired) >= 10:
    paired_outpath = os.path.join(OUTDIR, "paired_locus_delta.tsv")
    paired.to_csv(paired_outpath, sep="\t", index=False)
    print(f"  Saved: {paired_outpath}")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Per-read vs Per-locus R² comparison")
print("=" * 70)

summary_rows = []
for _, row in res_df.iterrows():
    if "note" in row and pd.notna(row.get("note")):
        continue
    is_perlocus = "Per-locus" in str(row.get("label", "")) or "WLS" in str(row.get("label", ""))
    level = "locus" if is_perlocus else "read"
    summary_rows.append({
        "Analysis": row["label"],
        "Level": level,
        "N": row["N"],
        "rho": row["spearman_rho"],
        "rho_P": row["spearman_p"],
        "R2_pct": row["ols_R2"] * 100,
        "Beta": row["ols_beta"],
        "Beta_P": row["ols_beta_p"],
    })

sum_df = pd.DataFrame(summary_rows)
print(f"\n{'Analysis':<60} {'Level':>6} {'N':>7} {'rho':>7} {'R²%':>7} {'Beta':>7} {'Beta P':>12}")
print("-" * 112)
for _, row in sum_df.iterrows():
    print(f"{row['Analysis']:<60} {row['Level']:>6} {row['N']:>7} "
          f"{row['rho']:>7.4f} {row['R2_pct']:>6.2f}% {row['Beta']:>7.3f} {row['Beta_P']:>12.2e}")

print("\n\nDone.")
