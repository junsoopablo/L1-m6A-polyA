#!/usr/bin/env python3
"""
Check whether m6A/kb has a nonlinear relationship with read length in L1 reads.

1. Load Part3 L1 and Ctrl per-read caches.
2. Bin by read length and compute m6A/kb per bin.
3. Check if L1/Ctrl m6A/kb ratio varies across RL bins.
4. OLS: polya ~ m6a_kb + is_stress + m6a_kb:is_stress + RL + RL^2
   for HeLa / HeLa-Ars.

Output: topic_09_regulatory_expression/rl_nonlinearity_results.txt
"""

import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -- Paths -------------------------------------------------------------------
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
ANALYSIS = os.path.join(BASE, "analysis/01_exploration")
L1_CACHE = os.path.join(ANALYSIS, "topic_05_cellline/part3_l1_per_read_cache")
CTRL_CACHE = os.path.join(ANALYSIS, "topic_05_cellline/part3_ctrl_per_read_cache")
RESULTS_GROUP = os.path.join(BASE, "results_group")
OUTDIR = os.path.join(ANALYSIS, "topic_09_regulatory_expression")
OUTFILE = os.path.join(OUTDIR, "rl_nonlinearity_results.txt")

os.makedirs(OUTDIR, exist_ok=True)

# -- RL bins -----------------------------------------------------------------
RL_BINS = [0, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, np.inf]
RL_LABELS = ["0-300", "300-500", "500-750", "750-1K", "1K-1.5K",
             "1.5K-2K", "2K-3K", "3K-5K", "5K+"]

out_lines = []

def prt(line=""):
    out_lines.append(line)
    print(line)


# ============================================================================
# 1. Load Part3 caches
# ============================================================================
prt("=" * 90)
prt("SECTION 1: Load Part3 per-read caches")
prt("=" * 90)

def load_cache(cache_dir, pattern, label):
    files = sorted(glob.glob(os.path.join(cache_dir, pattern)))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        basename = os.path.basename(f)
        group = basename.replace("_l1_per_read.tsv", "").replace("_ctrl_per_read.tsv", "")
        df["group"] = group
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["m6a_kb"] = combined["m6a_sites_high"] / combined["read_length"] * 1000
    combined["psi_kb"] = combined["psi_sites_high"] / combined["read_length"] * 1000
    prt(f"  {label}: {len(files)} files, {len(combined):,} reads")
    return combined

l1_df = load_cache(L1_CACHE, "*_l1_per_read.tsv", "L1")
ctrl_df = load_cache(CTRL_CACHE, "*_ctrl_per_read.tsv", "Control")

prt(f"  L1 read_length: median={l1_df['read_length'].median():.0f}, "
    f"mean={l1_df['read_length'].mean():.0f}")
prt(f"  Ctrl read_length: median={ctrl_df['read_length'].median():.0f}, "
    f"mean={ctrl_df['read_length'].mean():.0f}")
prt()

# ============================================================================
# 2. Bin by read length -> m6A/kb per bin
# ============================================================================
prt("=" * 90)
prt("SECTION 2: m6A/kb by read-length bin (L1 vs Control)")
prt("=" * 90)

l1_df["rl_bin"] = pd.cut(l1_df["read_length"], bins=RL_BINS, labels=RL_LABELS, right=False)
ctrl_df["rl_bin"] = pd.cut(ctrl_df["read_length"], bins=RL_BINS, labels=RL_LABELS, right=False)

def bin_stats(df, label):
    rows = []
    for b in RL_LABELS:
        sub = df[df["rl_bin"] == b]
        n = len(sub)
        if n == 0:
            rows.append({"bin": b, "n": 0, "mean_m6a_kb": np.nan,
                          "median_m6a_kb": np.nan, "mean_rl": np.nan})
            continue
        rows.append({
            "bin": b,
            "n": n,
            "mean_m6a_kb": sub["m6a_kb"].mean(),
            "median_m6a_kb": sub["m6a_kb"].median(),
            "mean_rl": sub["read_length"].mean(),
        })
    return pd.DataFrame(rows)

l1_bins = bin_stats(l1_df, "L1")
ctrl_bins = bin_stats(ctrl_df, "Ctrl")

# Merge for side-by-side table
merged = l1_bins.merge(ctrl_bins, on="bin", suffixes=("_l1", "_ctrl"))
merged["ratio_mean"] = merged["mean_m6a_kb_l1"] / merged["mean_m6a_kb_ctrl"]
merged["ratio_median"] = merged["median_m6a_kb_l1"] / merged["median_m6a_kb_ctrl"]

header = f"{'RL bin':<10} {'n_L1':>8} {'n_Ctrl':>8} {'mean_L1':>9} {'mean_Ctrl':>10} {'ratio':>7} " \
         f"{'med_L1':>8} {'med_Ctrl':>9} {'med_ratio':>9}"
prt(header)
prt("-" * len(header))
for _, r in merged.iterrows():
    prt(f"{r['bin']:<10} {int(r['n_l1']):>8,} {int(r['n_ctrl']):>8,} "
        f"{r['mean_m6a_kb_l1']:>9.2f} {r['mean_m6a_kb_ctrl']:>10.2f} {r['ratio_mean']:>7.3f} "
        f"{r['median_m6a_kb_l1']:>8.2f} {r['median_m6a_kb_ctrl']:>9.2f} {r['ratio_median']:>9.3f}")

# Overall
prt(f"\n  Overall L1 m6A/kb: mean={l1_df['m6a_kb'].mean():.3f}, median={l1_df['m6a_kb'].median():.3f}")
prt(f"  Overall Ctrl m6A/kb: mean={ctrl_df['m6a_kb'].mean():.3f}, median={ctrl_df['m6a_kb'].median():.3f}")
prt(f"  Overall ratio (mean): {l1_df['m6a_kb'].mean() / ctrl_df['m6a_kb'].mean():.3f}")
prt()

# ============================================================================
# 3. Test if ratio varies across bins (heterogeneity)
# ============================================================================
prt("=" * 90)
prt("SECTION 3: Does L1/Ctrl m6A/kb ratio vary across RL bins?")
prt("=" * 90)

# Spearman correlation: bin midpoint vs ratio
bin_midpoints = [150, 400, 625, 875, 1250, 1750, 2500, 4000, 6500]
valid = merged.dropna(subset=["ratio_mean"])
if len(valid) >= 3:
    r_sp, p_sp = stats.spearmanr(bin_midpoints[:len(valid)], valid["ratio_mean"].values)
    prt(f"  Spearman(bin midpoint, L1/Ctrl mean ratio): r={r_sp:.3f}, p={p_sp:.4f}")
else:
    r_sp, p_sp = np.nan, np.nan
    prt("  Not enough bins for Spearman test")

# Linear regression: m6a_kb ~ read_length + is_l1 + read_length:is_l1
prt("\n  OLS: m6a_kb ~ read_length + is_l1 + read_length:is_l1")
combined = pd.concat([
    l1_df[["read_length", "m6a_kb"]].assign(is_l1=1),
    ctrl_df[["read_length", "m6a_kb"]].assign(is_l1=0),
], ignore_index=True)

# Subsample for speed
if len(combined) > 200_000:
    combined_sub = combined.sample(n=200_000, random_state=42)
else:
    combined_sub = combined

model1 = smf.ols("m6a_kb ~ read_length + is_l1 + read_length:is_l1", data=combined_sub).fit()
prt(f"    is_l1 coef = {model1.params['is_l1']:.4f} (p={model1.pvalues['is_l1']:.2e})")
prt(f"    read_length coef = {model1.params['read_length']:.6f} (p={model1.pvalues['read_length']:.2e})")
prt(f"    read_length:is_l1 coef = {model1.params['read_length:is_l1']:.6f} "
    f"(p={model1.pvalues['read_length:is_l1']:.2e})")
prt(f"    R^2 = {model1.rsquared:.4f}")

# Check quadratic
prt("\n  OLS with quadratic: m6a_kb ~ read_length + I(read_length**2) + is_l1 + read_length:is_l1")
model2 = smf.ols("m6a_kb ~ read_length + I(read_length**2) + is_l1 + read_length:is_l1",
                  data=combined_sub).fit()
prt(f"    read_length coef = {model2.params['read_length']:.6f} (p={model2.pvalues['read_length']:.2e})")
prt(f"    read_length^2 coef = {model2.params['I(read_length ** 2)']:.9f} "
    f"(p={model2.pvalues['I(read_length ** 2)']:.2e})")
prt(f"    is_l1 coef = {model2.params['is_l1']:.4f} (p={model2.pvalues['is_l1']:.2e})")
prt(f"    read_length:is_l1 coef = {model2.params['read_length:is_l1']:.6f} "
    f"(p={model2.pvalues['read_length:is_l1']:.2e})")
prt(f"    R^2 = {model2.rsquared:.4f}")
prt()

# ============================================================================
# 4. L1-only: m6A/kb ~ read_length nonlinearity
# ============================================================================
prt("=" * 90)
prt("SECTION 4: L1-only m6A/kb ~ read_length nonlinearity")
prt("=" * 90)

if len(l1_df) > 100_000:
    l1_sub = l1_df.sample(n=100_000, random_state=42)
else:
    l1_sub = l1_df

# Linear
m_lin = smf.ols("m6a_kb ~ read_length", data=l1_sub).fit()
prt(f"  Linear: m6a_kb ~ read_length")
prt(f"    read_length coef = {m_lin.params['read_length']:.6f} (p={m_lin.pvalues['read_length']:.2e})")
prt(f"    R^2 = {m_lin.rsquared:.6f}")

# Quadratic
m_quad = smf.ols("m6a_kb ~ read_length + I(read_length**2)", data=l1_sub).fit()
prt(f"  Quadratic: m6a_kb ~ read_length + read_length^2")
prt(f"    read_length coef = {m_quad.params['read_length']:.6f} (p={m_quad.pvalues['read_length']:.2e})")
prt(f"    read_length^2 coef = {m_quad.params['I(read_length ** 2)']:.9f} "
    f"(p={m_quad.pvalues['I(read_length ** 2)']:.2e})")
prt(f"    R^2 = {m_quad.rsquared:.6f}")

# Log
l1_sub_log = l1_sub[l1_sub["read_length"] > 0].copy()
l1_sub_log["log_rl"] = np.log(l1_sub_log["read_length"])
m_log = smf.ols("m6a_kb ~ log_rl", data=l1_sub_log).fit()
prt(f"  Log: m6a_kb ~ log(read_length)")
prt(f"    log_rl coef = {m_log.params['log_rl']:.4f} (p={m_log.pvalues['log_rl']:.2e})")
prt(f"    R^2 = {m_log.rsquared:.6f}")

# Spearman
r_sp_l1, p_sp_l1 = stats.spearmanr(l1_sub["read_length"], l1_sub["m6a_kb"])
prt(f"\n  Spearman(read_length, m6a_kb) for L1: r={r_sp_l1:.4f}, p={p_sp_l1:.2e}")

# Same for Ctrl
if len(ctrl_df) > 100_000:
    ctrl_sub = ctrl_df.sample(n=100_000, random_state=42)
else:
    ctrl_sub = ctrl_df

r_sp_c, p_sp_c = stats.spearmanr(ctrl_sub["read_length"], ctrl_sub["m6a_kb"])
prt(f"  Spearman(read_length, m6a_kb) for Ctrl: r={r_sp_c:.4f}, p={p_sp_c:.2e}")
prt()

# ============================================================================
# 5. OLS: polya ~ m6a_kb + is_stress + m6a_kb:is_stress + RL + RL^2 (HeLa)
# ============================================================================
prt("=" * 90)
prt("SECTION 5: OLS polya ~ m6a_kb + stress + m6a_kb:stress + RL + RL^2 (HeLa)")
prt("=" * 90)

# Load L1 summaries for HeLa groups
hela_groups = ["HeLa_1", "HeLa_2", "HeLa_3", "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
summary_dfs = []
for grp in hela_groups:
    fpath = os.path.join(RESULTS_GROUP, grp, "g_summary", f"{grp}_L1_summary.tsv")
    if not os.path.exists(fpath):
        prt(f"  WARNING: {fpath} not found, skipping")
        continue
    df = pd.read_csv(fpath, sep="\t")
    # Filter PASS only
    df = df[df["qc_tag"] == "PASS"].copy()
    df["group"] = grp
    summary_dfs.append(df)
    prt(f"  Loaded {grp}: {len(df):,} PASS reads")

summary = pd.concat(summary_dfs, ignore_index=True)
prt(f"  Total HeLa PASS L1: {len(summary):,}")

# Identify stress
summary["is_stress"] = summary["group"].str.contains("Ars").astype(int)
prt(f"  Normal: {(summary['is_stress']==0).sum():,}, Stress: {(summary['is_stress']==1).sum():,}")

# Load HeLa Part3 caches for m6A/kb
hela_cache_groups = ["HeLa_1", "HeLa_2", "HeLa_3", "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
cache_dfs = []
for grp in hela_cache_groups:
    fpath = os.path.join(L1_CACHE, f"{grp}_l1_per_read.tsv")
    if not os.path.exists(fpath):
        prt(f"  WARNING: cache {fpath} not found")
        continue
    df = pd.read_csv(fpath, sep="\t")
    df["group"] = grp
    cache_dfs.append(df)

cache_hela = pd.concat(cache_dfs, ignore_index=True)
cache_hela["m6a_kb"] = cache_hela["m6a_sites_high"] / cache_hela["read_length"] * 1000

prt(f"  HeLa Part3 cache: {len(cache_hela):,} reads")

# Merge on read_id
merged_hela = summary.merge(
    cache_hela[["read_id", "m6a_kb", "m6a_sites_high"]],
    on="read_id", how="inner"
)
prt(f"  Merged (summary x cache): {len(merged_hela):,} reads")
prt(f"  Normal: {(merged_hela['is_stress']==0).sum():,}, Stress: {(merged_hela['is_stress']==1).sum():,}")

# Filter: need valid polya_length
merged_hela = merged_hela[merged_hela["polya_length"].notna() & (merged_hela["polya_length"] > 0)].copy()
prt(f"  After polya filter: {len(merged_hela):,}")

# Quick summary by condition
for cond, cname in [(0, "Normal"), (1, "Stress")]:
    sub = merged_hela[merged_hela["is_stress"] == cond]
    prt(f"  {cname}: n={len(sub):,}, polya median={sub['polya_length'].median():.1f}, "
        f"m6a_kb median={sub['m6a_kb'].median():.2f}, RL median={sub['read_length'].median():.0f}")

prt()

# -- Model A: Original (no RL control) --------------------------------------
prt("  --- Model A: polya ~ m6a_kb * is_stress (no RL control) ---")
mA = smf.ols("polya_length ~ m6a_kb + is_stress + m6a_kb:is_stress", data=merged_hela).fit()
for var in ["Intercept", "m6a_kb", "is_stress", "m6a_kb:is_stress"]:
    prt(f"    {var:30s} coef={mA.params[var]:>8.3f}  p={mA.pvalues[var]:.2e}")
prt(f"    R^2 = {mA.rsquared:.4f}, n = {int(mA.nobs)}")
prt()

# -- Model B: + read_length -------------------------------------------------
prt("  --- Model B: polya ~ m6a_kb * is_stress + read_length ---")
mB = smf.ols("polya_length ~ m6a_kb + is_stress + m6a_kb:is_stress + read_length",
             data=merged_hela).fit()
for var in ["Intercept", "m6a_kb", "is_stress", "m6a_kb:is_stress", "read_length"]:
    prt(f"    {var:30s} coef={mB.params[var]:>8.3f}  p={mB.pvalues[var]:.2e}")
prt(f"    R^2 = {mB.rsquared:.4f}, n = {int(mB.nobs)}")
prt()

# -- Model C: + read_length + read_length^2 ---------------------------------
prt("  --- Model C: polya ~ m6a_kb * is_stress + read_length + read_length^2 ---")
merged_hela["rl_sq"] = merged_hela["read_length"] ** 2
mC = smf.ols("polya_length ~ m6a_kb + is_stress + m6a_kb:is_stress + read_length + rl_sq",
             data=merged_hela).fit()
for var in ["Intercept", "m6a_kb", "is_stress", "m6a_kb:is_stress", "read_length", "rl_sq"]:
    prt(f"    {var:30s} coef={mC.params[var]:>8.4f}  p={mC.pvalues[var]:.2e}")
prt(f"    R^2 = {mC.rsquared:.4f}, n = {int(mC.nobs)}")
prt()

# -- Model D: + read_length + read_length^2 + m6a_kb:read_length ------------
prt("  --- Model D: + m6a_kb:read_length (RL-dependent m6A effect?) ---")
mD = smf.ols("polya_length ~ m6a_kb + is_stress + m6a_kb:is_stress + read_length + rl_sq + m6a_kb:read_length",
             data=merged_hela).fit()
for var in ["Intercept", "m6a_kb", "is_stress", "m6a_kb:is_stress", "read_length", "rl_sq", "m6a_kb:read_length"]:
    prt(f"    {var:30s} coef={mD.params[var]:>8.4f}  p={mD.pvalues[var]:.2e}")
prt(f"    R^2 = {mD.rsquared:.4f}, n = {int(mD.nobs)}")
prt()

# -- Model E: Full with 3-way: m6a_kb:is_stress:read_length -----------------
prt("  --- Model E: + m6a_kb:is_stress:read_length (stress-specific RL modulation?) ---")
mE = smf.ols("polya_length ~ m6a_kb * is_stress * read_length + rl_sq",
             data=merged_hela).fit()
prt(f"  Full model summary:")
for var in mE.params.index:
    prt(f"    {var:40s} coef={mE.params[var]:>10.5f}  p={mE.pvalues[var]:.2e}")
prt(f"    R^2 = {mE.rsquared:.4f}, n = {int(mE.nobs)}")
prt()

# ============================================================================
# 6. Comparison: key coefficient stability across models
# ============================================================================
prt("=" * 90)
prt("SECTION 6: Key coefficient stability across models")
prt("=" * 90)

prt(f"{'Model':<10} {'m6a_kb:stress coef':>20} {'p-value':>12} {'R^2':>8}")
prt("-" * 52)
for name, m in [("A (bare)", mA), ("B (+RL)", mB), ("C (+RL^2)", mC), ("D (+m6a:RL)", mD)]:
    coef = m.params["m6a_kb:is_stress"]
    pval = m.pvalues["m6a_kb:is_stress"]
    prt(f"{name:<10} {coef:>20.4f} {pval:>12.2e} {m.rsquared:>8.4f}")

# Model E has the 3-way interaction; extract the 2-way
coef_e = mE.params["m6a_kb:is_stress"]
pval_e = mE.pvalues["m6a_kb:is_stress"]
prt(f"{'E (3-way)':<10} {coef_e:>20.4f} {pval_e:>12.2e} {mE.rsquared:>8.4f}")
prt()

# ============================================================================
# 7. RL-stratified m6A-polyA correlation (within bins)
# ============================================================================
prt("=" * 90)
prt("SECTION 7: RL-stratified m6A-polyA Spearman correlation (HeLa stress only)")
prt("=" * 90)

ars_data = merged_hela[merged_hela["is_stress"] == 1].copy()
ars_data["rl_bin"] = pd.cut(ars_data["read_length"], bins=RL_BINS, labels=RL_LABELS, right=False)

prt(f"{'RL bin':<10} {'n':>7} {'Spearman r':>11} {'p':>12} {'median polyA':>13} {'median m6a/kb':>14}")
prt("-" * 70)
for b in RL_LABELS:
    sub = ars_data[ars_data["rl_bin"] == b]
    n = len(sub)
    if n < 10:
        prt(f"{b:<10} {n:>7}   (too few)")
        continue
    r, p = stats.spearmanr(sub["m6a_kb"], sub["polya_length"])
    prt(f"{b:<10} {n:>7} {r:>11.4f} {p:>12.2e} {sub['polya_length'].median():>13.1f} "
        f"{sub['m6a_kb'].median():>14.2f}")

# Same for normal
prt()
prt("  (Normal HeLa for comparison)")
norm_data = merged_hela[merged_hela["is_stress"] == 0].copy()
norm_data["rl_bin"] = pd.cut(norm_data["read_length"], bins=RL_BINS, labels=RL_LABELS, right=False)

prt(f"{'RL bin':<10} {'n':>7} {'Spearman r':>11} {'p':>12} {'median polyA':>13} {'median m6a/kb':>14}")
prt("-" * 70)
for b in RL_LABELS:
    sub = norm_data[norm_data["rl_bin"] == b]
    n = len(sub)
    if n < 10:
        prt(f"{b:<10} {n:>7}   (too few)")
        continue
    r, p = stats.spearmanr(sub["m6a_kb"], sub["polya_length"])
    prt(f"{b:<10} {n:>7} {r:>11.4f} {p:>12.2e} {sub['polya_length'].median():>13.1f} "
        f"{sub['m6a_kb'].median():>14.2f}")

prt()

# ============================================================================
# 8. Summary & interpretation
# ============================================================================
prt("=" * 90)
prt("SECTION 8: Summary")
prt("=" * 90)

# Check if RL^2 is significant in Model C
rl_sq_p = mC.pvalues["rl_sq"]
prt(f"  1. m6A/kb ~ read_length nonlinearity:")
prt(f"     - Quadratic term (RL^2) p = {rl_sq_p:.2e} in polya OLS")
prt(f"     - Spearman(RL, m6a_kb) L1: r={r_sp_l1:.4f}")
prt(f"     - Spearman(RL, m6a_kb) Ctrl: r={r_sp_c:.4f}")

prt(f"\n  2. L1/Ctrl m6A/kb ratio across RL bins:")
ratio_vals = merged[["bin", "ratio_mean"]].dropna()
prt(f"     - Range: {ratio_vals['ratio_mean'].min():.3f} - {ratio_vals['ratio_mean'].max():.3f}")
prt(f"     - Spearman(bin, ratio): r={r_sp:.3f}, p={p_sp:.4f}")

prt(f"\n  3. m6a_kb:is_stress interaction robustness:")
prt(f"     - Model A (no RL):    coef={mA.params['m6a_kb:is_stress']:.4f}, p={mA.pvalues['m6a_kb:is_stress']:.2e}")
prt(f"     - Model C (+RL+RL^2): coef={mC.params['m6a_kb:is_stress']:.4f}, p={mC.pvalues['m6a_kb:is_stress']:.2e}")
coef_change = abs(mC.params['m6a_kb:is_stress'] - mA.params['m6a_kb:is_stress'])
coef_pct = coef_change / abs(mA.params['m6a_kb:is_stress']) * 100
prt(f"     - Coefficient change: {coef_change:.4f} ({coef_pct:.1f}%)")

prt()

# -- Save -------------------------------------------------------------------
with open(OUTFILE, "w") as f:
    f.write("\n".join(out_lines) + "\n")

prt(f"Results saved to: {OUTFILE}")
