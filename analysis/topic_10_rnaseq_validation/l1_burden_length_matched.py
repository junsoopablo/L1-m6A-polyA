#!/usr/bin/env python3
"""
L1 burden vs gene length confound analysis.

Goal: Determine whether intronic L1 density has an independent effect on
gene expression change under arsenite stress, beyond what gene length alone
can explain.

Approaches:
1. Gene length-binned analysis (deciles)
2. Propensity score matching (length ±10%)
3. Multiple regression with interaction
4. Residual correlation analysis

Author: Claude Code
Date: 2026-02-17
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation"
INPUT = f"{BASE}/l1_burden_analysis/gene_l1_burden_rnaseq.tsv"
OUT_DIR = f"{BASE}/l1_burden_analysis"
OUT_PDF = f"{OUT_DIR}/l1_burden_length_matched.pdf"
OUT_PNG = f"{OUT_DIR}/l1_burden_length_matched.png"

# ── Load data ──
df = pd.read_csv(INPUT, sep='\t', index_col=0)
print(f"Loaded {len(df)} genes")
print(f"Columns: {list(df.columns)}")

# Basic stats
print(f"\n=== Basic Statistics ===")
print(f"Gene length: median={df['gene_length'].median():.0f}, "
      f"mean={df['gene_length'].mean():.0f}, range={df['gene_length'].min()}-{df['gene_length'].max()}")
print(f"Ancient L1 density: median={df['ancient_l1_density'].median():.4f}, "
      f"mean={df['ancient_l1_density'].mean():.4f}")
print(f"log2FC: median={df['log2FoldChange'].median():.4f}, "
      f"mean={df['log2FoldChange'].mean():.4f}")

# Genes with any L1
has_l1 = df['ancient_l1_density'] > 0
print(f"Genes with any ancient L1: {has_l1.sum()} ({has_l1.mean()*100:.1f}%)")
print(f"Genes without L1: {(~has_l1).sum()} ({(~has_l1).mean()*100:.1f}%)")

# Correlation: gene_length vs L1 density
r_len_l1, p_len_l1 = pearsonr(df['gene_length'], df['ancient_l1_density'])
r_len_l1_s, p_len_l1_s = spearmanr(df['gene_length'], df['ancient_l1_density'])
print(f"\nGene length vs ancient L1 density: Pearson r={r_len_l1:.3f} (P={p_len_l1:.2e}), "
      f"Spearman rho={r_len_l1_s:.3f} (P={p_len_l1_s:.2e})")

# Correlation: gene_length vs log2FC
r_len_fc, p_len_fc = pearsonr(df['gene_length'], df['log2FoldChange'])
print(f"Gene length vs log2FC: Pearson r={r_len_fc:.3f} (P={p_len_fc:.2e})")

# Correlation: L1 density vs log2FC
r_l1_fc, p_l1_fc = pearsonr(df['ancient_l1_density'], df['log2FoldChange'])
print(f"L1 density vs log2FC: Pearson r={r_l1_fc:.3f} (P={p_l1_fc:.2e})")


# ============================================================
# ANALYSIS 1: Gene length-binned analysis (deciles)
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 1: Gene Length-Binned Analysis (Deciles)")
print("="*70)

# Create length deciles
df['length_decile'] = pd.qcut(df['gene_length'], 10, labels=False, duplicates='drop')

bin_results = []
for d in sorted(df['length_decile'].unique()):
    sub = df[df['length_decile'] == d].copy()
    n_total = len(sub)
    len_range = f"{sub['gene_length'].min():,}-{sub['gene_length'].max():,}"
    len_median = sub['gene_length'].median()

    # Genes with L1 in this bin
    has_l1_bin = sub['ancient_l1_density'] > 0
    n_with_l1 = has_l1_bin.sum()

    if n_with_l1 < 10 or (~has_l1_bin).sum() < 10:
        print(f"  Decile {d}: {len_range} bp (n={n_total}), "
              f"L1+={n_with_l1}, L1-={n_total - n_with_l1} → SKIPPED (too few)")
        continue

    # Split at median L1 density among genes WITH L1 in this bin
    # Strategy: compare genes with L1 (any density) vs without L1
    fc_with = sub.loc[has_l1_bin, 'log2FoldChange']
    fc_without = sub.loc[~has_l1_bin, 'log2FoldChange']

    diff = fc_with.mean() - fc_without.mean()
    stat, pval = mannwhitneyu(fc_with, fc_without, alternative='two-sided')

    # Also: within L1+ genes, split at median density
    if n_with_l1 >= 20:
        l1_genes = sub.loc[has_l1_bin].copy()
        med_density = l1_genes['ancient_l1_density'].median()
        high_l1 = l1_genes[l1_genes['ancient_l1_density'] >= med_density]['log2FoldChange']
        low_l1 = l1_genes[l1_genes['ancient_l1_density'] < med_density]['log2FoldChange']
        diff_within = high_l1.mean() - low_l1.mean()
        _, p_within = mannwhitneyu(high_l1, low_l1, alternative='two-sided')
    else:
        diff_within = np.nan
        p_within = np.nan

    bin_results.append({
        'decile': d,
        'len_range': len_range,
        'len_median': len_median,
        'n_total': n_total,
        'n_with_l1': n_with_l1,
        'n_without_l1': n_total - n_with_l1,
        'fc_with_mean': fc_with.mean(),
        'fc_without_mean': fc_without.mean(),
        'diff': diff,
        'pval': pval,
        'se_diff': np.sqrt(fc_with.var()/len(fc_with) + fc_without.var()/len(fc_without)),
        'diff_within_l1': diff_within,
        'p_within_l1': p_within,
    })

    print(f"  Decile {d}: {len_range} bp (n={n_total}), "
          f"L1+ mean FC={fc_with.mean():.3f} (n={n_with_l1}), "
          f"L1- mean FC={fc_without.mean():.3f} (n={n_total - n_with_l1}), "
          f"Δ={diff:.3f}, P={pval:.3e}")

bin_df = pd.DataFrame(bin_results)

# Meta-analysis: inverse-variance weighted mean difference
if len(bin_df) > 0:
    valid = bin_df[bin_df['se_diff'] > 0].copy()
    weights = 1.0 / valid['se_diff']**2
    meta_diff = np.average(valid['diff'], weights=weights)
    meta_se = np.sqrt(1.0 / weights.sum())
    meta_z = meta_diff / meta_se
    meta_p = 2 * stats.norm.sf(abs(meta_z))
    print(f"\n  ** Fixed-effect meta-analysis (L1+ vs L1-): **")
    print(f"     Weighted mean Δlog2FC = {meta_diff:.4f} (SE={meta_se:.4f})")
    print(f"     Z = {meta_z:.2f}, P = {meta_p:.2e}")
    print(f"     (Negative = L1+ genes more downregulated)")

    # Within-L1 meta-analysis (high vs low density)
    valid_within = bin_df.dropna(subset=['diff_within_l1']).copy()
    if len(valid_within) > 0:
        # Use approximate SE for within-L1 comparison
        # This is a rough estimate; the main comparison above is more robust
        print(f"\n  ** Within-L1 genes (high vs low density, per-bin): **")
        for _, row in valid_within.iterrows():
            star = "***" if row['p_within_l1'] < 0.001 else "**" if row['p_within_l1'] < 0.01 else "*" if row['p_within_l1'] < 0.05 else "ns"
            print(f"     Decile {row['decile']}: Δ={row['diff_within_l1']:.3f}, P={row['p_within_l1']:.3e} {star}")

# Also: continuous L1 density within length bins
print("\n  ** Spearman correlation of L1 density vs log2FC within length bins: **")
within_corrs = []
for d in sorted(df['length_decile'].unique()):
    sub = df[df['length_decile'] == d]
    l1_sub = sub[sub['ancient_l1_density'] > 0]
    if len(l1_sub) >= 30:
        rho, p = spearmanr(l1_sub['ancient_l1_density'], l1_sub['log2FoldChange'])
        within_corrs.append({'decile': d, 'rho': rho, 'p': p, 'n': len(l1_sub)})
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"     Decile {d}: rho={rho:.3f}, P={p:.3e}, n={len(l1_sub)} {star}")


# ============================================================
# ANALYSIS 2: Propensity score matching (length ±10%)
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 2: Propensity Score Matching (Gene Length ±10%)")
print("="*70)

# Define high vs low L1 density
# Strategy: Compare genes with above-median L1 density vs genes with L1 density = 0
# But we need both groups to have enough genes at similar lengths.

# Alternative: among ALL genes, define high L1 = top quartile of density (among L1+ genes)
# and low L1 = L1 density = 0

l1_positive = df[df['ancient_l1_density'] > 0].copy()
l1_q75 = l1_positive['ancient_l1_density'].quantile(0.75)
l1_median = l1_positive['ancient_l1_density'].median()

high_l1 = df[df['ancient_l1_density'] >= l1_median].copy()  # top 50% of L1+ genes
low_l1 = df[df['ancient_l1_density'] == 0].copy()  # no L1

print(f"High L1 density (>= median of L1+ = {l1_median:.4f}): n={len(high_l1)}")
print(f"No L1 (density = 0): n={len(low_l1)}")

# Match each high-L1 gene to nearest low-L1 gene by length (±10%)
np.random.seed(42)
matched_pairs = []
used_low = set()

# Sort high_l1 randomly to avoid systematic bias
high_l1_shuffled = high_l1.sample(frac=1, random_state=42)

for idx, row in high_l1_shuffled.iterrows():
    gene_len = row['gene_length']
    len_lower = gene_len * 0.9
    len_upper = gene_len * 1.1

    # Find candidates in low_l1 within ±10% length, not already used
    candidates = low_l1[
        (low_l1['gene_length'] >= len_lower) &
        (low_l1['gene_length'] <= len_upper) &
        (~low_l1.index.isin(used_low))
    ]

    if len(candidates) == 0:
        continue

    # Pick closest in length
    best_idx = (candidates['gene_length'] - gene_len).abs().idxmin()
    used_low.add(best_idx)

    matched_pairs.append({
        'high_gene': idx,
        'low_gene': best_idx,
        'high_fc': row['log2FoldChange'],
        'low_fc': low_l1.loc[best_idx, 'log2FoldChange'],
        'high_len': gene_len,
        'low_len': low_l1.loc[best_idx, 'gene_length'],
        'high_density': row['ancient_l1_density'],
    })

matched_df = pd.DataFrame(matched_pairs)
print(f"Successfully matched: {len(matched_df)} pairs")
print(f"Length ratio (high/low): {(matched_df['high_len']/matched_df['low_len']).median():.3f} median")

if len(matched_df) > 0:
    # Paired comparison
    fc_diff = matched_df['high_fc'] - matched_df['low_fc']
    t_stat, t_pval = stats.ttest_1samp(fc_diff, 0)
    w_stat, w_pval = wilcoxon(fc_diff)

    print(f"\nMatched pair analysis (High L1 - No L1):")
    print(f"  Mean Δlog2FC = {fc_diff.mean():.4f} (SE={fc_diff.sem():.4f})")
    print(f"  Median Δlog2FC = {fc_diff.median():.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, P={t_pval:.2e}")
    print(f"  Wilcoxon signed-rank: P={w_pval:.2e}")
    print(f"  High L1 mean FC = {matched_df['high_fc'].mean():.4f}")
    print(f"  No L1 mean FC = {matched_df['low_fc'].mean():.4f}")

    # Verify length matching quality
    len_r, len_p = pearsonr(matched_df['high_len'], matched_df['low_len'])
    print(f"\n  Length matching quality:")
    print(f"  Pearson r(high_len, low_len) = {len_r:.4f}")
    print(f"  Mean |len_diff|/len = {((matched_df['high_len'] - matched_df['low_len']).abs() / matched_df['high_len']).mean():.3f}")

# Also: stricter matching (±5%)
matched_strict = []
used_low_strict = set()
for idx, row in high_l1_shuffled.iterrows():
    gene_len = row['gene_length']
    candidates = low_l1[
        (low_l1['gene_length'] >= gene_len * 0.95) &
        (low_l1['gene_length'] <= gene_len * 1.05) &
        (~low_l1.index.isin(used_low_strict))
    ]
    if len(candidates) == 0:
        continue
    best_idx = (candidates['gene_length'] - gene_len).abs().idxmin()
    used_low_strict.add(best_idx)
    matched_strict.append({
        'high_fc': row['log2FoldChange'],
        'low_fc': low_l1.loc[best_idx, 'log2FoldChange'],
    })

if len(matched_strict) > 0:
    ms_df = pd.DataFrame(matched_strict)
    fc_diff_strict = ms_df['high_fc'] - ms_df['low_fc']
    _, p_strict = wilcoxon(fc_diff_strict)
    print(f"\n  Strict matching (±5%): n={len(ms_df)}, "
          f"mean Δ={fc_diff_strict.mean():.4f}, Wilcoxon P={p_strict:.2e}")


# ============================================================
# ANALYSIS 3: Multiple regression
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 3: Multiple Regression")
print("="*70)

# Standardize predictors for interpretability
df_reg = df.copy()
df_reg['gene_length_z'] = (df_reg['gene_length'] - df_reg['gene_length'].mean()) / df_reg['gene_length'].std()
df_reg['l1_density_z'] = (df_reg['ancient_l1_density'] - df_reg['ancient_l1_density'].mean()) / df_reg['ancient_l1_density'].std()
df_reg['log_gene_length'] = np.log10(df_reg['gene_length'])
df_reg['log_gene_length_z'] = (df_reg['log_gene_length'] - df_reg['log_gene_length'].mean()) / df_reg['log_gene_length'].std()

# Model 1: gene_length only
m1 = smf.ols('log2FoldChange ~ gene_length_z', data=df_reg).fit()
print(f"\nModel 1: log2FC ~ gene_length (standardized)")
print(f"  R² = {m1.rsquared:.4f}, Adj R² = {m1.rsquared_adj:.4f}")
print(f"  gene_length_z: coef={m1.params['gene_length_z']:.4f}, P={m1.pvalues['gene_length_z']:.2e}")

# Model 2: L1 density only
m2 = smf.ols('log2FoldChange ~ l1_density_z', data=df_reg).fit()
print(f"\nModel 2: log2FC ~ L1_density (standardized)")
print(f"  R² = {m2.rsquared:.4f}, Adj R² = {m2.rsquared_adj:.4f}")
print(f"  l1_density_z: coef={m2.params['l1_density_z']:.4f}, P={m2.pvalues['l1_density_z']:.2e}")

# Model 3: both predictors (no interaction)
m3 = smf.ols('log2FoldChange ~ gene_length_z + l1_density_z', data=df_reg).fit()
print(f"\nModel 3: log2FC ~ gene_length + L1_density")
print(f"  R² = {m3.rsquared:.4f}, Adj R² = {m3.rsquared_adj:.4f}")
print(f"  gene_length_z: coef={m3.params['gene_length_z']:.4f}, P={m3.pvalues['gene_length_z']:.2e}")
print(f"  l1_density_z: coef={m3.params['l1_density_z']:.4f}, P={m3.pvalues['l1_density_z']:.2e}")

# Model 4: full interaction model
m4 = smf.ols('log2FoldChange ~ gene_length_z * l1_density_z', data=df_reg).fit()
print(f"\nModel 4: log2FC ~ gene_length * L1_density (interaction)")
print(f"  R² = {m4.rsquared:.4f}, Adj R² = {m4.rsquared_adj:.4f}")
for term in ['gene_length_z', 'l1_density_z', 'gene_length_z:l1_density_z']:
    print(f"  {term}: coef={m4.params[term]:.4f}, P={m4.pvalues[term]:.2e}")

# Model 5: use log(gene_length) as predictor (more linear relationship)
m5 = smf.ols('log2FoldChange ~ log_gene_length_z * l1_density_z', data=df_reg).fit()
print(f"\nModel 5: log2FC ~ log(gene_length) * L1_density")
print(f"  R² = {m5.rsquared:.4f}, Adj R² = {m5.rsquared_adj:.4f}")
for term in ['log_gene_length_z', 'l1_density_z', 'log_gene_length_z:l1_density_z']:
    print(f"  {term}: coef={m5.params[term]:.4f}, P={m5.pvalues[term]:.2e}")

# Model comparison: ANOVA (M1 vs M3)
from statsmodels.stats.anova import anova_lm
anova_res = anova_lm(m1, m3)
print(f"\nANOVA: Model 1 (length only) vs Model 3 (length + L1 density)")
print(anova_res)

# VIF (variance inflation factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df_reg[['gene_length_z', 'l1_density_z']].copy()
X['const'] = 1
vif_gl = variance_inflation_factor(X.values, 0)
vif_l1 = variance_inflation_factor(X.values, 1)
print(f"\nVariance Inflation Factors:")
print(f"  gene_length_z: VIF = {vif_gl:.2f}")
print(f"  l1_density_z: VIF = {vif_l1:.2f}")
print(f"  (VIF > 10 = severe collinearity; > 5 = moderate)")

# Model 6: Among L1+ genes only (to avoid zero-inflation)
df_l1pos = df_reg[df_reg['ancient_l1_density'] > 0].copy()
df_l1pos['l1_density_pos_z'] = (df_l1pos['ancient_l1_density'] - df_l1pos['ancient_l1_density'].mean()) / df_l1pos['ancient_l1_density'].std()
m6 = smf.ols('log2FoldChange ~ gene_length_z + l1_density_pos_z', data=df_l1pos).fit()
print(f"\nModel 6: L1+ genes only (n={len(df_l1pos)})")
print(f"  R² = {m6.rsquared:.4f}")
print(f"  gene_length_z: coef={m6.params['gene_length_z']:.4f}, P={m6.pvalues['gene_length_z']:.2e}")
print(f"  l1_density_z: coef={m6.params['l1_density_pos_z']:.4f}, P={m6.pvalues['l1_density_pos_z']:.2e}")

r_l1pos, p_l1pos = pearsonr(df_l1pos['gene_length'], df_l1pos['ancient_l1_density'])
print(f"  (gene_length vs L1_density in L1+ genes: r={r_l1pos:.3f})")


# ============================================================
# ANALYSIS 4: Residual correlation analysis
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 4: Residual Correlation Analysis")
print("="*70)

# Regress out gene_length from both variables
# Use log(gene_length) for better linearity
df_reg['log_gl'] = np.log10(df_reg['gene_length'])

# Residual of log2FC after removing gene_length effect
m_fc_len = smf.ols('log2FoldChange ~ log_gl', data=df_reg).fit()
df_reg['fc_resid'] = m_fc_len.resid

# Residual of L1 density after removing gene_length effect
m_l1_len = smf.ols('ancient_l1_density ~ log_gl', data=df_reg).fit()
df_reg['l1_resid'] = m_l1_len.resid

# Correlate residuals (= partial correlation)
r_resid, p_resid = pearsonr(df_reg['l1_resid'], df_reg['fc_resid'])
rho_resid, p_resid_s = spearmanr(df_reg['l1_resid'], df_reg['fc_resid'])

print(f"After regressing out log(gene_length) from both variables:")
print(f"  Pearson r(L1_density_resid, log2FC_resid) = {r_resid:.4f} (P={p_resid:.2e})")
print(f"  Spearman rho = {rho_resid:.4f} (P={p_resid_s:.2e})")
print(f"  (Original r without correction: {r_l1_fc:.4f})")
print(f"  Attenuation: {abs(r_resid)/abs(r_l1_fc)*100:.1f}% of original")

# Among L1+ genes only
l1pos_mask = df_reg['ancient_l1_density'] > 0
r_resid_pos, p_resid_pos = pearsonr(
    df_reg.loc[l1pos_mask, 'l1_resid'],
    df_reg.loc[l1pos_mask, 'fc_resid']
)
print(f"\n  Among L1+ genes only (n={l1pos_mask.sum()}):")
print(f"  Pearson r = {r_resid_pos:.4f} (P={p_resid_pos:.2e})")


# ============================================================
# ANALYSIS 5: Additional robustness checks
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 5: Robustness Checks")
print("="*70)

# 5a: L1 bp coverage instead of count-based density
df_reg['l1_bp_density'] = df_reg['ancient_l1_bp'] / df_reg['gene_length']
df_reg['l1_bp_density_z'] = (df_reg['l1_bp_density'] - df_reg['l1_bp_density'].mean()) / df_reg['l1_bp_density'].std()

m_bp = smf.ols('log2FoldChange ~ gene_length_z + l1_bp_density_z', data=df_reg).fit()
print(f"5a. Using L1 bp coverage (proportion of gene covered by L1):")
print(f"  gene_length_z: coef={m_bp.params['gene_length_z']:.4f}, P={m_bp.pvalues['gene_length_z']:.2e}")
print(f"  l1_bp_density_z: coef={m_bp.params['l1_bp_density_z']:.4f}, P={m_bp.pvalues['l1_bp_density_z']:.2e}")

# 5b: Tercile analysis (cleaner separation)
print(f"\n5b. Gene length tercile × L1 density tercile:")
df_reg['len_tercile'] = pd.qcut(df_reg['gene_length'], 3, labels=['Short', 'Medium', 'Long'])
# For L1 density: 0 vs low vs high (among L1+ genes)
l1pos_vals = df_reg.loc[df_reg['ancient_l1_density'] > 0, 'ancient_l1_density']
l1_med = l1pos_vals.median()

def l1_group(x):
    if x == 0:
        return 'None'
    elif x < l1_med:
        return 'Low'
    else:
        return 'High'

df_reg['l1_group'] = df_reg['ancient_l1_density'].apply(l1_group)

for lt in ['Short', 'Medium', 'Long']:
    sub = df_reg[df_reg['len_tercile'] == lt]
    means = sub.groupby('l1_group')['log2FoldChange'].agg(['mean', 'count'])
    print(f"  {lt} genes:")
    for g in ['None', 'Low', 'High']:
        if g in means.index:
            print(f"    L1={g}: mean FC={means.loc[g, 'mean']:.4f}, n={means.loc[g, 'count']:.0f}")
    # Test None vs High within this length tercile
    none_fc = sub[sub['l1_group'] == 'None']['log2FoldChange']
    high_fc = sub[sub['l1_group'] == 'High']['log2FoldChange']
    if len(none_fc) >= 10 and len(high_fc) >= 10:
        _, p = mannwhitneyu(none_fc, high_fc, alternative='two-sided')
        diff = high_fc.mean() - none_fc.mean()
        print(f"    None vs High: Δ={diff:.4f}, MWU P={p:.3e}")

# 5c: Quantile regression (median regression, robust to outliers)
print(f"\n5c. Quantile regression (median, tau=0.5):")
qr = smf.quantreg('log2FoldChange ~ gene_length_z + l1_density_z', data=df_reg).fit(q=0.5)
print(f"  gene_length_z: coef={qr.params['gene_length_z']:.4f}, P={qr.pvalues['gene_length_z']:.2e}")
print(f"  l1_density_z: coef={qr.params['l1_density_z']:.4f}, P={qr.pvalues['l1_density_z']:.2e}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY: L1 Burden vs Gene Length Confound")
print("="*70)

print(f"""
Key statistics:
  Gene length ↔ L1 density:  r = {r_len_l1:.3f} (very strong collinearity)
  Gene length ↔ log2FC:      r = {r_len_fc:.3f}
  L1 density ↔ log2FC:       r = {r_l1_fc:.3f}

Analysis 1 (Length-binned):
  Meta-analysis Δlog2FC = {meta_diff:.4f} (P={meta_p:.2e})
  {"SIGNIFICANT" if meta_p < 0.05 else "NOT significant"}: L1+ genes {"more downregulated" if meta_diff < 0 else "less downregulated"} within length bins

Analysis 2 (Length-matched pairs):
  {len(matched_df)} matched pairs (±10% length)
  Mean Δlog2FC = {fc_diff.mean():.4f}, Wilcoxon P = {w_pval:.2e}
  {"SIGNIFICANT" if w_pval < 0.05 else "NOT significant"}

Analysis 3 (Multiple regression):
  Full model (length + L1 density):
    gene_length: coef={m3.params['gene_length_z']:.4f}, P={m3.pvalues['gene_length_z']:.2e}
    L1_density:  coef={m3.params['l1_density_z']:.4f}, P={m3.pvalues['l1_density_z']:.2e}
  VIF = {vif_gl:.1f} ({"severe" if vif_gl > 10 else "moderate" if vif_gl > 5 else "acceptable"} collinearity)

Analysis 4 (Residual correlation):
  Partial r = {r_resid:.4f} (P={p_resid:.2e})
  Original r = {r_l1_fc:.4f} → {abs(r_resid)/abs(r_l1_fc)*100:.1f}% retained after length correction

CONCLUSION:
""")

# Determine conclusion
if abs(r_resid) > 0.02 and p_resid < 0.05 and w_pval < 0.05:
    conclusion = ("L1 density has a STATISTICALLY SIGNIFICANT independent effect on arsenite "
                  "gene expression change, beyond gene length. However, the effect size is "
                  f"small (partial r={r_resid:.4f}) and gene length accounts for most of the "
                  f"variance. The independent L1 contribution is {abs(r_resid)/abs(r_l1_fc)*100:.1f}% "
                  "of the original correlation.")
elif p_resid < 0.05:
    conclusion = ("L1 density shows a marginal independent effect (P<0.05), but the effect "
                  f"size is very small (partial r={r_resid:.4f}). Gene length is the dominant "
                  "predictor, and the L1 association is largely mediated through length.")
else:
    conclusion = ("L1 density does NOT have a significant independent effect on arsenite "
                  "gene expression change after controlling for gene length. The observed "
                  "L1-expression association is entirely explained by gene length confounding.")

print(f"  {conclusion}")


# ============================================================
# FIGURE
# ============================================================
print("\n\nGenerating figure...")

fig = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 3, hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.94, bottom=0.06)

# ── Panel A: Length-binned analysis (forest plot) ──
ax1 = fig.add_subplot(gs[0, 0])

if len(bin_df) > 0:
    y_pos = np.arange(len(bin_df))
    colors = ['#d62728' if p < 0.05 else '#999999' for p in bin_df['pval']]

    ax1.barh(y_pos, bin_df['diff'], xerr=1.96*bin_df['se_diff'],
             color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
             capsize=3, height=0.7)
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax1.axvline(meta_diff, color='blue', linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'Meta Δ={meta_diff:.3f}\n(P={meta_p:.1e})')

    # Y-axis labels: length range
    labels = [f"D{int(r['decile'])}: {r['len_range'].split('-')[0][:5]}-{r['len_range'].split('-')[1][:6]}"
              for _, r in bin_df.iterrows()]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel('Δlog2FC (L1+ − L1−)', fontsize=10)
    ax1.set_title('A. Length-binned L1 effect', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d62728', alpha=0.7, label='P < 0.05'),
                       Patch(facecolor='#999999', alpha=0.7, label='P ≥ 0.05')]
    ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], color='blue', linestyle='--',
               label=f'Meta Δ={meta_diff:.3f} (P={meta_p:.1e})')],
               fontsize=7, loc='lower right')

# ── Panel B: Matched pairs comparison ──
ax2 = fig.add_subplot(gs[0, 1])

if len(matched_df) > 0:
    # Violin + box plot
    data_for_plot = [matched_df['low_fc'].values, matched_df['high_fc'].values]
    parts = ax2.violinplot(data_for_plot, positions=[0, 1],
                           showmeans=False, showmedians=False, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#2ca02c', '#d62728'][i])
        pc.set_alpha(0.4)

    bp = ax2.boxplot(data_for_plot, positions=[0, 1], widths=0.3,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#d62728')
    bp['boxes'][1].set_alpha(0.6)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['No L1\n(matched)', 'High L1\n(density)'], fontsize=10)
    ax2.set_ylabel('log2FC (arsenite/mock)', fontsize=10)
    ax2.set_title('B. Length-matched pairs', fontsize=12, fontweight='bold')

    # Significance annotation
    y_max = max(np.percentile(matched_df['low_fc'], 95), np.percentile(matched_df['high_fc'], 95))
    sig_text = f'Δ={fc_diff.mean():.3f}\nP={w_pval:.1e}\nn={len(matched_df)}'
    ax2.annotate(sig_text, xy=(0.5, y_max + 0.1), fontsize=9,
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

# ── Panel C: Residual correlation ──
ax3 = fig.add_subplot(gs[0, 2])

# Subsample for visibility
np.random.seed(42)
n_plot = min(3000, len(df_reg))
idx_plot = np.random.choice(len(df_reg), n_plot, replace=False)
plot_data = df_reg.iloc[idx_plot]

# Color by L1 presence
colors_scatter = ['#d62728' if x > 0 else '#2ca02c' for x in plot_data['ancient_l1_density']]
ax3.scatter(plot_data['l1_resid'], plot_data['fc_resid'],
            c=colors_scatter, alpha=0.15, s=8, rasterized=True)

# Regression line
x_range = np.linspace(plot_data['l1_resid'].min(), plot_data['l1_resid'].max(), 100)
slope, intercept = np.polyfit(df_reg['l1_resid'], df_reg['fc_resid'], 1)
ax3.plot(x_range, slope * x_range + intercept, 'k-', linewidth=2,
         label=f'r={r_resid:.3f} (P={p_resid:.1e})')

ax3.set_xlabel('L1 density residual\n(gene length regressed out)', fontsize=10)
ax3.set_ylabel('log2FC residual\n(gene length regressed out)', fontsize=10)
ax3.set_title('C. Partial correlation', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='lower left')

# Legend for colors
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                        markersize=6, label=f'L1+ (n={has_l1.sum()})'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                        markersize=6, label=f'L1- (n={(~has_l1).sum()})')]
ax3.legend(handles=legend_elems + [plt.Line2D([0], [0], color='k',
           label=f'r={r_resid:.3f} (P={p_resid:.1e})')],
           fontsize=7, loc='lower left')

# ── Panel D: Gene length × L1 density × log2FC heatmap (terciles) ──
ax4 = fig.add_subplot(gs[1, 0])

# Create tercile heatmap
pivot_data = df_reg.groupby(['len_tercile', 'l1_group'])['log2FoldChange'].mean().unstack()
# Reorder columns
col_order = [c for c in ['None', 'Low', 'High'] if c in pivot_data.columns]
pivot_data = pivot_data[col_order]

im = ax4.imshow(pivot_data.values, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
ax4.set_xticks(range(len(col_order)))
ax4.set_xticklabels(col_order, fontsize=10)
ax4.set_yticks(range(len(pivot_data.index)))
ax4.set_yticklabels(pivot_data.index, fontsize=10)
ax4.set_xlabel('L1 density group', fontsize=10)
ax4.set_ylabel('Gene length tercile', fontsize=10)
ax4.set_title('D. Tercile mean log2FC', fontsize=12, fontweight='bold')

# Add text annotations
pivot_n = df_reg.groupby(['len_tercile', 'l1_group'])['log2FoldChange'].count().unstack()[col_order]
for i in range(len(pivot_data.index)):
    for j in range(len(col_order)):
        val = pivot_data.values[i, j]
        n = pivot_n.values[i, j]
        txt_color = 'white' if abs(val) > 0.15 else 'black'
        ax4.text(j, i, f'{val:.3f}\n(n={int(n)})', ha='center', va='center',
                 fontsize=8, color=txt_color, fontweight='bold')

plt.colorbar(im, ax=ax4, shrink=0.8, label='Mean log2FC')

# ── Panel E: Regression coefficients comparison ──
ax5 = fig.add_subplot(gs[1, 1])

models_to_plot = {
    'Length\nonly': (m1.params.get('gene_length_z', 0), m1.bse.get('gene_length_z', 0), 'gene_length'),
    'L1\nonly': (m2.params.get('l1_density_z', 0), m2.bse.get('l1_density_z', 0), 'l1_density'),
    'Length\n(adj.)': (m3.params.get('gene_length_z', 0), m3.bse.get('gene_length_z', 0), 'gene_length'),
    'L1\n(adj.)': (m3.params.get('l1_density_z', 0), m3.bse.get('l1_density_z', 0), 'l1_density'),
    'L1\n(L1+ only)': (m6.params.get('l1_density_pos_z', 0), m6.bse.get('l1_density_pos_z', 0), 'l1_density'),
}

x_positions = np.arange(len(models_to_plot))
coefs = [v[0] for v in models_to_plot.values()]
ses = [v[1] for v in models_to_plot.values()]
bar_colors = ['#1f77b4' if v[2] == 'gene_length' else '#d62728' for v in models_to_plot.values()]

bars = ax5.bar(x_positions, coefs, yerr=[1.96*s for s in ses],
               color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5,
               capsize=4, width=0.6)
ax5.axhline(0, color='black', linewidth=0.8)
ax5.set_xticks(x_positions)
ax5.set_xticklabels(models_to_plot.keys(), fontsize=9)
ax5.set_ylabel('Standardized coefficient\n(± 95% CI)', fontsize=10)
ax5.set_title('E. Regression coefficients', fontsize=12, fontweight='bold')

legend_elems2 = [Patch(facecolor='#1f77b4', alpha=0.7, label='Gene length'),
                 Patch(facecolor='#d62728', alpha=0.7, label='L1 density')]
ax5.legend(handles=legend_elems2, fontsize=9, loc='lower right')

# Add P-values
pvals_list = [
    m1.pvalues['gene_length_z'],
    m2.pvalues['l1_density_z'],
    m3.pvalues['gene_length_z'],
    m3.pvalues['l1_density_z'],
    m6.pvalues['l1_density_pos_z'],
]
for i, (c, p) in enumerate(zip(coefs, pvals_list)):
    offset = 0.02 if c >= 0 else -0.02
    va = 'bottom' if c >= 0 else 'top'
    ptext = f"P={p:.1e}" if p < 0.001 else f"P={p:.3f}"
    ax5.text(i, c + offset + (1.96*ses[i] if c >= 0 else -1.96*ses[i]),
             ptext, ha='center', va=va, fontsize=7, fontweight='bold')

# ── Panel F: Summary diagnostic ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary_text = (
    f"CONFOUND ANALYSIS SUMMARY\n"
    f"{'='*40}\n\n"
    f"Genes analyzed: {len(df):,}\n"
    f"  - With L1: {has_l1.sum():,} ({has_l1.mean()*100:.1f}%)\n"
    f"  - Without L1: {(~has_l1).sum():,}\n\n"
    f"Collinearity:\n"
    f"  length ↔ L1 density: r={r_len_l1:.3f}\n"
    f"  VIF = {vif_gl:.1f}\n\n"
    f"Raw correlations:\n"
    f"  length ↔ log2FC: r={r_len_fc:.3f}\n"
    f"  L1 ↔ log2FC: r={r_l1_fc:.3f}\n\n"
    f"After length correction:\n"
    f"  Partial r = {r_resid:.4f} (P={p_resid:.1e})\n"
    f"  Retained: {abs(r_resid)/abs(r_l1_fc)*100:.1f}% of original\n\n"
    f"Length-matched pairs (n={len(matched_df)}):\n"
    f"  Δlog2FC = {fc_diff.mean():.4f}\n"
    f"  Wilcoxon P = {w_pval:.1e}\n\n"
    f"Regression (adj. for length):\n"
    f"  L1 coef = {m3.params['l1_density_z']:.4f}\n"
    f"  L1 P = {m3.pvalues['l1_density_z']:.1e}\n"
)

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.set_title('F. Summary', fontsize=12, fontweight='bold')

fig.suptitle('L1 Burden vs Gene Length: Confound Analysis',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {OUT_PDF}")
print(f"Figure saved: {OUT_PNG}")

print("\n=== DONE ===")
