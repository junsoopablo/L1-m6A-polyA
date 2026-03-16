#!/usr/bin/env python3
"""
Read-length controlled test: Does psi status affect arsenite-induced
poly(A) shortening independently of read length?

Methods:
  1. Read length bin stratification: compare psi+ vs psi- within matched bins
  2. Propensity-style matching: match psi+ and psi- reads by read length
  3. Regression: poly(A) ~ arsenite * psi + read_length
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
ANALYSIS = PROJECT / 'analysis/01_exploration'

# =========================================================================
# 1. Load data
# =========================================================================
print("Loading coupled data...")
df = pd.read_csv(ANALYSIS / 'topic_02_polya/polya_modification_coupled.tsv', sep='\t')

l1 = df[df['l1_age'].isin(['young', 'ancient'])].copy()
l1['has_psi'] = l1['psi_sites_high'] > 0
l1['has_m6a'] = l1['m6a_sites_high'] > 0
l1['condition'] = l1['source'].map({'HeLa_L1': 'HeLa', 'HeLa-Ars_L1': 'HeLa-Ars'})
l1['is_ars'] = (l1['condition'] == 'HeLa-Ars').astype(int)

print(f"  L1 reads: {len(l1):,}")

# Quick check: read length distribution by psi status
print(f"\n  Read length by psi status:")
for cond in ['HeLa', 'HeLa-Ars']:
    for psi in [True, False]:
        sub = l1[(l1['condition'] == cond) & (l1['has_psi'] == psi)]
        print(f"    {cond} psi{'+'if psi else '-'}: n={len(sub):,}  "
              f"rdLen mean={sub['read_length'].mean():.0f}  median={sub['read_length'].median():.0f}")

# =========================================================================
# 2. Method 1: Read length bin stratification
# =========================================================================
print(f"\n{'='*90}")
print("Method 1: Stratified by Read Length Bins")
print(f"{'='*90}")

len_bins = [0, 500, 750, 1000, 1500, 2000, 3000, 50000]
len_labels = ['<500', '500-750', '750-1k', '1-1.5k', '1.5-2k', '2-3k', '3k+']

l1['len_bin'] = pd.cut(l1['read_length'], bins=len_bins, labels=len_labels)

print(f"\n{'Bin':<10} {'Psi':<5} {'HeLa n':>7} {'HeLa med':>9} {'Ars n':>6} {'Ars med':>9} "
      f"{'Δ':>7} {'p':>12}")
print("-" * 75)

bin_results = []
for b in len_labels:
    for psi_label, psi_val in [('psi+', True), ('psi-', False)]:
        hela = l1[(l1['len_bin'] == b) & (l1['condition'] == 'HeLa') & (l1['has_psi'] == psi_val)]
        ars = l1[(l1['len_bin'] == b) & (l1['condition'] == 'HeLa-Ars') & (l1['has_psi'] == psi_val)]
        if len(hela) >= 10 and len(ars) >= 10:
            h_med = hela['polya_length'].median()
            a_med = ars['polya_length'].median()
            delta = a_med - h_med
            _, p = stats.mannwhitneyu(hela['polya_length'], ars['polya_length'],
                                       alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{b:<10} {psi_label:<5} {len(hela):>7} {h_med:>9.1f} {len(ars):>6} {a_med:>9.1f} "
                  f"{delta:>+6.1f} {p:>11.2e}({sig})")
            bin_results.append({'bin': b, 'psi': psi_label, 'hela_n': len(hela),
                                'ars_n': len(ars), 'delta': delta, 'p': p})
        else:
            print(f"{b:<10} {psi_label:<5} {len(hela):>7} {'':>9} {len(ars):>6} {'':>9} "
                  f"{'n/a':>7} {'(too few)':>12}")

# Summarize within-bin psi+ vs psi- difference
print(f"\n  Within-bin differential (psi+ Δ - psi- Δ):")
print(f"  {'Bin':<10} {'psi+ Δ':>8} {'psi- Δ':>8} {'Diff':>8}")
print(f"  {'-'*38}")

diffs = []
for b in len_labels:
    psi_pos = [r for r in bin_results if r['bin'] == b and r['psi'] == 'psi+']
    psi_neg = [r for r in bin_results if r['bin'] == b and r['psi'] == 'psi-']
    if psi_pos and psi_neg:
        d_pos = psi_pos[0]['delta']
        d_neg = psi_neg[0]['delta']
        diff = d_pos - d_neg
        diffs.append(diff)
        print(f"  {b:<10} {d_pos:>+7.1f} {d_neg:>+7.1f} {diff:>+7.1f}")

if diffs:
    print(f"\n  Mean within-bin differential: {np.mean(diffs):+.1f} nt")
    print(f"  Median within-bin differential: {np.median(diffs):+.1f} nt")
    # Sign test
    n_neg = sum(1 for d in diffs if d < 0)
    n_pos = sum(1 for d in diffs if d > 0)
    p_sign = stats.binomtest(n_neg, n_neg + n_pos, 0.5).pvalue if (n_neg + n_pos) > 0 else 1.0
    print(f"  Sign test: {n_neg} negative / {n_pos} positive, p={p_sign:.3f}")

# =========================================================================
# 3. Method 2: Read-length matched comparison
# =========================================================================
print(f"\n{'='*90}")
print("Method 2: Read-Length Matched Comparison (1:1 Nearest Neighbor)")
print(f"{'='*90}")

def match_by_length(target_df, pool_df, tolerance=50):
    """Match each target read to nearest pool read by read_length."""
    pool_lens = pool_df['read_length'].values
    pool_polya = pool_df['polya_length'].values
    pool_idx = np.argsort(pool_lens)
    pool_lens_sorted = pool_lens[pool_idx]

    matched_polya = []
    used = set()
    for _, row in target_df.iterrows():
        tgt_len = row['read_length']
        # Binary search for closest
        pos = np.searchsorted(pool_lens_sorted, tgt_len)
        best_idx = None
        best_diff = float('inf')
        for candidate in range(max(0, pos - 5), min(len(pool_lens_sorted), pos + 5)):
            orig_idx = pool_idx[candidate]
            if orig_idx not in used:
                diff = abs(pool_lens_sorted[candidate] - tgt_len)
                if diff < best_diff and diff <= tolerance:
                    best_diff = diff
                    best_idx = orig_idx
        if best_idx is not None:
            used.add(best_idx)
            matched_polya.append(pool_polya[best_idx])
        else:
            matched_polya.append(np.nan)
    return np.array(matched_polya)

for cond_label, cond_name in [('HeLa', 'HeLa'), ('HeLa-Ars', 'HeLa-Ars')]:
    psi_pos = l1[(l1['condition'] == cond_name) & (l1['has_psi'])].copy()
    psi_neg = l1[(l1['condition'] == cond_name) & (~l1['has_psi'])].copy()

    # Match psi- to psi+ by read length
    matched_neg_polya = match_by_length(psi_pos, psi_neg, tolerance=100)
    valid = ~np.isnan(matched_neg_polya)
    n_matched = valid.sum()

    psi_pos_matched = psi_pos[valid]['polya_length'].values
    psi_neg_matched = matched_neg_polya[valid]

    if n_matched >= 20:
        _, p = stats.mannwhitneyu(psi_pos_matched, psi_neg_matched, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  {cond_label} (n_matched={n_matched}):")
        print(f"    psi+ poly(A) median: {np.median(psi_pos_matched):.1f}")
        print(f"    psi- poly(A) median (matched): {np.median(psi_neg_matched):.1f}")
        print(f"    Δ(psi+ - psi-): {np.median(psi_pos_matched) - np.median(psi_neg_matched):+.1f}")
        print(f"    MW p={p:.2e} ({sig})")

        # Read length check
        psi_pos_rdlen = psi_pos[valid]['read_length'].values
        psi_neg_rdlen_orig = l1[(l1['condition'] == cond_name) & (~l1['has_psi'])]['read_length']
        print(f"    Matched read length: psi+ mean={psi_pos[valid]['read_length'].mean():.0f}")

# =========================================================================
# 4. Method 3: OLS regression with read length control
# =========================================================================
print(f"\n{'='*90}")
print("Method 3: OLS Regression (poly(A) ~ arsenite * psi + read_length)")
print(f"{'='*90}")

from numpy.linalg import lstsq

for age_label, age_filter in [('L1 all', ['young', 'ancient']),
                                ('Ancient only', ['ancient'])]:
    subset = l1[l1['l1_age'].isin(age_filter)].copy()

    # Standardize read_length for numerical stability
    rl_mean = subset['read_length'].mean()
    rl_std = subset['read_length'].std()
    subset['rl_z'] = (subset['read_length'] - rl_mean) / rl_std

    # Design matrix: intercept, is_ars, has_psi, rl_z, is_ars*has_psi
    X = np.column_stack([
        np.ones(len(subset)),
        subset['is_ars'].values,
        subset['has_psi'].astype(int).values,
        subset['rl_z'].values,
        (subset['is_ars'] * subset['has_psi'].astype(int)).values,
    ])
    y = subset['polya_length'].values

    # OLS
    beta, residuals, rank, sv = lstsq(X, y, rcond=None)
    y_pred = X @ beta
    resid = y - y_pred
    n, p_params = X.shape
    mse = np.sum(resid**2) / (n - p_params)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - p_params)

    param_names = ['intercept', 'arsenite', 'psi', 'read_length(z)', 'arsenite*psi']
    print(f"\n  {age_label} (n={len(subset):,}):")
    print(f"  {'Parameter':<20} {'Coef':>8} {'SE':>8} {'t':>8} {'p':>12}")
    print(f"  {'-'*60}")
    for name, b, se, t, pv in zip(param_names, beta, se_beta, t_stats, p_vals):
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        print(f"  {name:<20} {b:>8.2f} {se:>8.2f} {t:>8.2f} {pv:>11.2e}({sig})")

    print(f"\n  Interpretation of arsenite*psi interaction:")
    interaction_coef = beta[4]
    interaction_p = p_vals[4]
    if interaction_p < 0.05:
        direction = "MORE" if interaction_coef < 0 else "LESS"
        print(f"    psi+ reads show {direction} poly(A) shortening under arsenite")
        print(f"    (additional {interaction_coef:+.1f} nt beyond main arsenite effect)")
    else:
        print(f"    No significant interaction (coef={interaction_coef:+.1f}, p={interaction_p:.2e})")
        print(f"    → After controlling read length, psi status does NOT independently")
        print(f"      affect the magnitude of arsenite-induced poly(A) shortening")

# =========================================================================
# 5. Method 4: Quantile regression / rank-based
# =========================================================================
print(f"\n{'='*90}")
print("Method 4: Poly(A) Residuals After Read Length Regression")
print(f"{'='*90}")

# Regress poly(A) ~ read_length within each condition, then compare residuals
for cond in ['HeLa', 'HeLa-Ars']:
    sub = l1[l1['condition'] == cond].copy()
    # Simple linear regression: poly(A) ~ read_length
    slope, intercept, r, p, se = stats.linregress(sub['read_length'], sub['polya_length'])
    sub['polya_resid'] = sub['polya_length'] - (intercept + slope * sub['read_length'])

    psi_pos_resid = sub[sub['has_psi']]['polya_resid']
    psi_neg_resid = sub[~sub['has_psi']]['polya_resid']

    _, p_mw = stats.mannwhitneyu(psi_pos_resid, psi_neg_resid, alternative='two-sided')
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "ns"

    print(f"\n  {cond}:")
    print(f"    poly(A) ~ read_length: r={r:.3f}, slope={slope:.3f}")
    print(f"    Residual poly(A) psi+: median={psi_pos_resid.median():+.1f}  mean={psi_pos_resid.mean():+.1f}")
    print(f"    Residual poly(A) psi-: median={psi_neg_resid.median():+.1f}  mean={psi_neg_resid.mean():+.1f}")
    print(f"    MW p={p_mw:.2e} ({sig})")

    # Store for comparison
    if cond == 'HeLa':
        hela_resid_pos = psi_pos_resid.median()
        hela_resid_neg = psi_neg_resid.median()
    else:
        ars_resid_pos = psi_pos_resid.median()
        ars_resid_neg = psi_neg_resid.median()

print(f"\n  Residual-based differential shortening:")
print(f"    psi+ residual change: {ars_resid_pos - hela_resid_pos:+.1f}")
print(f"    psi- residual change: {ars_resid_neg - hela_resid_neg:+.1f}")
print(f"    Interaction: {(ars_resid_pos - hela_resid_pos) - (ars_resid_neg - hela_resid_neg):+.1f}")

# =========================================================================
# 6. Dose-response recheck with read length control
# =========================================================================
print(f"\n{'='*90}")
print("Dose-Response Recheck: Psi Density Quartile (Read Length Controlled)")
print(f"{'='*90}")

l1_psi_pos = l1[l1['has_psi']].copy()

# Within each condition, regress poly(A) ~ read_length, get residuals
for cond in ['HeLa', 'HeLa-Ars']:
    sub = l1_psi_pos[l1_psi_pos['condition'] == cond].copy()
    slope, intercept, r, p, se = stats.linregress(sub['read_length'], sub['polya_length'])
    sub['polya_resid'] = sub['polya_length'] - (intercept + slope * sub['read_length'])
    l1_psi_pos.loc[sub.index, 'polya_resid'] = sub['polya_resid']

l1_psi_pos['psi_quartile'] = pd.qcut(l1_psi_pos['psi_per_kb'], q=4,
                                       labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])

print(f"\n  {'Quartile':<12} {'HeLa resid':>12} {'Ars resid':>12} {'Δresid':>10} {'Ars rdLen':>10}")
print(f"  {'-'*58}")

for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
    hela_q = l1_psi_pos[(l1_psi_pos['condition'] == 'HeLa') & (l1_psi_pos['psi_quartile'] == q)]
    ars_q = l1_psi_pos[(l1_psi_pos['condition'] == 'HeLa-Ars') & (l1_psi_pos['psi_quartile'] == q)]
    if len(hela_q) >= 10 and len(ars_q) >= 10:
        h_r = hela_q['polya_resid'].median()
        a_r = ars_q['polya_resid'].median()
        print(f"  {q:<12} {h_r:>+11.1f} {a_r:>+11.1f} {a_r - h_r:>+9.1f} {ars_q['read_length'].mean():>10.0f}")

# Spearman on residuals
ars_psi = l1_psi_pos[l1_psi_pos['condition'] == 'HeLa-Ars']
r_resid, p_resid = stats.spearmanr(ars_psi['psi_per_kb'], ars_psi['polya_resid'])
print(f"\n  HeLa-Ars psi+ reads: psi/kb vs polya_residual")
print(f"    Spearman r={r_resid:.3f}, p={p_resid:.2e}")

hela_psi = l1_psi_pos[l1_psi_pos['condition'] == 'HeLa']
r_h, p_h = stats.spearmanr(hela_psi['psi_per_kb'], hela_psi['polya_resid'])
print(f"  HeLa psi+ reads: psi/kb vs polya_residual")
print(f"    Spearman r={r_h:.3f}, p={p_h:.2e}")

# =========================================================================
# 7. Summary
# =========================================================================
print(f"\n{'='*90}")
print("FINAL SUMMARY")
print(f"{'='*90}")
print("""
  Question: After controlling for read length, does psi status independently
            affect arsenite-induced poly(A) shortening?
""")
