#!/usr/bin/env python3
"""
Sensitivity analysis: restrict to reads with high L1 body coverage
(overlap_length / read_length >= threshold) so most m6A sites are within L1.

Compares m6A/kb, OLS stress interaction, quartile dose-response, and decay zone
across body_fraction thresholds (0.8, 0.9, 0.95) vs. original (all reads).
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
SUMMARY_GLOB = os.path.join(BASE, 'results_group', '*', 'g_summary', '*_L1_summary.tsv')
L1_CACHE_DIR = os.path.join(BASE, 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache')
CTRL_CACHE_DIR = os.path.join(BASE, 'analysis/01_exploration/topic_05_cellline/part3_ctrl_per_read_cache')
OUT_DIR = os.path.join(BASE, 'analysis/01_exploration/topic_09_regulatory_expression')
OUT_FILE = os.path.join(OUT_DIR, 'sensitivity_body_m6a_results.txt')

os.makedirs(OUT_DIR, exist_ok=True)

# Groups to use (those that have Part3 cache)
CACHE_GROUPS = sorted([
    f.replace('_l1_per_read.tsv', '')
    for f in os.listdir(L1_CACHE_DIR) if f.endswith('_l1_per_read.tsv')
])

# HeLa group identification
HELA_NORMAL_GROUPS = [g for g in CACHE_GROUPS if g.startswith('HeLa_')]
HELA_ARS_GROUPS = [g for g in CACHE_GROUPS if g.startswith('HeLa-Ars_')]

print(f"Total cache groups: {len(CACHE_GROUPS)}")
print(f"HeLa normal: {HELA_NORMAL_GROUPS}")
print(f"HeLa Ars: {HELA_ARS_GROUPS}")

# ============================================================
# 1. Load L1 summaries
# ============================================================
print("\n--- Loading L1 summaries ---")
summary_files = glob.glob(SUMMARY_GLOB)
# Filter to groups that have Part3 cache
summary_dfs = []
for f in summary_files:
    group = os.path.basename(os.path.dirname(os.path.dirname(f)))
    if group in CACHE_GROUPS:
        df = pd.read_csv(f, sep='\t', low_memory=False)
        df['group'] = group
        summary_dfs.append(df)

summary = pd.concat(summary_dfs, ignore_index=True)
# Filter PASS only
summary = summary[summary['qc_tag'] == 'PASS'].copy()
summary['overlap_length'] = pd.to_numeric(summary['overlap_length'], errors='coerce')
summary['read_length'] = pd.to_numeric(summary['read_length'], errors='coerce')
summary['polya_length'] = pd.to_numeric(summary['polya_length'], errors='coerce')
summary['body_fraction'] = summary['overlap_length'] / summary['read_length']
print(f"Total PASS L1 reads (all groups): {len(summary)}")

# ============================================================
# 2. Load Part3 L1 per-read cache
# ============================================================
print("\n--- Loading Part3 L1 cache ---")
cache_dfs = []
for group in CACHE_GROUPS:
    fpath = os.path.join(L1_CACHE_DIR, f'{group}_l1_per_read.tsv')
    df = pd.read_csv(fpath, sep='\t')
    df['group'] = group
    cache_dfs.append(df)

cache = pd.concat(cache_dfs, ignore_index=True)
cache['m6a_kb'] = cache['m6a_sites_high'] / cache['read_length'] * 1000
cache['psi_kb'] = cache['psi_sites_high'] / cache['read_length'] * 1000
print(f"Total Part3 L1 cache reads: {len(cache)}")

# ============================================================
# 3. Load control cache for comparison
# ============================================================
print("\n--- Loading Part3 Control cache ---")
ctrl_dfs = []
for group in CACHE_GROUPS:
    fpath = os.path.join(CTRL_CACHE_DIR, f'{group}_ctrl_per_read.tsv')
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, sep='\t')
        df['group'] = group
        ctrl_dfs.append(df)

ctrl_cache = pd.concat(ctrl_dfs, ignore_index=True)
ctrl_cache['m6a_kb'] = ctrl_cache['m6a_sites_high'] / ctrl_cache['read_length'] * 1000
ctrl_m6a_kb_mean = ctrl_cache['m6a_kb'].mean()
ctrl_m6a_kb_median = ctrl_cache['m6a_kb'].median()
print(f"Total Part3 Control cache reads: {len(ctrl_cache)}")
print(f"Control m6A/kb: mean={ctrl_m6a_kb_mean:.3f}, median={ctrl_m6a_kb_median:.3f}")

# ============================================================
# 4. Merge summary + cache
# ============================================================
print("\n--- Merging summary + cache ---")
merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'psi_sites_high', 'm6a_kb', 'psi_kb']],
                       on='read_id', how='inner')
print(f"Merged reads: {len(merged)}")

# ============================================================
# Output collector
# ============================================================
lines = []
def out(s=''):
    lines.append(s)
    print(s)

out("=" * 90)
out("SENSITIVITY ANALYSIS: L1 BODY COVERAGE RESTRICTION ON m6A-Poly(A) RESULTS")
out("=" * 90)
out(f"Date: 2026-02-14")
out(f"Total PASS L1 reads (all groups with Part3 cache): {len(merged)}")
out(f"Control reads: {len(ctrl_cache)}")
out()

# ============================================================
# 4A. Body fraction distribution
# ============================================================
out("--- 4A. Body Fraction Distribution (all reads) ---")
q = merged['body_fraction'].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
out(f"  5th percentile:  {q[0.05]:.3f}")
out(f"  25th percentile: {q[0.25]:.3f}")
out(f"  50th percentile: {q[0.50]:.3f}")
out(f"  75th percentile: {q[0.75]:.3f}")
out(f"  95th percentile: {q[0.95]:.3f}")
out(f"  Mean:            {merged['body_fraction'].mean():.3f}")
out()

# Count reads passing each threshold
thresholds = [0.0, 0.8, 0.9, 0.95]
threshold_labels = ['All', '>=0.80', '>=0.90', '>=0.95']
out("  Threshold | N reads  | % of total")
out("  ----------|----------|----------")
for th, lab in zip(thresholds, threshold_labels):
    n = (merged['body_fraction'] >= th).sum()
    pct = n / len(merged) * 100
    out(f"  {lab:9s} | {n:8d} | {pct:6.1f}%")
out()

# ============================================================
# Helper: run all analyses for a given subset
# ============================================================
def run_analyses(df_sub, label, hela_normal_groups, hela_ars_groups):
    """Run m6A/kb, OLS, quartile, decay zone analyses on given subset."""
    results = {}

    # --- A. m6A/kb ---
    m6a_kb_mean = df_sub['m6a_kb'].mean()
    m6a_kb_median = df_sub['m6a_kb'].median()
    n_reads = len(df_sub)
    results['n_reads'] = n_reads
    results['m6a_kb_mean'] = m6a_kb_mean
    results['m6a_kb_median'] = m6a_kb_median
    results['m6a_kb_vs_ctrl'] = m6a_kb_mean / ctrl_m6a_kb_mean if ctrl_m6a_kb_mean > 0 else np.nan

    # --- HeLa subset ---
    hela_samples_normal = [f"{g}_1" for g in hela_normal_groups]
    hela_samples_ars = [f"{g}_1" for g in hela_ars_groups]

    hela = df_sub[df_sub['sample'].isin(hela_samples_normal + hela_samples_ars)].copy()
    hela['is_stress'] = hela['sample'].isin(hela_samples_ars).astype(int)
    hela_normal = hela[hela['is_stress'] == 0]
    hela_ars = hela[hela['is_stress'] == 1]
    results['n_hela_normal'] = len(hela_normal)
    results['n_hela_ars'] = len(hela_ars)

    # --- B. OLS ---
    if len(hela) > 50 and hela['is_stress'].nunique() == 2:
        y = hela['polya_length'].values
        X = pd.DataFrame({
            'm6a_kb': hela['m6a_kb'].values,
            'is_stress': hela['is_stress'].values,
            'interaction': hela['m6a_kb'].values * hela['is_stress'].values,
            'read_length': hela['read_length'].values
        })
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            idx_inter = list(X.columns).index('interaction')
            results['ols_inter_coef'] = model.params[idx_inter]
            results['ols_inter_pval'] = model.pvalues[idx_inter]
            results['ols_inter_se'] = model.bse[idx_inter]
            # Also get stress main effect
            idx_stress = list(X.columns).index('is_stress')
            results['ols_stress_coef'] = model.params[idx_stress]
            results['ols_stress_pval'] = model.pvalues[idx_stress]
            # m6a_kb main effect
            idx_m6a = list(X.columns).index('m6a_kb')
            results['ols_m6a_coef'] = model.params[idx_m6a]
            results['ols_m6a_pval'] = model.pvalues[idx_m6a]
            results['ols_rsq'] = model.rsquared
            results['ols_nobs'] = int(model.nobs)
        except Exception as e:
            results['ols_inter_coef'] = np.nan
            results['ols_inter_pval'] = np.nan
            results['ols_error'] = str(e)
    else:
        results['ols_inter_coef'] = np.nan
        results['ols_inter_pval'] = np.nan

    # --- C. Quartile dose-response under stress ---
    if len(hela_ars) > 20:
        hela_ars_q = hela_ars.copy()
        hela_ars_q['m6a_quartile'] = pd.qcut(hela_ars_q['m6a_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        q_medians = hela_ars_q.groupby('m6a_quartile')['polya_length'].median()
        if 'Q1' in q_medians.index and 'Q4' in q_medians.index:
            results['ars_q1_polya'] = q_medians['Q1']
            results['ars_q4_polya'] = q_medians['Q4']
            results['ars_q1q4_delta'] = q_medians['Q4'] - q_medians['Q1']
        else:
            results['ars_q1q4_delta'] = np.nan

        # Also for normal
        if len(hela_normal) > 20:
            hela_norm_q = hela_normal.copy()
            hela_norm_q['m6a_quartile'] = pd.qcut(hela_norm_q['m6a_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            q_medians_n = hela_norm_q.groupby('m6a_quartile')['polya_length'].median()
            if 'Q1' in q_medians_n.index and 'Q4' in q_medians_n.index:
                results['norm_q1_polya'] = q_medians_n['Q1']
                results['norm_q4_polya'] = q_medians_n['Q4']
                results['norm_q1q4_delta'] = q_medians_n['Q4'] - q_medians_n['Q1']
            else:
                results['norm_q1q4_delta'] = np.nan
    else:
        results['ars_q1q4_delta'] = np.nan
        results['norm_q1q4_delta'] = np.nan

    # --- D. Decay zone ---
    if len(hela_ars) > 20:
        hela_ars_q = hela_ars.copy()
        hela_ars_q['m6a_quartile'] = pd.qcut(hela_ars_q['m6a_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        hela_ars_q['decay'] = (hela_ars_q['polya_length'] < 30).astype(int)
        decay_by_q = hela_ars_q.groupby('m6a_quartile')['decay'].mean()
        if 'Q1' in decay_by_q.index and 'Q4' in decay_by_q.index:
            results['decay_q1'] = decay_by_q['Q1'] * 100
            results['decay_q4'] = decay_by_q['Q4'] * 100
            results['decay_q1q4_ratio'] = decay_by_q['Q1'] / decay_by_q['Q4'] if decay_by_q['Q4'] > 0 else np.nan
            # Fisher's exact test
            q1_reads = hela_ars_q[hela_ars_q['m6a_quartile'] == 'Q1']
            q4_reads = hela_ars_q[hela_ars_q['m6a_quartile'] == 'Q4']
            table = [[q1_reads['decay'].sum(), len(q1_reads) - q1_reads['decay'].sum()],
                     [q4_reads['decay'].sum(), len(q4_reads) - q4_reads['decay'].sum()]]
            _, p_fisher = stats.fisher_exact(table)
            results['decay_fisher_p'] = p_fisher
        else:
            results['decay_q1'] = np.nan
            results['decay_q4'] = np.nan
    else:
        results['decay_q1'] = np.nan
        results['decay_q4'] = np.nan

    return results


# ============================================================
# 5. Run analyses at each threshold
# ============================================================
all_results = {}
for th, lab in zip(thresholds, threshold_labels):
    if th == 0.0:
        df_sub = merged.copy()
    else:
        df_sub = merged[merged['body_fraction'] >= th].copy()

    out(f"\n{'=' * 70}")
    out(f"  THRESHOLD: {lab} (n={len(df_sub)})")
    out(f"{'=' * 70}")

    res = run_analyses(df_sub, lab, HELA_NORMAL_GROUPS, HELA_ARS_GROUPS)
    all_results[lab] = res

    out(f"  N reads:          {res['n_reads']}")
    out(f"  m6A/kb mean:      {res['m6a_kb_mean']:.3f}")
    out(f"  m6A/kb median:    {res['m6a_kb_median']:.3f}")
    out(f"  m6A/kb vs ctrl:   {res['m6a_kb_vs_ctrl']:.3f}x (ctrl mean={ctrl_m6a_kb_mean:.3f})")
    out()
    out(f"  HeLa normal n:    {res.get('n_hela_normal', 'N/A')}")
    out(f"  HeLa Ars n:       {res.get('n_hela_ars', 'N/A')}")
    out()
    out(f"  OLS stress*m6A/kb:  coef={res.get('ols_inter_coef', np.nan):.3f}, "
        f"p={res.get('ols_inter_pval', np.nan):.2e}, "
        f"SE={res.get('ols_inter_se', np.nan):.3f}")
    out(f"  OLS stress:         coef={res.get('ols_stress_coef', np.nan):.2f}, "
        f"p={res.get('ols_stress_pval', np.nan):.2e}")
    out(f"  OLS m6A/kb:         coef={res.get('ols_m6a_coef', np.nan):.3f}, "
        f"p={res.get('ols_m6a_pval', np.nan):.2e}")
    out(f"  OLS R-squared:      {res.get('ols_rsq', np.nan):.4f}")
    out(f"  OLS N:              {res.get('ols_nobs', 'N/A')}")
    out()
    out(f"  Ars quartile Q1 poly(A): {res.get('ars_q1_polya', np.nan):.1f}")
    out(f"  Ars quartile Q4 poly(A): {res.get('ars_q4_polya', np.nan):.1f}")
    out(f"  Ars Q1->Q4 delta:        {res.get('ars_q1q4_delta', np.nan):+.1f} nt")
    out(f"  Normal Q1->Q4 delta:     {res.get('norm_q1q4_delta', np.nan):+.1f} nt")
    out()
    out(f"  Decay zone (polyA<30) Ars Q1: {res.get('decay_q1', np.nan):.1f}%")
    out(f"  Decay zone (polyA<30) Ars Q4: {res.get('decay_q4', np.nan):.1f}%")
    out(f"  Decay Q1/Q4 ratio:            {res.get('decay_q1q4_ratio', np.nan):.2f}x")
    out(f"  Decay Fisher p:               {res.get('decay_fisher_p', np.nan):.2e}")

# ============================================================
# 6. Summary comparison table
# ============================================================
out("\n\n" + "=" * 90)
out("COMPARISON TABLE: Original vs. Body Coverage Thresholds")
out("=" * 90)
out(f"{'Metric':<40s}  {'All':>10s}  {'>=0.80':>10s}  {'>=0.90':>10s}  {'>=0.95':>10s}  {'Original':>10s}")
out("-" * 90)

# Original reference values from CLAUDE.md
orig = {
    'm6a_kb_mean': 5.17,
    'm6a_kb_vs_ctrl': 1.44,
    'ols_inter_coef': 3.17,
    'ols_inter_pval': 2.7e-05,
    'ars_q1q4_delta': 63.9,
    'decay_q1': 28.5,
    'decay_q4': 15.3,
    'decay_q1q4_ratio': 1.9,
}

def fmt(v, fmt_str='.2f'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:{fmt_str}}'

def fmt_p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.2e}'

metrics = [
    ('N reads (total)',         'n_reads',           'd', None),
    ('N reads (HeLa normal)',   'n_hela_normal',     'd', None),
    ('N reads (HeLa Ars)',      'n_hela_ars',        'd', None),
    ('m6A/kb mean',             'm6a_kb_mean',       '.3f', 5.17),
    ('m6A/kb vs Control',       'm6a_kb_vs_ctrl',    '.2f', 1.44),
    ('OLS stress*m6A coef',     'ols_inter_coef',    '.3f', 3.17),
    ('OLS stress*m6A p-val',    'ols_inter_pval',    'p',   2.7e-05),
    ('OLS R-squared',           'ols_rsq',           '.4f', None),
    ('Ars Q1 poly(A) (nt)',     'ars_q1_polya',      '.1f', None),
    ('Ars Q4 poly(A) (nt)',     'ars_q4_polya',      '.1f', None),
    ('Ars Q1->Q4 delta (nt)',   'ars_q1q4_delta',    '+.1f', 63.9),
    ('Normal Q1->Q4 delta (nt)','norm_q1q4_delta',   '+.1f', 28.9),
    ('Decay Q1 (%) Ars',        'decay_q1',          '.1f', 28.5),
    ('Decay Q4 (%) Ars',        'decay_q4',          '.1f', 15.3),
    ('Decay Q1/Q4 ratio',       'decay_q1q4_ratio',  '.2f', 1.9),
    ('Decay Fisher p',          'decay_fisher_p',     'p',  8.0e-09),
]

for metric_name, key, fmt_s, orig_val in metrics:
    vals = []
    for lab in threshold_labels:
        v = all_results[lab].get(key, np.nan)
        if fmt_s == 'p':
            vals.append(fmt_p(v))
        elif fmt_s == 'd':
            vals.append(f'{int(v):>10d}' if not (isinstance(v, float) and np.isnan(v)) else f'{"N/A":>10s}')
        else:
            vals.append(f'{fmt(v, fmt_s):>10s}')

    if orig_val is not None:
        if fmt_s == 'p':
            orig_str = fmt_p(orig_val)
        else:
            orig_str = fmt(orig_val, fmt_s)
    else:
        orig_str = '-'

    out(f"{metric_name:<40s}  {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}  {orig_str:>10s}")

out()
out("Original values from established analyses (CLAUDE.md):")
out("  m6A/kb L1 = 5.17 vs Ctrl 3.60 = 1.44x")
out("  OLS stress*m6A/kb: coef=+3.17, p=2.7e-05")
out("  Ars Q1->Q4 delta: +63.9 nt")
out("  Decay zone: Q1=28.5%, Q4=15.3%, ratio=1.9x, p=8e-09")

# ============================================================
# 7. Interpretation
# ============================================================
out("\n\n" + "=" * 90)
out("INTERPRETATION")
out("=" * 90)

# Check if body-restricted m6A/kb changes significantly
r_all = all_results['All']
r_80 = all_results['>=0.80']
r_90 = all_results['>=0.90']
r_95 = all_results['>=0.95']

out()
out("1. BODY FRACTION FILTER EFFECT ON m6A/kb:")
out(f"   All reads:    m6A/kb = {r_all['m6a_kb_mean']:.3f} (n={r_all['n_reads']})")
out(f"   >=0.80:       m6A/kb = {r_80['m6a_kb_mean']:.3f} (n={r_80['n_reads']}, "
    f"{r_80['n_reads']/r_all['n_reads']*100:.1f}%)")
out(f"   >=0.90:       m6A/kb = {r_90['m6a_kb_mean']:.3f} (n={r_90['n_reads']}, "
    f"{r_90['n_reads']/r_all['n_reads']*100:.1f}%)")
out(f"   >=0.95:       m6A/kb = {r_95['m6a_kb_mean']:.3f} (n={r_95['n_reads']}, "
    f"{r_95['n_reads']/r_all['n_reads']*100:.1f}%)")
out(f"   Control:      m6A/kb = {ctrl_m6a_kb_mean:.3f}")

# m6A enrichment direction
out()
out("2. m6A ENRICHMENT vs CONTROL:")
for lab in threshold_labels:
    r = all_results[lab]
    out(f"   {lab:9s}: {r['m6a_kb_vs_ctrl']:.2f}x")

# OLS
out()
out("3. OLS STRESS*m6A INTERACTION:")
for lab in threshold_labels:
    r = all_results[lab]
    coef = r.get('ols_inter_coef', np.nan)
    pval = r.get('ols_inter_pval', np.nan)
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    out(f"   {lab:9s}: coef={coef:+.3f}, p={pval:.2e} {sig}")
out(f"   {'Original':9s}: coef=+3.170, p=2.70e-05 ***")

# Quartile
out()
out("4. QUARTILE DOSE-RESPONSE (Ars Q1->Q4 delta):")
for lab in threshold_labels:
    r = all_results[lab]
    d = r.get('ars_q1q4_delta', np.nan)
    out(f"   {lab:9s}: {d:+.1f} nt")
out(f"   {'Original':9s}: +63.9 nt")

# Decay
out()
out("5. DECAY ZONE (Ars Q1 vs Q4):")
for lab in threshold_labels:
    r = all_results[lab]
    out(f"   {lab:9s}: Q1={r.get('decay_q1', np.nan):.1f}%, Q4={r.get('decay_q4', np.nan):.1f}%, "
        f"ratio={r.get('decay_q1q4_ratio', np.nan):.2f}x, p={r.get('decay_fisher_p', np.nan):.2e}")
out(f"   {'Original':9s}: Q1=28.5%, Q4=15.3%, ratio=1.90x, p=8.0e-09")

# Overall conclusion
out()
out("6. OVERALL CONCLUSION:")
# Check if results are robust
ols_robust = (not np.isnan(r_80.get('ols_inter_pval', np.nan)) and
              r_80['ols_inter_pval'] < 0.05)
quartile_robust = (not np.isnan(r_80.get('ars_q1q4_delta', np.nan)) and
                   r_80['ars_q1q4_delta'] > 30)
decay_robust = (not np.isnan(r_80.get('decay_fisher_p', np.nan)) and
                r_80['decay_fisher_p'] < 0.05)

if ols_robust and quartile_robust and decay_robust:
    out("   ROBUST: All key findings survive body_fraction >= 0.80 restriction.")
    out("   m6A-poly(A) protection under stress is NOT driven by flanking non-L1 m6A sites.")
elif ols_robust:
    out("   PARTIALLY ROBUST: OLS interaction survives, but quartile/decay may weaken.")
else:
    out("   WEAKENED: Results do not survive body coverage restriction. Flanking m6A")
    out("   sites may contribute to the observed effect.")

out()
out(f"   The body_fraction >= 0.80 filter retains reads where >= 80% of the read aligns")
out(f"   within the L1 element, ensuring most m6A sites are within L1 body rather than")
out(f"   flanking genomic sequence. If results persist, the m6A-poly(A) protection")
out(f"   relationship is genuinely L1-intrinsic.")

# ============================================================
# Save output
# ============================================================
with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"\n\nResults saved to: {OUT_FILE}")
