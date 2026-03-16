#!/usr/bin/env python3
"""
m6A-poly(A) stratified correlation analysis.

Computes Spearman correlation between m6A/kb and poly(A) length,
stratified by:
  - Condition: Normal vs Stress (HeLa vs HeLa-Ars)
  - Age: Young vs Ancient L1
  - Genomic context: Intronic vs Intergenic
  - All combinations thereof

Also extends to all 10+ cell lines for cross-CL validation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/m6a_polya_stratified'
OUT_DIR.mkdir(exist_ok=True)

# Young L1 subfamilies
YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

###############################################################################
# 1. Load all L1 summary files → get read_id, group, TE_group, gene_id, polya
###############################################################################
print("=== Loading L1 summary files ===")

summary_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'read_length', 'gene_id', 'polya_length',
        'qc_tag', 'TE_group', 'sample'
    ])
    # Extract group from sample (e.g., "HeLa_1_1" → "HeLa_1")
    df['group'] = df['sample'].str.rsplit('_', n=1).str[0]
    summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
print(f"  Total L1 reads from summaries: {len(summary):,}")

# Filter: qc_tag == PASS & polya_length > 0
summary = summary[summary['qc_tag'] == 'PASS'].copy()
summary = summary[summary['polya_length'] > 0].copy()
print(f"  After QC PASS + poly(A) > 0: {len(summary):,}")

# Classify age
def classify_age(gene_id):
    subfamily = gene_id.split('_dup')[0] if '_dup' in gene_id else gene_id
    return 'young' if subfamily in YOUNG_SUBFAMILIES else 'ancient'

summary['l1_age'] = summary['gene_id'].apply(classify_age)

# Classify genomic context (simplify TE_group)
def classify_context(tg):
    if pd.isna(tg):
        return 'other'
    tg = str(tg).lower()
    if 'intronic' in tg:
        return 'intronic'
    elif 'intergenic' in tg:
        return 'intergenic'
    else:
        return 'other'

summary['genomic_context'] = summary['TE_group'].apply(classify_context)

###############################################################################
# 2. Load Part3 per-read cache → get read_id, m6a_sites_high
###############################################################################
print("\n=== Loading Part3 per-read cache ===")

cache_rows = []
for f in sorted(CACHE_DIR.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_rows.append(df)

cache = pd.concat(cache_rows, ignore_index=True)
print(f"  Total cached reads: {len(cache):,}")

# Compute m6a_per_kb
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)

###############################################################################
# 3. Merge summary + cache on read_id
###############################################################################
print("\n=== Merging ===")

merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']],
                       on='read_id', how='inner')
print(f"  Merged reads: {len(merged):,}")

# Define cell line and condition
def get_cellline(group):
    # Strip replicate number: "HeLa_1" → "HeLa", "HeLa-Ars_1" → "HeLa-Ars"
    parts = group.rsplit('_', 1)
    return parts[0]

merged['cellline'] = merged['group'].apply(get_cellline)

# Condition: stress vs normal
merged['condition'] = merged['cellline'].apply(
    lambda x: 'stress' if 'Ars' in x else 'normal'
)

print(f"\n  Cell lines: {sorted(merged['cellline'].unique())}")
print(f"  Reads per condition: {merged['condition'].value_counts().to_dict()}")
print(f"  Reads per age: {merged['l1_age'].value_counts().to_dict()}")
print(f"  Reads per context: {merged['genomic_context'].value_counts().to_dict()}")

###############################################################################
# 4. Correlation function
###############################################################################
def compute_corr(df, label=''):
    """Compute Spearman r between m6a_per_kb and polya_length."""
    n = len(df)
    if n < 20:
        return {'label': label, 'n': n, 'r': np.nan, 'p': np.nan,
                'polya_median': np.nan, 'm6a_kb_median': np.nan}
    r, p = stats.spearmanr(df['m6a_per_kb'], df['polya_length'])
    return {
        'label': label,
        'n': n,
        'r': round(r, 4),
        'p': p,
        'polya_median': round(df['polya_length'].median(), 1),
        'm6a_kb_median': round(df['m6a_per_kb'].median(), 2),
        'm6a_kb_mean': round(df['m6a_per_kb'].mean(), 2),
        'polya_mean': round(df['polya_length'].mean(), 1),
    }

###############################################################################
# 5. Main stratified analysis — HeLa vs HeLa-Ars focus
###############################################################################
print("\n" + "="*80)
print("=== STRATIFIED m6A-poly(A) CORRELATION ===")
print("="*80)

results = []

# --- 5a. Overall by condition ---
print("\n--- By Condition (all L1) ---")
for cond in ['normal', 'stress']:
    # For HeLa comparison, use only HeLa/HeLa-Ars
    hela = merged[merged['cellline'].isin(['HeLa', 'HeLa-Ars'])]
    sub = hela[hela['condition'] == cond]
    r = compute_corr(sub, f'condition={cond}')
    r['stratification'] = 'condition'
    r['condition'] = cond
    r['age'] = 'all'
    r['context'] = 'all'
    results.append(r)
    sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
    print(f"  {cond:8s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}  "
          f"poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")

# --- 5b. By condition × age ---
print("\n--- By Condition × Age ---")
hela = merged[merged['cellline'].isin(['HeLa', 'HeLa-Ars'])]
for cond in ['normal', 'stress']:
    for age in ['young', 'ancient']:
        sub = hela[(hela['condition'] == cond) & (hela['l1_age'] == age)]
        r = compute_corr(sub, f'{cond}_{age}')
        r['stratification'] = 'condition×age'
        r['condition'] = cond
        r['age'] = age
        r['context'] = 'all'
        results.append(r)
        sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
        print(f"  {cond:8s} × {age:8s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}  "
              f"poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")

# --- 5c. By condition × genomic context ---
print("\n--- By Condition × Genomic Context ---")
for cond in ['normal', 'stress']:
    for ctx in ['intronic', 'intergenic']:
        sub = hela[(hela['condition'] == cond) & (hela['genomic_context'] == ctx)]
        r = compute_corr(sub, f'{cond}_{ctx}')
        r['stratification'] = 'condition×context'
        r['condition'] = cond
        r['age'] = 'all'
        r['context'] = ctx
        results.append(r)
        sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
        print(f"  {cond:8s} × {ctx:12s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}  "
              f"poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")

# --- 5d. Full 3-way: condition × age × context ---
print("\n--- By Condition × Age × Genomic Context ---")
for cond in ['normal', 'stress']:
    for age in ['young', 'ancient']:
        for ctx in ['intronic', 'intergenic']:
            sub = hela[(hela['condition'] == cond) &
                       (hela['l1_age'] == age) &
                       (hela['genomic_context'] == ctx)]
            r = compute_corr(sub, f'{cond}_{age}_{ctx}')
            r['stratification'] = 'condition×age×context'
            r['condition'] = cond
            r['age'] = age
            r['context'] = ctx
            results.append(r)
            if r['n'] >= 20:
                sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
                print(f"  {cond:8s} × {age:8s} × {ctx:12s}: n={r['n']:5d}  r={r['r']:+.4f}  "
                      f"p={r['p']:.2e} {sig}  poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")
            else:
                print(f"  {cond:8s} × {age:8s} × {ctx:12s}: n={r['n']:5d}  (too few)")

###############################################################################
# 6. Cross-CL validation — all normal cell lines
###############################################################################
print("\n" + "="*80)
print("=== CROSS-CL m6A-poly(A) CORRELATION (normal only) ===")
print("="*80)

cl_results = []
normal = merged[merged['condition'] == 'normal']

print("\n--- By Cell Line (all L1) ---")
for cl in sorted(normal['cellline'].unique()):
    sub = normal[normal['cellline'] == cl]
    r = compute_corr(sub, f'CL={cl}')
    r['stratification'] = 'cellline'
    r['cellline'] = cl
    r['age'] = 'all'
    r['context'] = 'all'
    cl_results.append(r)
    sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
    print(f"  {cl:12s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}  "
          f"poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")

# Add stress
stress = merged[merged['condition'] == 'stress']
for cl in sorted(stress['cellline'].unique()):
    sub = stress[stress['cellline'] == cl]
    r = compute_corr(sub, f'CL={cl}_stress')
    r['stratification'] = 'cellline'
    r['cellline'] = cl + '_stress'
    r['age'] = 'all'
    r['context'] = 'all'
    cl_results.append(r)
    sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
    print(f"  {cl+'(stress)':12s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}  "
          f"poly(A)={r['polya_median']:.0f}nt  m6A/kb={r['m6a_kb_median']:.1f}")

# Cross-CL by age
print("\n--- By Cell Line × Age ---")
for cl in sorted(normal['cellline'].unique()):
    for age in ['young', 'ancient']:
        sub = normal[(normal['cellline'] == cl) & (normal['l1_age'] == age)]
        r = compute_corr(sub, f'{cl}_{age}')
        r['stratification'] = 'cellline×age'
        r['cellline'] = cl
        r['age'] = age
        r['context'] = 'all'
        cl_results.append(r)
        if r['n'] >= 20:
            sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
            print(f"  {cl:12s} × {age:8s}: n={r['n']:5d}  r={r['r']:+.4f}  p={r['p']:.2e} {sig}")
        else:
            print(f"  {cl:12s} × {age:8s}: n={r['n']:5d}  (too few)")

###############################################################################
# 7. OLS with read length control — HeLa/HeLa-Ars
###############################################################################
print("\n" + "="*80)
print("=== OLS: poly(A) ~ m6A/kb + read_length + condition + age + context + interactions ===")
print("="*80)

from scipy.stats import zscore

hela = merged[merged['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
hela = hela[hela['genomic_context'].isin(['intronic', 'intergenic'])].copy()
hela['read_length_z'] = zscore(hela['read_length'])
hela['is_stress'] = (hela['condition'] == 'stress').astype(int)
hela['is_young'] = (hela['l1_age'] == 'young').astype(int)
hela['is_intergenic'] = (hela['genomic_context'] == 'intergenic').astype(int)

# Full OLS
import statsmodels.api as sm

X = hela[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young', 'is_intergenic']].copy()
# Interactions
X['stress_x_m6a'] = X['is_stress'] * X['m6a_per_kb']
X['young_x_m6a'] = X['is_young'] * X['m6a_per_kb']
X['intergenic_x_m6a'] = X['is_intergenic'] * X['m6a_per_kb']
X['stress_x_young'] = X['is_stress'] * X['is_young']
X = sm.add_constant(X)
y = hela['polya_length']

model = sm.OLS(y, X).fit()
print(model.summary2().tables[1].to_string())

# Save OLS results
ols_df = model.summary2().tables[1].reset_index()
ols_df.columns = ['variable', 'coef', 'se', 't', 'p', 'ci_low', 'ci_high']
ols_df.to_csv(OUT_DIR / 'm6a_polya_ols_full.tsv', sep='\t', index=False)

###############################################################################
# 8. Partial correlation (controlling read length)
###############################################################################
print("\n" + "="*80)
print("=== PARTIAL CORRELATION (m6A/kb ~ poly(A) | read_length) ===")
print("="*80)

def partial_spearman(df, x_col, y_col, z_col):
    """Spearman partial correlation controlling for z."""
    n = len(df)
    if n < 30:
        return np.nan, np.nan, n
    # Rank-based partial correlation
    rx = stats.spearmanr(df[x_col], df[z_col])[0]
    ry = stats.spearmanr(df[y_col], df[z_col])[0]
    rxy = stats.spearmanr(df[x_col], df[y_col])[0]
    # Partial r
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom == 0:
        return np.nan, np.nan, n
    r_partial = (rxy - rx * ry) / denom
    # Approximate p-value (t-test)
    t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2))
    p_val = 2 * stats.t.sf(abs(t_stat), df=n-3)
    return r_partial, p_val, n

partial_results = []
hela_full = merged[merged['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()

print("\n--- Partial r (controlling read_length) by Condition × Age × Context ---")
for cond in ['normal', 'stress']:
    for age in ['all', 'young', 'ancient']:
        for ctx in ['all', 'intronic', 'intergenic']:
            sub = hela_full[hela_full['condition'] == cond]
            if age != 'all':
                sub = sub[sub['l1_age'] == age]
            if ctx != 'all':
                sub = sub[sub['genomic_context'] == ctx]

            r_partial, p_val, n = partial_spearman(sub, 'm6a_per_kb', 'polya_length', 'read_length')

            row = {
                'condition': cond, 'age': age, 'context': ctx,
                'n': n, 'r_partial': round(r_partial, 4) if not np.isnan(r_partial) else np.nan,
                'p': p_val
            }
            partial_results.append(row)

            if not np.isnan(r_partial):
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                print(f"  {cond:8s} × {age:8s} × {ctx:12s}: n={n:5d}  r_partial={r_partial:+.4f}  "
                      f"p={p_val:.2e} {sig}")
            else:
                print(f"  {cond:8s} × {age:8s} × {ctx:12s}: n={n:5d}  (insufficient)")

###############################################################################
# 9. Save all results
###############################################################################
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / 'm6a_polya_hela_stratified.tsv', sep='\t', index=False)

cl_df = pd.DataFrame(cl_results)
cl_df.to_csv(OUT_DIR / 'm6a_polya_crosscl.tsv', sep='\t', index=False)

partial_df = pd.DataFrame(partial_results)
partial_df.to_csv(OUT_DIR / 'm6a_polya_partial_corr.tsv', sep='\t', index=False)

print(f"\n=== Saved to {OUT_DIR} ===")
print(f"  m6a_polya_hela_stratified.tsv")
print(f"  m6a_polya_crosscl.tsv")
print(f"  m6a_polya_partial_corr.tsv")
print(f"  m6a_polya_ols_full.tsv")
