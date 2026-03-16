#!/usr/bin/env python3
"""
Confounder check for ChromHMM × L1 m6A analysis.

Key confounders to verify:
1. Read length by chromatin state (longer reads → more m6A sites/kb artifact?)
2. Intronic vs intergenic distribution by state
3. L1 subfamily composition by state
4. Normal conditions only (exclude HeLa-Ars for m6A comparisons)
5. Read length-matched m6A comparison
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'

###############################################################################
# Load annotated data
###############################################################################
print("=== Loading annotated data ===")
df = pd.read_csv(OUT_DIR / 'l1_chromhmm_annotated.tsv', sep='\t')
print(f"  Total reads: {len(df):,}")

# We also need read_length from the original summary
RESULTS = BASE / 'results_group'
summary_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    s = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length'])
    summary_rows.append(s)
rl = pd.concat(summary_rows, ignore_index=True).drop_duplicates('read_id')

df = df.merge(rl, on='read_id', how='left')

ancient = df[df['l1_age'] == 'ancient'].copy()
print(f"  Ancient L1: {len(ancient):,}")

###############################################################################
# 1. Read length by chromatin state
###############################################################################
print("\n--- 1. Read Length by Chromatin State (Ancient L1) ---")
for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    sub = ancient[ancient['chromhmm_group'] == grp]
    if len(sub) >= 10:
        print(f"  {grp:20s}: n={len(sub):5,}, median RL={sub['read_length'].median():.0f}bp, "
              f"mean={sub['read_length'].mean():.0f}bp")

kw_stat, kw_p = stats.kruskal(
    *[ancient[ancient['chromhmm_group']==g]['read_length'].values
      for g in ['Promoter','Enhancer','Transcribed','Repressed','Heterochromatin','Quiescent']
      if len(ancient[ancient['chromhmm_group']==g]) >= 10]
)
print(f"  KW test: H={kw_stat:.1f}, p={kw_p:.2e}")

###############################################################################
# 2. m6A/kb by state — NORMAL conditions only
###############################################################################
print("\n--- 2. m6A/kb by State (Normal Conditions Only, Ancient) ---")
anc_normal = ancient[ancient['condition'] == 'normal']

for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    sub = anc_normal[anc_normal['chromhmm_group'] == grp]
    if len(sub) >= 10:
        print(f"  {grp:20s}: n={len(sub):5,}, median m6A/kb={sub['m6a_per_kb'].median():.2f}, "
              f"median RL={sub['read_length'].median():.0f}bp")

###############################################################################
# 3. Read-length matched m6A comparison
###############################################################################
print("\n--- 3. Read-Length Matched m6A Comparison ---")
print("\nAncient L1, normal conditions, by read length bin:")

anc_normal['rl_bin'] = pd.cut(anc_normal['read_length'],
                                bins=[0, 500, 1000, 2000, 5000, 100000],
                                labels=['0-500', '500-1K', '1K-2K', '2K-5K', '5K+'])

for rl_bin in ['0-500', '500-1K', '1K-2K', '2K-5K']:
    rl_sub = anc_normal[anc_normal['rl_bin'] == rl_bin]
    if len(rl_sub) < 50:
        continue
    print(f"\n  Read length {rl_bin} (n={len(rl_sub):,}):")
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Quiescent']:
        gs = rl_sub[rl_sub['chromhmm_group'] == grp]
        if len(gs) >= 10:
            print(f"    {grp:20s}: n={len(gs):4,}, median m6A/kb={gs['m6a_per_kb'].median():.2f}")

###############################################################################
# 4. Genomic context distribution by state
###############################################################################
print("\n--- 4. Genomic Context by Chromatin State (Ancient) ---")
for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    sub = ancient[ancient['chromhmm_group'] == grp]
    if len(sub) < 10:
        continue
    ctx = sub['genomic_context'].value_counts()
    total = len(sub)
    ctx_str = ', '.join([f"{c}={n}({100*n/total:.0f}%)" for c, n in ctx.items()])
    print(f"  {grp:20s}: {ctx_str}")

###############################################################################
# 5. L1 subfamily composition by state
###############################################################################
print("\n--- 5. Top L1 Subfamilies by Chromatin State (Ancient) ---")
ancient['subfamily'] = ancient['gene_id'].str.split('_dup').str[0]

for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Quiescent']:
    sub = ancient[ancient['chromhmm_group'] == grp]
    if len(sub) < 10:
        continue
    top_sf = sub['subfamily'].value_counts().head(5)
    top_str = ', '.join([f"{sf}={n}" for sf, n in top_sf.items()])
    print(f"  {grp:20s} (n={len(sub):,}): {top_str}")

###############################################################################
# 6. Arsenite shortening: read-length matched
###############################################################################
print("\n--- 6. Arsenite Shortening by State: Read-Length Matched ---")
print("\nAncient L1, HeLa vs HeLa-Ars, 500-2000bp reads only:")

hela_ancient = ancient[(ancient['cellline'] == 'HeLa') &
                        (ancient['read_length'] >= 500) &
                        (ancient['read_length'] <= 2000)]
ars_ancient = ancient[(ancient['cellline'] == 'HeLa-Ars') &
                       (ancient['read_length'] >= 500) &
                       (ancient['read_length'] <= 2000)]

print(f"  HeLa 500-2K: {len(hela_ancient):,}, HeLa-Ars 500-2K: {len(ars_ancient):,}")

for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    h = hela_ancient[hela_ancient['chromhmm_group'] == grp]['polya_length']
    ha = ars_ancient[ars_ancient['chromhmm_group'] == grp]['polya_length']
    if len(h) >= 5 and len(ha) >= 5:
        delta = ha.median() - h.median()
        _, p = stats.mannwhitneyu(h, ha, alternative='two-sided')
        print(f"  {grp:20s}: HeLa={h.median():.0f}nt (n={len(h)}), "
              f"Ars={ha.median():.0f}nt (n={len(ha)}), "
              f"Δ={delta:+.1f}nt, p={p:.2e}")

###############################################################################
# 7. OLS: poly(A) ~ condition * chromhmm_group * m6A/kb
###############################################################################
print("\n--- 7. OLS Regression: poly(A) ~ condition × regulatory × m6A/kb ---")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # HeLa + HeLa-Ars ancient only
    hela_all = ancient[ancient['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
    hela_all['is_stress'] = (hela_all['condition'] == 'stress').astype(int)
    hela_all['is_regulatory'] = hela_all['chromhmm_group'].isin(['Enhancer', 'Promoter']).astype(int)

    # OLS with interaction
    model = ols('polya_length ~ is_stress * is_regulatory * m6a_per_kb', data=hela_all).fit()
    print(f"\n  N={len(hela_all):,}")
    print(f"  R²={model.rsquared:.4f}")
    print("\n  Coefficients:")
    for param in model.params.index:
        coef = model.params[param]
        p = model.pvalues[param]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {param:50s}: {coef:+8.2f} (p={p:.2e}) {sig}")

except ImportError:
    print("  statsmodels not available, skipping OLS")

###############################################################################
# 8. Baseline poly(A) by state (normal conditions)
###############################################################################
print("\n--- 8. Baseline Poly(A) by State (Normal, Ancient) ---")
for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    sub = anc_normal[anc_normal['chromhmm_group'] == grp]
    if len(sub) >= 10:
        print(f"  {grp:20s}: median poly(A)={sub['polya_length'].median():.0f}nt, "
              f"mean={sub['polya_length'].mean():.0f}nt (n={len(sub):,})")

print("\n=== DONE ===")
