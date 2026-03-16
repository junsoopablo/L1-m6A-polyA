#!/usr/bin/env python3
"""
Final OLS: poly(A) ~ stress × chromatin × m6A/kb
Tests whether heterochromatin provides independent protection from m6A.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'

# Load annotated data
df = pd.read_csv(OUT_DIR / 'l1_chromhmm_annotated.tsv', sep='\t')
RESULTS = BASE / 'results_group'
rl_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    s = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length'])
    rl_rows.append(s)
rl = pd.concat(rl_rows, ignore_index=True).drop_duplicates('read_id')
df = df.merge(rl, on='read_id', how='left')

# HeLa + HeLa-Ars ancient only
ancient = df[df['l1_age'] == 'ancient'].copy()
hela = ancient[ancient['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
hela['is_stress'] = (hela['condition'] == 'stress').astype(int)

print(f"HeLa + HeLa-Ars ancient: {len(hela):,}")
print(f"  Stress: {hela['is_stress'].sum():,}, Normal: {(~hela['is_stress'].astype(bool)).sum():,}")

###############################################################################
# Model 1: Original (stress × m6A/kb)
###############################################################################
print("\n" + "="*70)
print("MODEL 1: poly(A) ~ stress × m6A/kb (baseline)")
print("="*70)

m1 = ols('polya_length ~ is_stress * m6a_per_kb', data=hela).fit()
print(f"N={m1.nobs:.0f}, R²={m1.rsquared:.4f}")
for p in m1.params.index:
    sig = '***' if m1.pvalues[p]<0.001 else '**' if m1.pvalues[p]<0.01 else '*' if m1.pvalues[p]<0.05 else 'ns'
    print(f"  {p:45s}: {m1.params[p]:+8.2f} (p={m1.pvalues[p]:.2e}) {sig}")

###############################################################################
# Model 2: + chromatin categories (3-way: regulatory, heterochromatin, other)
###############################################################################
print("\n" + "="*70)
print("MODEL 2: + is_regulatory + is_heterochromatin (main effects)")
print("="*70)

hela['is_regulatory'] = hela['chromhmm_group'].isin(['Enhancer', 'Promoter']).astype(int)
hela['is_het'] = (hela['chromhmm_group'] == 'Heterochromatin').astype(int)

m2 = ols('polya_length ~ is_stress * m6a_per_kb + is_regulatory + is_het', data=hela).fit()
print(f"N={m2.nobs:.0f}, R²={m2.rsquared:.4f}, ΔR² vs M1={m2.rsquared-m1.rsquared:.4f}")
for p in m2.params.index:
    sig = '***' if m2.pvalues[p]<0.001 else '**' if m2.pvalues[p]<0.01 else '*' if m2.pvalues[p]<0.05 else 'ns'
    print(f"  {p:45s}: {m2.params[p]:+8.2f} (p={m2.pvalues[p]:.2e}) {sig}")

###############################################################################
# Model 3: Full interactions (stress × regulatory, stress × het, stress × m6A)
###############################################################################
print("\n" + "="*70)
print("MODEL 3: stress × regulatory + stress × het + stress × m6A/kb")
print("="*70)

m3 = ols('polya_length ~ is_stress * m6a_per_kb + is_stress * is_regulatory + is_stress * is_het',
         data=hela).fit()
print(f"N={m3.nobs:.0f}, R²={m3.rsquared:.4f}, ΔR² vs M1={m3.rsquared-m1.rsquared:.4f}")
for p in m3.params.index:
    sig = '***' if m3.pvalues[p]<0.001 else '**' if m3.pvalues[p]<0.01 else '*' if m3.pvalues[p]<0.05 else 'ns'
    print(f"  {p:45s}: {m3.params[p]:+8.2f} (p={m3.pvalues[p]:.2e}) {sig}")

###############################################################################
# Model 4: + read_length control
###############################################################################
print("\n" + "="*70)
print("MODEL 4: M3 + read_length (confounder control)")
print("="*70)

m4 = ols('polya_length ~ is_stress * m6a_per_kb + is_stress * is_regulatory + '
         'is_stress * is_het + read_length', data=hela).fit()
print(f"N={m4.nobs:.0f}, R²={m4.rsquared:.4f}, ΔR² vs M1={m4.rsquared-m1.rsquared:.4f}")
for p in m4.params.index:
    sig = '***' if m4.pvalues[p]<0.001 else '**' if m4.pvalues[p]<0.01 else '*' if m4.pvalues[p]<0.05 else 'ns'
    print(f"  {p:45s}: {m4.params[p]:+8.2f} (p={m4.pvalues[p]:.2e}) {sig}")

###############################################################################
# Model 5: 4-way full model (stress × reg × m6A, stress × het × m6A)
###############################################################################
print("\n" + "="*70)
print("MODEL 5: Full 3-way interactions")
print("="*70)

m5 = ols('polya_length ~ is_stress * m6a_per_kb * is_regulatory + '
         'is_stress * m6a_per_kb * is_het + read_length', data=hela).fit()
print(f"N={m5.nobs:.0f}, R²={m5.rsquared:.4f}")
for p in m5.params.index:
    sig = '***' if m5.pvalues[p]<0.001 else '**' if m5.pvalues[p]<0.01 else '*' if m5.pvalues[p]<0.05 else 'ns'
    print(f"  {p:45s}: {m5.params[p]:+8.2f} (p={m5.pvalues[p]:.2e}) {sig}")

###############################################################################
# Summary: Key test — is heterochromatin immunity independent of m6A?
###############################################################################
print("\n" + "="*70)
print("SUMMARY: Independence Tests")
print("="*70)

# Het vs non-het: m6A/kb comparison
het = hela[hela['is_het'] == 1]
nonhet = hela[hela['is_het'] == 0]
print(f"\n  Het m6A/kb: median={het['m6a_per_kb'].median():.2f} (n={len(het)})")
print(f"  Non-het m6A/kb: median={nonhet['m6a_per_kb'].median():.2f} (n={len(nonhet)})")
_, p = stats.mannwhitneyu(het['m6a_per_kb'], nonhet['m6a_per_kb'])
print(f"  MW p={p:.2e}")

# Regulatory vs non-reg: m6A/kb comparison
reg = hela[hela['is_regulatory'] == 1]
nonreg = hela[(hela['is_regulatory'] == 0) & (hela['is_het'] == 0)]
print(f"\n  Regulatory m6A/kb: median={reg['m6a_per_kb'].median():.2f} (n={len(reg)})")
print(f"  Other m6A/kb: median={nonreg['m6a_per_kb'].median():.2f} (n={len(nonreg)})")

# Het: stress vs normal poly(A)
het_h = het[het['is_stress'] == 0]
het_a = het[het['is_stress'] == 1]
print(f"\n  Het Normal: poly(A)={het_h['polya_length'].median():.0f} (n={len(het_h)})")
print(f"  Het Stress: poly(A)={het_a['polya_length'].median():.0f} (n={len(het_a)})")
if len(het_h) >= 5 and len(het_a) >= 5:
    _, p = stats.mannwhitneyu(het_h['polya_length'], het_a['polya_length'])
    print(f"  MW p={p:.2e}")

# m6A-matched het vs non-het under stress
print("\n  m6A-matched heterochromatin test (stress only):")
stress_all = hela[hela['is_stress'] == 1]
stress_het = stress_all[stress_all['is_het'] == 1]
stress_nonhet = stress_all[stress_all['is_het'] == 0]
if len(stress_het) >= 10:
    # Match m6A/kb distribution
    het_q25, het_q75 = stress_het['m6a_per_kb'].quantile([0.25, 0.75])
    matched_nonhet = stress_nonhet[
        (stress_nonhet['m6a_per_kb'] >= het_q25) &
        (stress_nonhet['m6a_per_kb'] <= het_q75)
    ]
    print(f"  Het stress: poly(A)={stress_het['polya_length'].median():.0f} "
          f"(n={len(stress_het)}, m6A/kb={stress_het['m6a_per_kb'].median():.2f})")
    print(f"  Matched non-het stress: poly(A)={matched_nonhet['polya_length'].median():.0f} "
          f"(n={len(matched_nonhet)}, m6A/kb={matched_nonhet['m6a_per_kb'].median():.2f})")
    _, p = stats.mannwhitneyu(stress_het['polya_length'], matched_nonhet['polya_length'])
    print(f"  MW p={p:.2e}")

###############################################################################
# Save model summary
###############################################################################
print("\n=== Saving ===")
with open(OUT_DIR / 'ols_model_comparison.txt', 'w') as f:
    for name, model in [('M1_baseline', m1), ('M2_main_effects', m2),
                         ('M3_interactions', m3), ('M4_rl_control', m4),
                         ('M5_full_3way', m5)]:
        f.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
        f.write(model.summary().as_text())
        f.write('\n')
print("  Saved: ols_model_comparison.txt")
print("\n=== DONE ===")
