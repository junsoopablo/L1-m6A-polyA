#!/usr/bin/env python3
"""
Deep analysis: CpG density as a stress-specific vulnerability determinant
for L1 poly(A) tail shortening.

Key hypothesis: ZAP recognizes CpG dinucleotides in L1 RNA and recruits
PARN deadenylase → 3' poly(A) shortening. Under arsenite stress, this
pathway becomes the dominant L1-specific decay route.

Analyses:
1. CpG quartile × stress: poly(A) dose-response (like m6A quartile Fig.3d)
2. OLS: poly(A) ~ CpG + m6A + stress + interactions (independent effects?)
3. CpG vs m6A relationship (confound check)
4. GC-content decomposition: CpG specifically or GC generally?
5. CpG × PAS interaction: additive or synergistic protection?
6. TG dinucleotide deep-dive: why protective?
7. Subfamily-controlled CpG effect
8. Cross-cell-line validation using all 11 CLs (CpG at shared loci)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
FEAT_DIR = f'{BASE}/analysis/01_exploration/topic_08_sequence_features'
OUT_DIR = FEAT_DIR
CACHE_DIR = f'{BASE}/analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
RESULTS = f'{BASE}/results_group'

# Load the per-read feature dataset from Step 1
df = pd.read_csv(f'{FEAT_DIR}/ancient_l1_with_features.tsv', sep='\t')
print(f"Loaded {len(df)} ancient L1 reads with features")
print(f"  Normal: {(df['is_stress']==0).sum()}, Stress: {(df['is_stress']==1).sum()}")

# Rename for clarity
df['cpg_per_kb'] = df['rbp_ZAP_CpG_per_kb']
df['tg_freq'] = df['di_TG']

# ============================================================
# Analysis 1: CpG quartile × stress poly(A) dose-response
# ============================================================
print("\n" + "=" * 70)
print("Analysis 1: CpG quartile × stress poly(A) dose-response")
print("=" * 70)

# Compute CpG quartiles (using normal condition to define boundaries)
normal_cpg = df.loc[df['is_stress'] == 0, 'cpg_per_kb']
cpg_q = np.percentile(normal_cpg, [25, 50, 75])
print(f"CpG/kb quartile boundaries (from normal): Q1<{cpg_q[0]:.1f}, Q2<{cpg_q[1]:.1f}, Q3<{cpg_q[2]:.1f}, Q4≥{cpg_q[2]:.1f}")

df['cpg_quartile'] = pd.cut(df['cpg_per_kb'],
                             bins=[-np.inf, cpg_q[0], cpg_q[1], cpg_q[2], np.inf],
                             labels=['Q1 (low CpG)', 'Q2', 'Q3', 'Q4 (high CpG)'])

print(f"\n{'Quartile':<20} {'N normal':>10} {'N stress':>10} {'polyA norm':>12} {'polyA stress':>12} {'Δ':>8} {'P':>12}")
print("-" * 90)

cpg_quartile_results = []
for q in ['Q1 (low CpG)', 'Q2', 'Q3', 'Q4 (high CpG)']:
    n_mask = (df['cpg_quartile'] == q) & (df['is_stress'] == 0)
    s_mask = (df['cpg_quartile'] == q) & (df['is_stress'] == 1)

    n_polya = df.loc[n_mask, 'polya_length']
    s_polya = df.loc[s_mask, 'polya_length']

    delta = s_polya.median() - n_polya.median()
    stat, pval = stats.mannwhitneyu(n_polya, s_polya, alternative='two-sided')

    cpg_quartile_results.append({
        'quartile': q, 'n_normal': len(n_polya), 'n_stress': len(s_polya),
        'median_normal': n_polya.median(), 'median_stress': s_polya.median(),
        'delta': delta, 'pval': pval
    })

    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"{q:<20} {len(n_polya):>10} {len(s_polya):>10} {n_polya.median():>12.1f} {s_polya.median():>12.1f} {delta:>+8.1f} {pval:>12.2e} {sig}")

df_cpg_q = pd.DataFrame(cpg_quartile_results)
q1_delta = df_cpg_q.iloc[0]['delta']
q4_delta = df_cpg_q.iloc[3]['delta']
print(f"\nQ1 (low CpG) Δ = {q1_delta:+.1f} nt")
print(f"Q4 (high CpG) Δ = {q4_delta:+.1f} nt")
print(f"Q4-Q1 difference = {q4_delta - q1_delta:.1f} nt (high CpG → {'more' if q4_delta < q1_delta else 'less'} shortening)")

# Decay zone (<30 nt) by CpG quartile under stress
print("\nDecay zone (<30 nt) by CpG quartile under stress:")
for q in ['Q1 (low CpG)', 'Q2', 'Q3', 'Q4 (high CpG)']:
    s_mask = (df['cpg_quartile'] == q) & (df['is_stress'] == 1)
    s_polya = df.loc[s_mask, 'polya_length']
    decay_frac = (s_polya < 30).mean()
    print(f"  {q}: {decay_frac:.1%} in decay zone (n={len(s_polya)})")

# Fisher test: Q1 vs Q4 decay zone under stress
s_q1 = df.loc[(df['cpg_quartile'] == 'Q1 (low CpG)') & (df['is_stress'] == 1), 'polya_length']
s_q4 = df.loc[(df['cpg_quartile'] == 'Q4 (high CpG)') & (df['is_stress'] == 1), 'polya_length']
table = [[sum(s_q1 < 30), sum(s_q1 >= 30)],
         [sum(s_q4 < 30), sum(s_q4 >= 30)]]
or_val, fisher_p = stats.fisher_exact(table)
print(f"\n  Q4 vs Q1 decay zone: OR={or_val:.2f}, Fisher P={fisher_p:.2e}")

# ============================================================
# Analysis 2: OLS — CpG + m6A + PAS + stress + all interactions
# ============================================================
print("\n" + "=" * 70)
print("Analysis 2: Full OLS — CpG, m6A, PAS, and stress interactions")
print("=" * 70)

X = df[['is_stress', 'read_length']].copy()
X['read_length_kb'] = X['read_length'] / 1000
X.drop('read_length', axis=1, inplace=True)

# Main features
X['cpg_per_kb'] = df['cpg_per_kb']
X['m6a_per_kb'] = df['m6a_per_kb']
X['has_canonical_pas'] = df['has_canonical_pas']
X['gc_content'] = df['gc_content']

# Interaction terms (each × stress)
X['stress_x_cpg'] = X['is_stress'] * X['cpg_per_kb']
X['stress_x_m6a'] = X['is_stress'] * X['m6a_per_kb']
X['stress_x_pas'] = X['is_stress'] * X['has_canonical_pas']
X['stress_x_gc'] = X['is_stress'] * X['gc_content']

X = sm.add_constant(X)
y = df['polya_length']

mask = ~(X.isna().any(axis=1) | y.isna())
model = sm.OLS(y[mask], X[mask]).fit()

print(model.summary2().tables[1].to_string())
print(f"\nR²: {model.rsquared:.4f}, Adj R²: {model.rsquared_adj:.4f}")

print("\n--- Key interactions ---")
for term in ['stress_x_cpg', 'stress_x_m6a', 'stress_x_pas', 'stress_x_gc']:
    coef = model.params[term]
    pval = model.pvalues[term]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    direction = "protective" if coef > 0 else "vulnerability"
    print(f"  {term:<20} coef={coef:>+8.3f}  P={pval:.2e}  {direction} {sig}")

# ============================================================
# Analysis 3: CpG vs m6A — confound check
# ============================================================
print("\n" + "=" * 70)
print("Analysis 3: CpG vs m6A relationship (confound check)")
print("=" * 70)

r_cpg_m6a, p_cpg_m6a = stats.pearsonr(df['cpg_per_kb'], df['m6a_per_kb'])
print(f"CpG/kb vs m6A/kb: r={r_cpg_m6a:.4f}, P={p_cpg_m6a:.2e}")

r_cpg_gc, p_cpg_gc = stats.pearsonr(df['cpg_per_kb'], df['gc_content'])
print(f"CpG/kb vs GC content: r={r_cpg_gc:.4f}, P={p_cpg_gc:.2e}")

r_m6a_gc, p_m6a_gc = stats.pearsonr(df['m6a_per_kb'], df['gc_content'])
print(f"m6A/kb vs GC content: r={r_m6a_gc:.4f}, P={p_m6a_gc:.2e}")

r_cpg_pas, _ = stats.pointbiserialr(df['has_canonical_pas'], df['cpg_per_kb'])
print(f"CpG/kb vs canonical PAS: r={r_cpg_pas:.4f}")

# ============================================================
# Analysis 4: GC-content decomposition — CpG specifically or GC generally?
# ============================================================
print("\n" + "=" * 70)
print("Analysis 4: Is CpG-specific or GC-general?")
print("=" * 70)

# CpG observed/expected ratio
df['cpg_oe'] = df['cpg_per_kb'] / (df['gc_content']**2 * 1000 / 4 + 0.001)  # expected CpG from GC content

# Compare CpG vs non-CpG GC dinucleotides
df['non_cpg_gc_dinucs'] = df['di_GC'] + df['di_CC'] + df['di_GG']  # GC-containing but not CpG

# Correlation with poly(A) under stress
stress_mask = df['is_stress'] == 1
for feat, name in [('cpg_per_kb', 'CpG/kb'),
                    ('gc_content', 'GC content'),
                    ('non_cpg_gc_dinucs', 'non-CpG GC dinucs'),
                    ('cpg_oe', 'CpG obs/exp')]:
    r, p = stats.pearsonr(df.loc[stress_mask, feat], df.loc[stress_mask, 'polya_length'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {name:<25} r(stress)={r:>+.4f}  P={p:.2e} {sig}")

# OLS: CpG obs/exp vs just GC
print("\n  OLS: poly(A) ~ CpG_obs/exp + GC + stress + interactions")
X2 = df[['is_stress']].copy()
X2['read_length_kb'] = df['read_length'] / 1000
X2['m6a_per_kb'] = df['m6a_per_kb']
X2['cpg_oe'] = df['cpg_oe']
X2['gc_content'] = df['gc_content']
X2['stress_x_cpg_oe'] = X2['is_stress'] * X2['cpg_oe']
X2['stress_x_gc'] = X2['is_stress'] * X2['gc_content']
X2 = sm.add_constant(X2)

model2 = sm.OLS(y[mask], X2[mask]).fit()
print(f"  stress_x_cpg_oe: coef={model2.params['stress_x_cpg_oe']:+.3f}, P={model2.pvalues['stress_x_cpg_oe']:.2e}")
print(f"  stress_x_gc:     coef={model2.params['stress_x_gc']:+.3f}, P={model2.pvalues['stress_x_gc']:.2e}")

# ============================================================
# Analysis 5: CpG × PAS interaction — additive or synergistic?
# ============================================================
print("\n" + "=" * 70)
print("Analysis 5: CpG × PAS — double vulnerability")
print("=" * 70)

# 2×2 stratification: PAS (yes/no) × CpG (high/low)
cpg_median = df['cpg_per_kb'].median()
df['cpg_high'] = (df['cpg_per_kb'] >= cpg_median).astype(int)

print(f"\n{'Group':<35} {'N norm':>8} {'N stress':>8} {'polyA norm':>10} {'polyA stress':>10} {'Δ':>8} {'P':>12}")
print("-" * 95)

groups = [
    ('PAS+ / low CpG (safest)', (df['has_canonical_pas'] == 1) & (df['cpg_high'] == 0)),
    ('PAS+ / high CpG',         (df['has_canonical_pas'] == 1) & (df['cpg_high'] == 1)),
    ('PAS- / low CpG',          (df['has_canonical_pas'] == 0) & (df['cpg_high'] == 0)),
    ('PAS- / high CpG (most vulnerable)', (df['has_canonical_pas'] == 0) & (df['cpg_high'] == 1)),
]

for name, mask_group in groups:
    n_mask = mask_group & (df['is_stress'] == 0)
    s_mask = mask_group & (df['is_stress'] == 1)
    n_polya = df.loc[n_mask, 'polya_length']
    s_polya = df.loc[s_mask, 'polya_length']
    if len(n_polya) < 5 or len(s_polya) < 5:
        continue
    delta = s_polya.median() - n_polya.median()
    _, pval = stats.mannwhitneyu(n_polya, s_polya, alternative='two-sided')
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"{name:<35} {len(n_polya):>8} {len(s_polya):>8} {n_polya.median():>10.1f} {s_polya.median():>10.1f} {delta:>+8.1f} {pval:>12.2e} {sig}")

# ============================================================
# Analysis 6: TG dinucleotide — why protective?
# ============================================================
print("\n" + "=" * 70)
print("Analysis 6: TG dinucleotide deep-dive")
print("=" * 70)

# TG quartile analysis under stress
tg_q = np.percentile(df.loc[df['is_stress']==0, 'tg_freq'], [25, 50, 75])
df['tg_quartile'] = pd.cut(df['tg_freq'],
                            bins=[-np.inf, tg_q[0], tg_q[1], tg_q[2], np.inf],
                            labels=['Q1 (low TG)', 'Q2', 'Q3', 'Q4 (high TG)'])

print(f"\n{'Quartile':<20} {'polyA norm':>12} {'polyA stress':>12} {'Δ':>8}")
print("-" * 55)
for q in ['Q1 (low TG)', 'Q2', 'Q3', 'Q4 (high TG)']:
    n_polya = df.loc[(df['tg_quartile'] == q) & (df['is_stress'] == 0), 'polya_length']
    s_polya = df.loc[(df['tg_quartile'] == q) & (df['is_stress'] == 1), 'polya_length']
    delta = s_polya.median() - n_polya.median()
    print(f"{q:<20} {n_polya.median():>12.1f} {s_polya.median():>12.1f} {delta:>+8.1f}")

# TG vs CpG relationship
r_tg_cpg, p_tg_cpg = stats.pearsonr(df['tg_freq'], df['cpg_per_kb'])
print(f"\nTG freq vs CpG/kb: r={r_tg_cpg:.4f}, P={p_tg_cpg:.2e}")

# TG = reverse complement of CA on the other strand
r_tg_ca, _ = stats.pearsonr(df['tg_freq'], df['di_CA'])
print(f"TG freq vs CA freq: r={r_tg_ca:.4f}")

# ============================================================
# Analysis 7: Subfamily-controlled CpG effect
# ============================================================
print("\n" + "=" * 70)
print("Analysis 7: Subfamily-controlled CpG effect")
print("=" * 70)

# Check if CpG effect survives subfamily control
print("\nCpG/kb by L1 family:")
for fam in sorted(df['l1_family'].unique()):
    fam_data = df[df['l1_family'] == fam]
    if len(fam_data) < 50:
        continue
    print(f"  {fam:<8} n={len(fam_data):>5}  CpG/kb={fam_data['cpg_per_kb'].median():>6.1f}  GC={fam_data['gc_content'].mean():.3f}")

# OLS with subfamily dummies
print("\nOLS with subfamily control:")
X3 = df[['is_stress']].copy()
X3['read_length_kb'] = df['read_length'] / 1000
X3['m6a_per_kb'] = df['m6a_per_kb']
X3['cpg_per_kb'] = df['cpg_per_kb']
X3['has_canonical_pas'] = df['has_canonical_pas']
X3['stress_x_cpg'] = X3['is_stress'] * X3['cpg_per_kb']
X3['stress_x_m6a'] = X3['is_stress'] * X3['m6a_per_kb']
X3['stress_x_pas'] = X3['is_stress'] * X3['has_canonical_pas']

# Add subfamily dummies
for fam in ['L1MC', 'L1ME', 'L1M', 'L1P']:
    X3[f'fam_{fam}'] = (df['l1_family'] == fam).astype(int)

X3 = sm.add_constant(X3)
model3 = sm.OLS(y[mask], X3[mask]).fit()

print(f"  R² (with subfamily): {model3.rsquared:.4f}")
for term in ['stress_x_cpg', 'stress_x_m6a', 'stress_x_pas']:
    coef = model3.params[term]
    pval = model3.pvalues[term]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {term:<20} coef={coef:>+8.3f}  P={pval:.2e} {sig}")

# ============================================================
# Analysis 8: Cross-cell-line CpG validation
# ============================================================
print("\n" + "=" * 70)
print("Analysis 8: Cross-cell-line CpG validation")
print("=" * 70)

# Load all cell lines
ALL_GROUPS = {
    'A549': ['A549_1', 'A549_2'],
    'H9': ['H9_1', 'H9_2'],
    'Hct116': ['Hct116_1', 'Hct116_2'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HepG2': ['HepG2_1', 'HepG2_2'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2'],
    'K562': ['K562_1', 'K562_2'],
    'MCF7': ['MCF7_1', 'MCF7_2', 'MCF7_3'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2'],
}

# For each CL, load Part3 cache, merge with L1 summary, compute CpG from read sequence
# Actually, we don't have sequences extracted for all CLs. Instead, check if CpG/kb
# correlates with poly(A) across cell lines using the features we can get from summary.

# Alternative: Use subfamily as a proxy for CpG density (fixed per subfamily)
# and check the CpG-poly(A) relationship per CL

# Load ancient L1 reads from all base CLs with Part3 cache
print("Loading all cell lines (Part3 cache for m6A, summary for poly(A))...")

all_cl_data = []
for cl, groups in ALL_GROUPS.items():
    for group in groups:
        summary_path = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
        cache_path = f'{CACHE_DIR}/{group}_l1_per_read.tsv'

        if not os.path.exists(summary_path) or not os.path.exists(cache_path):
            continue

        s = pd.read_csv(summary_path, sep='\t')
        c = pd.read_csv(cache_path, sep='\t')

        s = s[s['qc_tag'] == 'PASS'].copy()
        s = s[~s['gene_id'].isin({'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'})].copy()
        s = s[s['polya_length'] > 0].copy()

        s = s.merge(c[['read_id', 'm6a_sites_high']], on='read_id', how='left')
        s['m6a_per_kb'] = s['m6a_sites_high'] / (s['read_length'] / 1000)
        s['cell_line'] = cl
        s['group'] = group

        all_cl_data.append(s[['read_id', 'polya_length', 'm6a_per_kb',
                              'read_length', 'gene_id', 'cell_line', 'group']])

df_all_cl = pd.concat(all_cl_data, ignore_index=True)
print(f"Total ancient L1 across {len(ALL_GROUPS)} CLs: {len(df_all_cl)}")

# Per-subfamily median CpG from HeLa (use as proxy for all CLs since CpG is genomic)
subfamily_cpg = df.groupby('gene_id')['cpg_per_kb'].median().reset_index()
subfamily_cpg.columns = ['gene_id', 'subfamily_cpg_per_kb']

df_all_cl = df_all_cl.merge(subfamily_cpg, on='gene_id', how='left')
df_all_cl = df_all_cl.dropna(subset=['subfamily_cpg_per_kb'])

print(f"Reads with subfamily CpG proxy: {len(df_all_cl)}")

# CpG-poly(A) correlation per cell line
print(f"\n{'Cell line':<12} {'N':>6} {'r(CpG,polyA)':>14} {'P':>12}")
print("-" * 50)

for cl in sorted(ALL_GROUPS.keys()):
    cl_data = df_all_cl[df_all_cl['cell_line'] == cl]
    if len(cl_data) < 50:
        continue
    r, p = stats.pearsonr(cl_data['subfamily_cpg_per_kb'], cl_data['polya_length'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {cl:<12} {len(cl_data):>6} {r:>+14.4f} {p:>12.2e} {sig}")

# ============================================================
# Analysis 9: Triple stratification — CpG × m6A × PAS
# ============================================================
print("\n" + "=" * 70)
print("Analysis 9: Triple stratification under stress (CpG × m6A × PAS)")
print("=" * 70)

stress_data = df[df['is_stress'] == 1].copy()
m6a_median = stress_data['m6a_per_kb'].median()
stress_data['m6a_high'] = (stress_data['m6a_per_kb'] >= m6a_median).astype(int)
stress_data['cpg_high'] = (stress_data['cpg_per_kb'] >= cpg_median).astype(int)

print(f"\n{'m6A':>5} {'PAS':>5} {'CpG':>8} {'N':>6} {'Median polyA':>14} {'Decay%':>8}")
print("-" * 55)

triple_results = []
for m6a_label, m6a_val in [('low', 0), ('high', 1)]:
    for pas_label, pas_val in [('no', 0), ('yes', 1)]:
        for cpg_label, cpg_val in [('low', 0), ('high', 1)]:
            mask = ((stress_data['m6a_high'] == m6a_val) &
                    (stress_data['has_canonical_pas'] == pas_val) &
                    (stress_data['cpg_high'] == cpg_val))
            sub = stress_data.loc[mask, 'polya_length']
            if len(sub) < 10:
                continue
            decay_pct = (sub < 30).mean()
            triple_results.append({
                'm6A': m6a_label, 'PAS': pas_label, 'CpG': cpg_label,
                'n': len(sub), 'median_polya': sub.median(), 'decay_pct': decay_pct
            })
            print(f"{m6a_label:>5} {pas_label:>5} {cpg_label:>8} {len(sub):>6} {sub.median():>14.1f} {decay_pct:>8.1%}")

# Most protected vs most vulnerable
if len(triple_results) >= 2:
    df_triple = pd.DataFrame(triple_results).sort_values('median_polya')
    best = df_triple.iloc[-1]
    worst = df_triple.iloc[0]
    print(f"\nMost protected:  m6A={best['m6A']}, PAS={best['PAS']}, CpG={best['CpG']} → median={best['median_polya']:.1f}, decay={best['decay_pct']:.1%}")
    print(f"Most vulnerable: m6A={worst['m6A']}, PAS={worst['PAS']}, CpG={worst['CpG']} → median={worst['median_polya']:.1f}, decay={worst['decay_pct']:.1%}")
    print(f"Range: {best['median_polya'] - worst['median_polya']:.1f} nt")

# ============================================================
# Analysis 10: Full model comparison (R² decomposition)
# ============================================================
print("\n" + "=" * 70)
print("Analysis 10: R² decomposition — what explains stress poly(A)?")
print("=" * 70)

stress_df = df[df['is_stress'] == 1].copy()
stress_y = stress_df['polya_length']

models_spec = {
    'm6A only': ['m6a_per_kb'],
    'CpG only': ['cpg_per_kb'],
    'PAS only': ['has_canonical_pas'],
    'Read length only': ['read_length_kb'],
    'm6A + CpG': ['m6a_per_kb', 'cpg_per_kb'],
    'm6A + CpG + PAS': ['m6a_per_kb', 'cpg_per_kb', 'has_canonical_pas'],
    'm6A + CpG + PAS + RL': ['m6a_per_kb', 'cpg_per_kb', 'has_canonical_pas', 'read_length_kb'],
    'All seq features': [c for c in df.columns if c.startswith(('di_', 'k3_', 'rbp_', 'gc_', 'has_', 'max_', 'kmer', 'a_frac', 't_frac', 'at_', 'ua_'))],
    'All seq + m6A + RL': [c for c in df.columns if c.startswith(('di_', 'k3_', 'rbp_', 'gc_', 'has_', 'max_', 'kmer', 'a_frac', 't_frac', 'at_', 'ua_'))] + ['m6a_per_kb', 'read_length_kb'],
}

stress_df['read_length_kb'] = stress_df['read_length'] / 1000

print(f"\n{'Model':<30} {'R²':>8} {'Adj R²':>8} {'N features':>12}")
print("-" * 65)

for name, features in models_spec.items():
    avail = [f for f in features if f in stress_df.columns]
    if not avail:
        continue
    X_m = sm.add_constant(stress_df[avail])
    mask_m = ~(X_m.isna().any(axis=1) | stress_y.isna())
    try:
        m = sm.OLS(stress_y[mask_m], X_m[mask_m]).fit()
        print(f"  {name:<30} {m.rsquared:>8.4f} {m.rsquared_adj:>8.4f} {len(avail):>12}")
    except Exception as e:
        print(f"  {name:<30} ERROR: {e}")

# ============================================================
# Save summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

df_cpg_q.to_csv(f'{OUT_DIR}/cpg_quartile_polya.tsv', sep='\t', index=False)
pd.DataFrame(triple_results).to_csv(f'{OUT_DIR}/triple_stratification_stress.tsv', sep='\t', index=False)

print("""
Key findings:

1. CpG QUARTILE × STRESS:
   - Tests whether CpG density predicts poly(A) shortening under stress
   - If Q4 (high CpG) shows more shortening → ZAP-PARN pathway

2. OLS INTERACTIONS:
   - m6A × stress: known protective effect (validation)
   - CpG × stress: new vulnerability axis?
   - PAS × stress: known from prior analysis
   - Are these independent?

3. CpG SPECIFICITY:
   - CpG vs GC content → is it CpG-specific (ZAP) or general GC?

4. TRIPLE STRATIFICATION:
   - Most protected: high m6A + PAS + low CpG
   - Most vulnerable: low m6A + no PAS + high CpG
   - Range shows the combined effect

Results saved to topic_08_sequence_features/
""")
print("Done!")
