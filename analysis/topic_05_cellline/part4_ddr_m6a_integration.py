#!/usr/bin/env python3
"""
Part 4 Integration: DDR gene × m6A × arsenite poly(A).

Integrates three axes:
  1. L1 m6A enrichment (Part 1)
  2. Arsenite poly(A) shortening (Part 2)
  3. DDR host gene context (Part 4)

To show: "DDR gene 내 m6A-marked L1 → stress-sensitive regulatory element"

Figures:
  A: DDR vs non-DDR host gene L1: m6A/kb comparison (all CLs)
  B: DDR vs non-DDR: arsenite poly(A) delta
  C: m6A quartile × poly(A) in DDR genes (HeLa vs HeLa-Ars)
  D: BRCA1 case study: reads with m6A and poly(A)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_L1 = TOPICDIR / 'part3_l1_per_read_cache'
RESULTS = PROJECT / 'results_group'
OUTDIR = TOPICDIR / 'part4_ddr_m6a_integration'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

DDR_GENES = {'BRCA1', 'BRCA2', 'ATR', 'ATM', 'ATRX', 'RAD51', 'RAD50',
             'FANCC', 'FANCD2', 'FANCI', 'ZRANB3', 'SMARCAL1', 'RB1',
             'NBN', 'BLM', 'WRN', 'XRCC5', 'XRCC6', 'CHEK1', 'CHEK2',
             'PARP1', 'TP53', 'MRE11'}

# Extended DDR-related gene set from enrichment results
DDR_EXTENDED = DDR_GENES | {
    'ERCC1', 'ERCC4', 'XRCC4', 'POLQ', 'RPA1', 'PALB2',
    'LIG4', 'TANK', 'MDC1', 'TOPBP1', 'REV3L', 'POLE',
    'RAD51B', 'RAD51C', 'SLX4', 'EME1', 'GEN1', 'MUS81',
    'RECQL4', 'RECQL5', 'ABRAXAS1', 'BARD1', 'BABAM1'
}

# ==========================================================================
# 1. Load ALL cell lines: Part3 cache + L1 summary
# ==========================================================================
print("=== Loading Part3 cache + L1 summary (ALL cell lines) ===")

# All cache files
cache_dfs = []
for f in sorted(CACHE_L1.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t')
    group = f.stem.replace('_l1_per_read', '')
    df['group'] = group
    cache_dfs.append(df)
cache = pd.concat(cache_dfs, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
print(f"  Cache: {len(cache):,} reads from {cache['group'].nunique()} groups")

# All summary files
summ_dfs = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    group = df['sample'].str.rsplit('_', n=1).str[0].iloc[0] if len(df) > 0 else ''
    df['group'] = group
    summ_dfs.append(df[['read_id', 'polya_length', 'gene_id', 'TE_group',
                         'overlapping_genes', 'group']])
summ = pd.concat(summ_dfs, ignore_index=True)
print(f"  Summary: {len(summ):,} reads")

# Classify
summ['l1_age'] = summ['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG else 'ancient')

def classify_context(tg):
    if pd.isna(tg):
        return 'other'
    tg = str(tg).lower()
    if 'intronic' in tg:
        return 'intronic'
    elif 'intergenic' in tg:
        return 'intergenic'
    return 'other'

summ['genomic_context'] = summ['TE_group'].apply(classify_context)

# Host gene: use overlapping_genes for intronic reads
summ['host_gene'] = summ['overlapping_genes'].fillna('')

# DDR classification
def classify_host(gene):
    if not gene or gene == '' or pd.isna(gene):
        return 'none'
    genes = set(str(gene).split(','))
    if genes & DDR_GENES:
        return 'DDR_core'
    if genes & DDR_EXTENDED:
        return 'DDR_extended'
    return 'non-DDR'

summ['ddr_status'] = summ['host_gene'].apply(classify_host)

# Merge cache + summary
merged = cache.merge(summ, on=['read_id', 'group'], how='inner')
merged = merged[merged['polya_length'] > 0].copy()

# Cell line and condition
def get_cellline(group):
    return group.rsplit('_', 1)[0]
merged['cellline'] = merged['group'].apply(get_cellline)
merged['condition'] = merged['cellline'].apply(
    lambda x: 'stress' if 'Ars' in x else 'normal')

print(f"  Merged: {len(merged):,} reads")
print(f"  Intronic: {(merged['genomic_context']=='intronic').sum():,}")
print(f"  DDR status: {merged['ddr_status'].value_counts().to_dict()}")

# Focus on intronic reads (host gene relevant)
intronic = merged[merged['genomic_context'] == 'intronic'].copy()
print(f"\n  Intronic reads: {len(intronic):,}")
print(f"  DDR core: {(intronic['ddr_status']=='DDR_core').sum()}")
print(f"  DDR extended: {(intronic['ddr_status']=='DDR_extended').sum()}")
print(f"  Non-DDR intronic: {(intronic['ddr_status']=='non-DDR').sum()}")

# ==========================================================================
# 2. Panel A: DDR vs non-DDR m6A/kb comparison (ALL cell lines, intronic)
# ==========================================================================
print("\n" + "="*60)
print("=== Panel A: DDR vs non-DDR m6A/kb ===")
print("="*60)

ddr_reads = intronic[intronic['ddr_status'].isin(['DDR_core', 'DDR_extended'])]
non_ddr_reads = intronic[intronic['ddr_status'] == 'non-DDR']

print(f"\n  DDR (core+ext): n={len(ddr_reads)}, m6A/kb median={ddr_reads['m6a_per_kb'].median():.2f}, "
      f"mean={ddr_reads['m6a_per_kb'].mean():.2f}")
print(f"  Non-DDR:        n={len(non_ddr_reads)}, m6A/kb median={non_ddr_reads['m6a_per_kb'].median():.2f}, "
      f"mean={non_ddr_reads['m6a_per_kb'].mean():.2f}")

mw = stats.mannwhitneyu(ddr_reads['m6a_per_kb'], non_ddr_reads['m6a_per_kb'], alternative='two-sided')
print(f"  MW U-test p={mw.pvalue:.4e}")

# Per-gene DDR m6A/kb
print("\n  --- Per DDR gene m6A/kb ---")
ddr_gene_stats = []
for gene in sorted(DDR_GENES):
    g_reads = intronic[intronic['host_gene'].str.contains(gene, na=False)]
    if len(g_reads) > 0:
        ddr_gene_stats.append({
            'gene': gene, 'n_reads': len(g_reads),
            'n_cl': g_reads['cellline'].nunique(),
            'm6a_kb_median': g_reads['m6a_per_kb'].median(),
            'm6a_kb_mean': g_reads['m6a_per_kb'].mean(),
            'polya_median': g_reads['polya_length'].median(),
            'has_hela': 'HeLa' in g_reads['cellline'].values,
            'has_ars': 'HeLa-Ars' in g_reads['cellline'].values,
        })
        print(f"    {gene:12s}: n={len(g_reads):3d}  CL={g_reads['cellline'].nunique()}  "
              f"m6A/kb={g_reads['m6a_per_kb'].median():.1f}  poly(A)={g_reads['polya_length'].median():.0f}")

ddr_stats_df = pd.DataFrame(ddr_gene_stats)
ddr_stats_df.to_csv(OUTDIR / 'ddr_gene_m6a_stats.tsv', sep='\t', index=False)

# ==========================================================================
# 3. Panel B: DDR vs non-DDR arsenite poly(A) delta (HeLa only)
# ==========================================================================
print("\n" + "="*60)
print("=== Panel B: Arsenite poly(A) delta — DDR vs non-DDR ===")
print("="*60)

hela_intronic = intronic[intronic['cellline'].isin(['HeLa', 'HeLa-Ars'])]

# Overall comparison
delta_results = []
for status_label, status_filter in [
    ('DDR (core+ext)', ['DDR_core', 'DDR_extended']),
    ('Non-DDR intronic', ['non-DDR']),
]:
    sub = hela_intronic[hela_intronic['ddr_status'].isin(status_filter)]
    hela_vals = sub[sub['condition'] == 'normal']['polya_length']
    ars_vals = sub[sub['condition'] == 'stress']['polya_length']

    if len(hela_vals) >= 5 and len(ars_vals) >= 5:
        delta = ars_vals.median() - hela_vals.median()
        mw = stats.mannwhitneyu(hela_vals, ars_vals, alternative='two-sided')
        print(f"  {status_label:25s}: HeLa={hela_vals.median():.1f}nt (n={len(hela_vals)}), "
              f"Ars={ars_vals.median():.1f}nt (n={len(ars_vals)}), "
              f"Δ={delta:+.1f}nt, MW p={mw.pvalue:.2e}")
        delta_results.append({
            'category': status_label,
            'n_hela': len(hela_vals), 'n_ars': len(ars_vals),
            'polya_hela': hela_vals.median(), 'polya_ars': ars_vals.median(),
            'delta': delta, 'mw_p': mw.pvalue
        })

delta_df = pd.DataFrame(delta_results)
delta_df.to_csv(OUTDIR / 'ddr_vs_nonddr_arsenite_delta.tsv', sep='\t', index=False)

# ==========================================================================
# 4. Panel C: m6A quartile × poly(A) in DDR genes (HeLa vs HeLa-Ars)
# ==========================================================================
print("\n" + "="*60)
print("=== Panel C: m6A quartile × poly(A) in DDR genes ===")
print("="*60)

# Use all intronic reads for quartile computation, then filter DDR
hela_all_intronic = intronic[intronic['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()

# Quartiles computed on all intronic reads
hela_all_intronic['m6a_q'] = pd.qcut(hela_all_intronic['m6a_per_kb'], q=4,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4'])

# DDR genes: m6A quartile × condition
ddr_hela = hela_all_intronic[hela_all_intronic['ddr_status'].isin(['DDR_core', 'DDR_extended'])]
nonddr_hela = hela_all_intronic[hela_all_intronic['ddr_status'] == 'non-DDR']

print(f"\n  DDR intronic HeLa/Ars: {len(ddr_hela)} reads")
print(f"  Non-DDR intronic HeLa/Ars: {len(nonddr_hela)} reads")

if len(ddr_hela) >= 20:
    print("\n  DDR genes — m6A quartile × condition:")
    for cond in ['normal', 'stress']:
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = ddr_hela[(ddr_hela['condition'] == cond) & (ddr_hela['m6a_q'] == q)]
            if len(sub) >= 3:
                print(f"    {cond:8s} {q}: n={len(sub):3d}  poly(A)={sub['polya_length'].median():.1f}")

# Non-DDR comparison
print("\n  Non-DDR — m6A quartile × condition:")
for cond in ['normal', 'stress']:
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = nonddr_hela[(nonddr_hela['condition'] == cond) & (nonddr_hela['m6a_q'] == q)]
        if len(sub) >= 3:
            print(f"    {cond:8s} {q}: n={len(sub):3d}  poly(A)={sub['polya_length'].median():.1f}")

# ==========================================================================
# 5. Panel D: BRCA1 case study
# ==========================================================================
print("\n" + "="*60)
print("=== Panel D: BRCA1 case study ===")
print("="*60)

brca1 = intronic[intronic['host_gene'].str.contains('BRCA1', na=False)]
print(f"\n  BRCA1 total reads: {len(brca1)}")
print(f"  Cell lines: {sorted(brca1['cellline'].unique())}")
print(f"  Groups: {sorted(brca1['group'].unique())}")

if len(brca1) > 0:
    print(f"\n  m6A/kb: median={brca1['m6a_per_kb'].median():.2f}, mean={brca1['m6a_per_kb'].mean():.2f}")
    print(f"  poly(A): median={brca1['polya_length'].median():.1f}")

    # HeLa vs HeLa-Ars within BRCA1
    brca1_hela = brca1[brca1['cellline'] == 'HeLa']
    brca1_ars = brca1[brca1['cellline'] == 'HeLa-Ars']
    if len(brca1_hela) > 0 and len(brca1_ars) > 0:
        print(f"\n  BRCA1 HeLa: n={len(brca1_hela)}, poly(A)={brca1_hela['polya_length'].median():.1f}nt, "
              f"m6A/kb={brca1_hela['m6a_per_kb'].median():.1f}")
        print(f"  BRCA1 Ars:  n={len(brca1_ars)}, poly(A)={brca1_ars['polya_length'].median():.1f}nt, "
              f"m6A/kb={brca1_ars['m6a_per_kb'].median():.1f}")
        delta = brca1_ars['polya_length'].median() - brca1_hela['polya_length'].median()
        print(f"  BRCA1 Δpoly(A) = {delta:+.1f}nt")

    # BRCA1 per cell line
    print("\n  BRCA1 per cell line:")
    for cl in sorted(brca1['cellline'].unique()):
        sub = brca1[brca1['cellline'] == cl]
        print(f"    {cl:12s}: n={len(sub):3d}  m6A/kb={sub['m6a_per_kb'].median():.1f}  "
              f"poly(A)={sub['polya_length'].median():.0f}")

# Save BRCA1 data
brca1.to_csv(OUTDIR / 'brca1_reads.tsv', sep='\t', index=False)

# ==========================================================================
# 6. Top DDR genes with arsenite data
# ==========================================================================
print("\n" + "="*60)
print("=== Top DDR genes: HeLa vs HeLa-Ars ===")
print("="*60)

ddr_ars_results = []
hela_intronic_ddr = hela_intronic[hela_intronic['ddr_status'].isin(['DDR_core', 'DDR_extended'])]

# Per-gene arsenite comparison
for gene in sorted(DDR_EXTENDED):
    g_reads = hela_intronic[hela_intronic['host_gene'].str.contains(gene, na=False)]
    if len(g_reads) < 3:
        continue
    hela_g = g_reads[g_reads['condition'] == 'normal']
    ars_g = g_reads[g_reads['condition'] == 'stress']
    if len(hela_g) >= 1 and len(ars_g) >= 1:
        ddr_ars_results.append({
            'gene': gene,
            'n_hela': len(hela_g), 'n_ars': len(ars_g),
            'polya_hela': hela_g['polya_length'].median(),
            'polya_ars': ars_g['polya_length'].median(),
            'delta': ars_g['polya_length'].median() - hela_g['polya_length'].median(),
            'm6a_kb_hela': hela_g['m6a_per_kb'].median(),
            'm6a_kb_ars': ars_g['m6a_per_kb'].median(),
            'in_DDR_core': gene in DDR_GENES,
        })

if ddr_ars_results:
    ddr_ars_df = pd.DataFrame(ddr_ars_results).sort_values('delta')
    ddr_ars_df.to_csv(OUTDIR / 'ddr_gene_arsenite_polya.tsv', sep='\t', index=False)
    print("\n  DDR genes with HeLa & HeLa-Ars data:")
    for _, row in ddr_ars_df.iterrows():
        core = '*' if row['in_DDR_core'] else ' '
        print(f"  {core}{row['gene']:12s}: HeLa={row['polya_hela']:.0f}nt(n={row['n_hela']}), "
              f"Ars={row['polya_ars']:.0f}nt(n={row['n_ars']}), "
              f"Δ={row['delta']:+.0f}nt, m6A/kb={row['m6a_kb_hela']:.1f}/{row['m6a_kb_ars']:.1f}")

# ==========================================================================
# 7. OLS: poly(A) ~ ars + m6A/kb + DDR_status + interactions (HeLa only)
# ==========================================================================
print("\n" + "="*60)
print("=== OLS: DDR × m6A × arsenite interaction ===")
print("="*60)

import statsmodels.api as sm
from scipy.stats import zscore

ols_data = hela_all_intronic.copy()
ols_data = ols_data[ols_data['ddr_status'].isin(['DDR_core', 'DDR_extended', 'non-DDR'])].copy()
ols_data['is_ddr'] = ols_data['ddr_status'].isin(['DDR_core', 'DDR_extended']).astype(int)
ols_data['is_stress'] = (ols_data['condition'] == 'stress').astype(int)
ols_data['is_young'] = (ols_data['l1_age'] == 'young').astype(int)
ols_data['rdlen_z'] = zscore(ols_data['read_length'])
ols_data['stress_x_m6a'] = ols_data['is_stress'] * ols_data['m6a_per_kb']
ols_data['ddr_x_stress'] = ols_data['is_ddr'] * ols_data['is_stress']
ols_data['ddr_x_m6a'] = ols_data['is_ddr'] * ols_data['m6a_per_kb']
ols_data['ddr_x_stress_x_m6a'] = ols_data['is_ddr'] * ols_data['is_stress'] * ols_data['m6a_per_kb']

X_cols = ['is_stress', 'm6a_per_kb', 'rdlen_z', 'is_young', 'is_ddr',
          'stress_x_m6a', 'ddr_x_stress', 'ddr_x_m6a', 'ddr_x_stress_x_m6a']
X = sm.add_constant(ols_data[X_cols])
y = ols_data['polya_length']
model = sm.OLS(y, X).fit()

coef_names = ['intercept', 'stress', 'm6A/kb', 'read_length_z', 'young', 'DDR',
              'stress×m6A/kb', 'DDR×stress', 'DDR×m6A/kb', 'DDR×stress×m6A/kb']
print(f"\n  N = {model.nobs:.0f}, R² = {model.rsquared:.4f}")
for name, coef, se, pval in zip(coef_names, model.params, model.bse, model.pvalues):
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {name:25s}: coef={coef:+8.2f}  SE={se:6.2f}  p={pval:.2e} {sig}")

# Save OLS
ols_table = pd.DataFrame({
    'variable': coef_names,
    'coefficient': model.params.values,
    'se': model.bse.values,
    'p_value': model.pvalues.values,
})
ols_table.to_csv(OUTDIR / 'ddr_m6a_ars_ols.tsv', sep='\t', index=False)

# ==========================================================================
# 8. Generate figures
# ==========================================================================
print("\n=== Generating figures ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Panel A: DDR vs non-DDR m6A/kb ---
ax = axes[0, 0]
data_A = [ddr_reads['m6a_per_kb'].values, non_ddr_reads['m6a_per_kb'].values]
parts = ax.violinplot(data_A, positions=[0, 1], showmedians=True, showextrema=False)
colors_A = ['#C44E52', '#4C72B0']
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(colors_A[i])
    body.set_alpha(0.7)
parts['cmedians'].set_color('black')
for i, (vals, label) in enumerate(zip(data_A, ['DDR', 'non-DDR'])):
    med = np.median(vals)
    ax.text(i, med + 0.3, f'{med:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'DDR\n(n={len(ddr_reads):,})', f'non-DDR\n(n={len(non_ddr_reads):,})'],
                    fontsize=11)
ax.set_ylabel('m6A/kb', fontsize=11)
ax.set_title('A. m6A Density: DDR vs non-DDR Host Genes\n(all CLs, intronic L1)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 20)
mw_A = stats.mannwhitneyu(ddr_reads['m6a_per_kb'], non_ddr_reads['m6a_per_kb'])
ax.text(0.02, 0.98, f'MW p={mw_A.pvalue:.2e}', transform=ax.transAxes,
        va='top', fontsize=9, fontstyle='italic')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: Arsenite poly(A) delta ---
ax = axes[0, 1]
categories = ['DDR intronic', 'non-DDR intronic']
hela_medians = [delta_df.iloc[0]['polya_hela'], delta_df.iloc[1]['polya_hela']]
ars_medians = [delta_df.iloc[0]['polya_ars'], delta_df.iloc[1]['polya_ars']]
deltas = [delta_df.iloc[0]['delta'], delta_df.iloc[1]['delta']]

x = np.arange(2)
w = 0.35
bars1 = ax.bar(x - w/2, hela_medians, w, color='#4C72B0', alpha=0.8,
               edgecolor='black', linewidth=0.5, label='HeLa')
bars2 = ax.bar(x + w/2, ars_medians, w, color='#C44E52', alpha=0.8,
               edgecolor='black', linewidth=0.5, label='HeLa-Ars')

for i in range(2):
    ax.text(i, max(hela_medians[i], ars_medians[i]) + 5,
            f'Δ={deltas[i]:+.0f}nt', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Median Poly(A) Length (nt)', fontsize=11)
ax.set_title('B. Arsenite Poly(A) Shortening\nDDR vs non-DDR Host Genes',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, max(hela_medians + ars_medians) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel C: m6A quartile × poly(A) for DDR + non-DDR under stress ---
ax = axes[1, 0]
qc_results = []
for status, label, color in [
    (['DDR_core', 'DDR_extended'], 'DDR', '#C44E52'),
    (['non-DDR'], 'non-DDR', '#4C72B0'),
]:
    for cond in ['normal', 'stress']:
        sub = hela_all_intronic[(hela_all_intronic['ddr_status'].isin(status)) &
                                 (hela_all_intronic['condition'] == cond)]
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            qsub = sub[sub['m6a_q'] == q]
            if len(qsub) >= 3:
                qc_results.append({
                    'status': label, 'condition': cond, 'quartile': q,
                    'n': len(qsub), 'polya_median': qsub['polya_length'].median(),
                })

qc_df = pd.DataFrame(qc_results)
qc_df.to_csv(OUTDIR / 'ddr_quartile_polya.tsv', sep='\t', index=False)

# Plot: non-DDR stress vs non-DDR normal (to show the protection effect)
nonddr_normal = qc_df[(qc_df['status'] == 'non-DDR') & (qc_df['condition'] == 'normal')]
nonddr_stress = qc_df[(qc_df['status'] == 'non-DDR') & (qc_df['condition'] == 'stress')]

x = np.arange(4)
if len(nonddr_normal) == 4 and len(nonddr_stress) == 4:
    ax.plot(x, nonddr_normal['polya_median'].values, 'o-',
            color='#4C72B0', linewidth=2, markersize=8, label='non-DDR Normal')
    ax.plot(x, nonddr_stress['polya_median'].values, 's-',
            color='#C44E52', linewidth=2, markersize=8, label='non-DDR Stress')

# DDR if enough data
ddr_normal = qc_df[(qc_df['status'] == 'DDR') & (qc_df['condition'] == 'normal')]
ddr_stress = qc_df[(qc_df['status'] == 'DDR') & (qc_df['condition'] == 'stress')]
if len(ddr_normal) == 4:
    ax.plot(x, ddr_normal['polya_median'].values, 'o--',
            color='#4C72B0', linewidth=1.5, markersize=6, alpha=0.6, label='DDR Normal')
if len(ddr_stress) == 4:
    ax.plot(x, ddr_stress['polya_median'].values, 's--',
            color='#C44E52', linewidth=1.5, markersize=6, alpha=0.6, label='DDR Stress')

ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'], fontsize=10)
ax.set_xlabel('m6A/kb Quartile', fontsize=11)
ax.set_ylabel('Median Poly(A) Length (nt)', fontsize=11)
ax.set_title('C. m6A Dose-Response: DDR vs non-DDR\n(intronic L1, HeLa/Ars)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel D: BRCA1 scatter — m6A/kb vs poly(A) colored by condition ---
ax = axes[1, 1]
if len(brca1) > 0:
    brca1_hela_plot = brca1[brca1['condition'] == 'normal']
    brca1_ars_plot = brca1[brca1['condition'] == 'stress']

    if len(brca1_hela_plot) > 0:
        ax.scatter(brca1_hela_plot['m6a_per_kb'], brca1_hela_plot['polya_length'],
                   c='#4C72B0', alpha=0.6, s=30, label=f'HeLa (n={len(brca1_hela_plot)})')
    if len(brca1_ars_plot) > 0:
        ax.scatter(brca1_ars_plot['m6a_per_kb'], brca1_ars_plot['polya_length'],
                   c='#C44E52', alpha=0.6, s=30, label=f'HeLa-Ars (n={len(brca1_ars_plot)})')

    # All other CLs
    brca1_other = brca1[~brca1['cellline'].isin(['HeLa', 'HeLa-Ars'])]
    if len(brca1_other) > 0:
        ax.scatter(brca1_other['m6a_per_kb'], brca1_other['polya_length'],
                   c='#888888', alpha=0.4, s=20, label=f'Other CLs (n={len(brca1_other)})')

    ax.set_xlabel('m6A/kb', fontsize=11)
    ax.set_ylabel('Poly(A) Length (nt)', fontsize=11)
    ax.set_title('D. BRCA1: m6A vs Poly(A)\n(per read, all CLs)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig_ddr_m6a_integration.png', dpi=200)
plt.close()
print(f"\n  Saved: fig_ddr_m6a_integration.png")

# ==========================================================================
# 9. Summary
# ==========================================================================
print("\n" + "="*60)
print("=== SUMMARY ===")
print("="*60)
print(f"\nOutput: {OUTDIR}")
print(f"Figures: fig_ddr_m6a_integration.png (4 panels)")
print(f"TSVs:")
print(f"  ddr_gene_m6a_stats.tsv          — Per DDR gene m6A/kb")
print(f"  ddr_vs_nonddr_arsenite_delta.tsv — DDR vs non-DDR poly(A) Δ")
print(f"  ddr_quartile_polya.tsv           — Quartile × DDR × condition")
print(f"  ddr_m6a_ars_ols.tsv              — OLS with DDR interaction")
print(f"  ddr_gene_arsenite_polya.tsv      — Per DDR gene arsenite effect")
print(f"  brca1_reads.tsv                  — BRCA1 per-read data")
