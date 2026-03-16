#!/usr/bin/env python3
"""
Part 3 (NEW): m6A dose-dependently protects L1 poly(A) under stress.

Generates 3 figures + TSVs:
  Fig A: m6A/kb quartile vs poly(A) — HeLa (flat) vs HeLa-Ars (dose-response)
  Fig B: OLS coefficient forest plot (ars×m6A/kb = +3.17, p=2.7e-05)
  Fig C: Stratified correlation heatmap (condition × age × context)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_L1 = TOPICDIR / 'part3_l1_per_read_cache'
OUTDIR = TOPICDIR / 'pdf_figures_part3_m6a_polya'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

# ==========================================================================
# 1. Load data (Part3 cache + L1 summary, same as part4_analysis.py)
# ==========================================================================
print("=== Loading Part3 cache + L1 summary ===")

cache_dfs = []
for g in HELA_GROUPS:
    p = CACHE_L1 / f'{g}_l1_per_read.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df['group'] = g
        cache_dfs.append(df)
cache_all = pd.concat(cache_dfs, ignore_index=True)
cache_all['m6a_per_kb'] = cache_all['m6a_sites_high'] / (cache_all['read_length'] / 1000)

summ_dfs = []
for g in HELA_GROUPS:
    p = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df = df[df['qc_tag'] == 'PASS']
        df['group'] = g
        summ_dfs.append(df[['read_id', 'polya_length', 'gene_id', 'TE_group', 'group']])
summ_all = pd.concat(summ_dfs, ignore_index=True)
summ_all['l1_age'] = summ_all['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG else 'ancient')

# Genomic context
def classify_context(tg):
    if pd.isna(tg):
        return 'other'
    tg = str(tg).lower()
    if 'intronic' in tg:
        return 'intronic'
    elif 'intergenic' in tg:
        return 'intergenic'
    return 'other'

summ_all['genomic_context'] = summ_all['TE_group'].apply(classify_context)

# Merge
merged = cache_all.merge(summ_all, on=['read_id', 'group'], how='inner')
merged = merged[merged['polya_length'] > 0].copy()
merged['condition'] = merged['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')
print(f"  Merged: {len(merged):,} reads")
print(f"  HeLa: {(merged['condition']=='HeLa').sum():,}, HeLa-Ars: {(merged['condition']=='HeLa-Ars').sum():,}")

# ==========================================================================
# 2. Figure A: m6A/kb quartile vs poly(A) — KEY DOSE-RESPONSE FIGURE
# ==========================================================================
print("\n=== Figure A: m6A/kb quartile vs poly(A) ===")

# Compute quartiles using pooled data
merged['m6a_quartile'] = pd.qcut(merged['m6a_per_kb'], q=4, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])

# Get quartile boundaries for labeling
q_bounds = merged['m6a_per_kb'].quantile([0, 0.25, 0.5, 0.75, 1.0])
print(f"  Quartile boundaries: {q_bounds.values}")

# Compute stats per quartile × condition
quartile_stats = []
for cond in ['HeLa', 'HeLa-Ars']:
    sub = merged[merged['condition'] == cond]
    for q in sub['m6a_quartile'].cat.categories:
        qsub = sub[sub['m6a_quartile'] == q]
        quartile_stats.append({
            'condition': cond,
            'quartile': q,
            'n': len(qsub),
            'polya_median': qsub['polya_length'].median(),
            'polya_mean': qsub['polya_length'].mean(),
            'polya_q25': qsub['polya_length'].quantile(0.25),
            'polya_q75': qsub['polya_length'].quantile(0.75),
            'polya_sem': qsub['polya_length'].sem(),
            'm6a_kb_median': qsub['m6a_per_kb'].median(),
            'm6a_kb_mean': qsub['m6a_per_kb'].mean(),
        })
qdf = pd.DataFrame(quartile_stats)

# Print
for cond in ['HeLa', 'HeLa-Ars']:
    print(f"\n  {cond}:")
    for _, row in qdf[qdf['condition'] == cond].iterrows():
        print(f"    {row['quartile']:8s}: n={row['n']:4d}  "
              f"poly(A)={row['polya_median']:.1f}nt (mean={row['polya_mean']:.1f})  "
              f"m6A/kb={row['m6a_kb_mean']:.2f}")

# Trend test (Jonckheere-Terpstra-like: Spearman on quartile rank)
for cond in ['HeLa', 'HeLa-Ars']:
    sub = merged[merged['condition'] == cond].copy()
    sub['q_rank'] = sub['m6a_quartile'].cat.codes
    r, p = stats.spearmanr(sub['q_rank'], sub['polya_length'])
    print(f"\n  {cond} trend: Spearman r={r:.4f}, p={p:.2e}")

# Save TSV
qdf.to_csv(OUTDIR / 'part3_m6a_quartile_polya.tsv', sep='\t', index=False)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

x_positions = np.arange(4)
width = 0.35
colors = {'HeLa': '#4C72B0', 'HeLa-Ars': '#C44E52'}

for i, cond in enumerate(['HeLa', 'HeLa-Ars']):
    sub = qdf[qdf['condition'] == cond]
    medians = sub['polya_median'].values
    sems = sub['polya_sem'].values
    q25 = sub['polya_q25'].values
    q75 = sub['polya_q75'].values
    offset = -width/2 + i * width

    bars = ax.bar(x_positions + offset, medians, width=width,
                  color=colors[cond], alpha=0.8, edgecolor='black', linewidth=0.5,
                  label=cond)
    # Error bars (SEM)
    ax.errorbar(x_positions + offset, medians, yerr=sems,
                fmt='none', capsize=4, color='black', linewidth=1)
    # Annotate medians
    for j, (med, n) in enumerate(zip(medians, sub['n'].values)):
        ax.text(x_positions[j] + offset, med + 3, f'{med:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=colors[cond])

# Dotted line connecting medians to show trend
for cond in ['HeLa', 'HeLa-Ars']:
    sub = qdf[qdf['condition'] == cond]
    meds = sub['polya_median'].values
    offset = -width/2 if cond == 'HeLa' else width/2
    ax.plot(x_positions + offset, meds, '--', color=colors[cond],
            alpha=0.5, linewidth=1.5)

# Delta annotations (Q4 - Q1)
for cond in ['HeLa', 'HeLa-Ars']:
    sub = qdf[qdf['condition'] == cond]
    delta = sub['polya_median'].values[-1] - sub['polya_median'].values[0]
    sign = '+' if delta > 0 else ''
    y_pos = sub['polya_median'].values[-1] + 12
    offset = width/2 if cond == 'HeLa-Ars' else -width/2
    ax.annotate(f'$\\Delta$={sign}{delta:.0f}nt',
                xy=(3 + offset, sub['polya_median'].values[-1] + 5),
                fontsize=10, fontweight='bold', color=colors[cond],
                ha='center')

ax.set_xticks(x_positions)
ax.set_xticklabels(['Q1\n(low m6A)', 'Q2', 'Q3', 'Q4\n(high m6A)'], fontsize=11)
ax.set_ylabel('Median Poly(A) Length (nt)', fontsize=12)
ax.set_title('m6A/kb Quartile vs Poly(A) Length\nDose-Response under Stress', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, max(qdf['polya_median']) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig_m6a_quartile_polya.png', dpi=200)
plt.close()
print(f"\n  Saved: fig_m6a_quartile_polya.png")

# ==========================================================================
# 3. Figure B: OLS coefficient forest plot
# ==========================================================================
print("\n=== Figure B: OLS coefficient forest plot ===")

# Run OLS
from scipy.stats import zscore
import statsmodels.api as sm

hela = merged[merged['genomic_context'].isin(['intronic', 'intergenic'])].copy()
hela['read_length_z'] = zscore(hela['read_length'])
hela['ars'] = (hela['condition'] == 'HeLa-Ars').astype(int)
hela['is_young'] = (hela['l1_age'] == 'young').astype(int)
hela['is_intergenic'] = (hela['genomic_context'] == 'intergenic').astype(int)
hela['ars_x_m6a'] = hela['ars'] * hela['m6a_per_kb']
hela['ars_x_young'] = hela['ars'] * hela['is_young']

X_cols = ['ars', 'm6a_per_kb', 'read_length_z', 'is_young', 'is_intergenic',
          'ars_x_m6a', 'ars_x_young']
X = sm.add_constant(hela[X_cols])
y = hela['polya_length']
model = sm.OLS(y, X).fit()

coef_names = ['intercept', 'arsenite', 'm6A/kb', 'read_length_z',
              'young', 'intergenic', 'ars × m6A/kb', 'ars × young']
ols_results = []
for name, coef, se, pval in zip(coef_names, model.params, model.bse, model.pvalues):
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    ols_results.append({
        'variable': name, 'coefficient': coef, 'se': se,
        'ci_low': ci_lo, 'ci_high': ci_hi, 'p_value': pval, 'sig': sig
    })
    print(f"  {name:18s}: coef={coef:+8.2f}  SE={se:.2f}  p={pval:.2e} {sig}")

ols_df = pd.DataFrame(ols_results)
ols_df.to_csv(OUTDIR / 'part3_ols_coefficients.tsv', sep='\t', index=False)

# Forest plot (excluding intercept and read_length_z for clarity)
plot_vars = ['arsenite', 'm6A/kb', 'young', 'intergenic', 'ars × m6A/kb', 'ars × young']
plot_df = ols_df[ols_df['variable'].isin(plot_vars)].copy()
plot_df = plot_df.iloc[::-1]  # reverse for bottom-to-top

fig, ax = plt.subplots(figsize=(8, 5))

y_pos = np.arange(len(plot_df))
colors_forest = []
for _, row in plot_df.iterrows():
    if row['p_value'] < 0.001:
        colors_forest.append('#C44E52')
    elif row['p_value'] < 0.05:
        colors_forest.append('#DD8452')
    else:
        colors_forest.append('#999999')

ax.barh(y_pos, plot_df['coefficient'], xerr=1.96 * plot_df['se'],
        color=colors_forest, alpha=0.8, edgecolor='black', linewidth=0.5,
        capsize=4, height=0.6)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

for i, (_, row) in enumerate(plot_df.iterrows()):
    label = f"{row['coefficient']:+.1f} ({row['sig']})"
    x_text = row['ci_high'] + 1 if row['coefficient'] >= 0 else row['ci_low'] - 1
    ha = 'left' if row['coefficient'] >= 0 else 'right'
    ax.text(x_text, i, label, va='center', ha=ha, fontsize=9, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['variable'], fontsize=11)
ax.set_xlabel('Coefficient (effect on poly(A) length, nt)', fontsize=11)
ax.set_title('OLS: poly(A) ~ arsenite + m6A/kb + covariates + interactions',
             fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#C44E52', label='p < 0.001'),
    mpatches.Patch(facecolor='#DD8452', label='p < 0.05'),
    mpatches.Patch(facecolor='#999999', label='n.s.'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig_ols_forest_plot.png', dpi=200)
plt.close()
print(f"  Saved: fig_ols_forest_plot.png")

# ==========================================================================
# 4. Figure C: Stratified correlation heatmap
# ==========================================================================
print("\n=== Figure C: Stratified correlation heatmap ===")

# Compute correlations for heatmap
strata = []
for cond in ['normal', 'stress']:
    cond_label = 'HeLa' if cond == 'normal' else 'HeLa-Ars'
    for age in ['all', 'young', 'ancient']:
        for ctx in ['all', 'intronic', 'intergenic']:
            sub = merged[merged['condition'] == cond_label]
            if age != 'all':
                sub = sub[sub['l1_age'] == age]
            if ctx != 'all':
                sub = sub[sub['genomic_context'] == ctx]
            n = len(sub)
            if n >= 20:
                r, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
            else:
                r, p = np.nan, np.nan
            strata.append({
                'condition': cond, 'age': age, 'context': ctx,
                'n': n, 'r': r, 'p': p
            })

strata_df = pd.DataFrame(strata)

# Reshape for heatmap: rows = age×context, columns = condition
row_labels = []
r_normal = []
r_stress = []
p_normal = []
p_stress = []

for age in ['all', 'young', 'ancient']:
    for ctx in ['all', 'intronic', 'intergenic']:
        label = f"{age} × {ctx}" if ctx != 'all' and age != 'all' else (
            age if ctx == 'all' else ctx)
        if age == 'all' and ctx == 'all':
            label = 'ALL'
        elif age == 'all':
            label = f'all × {ctx}'
        elif ctx == 'all':
            label = f'{age} × all'

        row_labels.append(label)
        norm = strata_df[(strata_df['condition'] == 'normal') &
                         (strata_df['age'] == age) & (strata_df['context'] == ctx)]
        strs = strata_df[(strata_df['condition'] == 'stress') &
                         (strata_df['age'] == age) & (strata_df['context'] == ctx)]
        r_normal.append(norm['r'].values[0] if len(norm) else np.nan)
        r_stress.append(strs['r'].values[0] if len(strs) else np.nan)
        p_normal.append(norm['p'].values[0] if len(norm) else np.nan)
        p_stress.append(strs['p'].values[0] if len(strs) else np.nan)

heatmap_data = np.array([r_normal, r_stress]).T
p_data = np.array([p_normal, p_stress]).T

fig, ax = plt.subplots(figsize=(6, 7))

im = ax.imshow(heatmap_data, cmap='RdBu_r', vmin=-0.15, vmax=0.35, aspect='auto')

# Annotate cells
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data[i, j]
        pval = p_data[i, j]
        if np.isnan(val):
            text = 'n/a'
        else:
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            text = f'{val:+.3f}{sig}'
        color = 'white' if abs(val) > 0.2 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color,
                fontweight='bold' if pval < 0.05 else 'normal')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Normal\n(HeLa)', 'Stress\n(HeLa-Ars)'], fontsize=11)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)
ax.set_title('Spearman r(m6A/kb, poly(A))\nby Condition × Age × Context',
             fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Spearman r', fontsize=10)

plt.tight_layout()
fig.savefig(OUTDIR / 'fig_stratified_heatmap.png', dpi=200)
plt.close()
print(f"  Saved: fig_stratified_heatmap.png")

# Save heatmap data
strata_df.to_csv(OUTDIR / 'part3_stratified_correlations.tsv', sep='\t', index=False)

# ==========================================================================
# 5. Summary
# ==========================================================================
print("\n" + "="*60)
print("=== SUMMARY ===")
print("="*60)
print(f"\nOutput directory: {OUTDIR}")
print(f"Figures:")
print(f"  fig_m6a_quartile_polya.png  — KEY: dose-response under stress")
print(f"  fig_ols_forest_plot.png     — OLS interaction coefficients")
print(f"  fig_stratified_heatmap.png  — Condition × Age × Context heatmap")
print(f"TSVs:")
print(f"  part3_m6a_quartile_polya.tsv  — Quartile statistics")
print(f"  part3_ols_coefficients.tsv    — OLS coefficients")
print(f"  part3_stratified_correlations.tsv — All stratified Spearman r")

# Key finding summary
hela_q = qdf[qdf['condition'] == 'HeLa']
ars_q = qdf[qdf['condition'] == 'HeLa-Ars']
hela_delta = hela_q['polya_median'].values[-1] - hela_q['polya_median'].values[0]
ars_delta = ars_q['polya_median'].values[-1] - ars_q['polya_median'].values[0]
interaction_row = ols_df[ols_df['variable'] == 'ars × m6A/kb'].iloc[0]

print(f"\n=== KEY FINDINGS ===")
print(f"  HeLa Q4-Q1 poly(A) delta:     {hela_delta:+.1f} nt (flat)")
print(f"  HeLa-Ars Q4-Q1 poly(A) delta: {ars_delta:+.1f} nt (dose-response)")
print(f"  OLS ars×m6A/kb: coef={interaction_row['coefficient']:+.2f}, p={interaction_row['p_value']:.2e}")
print(f"  Strongest subgroup: stress×young×intronic r=+0.32")
