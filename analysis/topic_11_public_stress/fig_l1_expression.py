#!/usr/bin/env python3
"""
Figure: L1 expression changes under stress (GSE277764 ONT cDNA)
Young vs Ancient L1 RPM across Untreated, Arsenite, HeatShock.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_11_public_stress'
df = pd.read_csv(f'{OUTDIR}/all_l1_subfamily_counts.tsv', sep='\t')

YOUNG = {'L1HS', 'L1PA2', 'L1PA3'}
SAMPLES_META = {
    'UN_rep1': ('Untreated', 1), 'UN_rep2': ('Untreated', 2),
    'SA_rep1': ('Arsenite', 1), 'SA_rep2': ('Arsenite', 2),
    'HS_rep1': ('HeatShock', 1), 'HS_rep2': ('HeatShock', 2),
}

df['age'] = df['subfamily'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')

# Aggregate by sample × age
agg = df.groupby(['sample', 'condition', 'age']).agg(
    l1_reads=('count', 'sum'),
    total_reads=('total_reads', 'first'),
).reset_index()
agg['rpm'] = agg['l1_reads'] / agg['total_reads'] * 1e6
agg['rep'] = agg['sample'].map(lambda s: SAMPLES_META.get(s, ('?', 0))[1])

# Also compute per-subfamily RPM for top young families
subfam_agg = df.groupby(['sample', 'condition', 'subfamily']).agg(
    l1_reads=('count', 'sum'),
    total_reads=('total_reads', 'first'),
).reset_index()
subfam_agg['rpm'] = subfam_agg['l1_reads'] / subfam_agg['total_reads'] * 1e6

# === Figure ===
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Panel A: Young L1 RPM by condition
ax = axes[0]
conditions = ['Untreated', 'Arsenite', 'HeatShock']
colors = {'Untreated': '#636363', 'Arsenite': '#e6550d', 'HeatShock': '#3182bd'}
x_pos = np.arange(len(conditions))
for i, cond in enumerate(conditions):
    sub = agg[(agg['condition'] == cond) & (agg['age'] == 'Young')]
    rpms = sub['rpm'].values
    ax.bar(i, rpms.mean(), color=colors[cond], alpha=0.7, width=0.6)
    ax.scatter([i]*len(rpms), rpms, color='black', s=30, zorder=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(conditions, rotation=30, ha='right')
ax.set_ylabel('Young L1 RPM')
ax.set_title('Young L1 (L1HS/PA2/PA3)')
ax.spines[['top', 'right']].set_visible(False)

# Panel B: Ancient L1 RPM by condition
ax = axes[1]
for i, cond in enumerate(conditions):
    sub = agg[(agg['condition'] == cond) & (agg['age'] == 'Ancient')]
    rpms = sub['rpm'].values
    ax.bar(i, rpms.mean(), color=colors[cond], alpha=0.7, width=0.6)
    ax.scatter([i]*len(rpms), rpms, color='black', s=30, zorder=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(conditions, rotation=30, ha='right')
ax.set_ylabel('Ancient L1 RPM')
ax.set_title('Ancient L1')
ax.spines[['top', 'right']].set_visible(False)

# Panel C: Fold change heatmap (Stress / Untreated) for top subfamilies
ax = axes[2]
top_subfams = ['L1HS', 'L1PA2', 'L1PA3', 'L1PA4', 'L1PA5', 'L1PA6', 'L1PA7',
               'L1M1', 'L1M4c', 'L1M5', 'L1MB7', 'L1MC4', 'L1ME1']

fc_data = []
for subfam in top_subfams:
    un_rpms = subfam_agg[(subfam_agg['condition'] == 'Untreated') & (subfam_agg['subfamily'] == subfam)]['rpm']
    un_mean = un_rpms.mean() if len(un_rpms) > 0 else 0
    for cond in ['Arsenite', 'HeatShock']:
        st_rpms = subfam_agg[(subfam_agg['condition'] == cond) & (subfam_agg['subfamily'] == subfam)]['rpm']
        st_mean = st_rpms.mean() if len(st_rpms) > 0 else 0
        fc = st_mean / un_mean if un_mean > 0 else np.nan
        fc_data.append({'subfamily': subfam, 'condition': cond, 'fc': fc})

fc_df = pd.DataFrame(fc_data)
fc_pivot = fc_df.pivot(index='subfamily', columns='condition', values='fc')
fc_pivot = fc_pivot.reindex(top_subfams)

im = ax.imshow(fc_pivot.values, cmap='RdBu_r', vmin=0.7, vmax=1.3, aspect='auto')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Arsenite', 'HeatShock'])
ax.set_yticks(range(len(top_subfams)))
ax.set_yticklabels(top_subfams, fontsize=8)
ax.set_title('Fold Change\n(Stress / Untreated)')

# Add FC text
for i in range(len(top_subfams)):
    for j in range(2):
        val = fc_pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(val - 1) > 0.15 else 'black')

plt.colorbar(im, ax=ax, shrink=0.8, label='Fold Change')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_l1_expression_stress.pdf', bbox_inches='tight', dpi=150)
plt.savefig(f'{OUTDIR}/fig_l1_expression_stress.png', bbox_inches='tight', dpi=150)
print(f'Saved: {OUTDIR}/fig_l1_expression_stress.pdf')

# Print summary table
print('\n=== Summary Table ===')
print(f"{'Subfamily':<10} {'Ars FC':>8} {'HS FC':>8} {'Age':<8}")
print('-' * 40)
for subfam in top_subfams:
    row = fc_pivot.loc[subfam] if subfam in fc_pivot.index else pd.Series({'Arsenite': np.nan, 'HeatShock': np.nan})
    age = 'Young' if subfam in YOUNG else 'Ancient'
    ars = row.get('Arsenite', np.nan)
    hs = row.get('HeatShock', np.nan)
    ars_s = f'{ars:.3f}' if not np.isnan(ars) else 'N/A'
    hs_s = f'{hs:.3f}' if not np.isnan(hs) else 'N/A'
    print(f'  {subfam:<10} {ars_s:>8} {hs_s:>8} {age:<8}')
