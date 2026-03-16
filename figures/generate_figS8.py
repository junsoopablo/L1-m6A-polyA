#!/usr/bin/env python3
"""
Supplementary Figure S8: CHX rescue of arsenite-induced L1 poly(A) shortening.
Panels: (a) ECDF of poly(A) length: mock vs Ars vs Ars+CHX
        (b) Paired bar: ancient vs young poly(A) by condition
        (c) m6A-stratified rescue: low vs high m6A rescue magnitude
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats
from fig_style import *

setup_style()

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA = '/vault/external-datasets/2026/PRJNA842344_HeLA_under_oxidative-stress_RNA002/xrn1_analysis/analysis'

# ── Load data ──
df = pd.read_csv(f'{DATA}/ars_chx_analysis.tsv', sep='\t')

# Focus on the three relevant conditions (within-experiment)
conds = ['mock', 'Ars_mock', 'Ars+CHX']
df3 = df[df['condition'].isin(conds)].copy()
df3['condition_label'] = df3['condition'].map({
    'mock': 'Mock', 'Ars_mock': 'Arsenite', 'Ars+CHX': 'Ars+CHX'
})

# ── Create figure: 1 row, 3 columns ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.33))
gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.45,
                      left=0.08, right=0.97, top=0.88, bottom=0.18)

# ────────────────────────────────────────────
# Panel (a): ECDF of poly(A) — mock vs Ars vs Ars+CHX
# ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')

colors = {'Mock': C_NORMAL, 'Arsenite': C_STRESS, 'Ars+CHX': C_HIGHLIGHT}
for label in ['Mock', 'Arsenite', 'Ars+CHX']:
    sub = df3[df3['condition_label'] == label]['polya_length'].dropna()
    sorted_data = np.sort(sub.values)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ls = '--' if label == 'Mock' else '-'
    ax_a.step(sorted_data, y, where='post', color=colors[label],
              lw=1.2, ls=ls, label=f'{label} (n={len(sub)})')

ax_a.set_xlim(0, 300)
ax_a.set_xlabel('Poly(A) length (nt)')
ax_a.set_ylabel('Cumulative fraction')
ax_a.legend(fontsize=FS_LEGEND_SMALL, loc='lower right')

# Add median lines
for label in ['Mock', 'Arsenite', 'Ars+CHX']:
    sub = df3[df3['condition_label'] == label]['polya_length'].dropna()
    med = sub.median()
    ax_a.axvline(med, color=colors[label], ls=':', lw=0.7, alpha=0.7)

# Annotation: rescue arrow
ax_a.annotate('', xy=(93.6, 0.45), xytext=(65.9, 0.45),
              arrowprops=dict(arrowstyle='->', color=C_HIGHLIGHT, lw=1.5))
ax_a.text(80, 0.48, 'CHX\nrescue', fontsize=FS_LEGEND_SMALL, ha='center', color=C_HIGHLIGHT,
          fontweight='bold')

# ────────────────────────────────────────────
# Panel (b): Ancient vs Young poly(A) grouped bars
# ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')

# Compute medians by age × condition
age_data = []
for cond, clabel in [('mock', 'Mock'), ('Ars_mock', 'Ars'), ('Ars+CHX', 'Ars+CHX')]:
    sub = df3[df3['condition'] == cond]
    for age_label in ['ancient', 'young']:
        age_sub = sub[sub['age'] == age_label]
        if len(age_sub) >= 5:
            age_data.append({
                'condition': clabel, 'age': age_label,
                'median_polya': age_sub['polya_length'].median(),
                'n': len(age_sub)
            })

age_df = pd.DataFrame(age_data)

# Slope chart: 3 conditions × Ancient/Young connected lines
x = np.arange(3)  # Mock, Ars, Ars+CHX
cond_order = ['Mock', 'Ars', 'Ars+CHX']

anc = [age_df[(age_df['condition'] == c) & (age_df['age'] == 'ancient')]['median_polya'].values[0]
       if len(age_df[(age_df['condition'] == c) & (age_df['age'] == 'ancient')]) > 0 else 0
       for c in cond_order]
young_vals = []
for c in cond_order:
    ydf = age_df[(age_df['condition'] == c) & (age_df['age'] == 'young')]
    young_vals.append(ydf['median_polya'].values[0] if len(ydf) > 0 else np.nan)

# Ancient line
ax_b.plot(x, anc, 'o-', color=C_ANCIENT, lw=1.5, ms=7, zorder=3,
          markeredgecolor='white', markeredgewidth=0.5, label='Ancient')
# Young line (skip NaN)
valid_x_y = [xi for xi, v in zip(x, young_vals) if not np.isnan(v)]
valid_v_y = [v for v in young_vals if not np.isnan(v)]
if valid_v_y:
    ax_b.plot(valid_x_y, valid_v_y, 's-', color=C_YOUNG, lw=1.5, ms=7, zorder=3,
              markeredgecolor='white', markeredgewidth=0.5, label='Young')

# Value labels
for i, v in enumerate(anc):
    ax_b.text(i, v - 6, f'{v:.0f}', ha='center', va='top',
              fontsize=FS_ANNOT_SMALL, color=C_ANCIENT)
for xi, v in zip(valid_x_y, valid_v_y):
    ax_b.text(xi, v + 4, f'{v:.0f}', ha='center', va='bottom',
              fontsize=FS_ANNOT_SMALL, color=C_YOUNG)

ax_b.set_xticks(x)
ax_b.set_xticklabels(cond_order, fontsize=FS_ANNOT)
ax_b.set_ylabel('Median poly(A) (nt)')
ax_b.legend(fontsize=FS_LEGEND_SMALL, loc='upper right')

# Significance bracket for ancient Ars vs Ars+CHX
sig_y = max(anc) + 10
significance_bracket(ax_b, 1, 2, sig_y, 3, '***')

# ────────────────────────────────────────────
# Panel (c): m6A-stratified rescue bar chart
# ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'c')

# Split by m6A median
m6a_med = df3['m6a_per_kb'].median()
df3['m6a_group'] = np.where(df3['m6a_per_kb'] >= m6a_med, 'High m6A', 'Low m6A')

rescue_data = []
for grp in ['Low m6A', 'High m6A']:
    sub = df3[df3['m6a_group'] == grp]
    ars_pa = sub[sub['condition'] == 'Ars_mock']['polya_length'].dropna()
    chx_pa = sub[sub['condition'] == 'Ars+CHX']['polya_length'].dropna()
    mock_pa = sub[sub['condition'] == 'mock']['polya_length'].dropna()
    rescue = chx_pa.median() - ars_pa.median()
    _, p = stats.mannwhitneyu(ars_pa, chx_pa, alternative='two-sided')
    rescue_data.append({'group': grp, 'rescue_nt': rescue, 'p': p,
                        'ars_med': ars_pa.median(), 'chx_med': chx_pa.median(),
                        'mock_med': mock_pa.median()})

rdf = pd.DataFrame(rescue_data)

# Forest plot: horizontal point + CI (bootstrap rescue estimate)
colors_m6a = [C_CTRL, C_L1]
rescue_vals = rdf['rescue_nt'].values

# Bootstrap CI for rescue magnitude
np.random.seed(42)
ci_lo_vals = []
ci_hi_vals = []
for grp in ['Low m6A', 'High m6A']:
    sub = df3[df3['m6a_group'] == grp]
    ars_pa = sub[sub['condition'] == 'Ars_mock']['polya_length'].dropna().values
    chx_pa = sub[sub['condition'] == 'Ars+CHX']['polya_length'].dropna().values
    boot_rescues = []
    for _ in range(2000):
        a_boot = np.median(np.random.choice(ars_pa, size=len(ars_pa), replace=True))
        c_boot = np.median(np.random.choice(chx_pa, size=len(chx_pa), replace=True))
        boot_rescues.append(c_boot - a_boot)
    ci_lo_vals.append(np.percentile(boot_rescues, 2.5))
    ci_hi_vals.append(np.percentile(boot_rescues, 97.5))

forest_plot(ax_c, rdf['group'].values, rescue_vals, ci_lo_vals, ci_hi_vals,
            colors=colors_m6a, ref_line=0, horizontal=True, marker_size=60)
ax_c.set_xlabel('CHX rescue (nt)')
ax_c.invert_yaxis()

# Add p-values
for i, row in rdf.iterrows():
    sig = significance_text(row['p'])
    ax_c.text(ci_hi_vals[i] + 1, i, sig, ha='left', va='center',
              fontsize=FS_ANNOT, fontweight='bold')

# Add subtitle
ax_c.text(0.5, 1.08, 'm6A-independent rescue',
          transform=ax_c.transAxes, ha='center', fontsize=FS_ANNOT_SMALL,
          color='#555', style='italic')

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS8')
print(f'Saved figS8.pdf to {OUTDIR}')

# Print key stats
print('\n=== Key statistics for S8 ===')
for cond in ['mock', 'Ars_mock', 'Ars+CHX']:
    sub = df3[df3['condition'] == cond]['polya_length'].dropna()
    print(f'{cond:12s}: n={len(sub):5d}, median={sub.median():.1f}, mean={sub.mean():.1f}')

print('\nm6A-stratified rescue:')
for _, row in rdf.iterrows():
    print(f"  {row['group']:10s}: Ars={row['ars_med']:.1f} → CHX={row['chx_med']:.1f}, "
          f"Δ=+{row['rescue_nt']:.1f}nt, P={row['p']:.2e}")
