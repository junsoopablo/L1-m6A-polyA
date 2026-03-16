#!/usr/bin/env python3
"""
Generate Supplementary Figure S12: Cross-cell-line m6A consistency.
4 panels: (a) Young vs Ancient per CL, (b) Subfamily rank correlation heatmap,
(c) Per-locus correlation heatmap, (d) Per-CL median with replicates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations
from fig_style import *

setup_style()

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CACHE = PROJECT / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
SUMMARY_DIR = PROJECT / 'results_group'

CL_GROUPS = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'HCT116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SH-SY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Colors — use fig_style palette (C_YOUNG, C_ANCIENT already imported)
C_MAIN = C_NORMAL  # steel blue for overall bars

# Load data
all_reads = []
for cl, groups in CL_GROUPS.items():
    for grp in groups:
        cache_file = CACHE / f'{grp}_l1_per_read.tsv'
        if not cache_file.exists():
            print(f'  Missing: {cache_file}')
            continue
        df = pd.read_csv(cache_file, sep='\t',
                         usecols=['read_id', 'read_length', 'm6a_sites_high'])
        df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)

        summary_file = SUMMARY_DIR / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if summary_file.exists():
            summ = pd.read_csv(summary_file, sep='\t',
                               usecols=['read_id', 'transcript_id'])
            df = df.merge(summ, on='read_id', how='left')
        else:
            df['transcript_id'] = ''

        df['cell_line'] = cl
        df['group'] = grp
        all_reads.append(df)

data = pd.concat(all_reads, ignore_index=True)
data['subfamily'] = data['transcript_id'].str.split('_dup').str[0]
data['age'] = data['subfamily'].apply(
    lambda x: 'young' if x in YOUNG else 'ancient' if pd.notna(x) and x != '' else 'unknown')

print(f'Loaded {len(data)} reads from {data["cell_line"].nunique()} cell lines')

# Sort CLs by median m6A/kb for consistent ordering
cl_order = data.groupby('cell_line')['m6a_per_kb'].median().sort_values(ascending=False).index.tolist()

# ---- Figure ----
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 1.1))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# =====================================================================
# Panel (a): Young vs Ancient m6A/kb per cell line
# =====================================================================
ax_a = fig.add_subplot(gs[0, 0])

young_meds = []
ancient_meds = []
young_iqr = []
ancient_iqr = []
for cl in cl_order:
    sub = data[data['cell_line'] == cl]
    y = sub[sub['age'] == 'young']['m6a_per_kb']
    a = sub[sub['age'] == 'ancient']['m6a_per_kb']
    young_meds.append(y.median())
    ancient_meds.append(a.median())
    young_iqr.append([y.median() - y.quantile(0.25), y.quantile(0.75) - y.median()])
    ancient_iqr.append([a.median() - a.quantile(0.25), a.quantile(0.75) - a.median()])

x = np.arange(len(cl_order))

# Dumbbell: Ancient → Young connected dots per CL
dumbbell_plot(ax_a, cl_order, ancient_meds, young_meds,
              C_ANCIENT, C_YOUNG, label1='Ancient', label2='Young',
              horizontal=False, marker_size=40, line_width=1.5)
ax_a.set_xticklabels(cl_order, rotation=45, ha='right')
ax_a.set_ylabel('Median m⁶A/kb')
ax_a.legend(fontsize=FS_LEGEND, loc='upper right')
ax_a.set_title('Young > Ancient m⁶A: all 9 cell lines', fontweight='bold')

# Add significance stars
for i, cl in enumerate(cl_order):
    sub = data[data['cell_line'] == cl]
    y = sub[sub['age'] == 'young']['m6a_per_kb']
    a = sub[sub['age'] == 'ancient']['m6a_per_kb']
    if len(y) >= 10:
        _, p = stats.mannwhitneyu(y, a, alternative='greater')
        star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
        ymax = max(young_meds[i], ancient_meds[i])
        ax_a.text(i, ymax + 0.3, star, ha='center', fontsize=FS_ANNOT, color='grey')

# Summary ratio text box instead of per-bar labels
ratios = [young_meds[i] / ancient_meds[i] for i in range(len(cl_order))]
ax_a.text(0.98, 0.55, f'Young/Ancient ratio\n'
          f'median {np.median(ratios):.2f}×\n'
          f'range {min(ratios):.2f}–{max(ratios):.2f}×\n'
          f'CV = {np.std(ratios)/np.mean(ratios):.3f}\n'
          f'all P < 10⁻⁷',
          transform=ax_a.transAxes, ha='right', va='center', fontsize=FS_ANNOT,
          bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='#ccc'))

panel_label(ax_a, 'a')

# =====================================================================
# Panel (b): Subfamily rank correlation heatmap
# =====================================================================
ax_b = fig.add_subplot(gs[0, 1])

top_sf = data['subfamily'].value_counts().head(10).index.tolist()
sf_cl = data[data['subfamily'].isin(top_sf)].groupby(
    ['subfamily', 'cell_line'])['m6a_per_kb'].agg(['median', 'count']).reset_index()
sf_cl.columns = ['subfamily', 'cell_line', 'm6a_median', 'n_reads']

sf_pivot = sf_cl[sf_cl['n_reads'] >= 5].pivot_table(
    index='subfamily', columns='cell_line', values='m6a_median')

cls_sf = [c for c in cl_order if c in sf_pivot.columns]
sf_corr = pd.DataFrame(index=cls_sf, columns=cls_sf, dtype=float)
for cl1, cl2 in combinations(cls_sf, 2):
    shared = sf_pivot[[cl1, cl2]].dropna()
    if len(shared) >= 5:
        r, _ = stats.spearmanr(shared[cl1], shared[cl2])
        sf_corr.loc[cl1, cl2] = r
        sf_corr.loc[cl2, cl1] = r
np.fill_diagonal(sf_corr.values, 1.0)

im_b = ax_b.imshow(sf_corr.values.astype(float), cmap='YlOrRd', vmin=0.4, vmax=1.0, aspect='auto')
ax_b.set_xticks(range(len(cls_sf)))
ax_b.set_xticklabels(cls_sf, rotation=45, ha='right')
ax_b.set_yticks(range(len(cls_sf)))
ax_b.set_yticklabels(cls_sf)

# Annotate values
for i in range(len(cls_sf)):
    for j in range(len(cls_sf)):
        v = sf_corr.iloc[i, j]
        if pd.notna(v) and i != j:
            color = 'white' if v > 0.85 else 'black'
            ax_b.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FS_ANNOT_SMALL, color=color)

cb_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
cb_b.set_label('Spearman ρ', fontsize=FS_CBAR)
ax_b.set_title('Subfamily m⁶A/kb rank correlation', fontweight='bold')
panel_label(ax_b, 'b')

# =====================================================================
# Panel (c): Per-locus correlation heatmap
# =====================================================================
ax_c = fig.add_subplot(gs[1, 0])

locus_cl = data.groupby(['transcript_id', 'cell_line']).agg(
    m6a_median=('m6a_per_kb', 'median'),
    n_reads=('read_id', 'count')
).reset_index()
locus_cl_filt = locus_cl[locus_cl['n_reads'] >= 3]
locus_ncl = locus_cl_filt.groupby('transcript_id')['cell_line'].nunique()
multi_cl_loci = locus_ncl[locus_ncl >= 2].index

pivot = locus_cl_filt[locus_cl_filt['transcript_id'].isin(multi_cl_loci)].pivot_table(
    index='transcript_id', columns='cell_line', values='m6a_median')

cls_loc = [c for c in cl_order if c in pivot.columns]
loc_corr = pd.DataFrame(index=cls_loc, columns=cls_loc, dtype=float)
n_loci_matrix = pd.DataFrame(index=cls_loc, columns=cls_loc, dtype=float)
for cl1, cl2 in combinations(cls_loc, 2):
    shared = pivot[[cl1, cl2]].dropna()
    if len(shared) >= 10:
        r, _ = stats.spearmanr(shared[cl1], shared[cl2])
        loc_corr.loc[cl1, cl2] = r
        loc_corr.loc[cl2, cl1] = r
        n_loci_matrix.loc[cl1, cl2] = len(shared)
        n_loci_matrix.loc[cl2, cl1] = len(shared)
np.fill_diagonal(loc_corr.values, 1.0)

im_c = ax_c.imshow(loc_corr.values.astype(float), cmap='YlGnBu', vmin=0.5, vmax=1.0, aspect='auto')
ax_c.set_xticks(range(len(cls_loc)))
ax_c.set_xticklabels(cls_loc, rotation=45, ha='right')
ax_c.set_yticks(range(len(cls_loc)))
ax_c.set_yticklabels(cls_loc)

for i in range(len(cls_loc)):
    for j in range(len(cls_loc)):
        v = loc_corr.iloc[i, j]
        if pd.notna(v) and i != j:
            color = 'white' if v > 0.78 else 'black'
            ax_c.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FS_ANNOT_SMALL, color=color)

cb_c = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
cb_c.set_label('Spearman ρ', fontsize=FS_CBAR)
ax_c.set_title(f'Per-locus m⁶A/kb consistency (n={len(multi_cl_loci)} loci)', fontweight='bold')
panel_label(ax_c, 'c')

# =====================================================================
# Panel (d): Per-CL median with replicate dots
# =====================================================================
ax_d = fig.add_subplot(gs[1, 1])

cl_medians = []
for cl in cl_order:
    sub = data[data['cell_line'] == cl]
    cl_medians.append(sub['m6a_per_kb'].median())

# Strip plot: replicate dots + median tick (no bars)
for i, cl in enumerate(cl_order):
    groups_cl = CL_GROUPS[cl]
    rep_meds = []
    for grp in groups_cl:
        sub = data[(data['cell_line'] == cl) & (data['group'] == grp)]
        if len(sub) > 0:
            rep_meds.append(sub['m6a_per_kb'].median())
    jitter = np.linspace(-0.12, 0.12, len(rep_meds))
    ax_d.scatter([i + j for j in jitter], rep_meds, color=C_MAIN, s=25, zorder=5,
                 alpha=0.8, edgecolors='white', linewidths=0.4)
    # Median tick
    ax_d.hlines(cl_medians[i], i - 0.18, i + 0.18, color='black', lw=1.2, zorder=6)

ax_d.set_xticks(range(len(cl_order)))
ax_d.set_xticklabels(cl_order, rotation=45, ha='right')
ax_d.set_ylabel('Median m⁶A/kb')
ax_d.set_title('Per-cell-line m⁶A/kb with replicates', fontweight='bold')

# Annotate overall stats
m6a_arr = np.array(cl_medians)
cv = m6a_arr.std() / m6a_arr.mean()
ax_d.text(0.98, 0.95, f'CV = {cv:.3f}\nRange: {m6a_arr.min():.1f}–{m6a_arr.max():.1f}',
          transform=ax_d.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

panel_label(ax_d, 'd')

OUTDIR = os.path.dirname(os.path.abspath(__file__))
save_figure(fig, f'{OUTDIR}/figS12')
print(f'Saved: {OUTDIR}/figS12.pdf')
