#!/usr/bin/env python3
"""
Figure 2: Arsenite selectively shortens L1 poly(A).
Generates individual panel PDFs: fig2a.pdf ... fig2d.pdf
  (a) ECDF poly(A) — L1 vs Ctrl x HeLa vs Ars
  (b) Cleveland dot plot — Cross-CL validation
  (c) Violin+strip — Ancient vs Young poly(A) immunity
  (d) Violin+strip — ChromHMM x arsenite per chromatin group
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import glob
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
RESULTS = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load raw L1 poly(A) ──
raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'polya_length', 'qc_tag', 'gene_id'])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    raw_frames.append(tmp)
df_l1 = pd.concat(raw_frames, ignore_index=True)
df_l1 = df_l1[df_l1['qc_tag'] == 'PASS'].copy()
df_l1['is_young'] = df_l1['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')

# ── Load raw Control poly(A) ──
ctrl_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/i_control/*_control_summary.tsv')):
    group = os.path.basename(f).replace('_control_summary.tsv', '')
    cl = group.rsplit('_', 1)[0]
    if cl in ['HeLa', 'HeLa-Ars']:
        try:
            tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'polya_length'])
            tmp['cell_line'] = cl
            ctrl_frames.append(tmp)
        except Exception:
            pass
df_ctrl = pd.concat(ctrl_frames, ignore_index=True) if ctrl_frames else pd.DataFrame()

# ── Load cross-CL validation ──
df_crosscl = pd.read_csv(f'{BASE}/topic_05_cellline/pdf_figures_part4/part4_cross_cl_validation.tsv', sep='\t')

# ── Load ChromHMM annotated data ──
df_chrom = pd.read_csv(f'{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv', sep='\t')


# ═══════════════════════════════════════════
# Panel (a): ECDF — L1 vs Control poly(A), HeLa vs Ars
# ═══════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'b')  # new Fig 1b (was Fig 2a)

l1_hela = df_l1[df_l1['cell_line'] == 'HeLa']['polya_length'].dropna().values
l1_ars = df_l1[df_l1['cell_line'] == 'HeLa-Ars']['polya_length'].dropna().values
ctrl_hela = df_ctrl[df_ctrl['cell_line'] == 'HeLa']['polya_length'].dropna().values if len(df_ctrl) > 0 else np.array([])
ctrl_ars = df_ctrl[df_ctrl['cell_line'] == 'HeLa-Ars']['polya_length'].dropna().values if len(df_ctrl) > 0 else np.array([])

xlim_polya = 300

ecdf_plot(ax, np.clip(l1_hela, 0, xlim_polya), C_NORMAL,
          'L1 HeLa', lw=LW_DATA)
ecdf_plot(ax, np.clip(l1_ars, 0, xlim_polya), C_STRESS,
          'L1 Arsenite', lw=LW_DATA)
if len(ctrl_hela) > 0:
    ecdf_plot(ax, np.clip(ctrl_hela, 0, xlim_polya), C_CTRL,
              'non-L1 HeLa', lw=LW_DATA_SEC, ls='--')
if len(ctrl_ars) > 0:
    ecdf_plot(ax, np.clip(ctrl_ars, 0, xlim_polya), C_GREY,
              'non-L1 Arsenite', lw=LW_DATA_SEC, ls='--')

med_l1h = np.median(l1_hela)
med_l1a = np.median(l1_ars)
delta = med_l1a - med_l1h
ax.annotate('', xy=(med_l1a, 0.35), xytext=(med_l1h, 0.35),
              arrowprops=dict(arrowstyle='->', color=C_STRESS, lw=LW_DATA))
ax.text((med_l1h + med_l1a) / 2, 0.37,
          f'$\\Delta$={delta:.0f} nt', ha='center', va='bottom',
          fontsize=FS_ANNOT, color=C_STRESS)

ax.set_xlim(0, xlim_polya); ax.set_ylim(0, 1.02)
ax.set_xlabel('Poly(A) length (nt)'); ax.set_ylabel('Cumulative fraction')
ax.legend(fontsize=FS_ANNOT_SMALL, loc='lower right', handlelength=1.4)
save_figure(fig_a, f'{OUTDIR}/fig2a')
print(f"fig2a: {len(l1_hela):,} L1 HeLa + {len(l1_ars):,} L1 Ars + {len(ctrl_hela):,} Ctrl HeLa + {len(ctrl_ars):,} Ctrl Ars")


# ═══════════════════════════════════════════
# Panel (c): Violin+strip — Ancient vs Young immunity
# ═══════════════════════════════════════════
fig_b, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'c')  # new Fig 1c (was Fig 2c)

anc_hela = df_l1[(~df_l1['is_young']) & (df_l1['cell_line'] == 'HeLa')]['polya_length'].dropna().values
anc_ars = df_l1[(~df_l1['is_young']) & (df_l1['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna().values
yng_hela = df_l1[(df_l1['is_young']) & (df_l1['cell_line'] == 'HeLa')]['polya_length'].dropna().values
yng_ars = df_l1[(df_l1['is_young']) & (df_l1['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna().values

pos_b = [0, 0.7, 1.8, 2.5]
data_b = [np.clip(anc_hela, 0, 300), np.clip(anc_ars, 0, 300),
          np.clip(yng_hela, 0, 300), np.clip(yng_ars, 0, 300)]
colors_b = [C_NORMAL, C_STRESS, C_NORMAL, C_STRESS]
edge_colors_b = [C_ANCIENT, C_STRESS, C_YOUNG, C_STRESS]

for pos, data, fc, ec in zip(pos_b, data_b, colors_b, edge_colors_b):
    if len(data) > 5:
        vp = ax.violinplot([data], positions=[pos],
                             showmedians=False, showextrema=False, widths=0.50)
        for body in vp['bodies']:
            body.set_facecolor(fc)
            body.set_edgecolor(ec)
            body.set_alpha(0.25)
            body.set_linewidth(0.7)

add_strip(ax, data_b, pos_b, colors=colors_b,
          size=1.5, alpha=0.18, jitter=0.13)

for pos, data in zip(pos_b, data_b):
    if len(data) > 5:
        median_line(ax, data, pos, width=0.18, lw=LW_MEDIAN)

ax.set_xticks(pos_b)
ax.set_xticklabels([
    'Ancient\nHeLa',
    'Ancient\nArsenite',
    'Young\nHeLa',
    'Young\nArsenite',
], fontsize=FS_LEGEND_SMALL)
ax.set_ylabel('Poly(A) length (nt)')
ax.set_ylim(0, 300)

delta_anc = np.median(anc_ars) - np.median(anc_hela) if len(anc_ars) > 0 else 0
delta_yng = np.median(yng_ars) - np.median(yng_hela) if len(yng_ars) > 0 and len(yng_hela) > 0 else 0

significance_bracket(ax, 0, 0.7, 268, 3, '')
ax.text(0.35, 276, f'***  $\\Delta$={delta_anc:.0f} nt', ha='center', va='bottom',
          fontsize=FS_ANNOT_SMALL, color=C_STRESS)
if len(yng_hela) > 5 and len(yng_ars) > 5:
    significance_bracket(ax, 1.8, 2.5, 268, 3, '')
    ax.text(2.15, 276, f'ns  $\\Delta$={delta_yng:.0f} nt', ha='center', va='bottom',
              fontsize=FS_ANNOT_SMALL, color='#888888')

save_figure(fig_b, f'{OUTDIR}/fig2c')
print(f"fig2c: {len(anc_hela):,} + {len(anc_ars):,} ancient, {len(yng_hela):,} + {len(yng_ars):,} young")


# ═══════════════════════════════════════════
# Panel (b): Cleveland dot plot — Cross-CL validation
# ═══════════════════════════════════════════
fig_c, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'b')

cl_sorted = df_crosscl.sort_values('median_polya')
y_pos_c = np.arange(len(cl_sorted))

for i, (_, row) in enumerate(cl_sorted.iterrows()):
    ax.hlines(i, 0, row['median_polya'], color='#E8E8E8', lw=LW_REF, zorder=1)
    ax.scatter(row['median_polya'], i, color=C_NORMAL, s=S_POINT, zorder=3,
                 edgecolors='white', linewidths=0.4)

ars_median = 77.5
ax.axvline(x=ars_median, color=C_STRESS, linewidth=LW_DATA_SEC, ls='--', zorder=2)
ax.text(ars_median + 2, len(cl_sorted) - 0.5, 'HeLa\nArsenite',
        ha='left', va='top', fontsize=FS_ANNOT_SMALL, color=C_STRESS)

ax.set_yticks(y_pos_c)
ax.set_yticklabels(cl_sorted['cell_line'].values)
ax.set_xlabel('Median poly(A) length at arsenite-only loci (nt)')
ax.set_xlim(0, 180)

save_figure(fig_c, f'{OUTDIR}/fig2b')
print(f"fig2b: {len(cl_sorted)} cell lines")


# ═══════════════════════════════════════════
# Panel (d): ChromHMM paired violins — per chromatin group
# ═══════════════════════════════════════════
fig_d, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'f')  # new Fig 1f (was Fig 2d)

chrom_ancient = df_chrom[(df_chrom['l1_age'] == 'ancient') &
                          (df_chrom['cellline'].isin(['HeLa', 'HeLa-Ars']))].copy()

state_order = ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent']
state_labels = ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent']
state_base_colors = [C_L1, C_L1, C_GREY, C_GREY]

positions_normal = []
positions_stress = []
data_normal = []
data_stress = []
colors_normal_list = []
colors_stress_list = []

for i, st in enumerate(state_order):
    x_center = i * 1.2
    x_n = x_center - 0.22
    x_s = x_center + 0.22
    positions_normal.append(x_n)
    positions_stress.append(x_s)

    d_n = chrom_ancient[(chrom_ancient['chromhmm_group'] == st) &
                         (chrom_ancient['condition'] == 'normal')]['polya_length'].dropna().values
    d_s = chrom_ancient[(chrom_ancient['chromhmm_group'] == st) &
                         (chrom_ancient['condition'] == 'stress')]['polya_length'].dropna().values
    data_normal.append(np.clip(d_n, 0, 300))
    data_stress.append(np.clip(d_s, 0, 300))
    colors_normal_list.append(C_NORMAL)
    colors_stress_list.append(C_STRESS)

for pos_n, pos_s, d_n, d_s in zip(positions_normal, positions_stress, data_normal, data_stress):
    for pos, data, color in [(pos_n, d_n, C_NORMAL), (pos_s, d_s, C_STRESS)]:
        if len(data) > 10:
            vp = ax.violinplot([data], positions=[pos],
                                 showmedians=False, showextrema=False, widths=0.35)
            for body in vp['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.25)
                body.set_linewidth(0.4)

add_strip(ax, data_normal, positions_normal, colors=colors_normal_list,
          size=1.5, alpha=0.18, jitter=0.08)
add_strip(ax, data_stress, positions_stress, colors=colors_stress_list,
          size=1.5, alpha=0.18, jitter=0.08)

for pos, data in zip(positions_normal + positions_stress, data_normal + data_stress):
    if len(data) > 10:
        median_line(ax, data, pos, width=0.12, lw=LW_MEDIAN)

for i, st in enumerate(state_order):
    if len(data_normal[i]) > 10 and len(data_stress[i]) > 10:
        med_n = np.median(data_normal[i])
        med_s = np.median(data_stress[i])
        delta = med_s - med_n
        x_center = i * 1.2
        color = C_STRESS if abs(delta) > 10 else '#888888'
        ax.text(x_center, 300, f'$\\Delta$={delta:.0f} nt', ha='center',
                  fontsize=FS_ANNOT_SMALL, color=color, clip_on=False)

x_centers = [i * 1.2 for i in range(len(state_order))]
ax.set_xticks(x_centers)
ax.set_xticklabels(state_labels, fontsize=FS_ANNOT, rotation=0, ha='center')
ax.set_ylabel('Poly(A) length (nt)')
ax.set_ylim(0, 295)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_NORMAL, alpha=0.4, label='Normal'),
                   Patch(facecolor=C_STRESS, alpha=0.4, label='Arsenite')]
ax.legend(handles=legend_elements, fontsize=FS_ANNOT_SMALL, loc='upper left')

save_figure(fig_d, f'{OUTDIR}/fig2d')
print(f"fig2d: {sum(len(d) for d in data_normal):,} normal + {sum(len(d) for d in data_stress):,} stress chromatin reads")

print("\nAll Fig 2 panels saved.")
