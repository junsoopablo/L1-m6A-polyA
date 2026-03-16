#!/usr/bin/env python3
"""
Supplementary Figure S2: L1 Poly(A) Tail Cross-Cell-Line Variation.
  (a) ECDF — Poly(A) distributions across cell lines
  (b) Scatter — L1 vs Ctrl poly(A) median per cell line
  (c) Heatmap — Top 20 hotspot poly(A) across cell lines
Layout: [a | b] / [c (wide)]
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

# ── Load raw per-read data ──
raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'polya_length',
                                             'qc_tag', 'gene_id', 'transcript_id'])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['group'] = group
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    raw_frames.append(tmp)
df_raw = pd.concat(raw_frames, ignore_index=True)
df_raw = df_raw[df_raw['qc_tag'] == 'PASS'].copy()
df_raw['is_young'] = df_raw['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')

# Summaries
df_polya = pd.read_csv(f'{BASE}/topic_05_cellline/part2_l1_vs_control_polya.tsv', sep='\t')
df_hotspot = pd.read_csv(f'{BASE}/topic_05_cellline/part2_hotspot_consistency.tsv', sep='\t')

# ── Create figure: [a|b] top, [c wide] bottom ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.40,
                      left=0.09, right=0.97, top=0.96, bottom=0.08,
                      height_ratios=[1, 1.1])

# ────────────────────────────────────────────
# Panel (a): ECDF — Poly(A) distributions across cell lines
# ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')

cl_order_cdf = ['A549', 'H9', 'HEYA8', 'MCF7', 'MCF7-EV',
                'HepG2', 'K562', 'Hct116', 'SHSY5Y',
                'HeLa', 'HeLa-Ars']

# Background CLs in grey
bg_medians = []
for cl in cl_order_cdf:
    if cl in ['HeLa', 'HeLa-Ars']:
        continue
    vals = df_raw[df_raw['cell_line'] == cl]['polya_length'].dropna().values
    if len(vals) > 10:
        ecdf_plot(ax_a, np.clip(vals, 0, 300), '#CCCCCC', None, lw=0.6, ls='-')
        bg_medians.append(np.median(vals))

if bg_medians:
    ax_a.text(0.97, 0.55, f'9 unstressed CLs\nmedian {min(bg_medians):.0f}\u2013{max(bg_medians):.0f} nt',
              transform=ax_a.transAxes, ha='right', fontsize=FS_ANNOT_SMALL,
              color='#999999', va='top')

# Highlighted CLs
for cl, color, lw in [('HeLa', C_NORMAL, 1.5), ('HeLa-Ars', C_STRESS, 1.5)]:
    vals = df_raw[df_raw['cell_line'] == cl]['polya_length'].dropna().values
    if len(vals) > 10:
        ecdf_plot(ax_a, np.clip(vals, 0, 300), color,
                  f'{cl} ({np.median(vals):.0f} nt, n={len(vals):,})',
                  lw=lw, ls='-')

ax_a.annotate('', xy=(90, 0.50), xytext=(121, 0.50),
              arrowprops=dict(arrowstyle='->', color=C_STRESS, lw=1.5))
ax_a.text(105, 0.53, '$\\Delta$=\u221231 nt', ha='center', fontsize=FS_ANNOT,
          color=C_STRESS, fontweight='bold')

ax_a.set_xlabel('Poly(A) tail length (nt)')
ax_a.set_ylabel('Cumulative fraction')
ax_a.set_xlim(0, 220)
ax_a.set_ylim(0, 1.02)
ax_a.legend(fontsize=FS_ANNOT, loc='lower right')

# ────────────────────────────────────────────
# Panel (b): Scatter — Poly(A) L1 vs Ctrl per CL
# ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')

df_pa = df_polya.copy()
sc = ax_b.scatter(df_pa['ctrl_median'], df_pa['l1_median'], c=df_pa['delta'],
                  cmap='RdBu_r', s=S_POINT_LARGE, edgecolors='#2C3E50', linewidths=0.5,
                  zorder=3, vmin=10, vmax=60)

cluster_cls = set()
outlier_cls = set()
ctrl_vals = df_pa['ctrl_median'].values
l1_vals = df_pa['l1_median'].values
ctrl_med = np.median(ctrl_vals)
l1_med = np.median(l1_vals)
for _, row in df_pa.iterrows():
    dist = np.sqrt((row['ctrl_median'] - ctrl_med)**2 + (row['l1_median'] - l1_med)**2)
    if dist > 20:
        outlier_cls.add(row['cell_line'])
    else:
        cluster_cls.add(row['cell_line'])

try:
    from adjustText import adjust_text
    texts_b = []
    for _, row in df_pa.iterrows():
        if row['cell_line'] in outlier_cls:
            texts_b.append(ax_b.text(row['ctrl_median'], row['l1_median'], row['cell_line'],
                                     fontsize=FS_LEGEND_SMALL))
    adjust_text(texts_b, ax=ax_b, arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
                force_points=0.5, force_text=0.8, expand_points=(2.0, 2.0))
except ImportError:
    for _, row in df_pa.iterrows():
        if row['cell_line'] in outlier_cls:
            ax_b.annotate(row['cell_line'], (row['ctrl_median'], row['l1_median']),
                          fontsize=FS_LEGEND_SMALL, xytext=(4, 4), textcoords='offset points')

if cluster_cls:
    cluster_rows = df_pa[df_pa['cell_line'].isin(cluster_cls)]
    cx = cluster_rows['ctrl_median'].mean()
    cy = cluster_rows['l1_median'].mean()
    ax_b.annotate(f'{len(cluster_cls)} CLs', xy=(cx, cy), xytext=(cx - 18, cy + 20),
                  fontsize=FS_LEGEND_SMALL, color='#777777', style='italic',
                  arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.6))

xlim = [50, 170]
ylim = [50, 175]
ax_b.plot(xlim, xlim, 'k--', lw=0.5, alpha=0.4)
ax_b.plot(xlim, [l + 40 for l in xlim], ':', color=C_GREY, lw=0.5)
ax_b.text(135, 180, '+40 nt', fontsize=FS_LEGEND_SMALL, color=C_GREY, rotation=28)

ax_b.set_xlabel('Control poly(A) median (nt)')
ax_b.set_ylabel('L1 poly(A) median (nt)')
ax_b.set_xlim(xlim)
ax_b.set_ylim(ylim)

cb2 = fig.colorbar(sc, ax=ax_b, fraction=0.03, pad=0.04)
cb2.set_label('L1 $-$ Control (nt)', fontsize=FS_ANNOT)
cb2.ax.tick_params(labelsize=6)

# ────────────────────────────────────────────
# Panel (c): Heatmap — Top 20 hotspot poly(A) across CLs
# ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, :])
panel_label(ax_c, 'c', x=-0.05)

top20 = df_hotspot.nlargest(20, 'n_reads').sort_values('polya_median')
hotspot_loci = top20['locus'].tolist()
main_cls = ['MCF7', 'HepG2', 'A549', 'K562', 'HeLa', 'Hct116', 'H9', 'HEYA8', 'SHSY5Y']

mat_hot = np.full((len(hotspot_loci), len(main_cls)), np.nan)
for i, locus in enumerate(hotspot_loci):
    for j, cl in enumerate(main_cls):
        vals = df_raw[(df_raw['cell_line'] == cl) &
                      (df_raw['transcript_id'] == locus)]['polya_length'].dropna()
        if len(vals) >= 1:
            mat_hot[i, j] = vals.median()

im = ax_c.imshow(mat_hot, aspect='auto', cmap='viridis', vmin=40, vmax=180)
ax_c.set_xticks(range(len(main_cls)))
ax_c.set_xticklabels(main_cls, rotation=45, ha='right', fontsize=FS_ANNOT_SMALL)

subfam_names = [l.split('_dup')[0] for l in hotspot_loci]
seen = {}
y_labels = []
for s in subfam_names:
    seen[s] = seen.get(s, 0) + 1
    suffix = f' ({seen[s]})' if subfam_names.count(s) > 1 else ''
    y_labels.append(f'{s}{suffix}')
ax_c.set_yticks(range(len(hotspot_loci)))
ax_c.set_yticklabels(y_labels, fontsize=5.5)
ax_c.set_ylabel('Hotspot locus (by subfamily)')
ax_c.set_facecolor('#EEEEEE')

mean_cv = top20['between_cl_cv'].mean()
ax_c.set_title(f'Poly(A) is locus-intrinsic (cross-CL CV = {mean_cv:.2f})',
               fontsize=FS_ANNOT, pad=4, color='#555555')

cb = fig.colorbar(im, ax=ax_c, fraction=0.02, pad=0.02)
cb.set_label('Median poly(A) (nt)', fontsize=FS_ANNOT)
cb.ax.tick_params(labelsize=6)

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS2')
print("Fig S2 saved")

n_detected = np.sum(~np.isnan(mat_hot), axis=1)
always = np.sum(n_detected == len(main_cls))
print(f"\n[DATA] {always}/{len(hotspot_loci)} hotspots detected in ALL {len(main_cls)} CLs")
