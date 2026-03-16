#!/usr/bin/env python3
"""
Supplementary Figure S5: m6A spatial distribution within L1.
Panels: (a) Body vs Flanking decomposition (m6A level, motif density, m6A/kb)
        (b) m6A/kb along L1 consensus position (ancient L1, 5 regions)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
df_bf = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_validation/m6a_body_vs_flanking_persite_summary.tsv', sep='\t')
df_consensus = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_consensus_position/ancient_m6a_corrected_by_region.tsv', sep='\t')
# Use L1-internal m6A/kb (matches main text values; UTR regions have low overlap → low values)
df_consensus['median_m6a_per_kb'] = df_consensus['median_m6a_in_l1_per_kb']

# ── Create figure: 1 row, 2 columns ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.38))
gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.50,
                      left=0.10, right=0.97, top=0.92, bottom=0.15)

# ────────────────────────────────────────────
# Panel (a): Body vs Flanking decomposition
# ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')

# Extract "all" category
df_all = df_bf[df_bf['category'].str.contains('all', case=False, na=False)].iloc[0]

metrics = ['m6A level (%)', 'DRACH density (/kb)', 'm6A/kb']
body_vals = [float(df_all['rate_body']) * 100,
             float(df_all['motif_per_kb_body']),
             float(df_all['m6a_per_kb_body'])]
flank_vals = [float(df_all['rate_flank']) * 100,
              float(df_all['motif_per_kb_flank']),
              float(df_all['m6a_per_kb_flank'])]
ratios = [b / f for b, f in zip(body_vals, flank_vals)]

y_pos = np.arange(len(metrics))

for i, (m, bv, fv, ratio) in enumerate(zip(metrics, body_vals, flank_vals, ratios)):
    ax_a.plot([fv, bv], [i, i], color='#BDC3C7', lw=2, zorder=1)
    ax_a.scatter(bv, i, color=C_L1, s=S_POINT_LARGE, zorder=2, edgecolors='#2C3E50',
                 linewidths=0.5, marker='s', label='Body' if i == 0 else '')
    ax_a.scatter(fv, i, color='#F5B7B1', s=S_POINT_LARGE, zorder=2, edgecolors='#2C3E50',
                 linewidths=0.5, marker='o', label='Flanking' if i == 0 else '')
    color_txt = C_L1 if ratio > 1 else '#F5B7B1'
    ax_a.text(max(bv, fv) + 0.8, i, f'{ratio:.2f}x', va='center',
              fontsize=FS_ANNOT, fontweight='bold', color=color_txt)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(metrics, fontsize=FS_ANNOT)
ax_a.set_xlabel('Metric value')
ax_a.legend(fontsize=FS_ANNOT, loc='lower right')
ax_a.invert_yaxis()

ax_a.text(0.5, 1.08,
          'Body: lower m6A level, higher motif density -> net higher m6A/kb',
          transform=ax_a.transAxes, ha='center', fontsize=FS_ANNOT_SMALL, color='#555', style='italic')

# ────────────────────────────────────────────
# Panel (b): Consensus position m6A/kb — bar chart
# ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')

# Region labels (from TSV 'region' column)
regions = df_consensus['region'].values
m6a_vals = df_consensus['median_m6a_per_kb'].values
n_reads = df_consensus['n_reads'].values

# Short labels
short_labels = ["5'UTR", "ORF1", "ORF2-5'", "ORF2-3'", "3'UTR"]

# Colors: highlight ORF1 (max m6A)
colors_b = [C_GREY if i != 1 else C_L1 for i in range(len(regions))]

# Lollipop chart (vertical)
lollipop_plot(ax_b, short_labels, m6a_vals, colors=colors_b,
              horizontal=False, marker_size=50, stem_width=2.0, ref_value=0)

# Read count annotation
for i, (val, n) in enumerate(zip(m6a_vals, n_reads)):
    ax_b.text(i, val + 0.08, f'{val:.1f}', ha='center', va='bottom',
              fontsize=FS_ANNOT, fontweight='bold', color=colors_b[i])
    ax_b.text(i, -0.25, f'n={n:,}', ha='center', va='top',
              fontsize=5.5, color='#888888')

ax_b.set_xticks(np.arange(len(regions)))
ax_b.set_xticklabels(short_labels, fontsize=FS_ANNOT)
ax_b.set_ylabel('Median m6A/kb')
ax_b.set_ylim(0, max(m6a_vals) * 1.15)
ax_b.axhline(np.median(m6a_vals), color='#CCCCCC', ls=':', lw=0.7)

ax_b.text(0.97, 0.95, 'ORF1 vs ORF2\nMW P = 0.005', transform=ax_b.transAxes,
          ha='right', va='top', fontsize=FS_ANNOT, color='#888888')

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS5')
print("Fig S5 saved")

print(f"\n[DATA] Panel (a): Body/Flanking rates: {body_vals[0]:.1f}%/{flank_vals[0]:.1f}% = {ratios[0]:.2f}x")
print(f"[DATA] Panel (a): Body/Flanking motif: {body_vals[1]:.1f}/{flank_vals[1]:.1f} = {ratios[1]:.2f}x")
print(f"[DATA] Panel (a): Body/Flanking m6A/kb: {body_vals[2]:.1f}/{flank_vals[2]:.1f} = {ratios[2]:.2f}x")
print(f"[DATA] Panel (b): {len(regions)} consensus regions, {n_reads.sum():,} total reads")
for r, v, n in zip(short_labels, m6a_vals, n_reads):
    print(f"  {r}: m6A/kb={v:.2f}, n={n:,}")
