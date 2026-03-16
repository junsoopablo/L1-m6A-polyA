#!/usr/bin/env python3
"""
Figure 1d: RNA004 dorado orthogonal validation — DRACH m6A specificity in L1.

Left sub-panel: Grouped bar chart of m6A methylation rate (% A positions with
    m6A probability >= 204/255) at DRACH vs non-DRACH sites, for L1 and non-L1.
Right sub-panel: Donut chart showing DRACH fraction among 539 position-reproducible
    high-confidence m6A sites in L1.

Data sources:
  - drach_comparison_summary.tsv  (threshold-level aggregate rates)
  - reproducible_sites.tsv        (539 position-reproducible m6A sites)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyArrowPatch

# ── Import shared figure style ──
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')
from fig_style import (
    setup_style, panel_label, save_figure,
    C_L1, C_CTRL, C_TEXT, C_GREY,
    HALF_WIDTH, FS_ANNOT, FS_ANNOT_SMALL, FS_LEGEND, FS_LEGEND_SMALL
)
setup_style()

DATADIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Colors ──
C_DRACH    = '#CC6677'   # muted rose — DRACH (consistent: DRACH = primary signal)
C_NONDRACH = '#BBBBBB'   # neutral grey — non-DRACH (background)
C_L1_BAR   = '#CC6677'   # L1 accent (rose)
C_NL_BAR   = '#4477AA'   # non-L1 accent (steel)

# ═════════════════════════════════════════════════════════════
# 1. Load DRACH comparison data
# ═════════════════════════════════════════════════════════════
df_drach = pd.read_csv(f'{DATADIR}/drach_comparison_summary.tsv', sep='\t')
df_drach = df_drach.set_index('category')

# m6A methylation rate = n_above_204 / n_A_positions * 100 (%)
l1_drach_rate    = df_drach.loc['L1_DRACH', 'n_above_204'] / df_drach.loc['L1_DRACH', 'n_A_positions'] * 100
l1_nondrach_rate = df_drach.loc['L1_nonDRACH', 'n_above_204'] / df_drach.loc['L1_nonDRACH', 'n_A_positions'] * 100
nl_drach_rate    = df_drach.loc['nonL1_DRACH', 'n_above_204'] / df_drach.loc['nonL1_DRACH', 'n_A_positions'] * 100
nl_nondrach_rate = df_drach.loc['nonL1_nonDRACH', 'n_above_204'] / df_drach.loc['nonL1_nonDRACH', 'n_A_positions'] * 100

l1_ratio = l1_drach_rate / l1_nondrach_rate
nl_ratio = nl_drach_rate / nl_nondrach_rate

print(f"L1 DRACH m6A rate:     {l1_drach_rate:.2f}%")
print(f"L1 non-DRACH m6A rate: {l1_nondrach_rate:.2f}%")
print(f"L1 DRACH/non-DRACH:    {l1_ratio:.1f}x")
print(f"nonL1 DRACH m6A rate:     {nl_drach_rate:.2f}%")
print(f"nonL1 non-DRACH m6A rate: {nl_nondrach_rate:.2f}%")
print(f"nonL1 DRACH/non-DRACH:    {nl_ratio:.1f}x")

# ═════════════════════════════════════════════════════════════
# 2. Load reproducible sites — count DRACH vs non-DRACH
# ═════════════════════════════════════════════════════════════
df_repro = pd.read_csv(f'{DATADIR}/reproducible_sites.tsv', sep='\t')
n_total = len(df_repro)
n_drach = int(df_repro['is_drach'].sum())
n_nondrach = n_total - n_drach
pct_drach = n_drach / n_total * 100

print(f"\nReproducible sites: {n_total}")
print(f"  DRACH:     {n_drach} ({pct_drach:.1f}%)")
print(f"  non-DRACH: {n_nondrach} ({100 - pct_drach:.1f}%)")

# ═════════════════════════════════════════════════════════════
# 3. Create figure — 2 sub-panels side by side
# ═════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(HALF_WIDTH, HALF_WIDTH * 0.78))
gs = GridSpec(1, 2, width_ratios=[2.0, 1], wspace=0.40, figure=fig,
              left=0.14, right=0.96, bottom=0.15, top=0.87)

# ─── Left panel: Grouped bar chart ───
ax_bar = fig.add_subplot(gs[0])
panel_label(ax_bar, 'd', x=-0.22, y=1.12)

bar_width = 0.30
x_pos = np.array([0, 1])  # L1, non-L1

# All bars use consistent DRACH/non-DRACH colors but with L1/nonL1 hue tinting
# L1 group: rose-tinted, nonL1 group: blue-tinted
drach_vals = [l1_drach_rate, nl_drach_rate]
nondrach_vals = [l1_nondrach_rate, nl_nondrach_rate]

# Use hatching to distinguish DRACH vs non-DRACH within each group
# DRACH = solid fill, non-DRACH = lighter fill
group_colors_drach    = [C_L1_BAR, C_NL_BAR]
group_colors_nondrach = ['#E8B4BC', '#88CCEE']  # lighter versions

bars_drach = ax_bar.bar(x_pos - bar_width / 2, drach_vals, bar_width,
                        color=group_colors_drach, edgecolor='white', linewidth=0.5,
                        zorder=3)
bars_nondrach = ax_bar.bar(x_pos + bar_width / 2, nondrach_vals, bar_width,
                           color=group_colors_nondrach, edgecolor='white', linewidth=0.5,
                           zorder=3)

# Value labels on bars
for bar, val in zip(list(bars_drach) + list(bars_nondrach),
                    drach_vals + nondrach_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=FS_ANNOT_SMALL, color=C_TEXT)

# Ratio annotations with thin bracket
for i, ratio in enumerate([l1_ratio, nl_ratio]):
    x_l = x_pos[i] - bar_width / 2
    x_r = x_pos[i] + bar_width / 2
    y_top = drach_vals[i] + 1.0  # above the taller DRACH bar
    h = 0.3
    # Bracket
    ax_bar.plot([x_l, x_l, x_r, x_r], [y_top, y_top + h, y_top + h, y_top],
                lw=0.6, color=C_TEXT, zorder=4)
    ax_bar.text(x_pos[i], y_top + h + 0.15, f'{ratio:.1f}x',
                ha='center', va='bottom',
                fontsize=FS_ANNOT, fontweight='bold', color=C_TEXT)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(['L1', 'non-L1'])
ax_bar.set_ylabel('m6A rate (%)')
ax_bar.set_ylim(0, max(drach_vals) * 1.55)

# Legend — use semantic colors (dark = DRACH, light = non-DRACH)
legend_elements = [
    Patch(facecolor='#888888', edgecolor='none', label='DRACH'),
    Patch(facecolor='#CCCCCC', edgecolor='none', label='non-DRACH'),
]
ax_bar.legend(handles=legend_elements, fontsize=FS_LEGEND_SMALL,
              loc='upper left', frameon=False,
              handlelength=1.0, handleheight=0.8)

ax_bar.set_title('RNA004 dorado', fontsize=7, pad=4)

# ─── Right panel: Donut chart — reproducible sites motif composition ───
ax_pie = fig.add_subplot(gs[1])

wedge_colors = [C_DRACH, '#CCCCCC']
wedges, texts, autotexts = ax_pie.pie(
    [n_drach, n_nondrach],
    colors=wedge_colors,
    autopct='',  # manual labels
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2, 'width': 0.45},
    textprops={'fontsize': FS_ANNOT, 'color': C_TEXT},
)

# Center text: total count
ax_pie.text(0, 0, f'{n_total}', ha='center', va='center',
            fontsize=8, fontweight='bold', color=C_TEXT)
ax_pie.text(0, -0.18, 'sites', ha='center', va='center',
            fontsize=FS_LEGEND_SMALL, color=C_TEXT)

# DRACH percentage — inside the large wedge
ax_pie.text(-0.15, 0.62, f'{pct_drach:.0f}%', ha='center', va='center',
            fontsize=FS_ANNOT, fontweight='bold', color='white')

# non-DRACH — outside with a thin leader line
angle_nondrach = 90 + 360 * pct_drach / 100 + 360 * (100 - pct_drach) / 200
angle_rad = np.radians(90 - (360 * (1 - pct_drach / 100) / 2))  # midpoint of non-DRACH wedge
# Place label to the right of the small wedge
ax_pie.annotate(f'{100 - pct_drach:.0f}%',
                xy=(0.85 * np.cos(angle_rad), 0.85 * np.sin(angle_rad)),
                xytext=(1.35 * np.cos(angle_rad), 1.35 * np.sin(angle_rad)),
                fontsize=FS_ANNOT_SMALL, color=C_TEXT, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color=C_GREY, lw=0.5))

ax_pie.set_title('Reproducible\nm6A sites', fontsize=6.5, pad=4)

# Legend for pie
pie_legend = [
    Patch(facecolor=C_DRACH, edgecolor='white', label='DRACH'),
    Patch(facecolor='#CCCCCC', edgecolor='white', label='non-DRACH'),
]
ax_pie.legend(handles=pie_legend, fontsize=FS_LEGEND_SMALL,
              loc='lower center', bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=1, handlelength=1.0, handleheight=0.8)

# ═════════════════════════════════════════════════════════════
# 4. Save
# ═════════════════════════════════════════════════════════════
outpath = f'{OUTDIR}/fig1d_drach_validation'
save_figure(fig, outpath)
print(f"\nSaved: {outpath}.pdf, {outpath}.svg")
