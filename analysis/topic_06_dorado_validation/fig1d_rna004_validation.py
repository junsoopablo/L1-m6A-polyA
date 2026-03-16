#!/usr/bin/env python3
"""
Figure 1d: RNA004 dorado orthogonal validation — per-read m6A/kb violin + DRACH donut.

Left sub-panel:  Per-read m6A/kb violin (Young L1 / Ancient L1 / non-L1 Control)
Right sub-panel: Donut chart — 539 reproducible sites, 93% DRACH

Data sources:
  - dorado_per_read_m6a.tsv.gz  (per-read m6A/kb with age_class)
  - reproducible_sites.tsv       (539 position-reproducible m6A sites)
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Import shared figure style ──
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')
from fig_style import (
    setup_style, panel_label, save_figure, add_strip, median_line,
    significance_bracket, significance_text,
    C_L1, C_CTRL, C_TEXT, C_GREY,
    HALF_WIDTH, FS_ANNOT, FS_ANNOT_SMALL, FS_LEGEND, FS_LEGEND_SMALL
)
setup_style()

DATADIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results'
OUTDIR  = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures'

# ── Colors matching manuscript style ──
C_YOUNG   = '#CC6677'   # crimson/rose — Young L1 (same as C_L1)
C_ANCIENT = '#E8A0A0'   # salmon/light rose — Ancient L1
C_CONTROL = '#4477AA'   # steel blue — non-L1 control
C_DRACH   = '#CC6677'   # muted rose — DRACH
C_NONDRACH_PIE = '#CCCCCC'  # light grey — non-DRACH

# ── Y-axis display limit ──
YLIM_MAX = 26  # room for violins + significance brackets

# ═════════════════════════════════════════════════════════════
# 1. Load per-read m6A data
# ═════════════════════════════════════════════════════════════
df = pd.read_csv(f'{DATADIR}/dorado_per_read_m6a.tsv.gz', sep='\t')

# Separate groups
young   = df.loc[df['age_class'] == 'Young',   'm6a_per_kb'].values
ancient = df.loc[df['age_class'] == 'Ancient', 'm6a_per_kb'].values
ctrl    = df.loc[df['category']  == 'Control', 'm6a_per_kb'].values

# Combine all L1 for L1 vs Ctrl test
l1_all = df.loc[df['category'] == 'L1', 'm6a_per_kb'].values

print(f"Young L1:   n={len(young):,},  median={np.median(young):.2f}, mean={np.mean(young):.2f}")
print(f"Ancient L1: n={len(ancient):,}, median={np.median(ancient):.2f}, mean={np.mean(ancient):.2f}")
print(f"Control:    n={len(ctrl):,},  median={np.median(ctrl):.2f}, mean={np.mean(ctrl):.2f}")
print(f"L1 (all):   n={len(l1_all):,}, median={np.median(l1_all):.2f}")

# Mann-Whitney U test
stat_lc, p_lc = stats.mannwhitneyu(l1_all, ctrl, alternative='two-sided')
stat_yc, p_yc = stats.mannwhitneyu(young, ctrl, alternative='two-sided')
stat_ac, p_ac = stats.mannwhitneyu(ancient, ctrl, alternative='two-sided')
stat_ya, p_ya = stats.mannwhitneyu(young, ancient, alternative='two-sided')
print(f"\nMWU L1 vs Ctrl:      U={stat_lc:.0f}, P={p_lc:.2e}")
print(f"MWU Young vs Ctrl:   U={stat_yc:.0f}, P={p_yc:.2e}")
print(f"MWU Ancient vs Ctrl: U={stat_ac:.0f}, P={p_ac:.2e}")
print(f"MWU Young vs Ancient: U={stat_ya:.0f}, P={p_ya:.2e}")

# ═════════════════════════════════════════════════════════════
# 2. Load reproducible sites for donut
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
# 3. Create figure — violin only (DRACH bar removed)
# ═════════════════════════════════════════════════════════════
fig, ax_vln = plt.subplots(figsize=(3.0, 3.5))
fig.subplots_adjust(left=0.20, right=0.95, bottom=0.18, top=0.88)
panel_label(ax_vln, 'd', x=-0.22, y=1.10)

positions = [0, 1, 2]
data_list = [young, ancient, ctrl]
colors = [C_YOUNG, C_ANCIENT, C_CONTROL]
labels = ['Young\nL1', 'Ancient\nL1', 'non-L1']

# Manual KDE-based violin with boundary reflection at 0.
# Violin is hard-cut at the 95th percentile per group — no thin spikes.
MAX_HALF_WIDTH = 0.35  # max half-width of each violin

violin_tops = []  # store trim points for strip clipping
for i, (data, pos) in enumerate(zip(data_list, positions)):
    # Hard cut: violin ends at the 95th percentile of each group
    p95 = np.percentile(data, 95)
    trim_upper = min(p95, YLIM_MAX)
    violin_tops.append(trim_upper)

    # KDE grid only spans [0, trim_upper] — no wasted tail
    local_grid = np.linspace(0, trim_upper, 400)
    clipped = data[(data >= 0) & (data <= trim_upper)]

    # Reflect about 0 for boundary correction
    reflected = np.concatenate([clipped, -clipped])
    try:
        kde = gaussian_kde(reflected, bw_method='scott')
        density = kde(local_grid) * 2
    except Exception:
        kde = gaussian_kde(reflected, bw_method=0.5)
        density = kde(local_grid) * 2

    # Normalize density so max = MAX_HALF_WIDTH
    density_norm = density / density.max() * MAX_HALF_WIDTH

    # Draw filled violin — clean top edge, no spike
    ax_vln.fill_betweenx(local_grid, pos - density_norm, pos + density_norm,
                          facecolor=colors[i], alpha=0.55, edgecolor=colors[i],
                          linewidth=0.5, zorder=2)

# Jittered strip overlay — clip each group at its violin top
# so scatter points stay within the violin shape
strip_data = []
for data, vtop in zip(data_list, violin_tops):
    strip_data.append(data[data <= vtop])
add_strip(ax_vln, strip_data, positions, colors=colors, size=1.0, alpha=0.15,
          jitter=0.12, seed=42)

# Median lines + annotations (use full unclipped data for stats)
for i, (data, pos) in enumerate(zip(data_list, positions)):
    med = np.median(data)
    # White background line for contrast
    ax_vln.hlines(med, pos - 0.22, pos + 0.22, color='white', lw=2.0, zorder=4)
    # Thin dark line on top
    ax_vln.hlines(med, pos - 0.22, pos + 0.22, color='black', lw=0.7, zorder=5)
    # Median value annotation
    ax_vln.text(pos, med + 0.7, f'{med:.1f}',
                ha='center', va='bottom', fontsize=FS_ANNOT,
                fontweight='bold', color=colors[i])

# Significance brackets — positioned above violins with enough clearance
h_brack = 0.35
# Place Ancient-vs-Ctrl bracket above the taller of the two (Ancient/Ctrl)
brack1_y = max(violin_tops[1], violin_tops[2]) + 0.8
significance_bracket(ax_vln, 1, 2, brack1_y, h_brack,
                     f'P={p_ac:.1e}', fontsize=FS_ANNOT_SMALL)

# Place Young-vs-Ancient bracket above Young violin top
brack2_y = max(brack1_y + 1.8, violin_tops[0] + 0.8)
significance_bracket(ax_vln, 0, 1, brack2_y, h_brack,
                     f'P={p_ya:.1e}', fontsize=FS_ANNOT_SMALL)

# Set ylim to accommodate brackets with a little padding
ylim_top = brack2_y + 2.5

ax_vln.set_xticks(positions)
ax_vln.set_xticklabels(labels)
ax_vln.set_ylabel('m6A / kb (per read)')
ax_vln.set_ylim(-0.5, ylim_top)
ax_vln.set_yticks([0, 5, 10, 15, 20, 25])
ax_vln.set_xlim(-0.6, 2.6)
ax_vln.set_title('RNA004 dorado', fontsize=7, pad=6, fontstyle='italic')

# N counts below x-tick labels using axis transform
trans = ax_vln.get_xaxis_transform()
for i, (data, pos) in enumerate(zip(data_list, positions)):
    ax_vln.text(pos, -0.17, f'n={len(data):,}', transform=trans,
                ha='center', va='top', fontsize=FS_LEGEND_SMALL, color=C_GREY)

# DRACH annotation on violin panel (replaces removed bar chart)
ax_vln.text(0.98, 0.03, f'{pct_drach:.0f}% DRACH\n({n_total} reprod. sites)',
            transform=ax_vln.transAxes, ha='right', va='bottom',
            fontsize=FS_ANNOT_SMALL, color=C_GREY)

# ═════════════════════════════════════════════════════════════
# 4. Save
# ═════════════════════════════════════════════════════════════
outpath = f'{OUTDIR}/fig1d_rna004_validation'
save_figure(fig, outpath)
print(f"\nSaved: {outpath}.pdf, {outpath}.svg")
