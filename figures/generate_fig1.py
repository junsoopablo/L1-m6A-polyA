#!/usr/bin/env python3
"""
Figure 1: L1 RNA carries elevated m6A density.
Generates individual panel PDFs: fig1a.pdf ... fig1e.pdf
Then merges them into fig1.pdf
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

# ── Load data (shared) ──
df_per_group = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_validation/m6a_validation_per_group.tsv', sep='\t')
df_per_group['cell_line'] = df_per_group['group'].str.replace(r'_\d+$', '', regex=True)
# Filter to groups with complete motif data (base groups only)
df_per_group = df_per_group.dropna(subset=['l1_motif_per_kb', 'ctrl_motif_per_kb']).copy()
df_hist = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_validation/m6a_ml_probability_histogram.tsv', sep='\t')

raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    grp = os.path.basename(os.path.dirname(os.path.dirname(f)))
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'qc_tag', 'gene_id'])
    tmp['group'] = grp
    raw_frames.append(tmp)
df_raw = pd.concat(raw_frames, ignore_index=True)
df_raw = df_raw[df_raw['qc_tag'] == 'PASS'].copy()
df_raw['is_young'] = df_raw['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')

cache_frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    grp = os.path.basename(f).replace('_l1_per_read.tsv', '')
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    tmp['group'] = grp
    cache_frames.append(tmp)
df_l1_cache = pd.concat(cache_frames, ignore_index=True)
df_l1_cache['m6a_per_kb'] = df_l1_cache['m6a_sites_high'] / (df_l1_cache['read_length'] / 1000)

ctrl_frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_ctrl_per_read_cache/*_ctrl_per_read.tsv')):
    ctrl_frames.append(pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high']))
df_ctrl_cache = pd.concat(ctrl_frames, ignore_index=True)
df_ctrl_cache['m6a_per_kb'] = df_ctrl_cache['m6a_sites_high'] / (df_ctrl_cache['read_length'] / 1000)

df_l1_m6a = df_raw.merge(df_l1_cache[['read_id', 'm6a_per_kb']], on='read_id', how='inner')
young_m6a = df_l1_m6a[df_l1_m6a['is_young']]['m6a_per_kb'].dropna().values
ancient_m6a = df_l1_m6a[~df_l1_m6a['is_young']]['m6a_per_kb'].dropna().values
ctrl_m6a = df_ctrl_cache['m6a_per_kb'].dropna().values

# ═══════════════════════════════════════════
# Panel (a): ECDF m6A/kb
# ═══════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'a')

all_m6a = np.concatenate([young_m6a, ancient_m6a, ctrl_m6a])
xlim_a = np.ceil(np.percentile(all_m6a, 99.5))
xlim_a = max(8, min(15, xlim_a))
ecdf_plot(ax, np.clip(young_m6a, 0, xlim_a), C_YOUNG, 'Young')
ecdf_plot(ax, np.clip(ancient_m6a, 0, xlim_a), C_ANCIENT, 'Ancient')
ecdf_plot(ax, np.clip(ctrl_m6a, 0, xlim_a), C_CTRL, 'non-L1')

for data, color in [(young_m6a, C_YOUNG), (ancient_m6a, C_ANCIENT), (ctrl_m6a, C_CTRL)]:
    med = np.median(data)
    ax.axvline(med, color=color, ls=':', lw=0.7, alpha=0.7)

ax.set_xlim(0, xlim_a); ax.set_ylim(0, 1.02)
ax.set_xlabel('m6A sites per kb'); ax.set_ylabel('Cumulative fraction')
ax.legend(fontsize=FS_LEGEND_SMALL, loc='lower right')
save_figure(fig_a, f'{OUTDIR}/fig1a')
print(f"fig1a: {len(young_m6a)+len(ancient_m6a)+len(ctrl_m6a):,} reads")

# ═══════════════════════════════════════════
# Panel (b): Per-library m6A/kb scatter — Young vs Ancient L1
# ═══════════════════════════════════════════
fig_b, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'b')

# Compute per-group Young/Ancient median m6A/kb
df_l1_m6a_grp = df_raw.merge(df_l1_cache[['read_id', 'm6a_per_kb', 'group']],
                               on=['read_id', 'group'], how='inner')
_young_grp = (df_l1_m6a_grp[df_l1_m6a_grp['is_young']]
              .groupby('group').agg(young_m6a=('m6a_per_kb', 'median'),
                                    young_n=('m6a_per_kb', 'size'))
              .reset_index())
_anc_grp = (df_l1_m6a_grp[~df_l1_m6a_grp['is_young']]
            .groupby('group').agg(ancient_m6a=('m6a_per_kb', 'median'),
                                   ancient_n=('m6a_per_kb', 'size'))
            .reset_index())
df_b = (df_per_group[['group', 'ctrl_m6a_per_kb', 'cell_line']]
        .merge(_young_grp, on='group', how='left')
        .merge(_anc_grp, on='group', how='left'))

# Filter: require ≥10 reads for reliable median
MIN_N = 10
df_b_young = df_b[df_b['young_n'] >= MIN_N].copy()
df_b_anc = df_b[df_b['ancient_n'] >= MIN_N].copy()

ax.scatter(df_b_young['ctrl_m6a_per_kb'], df_b_young['young_m6a'],
           s=S_POINT, c=C_YOUNG, marker='^',
           edgecolors='white', linewidths=0.5, zorder=4, alpha=0.85,
           label=f'Young L1 ({len(df_b_young)} lib.)')
ax.scatter(df_b_anc['ctrl_m6a_per_kb'], df_b_anc['ancient_m6a'],
           s=S_POINT, c=C_ANCIENT, marker='o',
           edgecolors='white', linewidths=0.5, zorder=3, alpha=0.85,
           label=f'Ancient L1 ({len(df_b_anc)} lib.)')

all_x = df_b_anc['ctrl_m6a_per_kb'].values
all_y = np.concatenate([df_b_young['young_m6a'].dropna().values,
                        df_b_anc['ancient_m6a'].dropna().values])
lim_lo = min(all_x.min(), all_y.min()) - 0.5
lim_hi = max(all_x.max(), all_y.max()) + 0.5
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color='#C7C7C7', lw=LW_REF, ls='--')
ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel('non-L1 m6A/kb'); ax.set_ylabel('L1 m6A/kb')
ax.legend(fontsize=FS_LEGEND_SMALL, loc='lower right')

save_figure(fig_b, f'{OUTDIR}/fig1b')
print(f"fig1b: Young {len(df_b_young)} lib, Ancient {len(df_b_anc)} lib")

# ═══════════════════════════════════════════
# Panel (c): DRACH motif density vs m6A density (fold-enrichment comparison)
# ═══════════════════════════════════════════
fig_c, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'c')

# Per-library ratios
motif_ratios = df_per_group['l1_motif_per_kb'].values / df_per_group['ctrl_motif_per_kb'].values
m6a_ratios = df_per_group['l1_m6a_per_kb'].values / df_per_group['ctrl_m6a_per_kb'].values

positions = [0, 1]

# Paired dot plot: connect DRACH and m6A ratios for each library
for i in range(len(motif_ratios)):
    ax.plot(positions, [motif_ratios[i], m6a_ratios[i]], color='#BDC3C7',
            lw=LW_CONNECT, zorder=1, alpha=0.6)

ax.scatter([positions[0]] * len(motif_ratios), motif_ratios,
           s=18, color=C_CTRL, edgecolors='white', linewidths=0.4, zorder=4, alpha=0.8)
ax.scatter([positions[1]] * len(m6a_ratios), m6a_ratios,
           s=18, color=C_L1, edgecolors='white', linewidths=0.4, zorder=4, alpha=0.8)

# Reference line at 1.0
ax.axhline(1.0, color='#CCCCCC', lw=LW_REF, ls='--')

# Mean markers (diamond)
mean_motif = np.mean(motif_ratios)
mean_m6a = np.mean(m6a_ratios)
ax.scatter(positions[0], mean_motif, s=60, color=C_CTRL, edgecolors='#2C3E50',
           linewidths=0.8, zorder=5, marker='D')
ax.scatter(positions[1], mean_m6a, s=60, color=C_L1, edgecolors='#2C3E50',
           linewidths=0.8, zorder=5, marker='D')

# Mean value labels
ax.text(positions[0] - 0.15, mean_motif, f'{mean_motif:.2f}x', ha='right', va='center',
        fontsize=FS_ANNOT, fontweight='bold', color=C_CTRL)
ax.text(positions[1] + 0.15, mean_m6a, f'{mean_m6a:.2f}x', ha='left', va='center',
        fontsize=FS_ANNOT, fontweight='bold', color=C_L1)


ax.set_xticks(positions)
ax.set_xticklabels(['DRACH\nmotifs/kb', 'm6A\nsites/kb'], fontsize=FS_ANNOT)
ax.set_ylabel('L1 / non-L1 ratio')
ax.set_xlim(-0.4, 1.6); ax.set_ylim(0.85, max(m6a_ratios) + 0.15)

save_figure(fig_c, f'{OUTDIR}/fig1c')
print(f"fig1c: DRACH ratio={mean_motif:.3f}, m6A ratio={mean_m6a:.3f}")

# ═══════════════════════════════════════════
# Panel (d): Threshold-free enrichment sweep
# ═══════════════════════════════════════════
fig_d, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
prob = df_hist['prob_value'].values
l1_cumrev = np.cumsum(df_hist['l1_count'].values[::-1])[::-1]
ctrl_cumrev = np.cumsum(df_hist['ctrl_count'].values[::-1])[::-1]
ratio = (l1_cumrev / l1_cumrev[0]) / (ctrl_cumrev / ctrl_cumrev[0])

ax.plot(prob, ratio, color=C_L1, lw=1.0, zorder=2)
ax.fill_between(prob, 1.0, ratio, where=(ratio>1), alpha=0.12, color=C_L1, lw=0)
ax.axhline(1.0, color='#CCCCCC', lw=LW_REF, ls='--')

# Mark primary threshold (204/255 = 80%) and alternative
for thr, tx, ty in [(204, 8, -3), (128, -12, 8)]:
    if thr < len(ratio):
        lbl = f'{ratio[thr]:.2f}'
        is_primary = (thr == 204)
        ax.scatter(thr, ratio[thr], color=C_HIGHLIGHT if is_primary else C_GREY,
                   s=S_POINT if is_primary else 25, zorder=4,
                   edgecolors='white', linewidths=0.4)
        ax.annotate(f'{lbl}x', xy=(thr, ratio[thr]), xytext=(tx, ty),
                    textcoords='offset points', fontsize=FS_ANNOT,
                    color=C_HIGHLIGHT if is_primary else '#999999',
                    fontweight='bold' if is_primary else 'normal')

ax.set_xlim(0,255); ax.set_ylim(0.8,1.7)
ax.set_xlabel('ML probability threshold'); ax.set_ylabel('Enrichment (L1 / non-L1)')
ax.text(0.04, 0.96, 'Enrichment increases\nwith stringency',
        transform=ax.transAxes, fontsize=FS_ANNOT, va='top', color='#888888')
save_figure(fig_d, f'{OUTDIR}/figS15')
print(f"figS15 (threshold sweep): {len(prob)} threshold bins")

# NOTE: Fig 1d (RNA004 dorado) and Fig 1e-f (consensus hotspot) are generated by
# topic_06_dorado_validation/fig1d_rna004_validation.py and fig1ef_consensus_hotspot.py
# METTL3 KO panels were removed from the paper (2026-02-24).
print("\nAll Fig 1 panels saved (1a-c + figS15).")
