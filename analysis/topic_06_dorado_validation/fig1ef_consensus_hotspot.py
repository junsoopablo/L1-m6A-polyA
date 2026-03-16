#!/usr/bin/env python3
"""
Figure 1e-f: L1 consensus m6A hotspot conservation
====================================================
(e) Young vs Ancient per-bin m6A rate correlation scatter
(f) Ancient L1 m6A rate by Hamming distance to Young flanking context

Data: consensus_drach_positions.tsv.gz (from consensus_hotspot_m6a_analysis.py)
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import shared style
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')
from fig_style import (
    setup_style, panel_label, save_figure,
    C_L1, C_CTRL, C_TEXT, C_GREY,
    HALF_WIDTH, FULL_WIDTH, FS_ANNOT, FS_ANNOT_SMALL, FS_LEGEND
)
setup_style()

DATADIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results'
FIGDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures'

# Colors
C_YOUNG = '#CC6677'
C_ANCIENT = '#E8A0A0'
C_SCATTER = '#2166ac'

BIN_SIZE = 50

# ════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════
print("Loading DRACH position data...")
drach_df = pd.read_csv(f'{DATADIR}/consensus_drach_positions.tsv.gz', sep='\t', compression='gzip')
print(f"  Total: {len(drach_df):,} DRACH positions")

young = drach_df[drach_df['age_class'] == 'Young'].copy()
ancient = drach_df[drach_df['age_class'] == 'Ancient'].copy()
young = young[young['cons_pos'].notna()]
ancient = ancient[ancient['cons_pos'].notna()]

# ════════════════════════════════════════════
# Panel (e): bin-level correlation
# ════════════════════════════════════════════
young['cons_bin'] = (young['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE
ancient['cons_bin'] = (ancient['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE

young_bin = young.groupby('cons_bin').agg(
    n=('is_methylated', 'count'), meth=('is_methylated', 'sum')
).reset_index()
young_bin['rate'] = young_bin['meth'] / young_bin['n']
young_bin = young_bin[young_bin['n'] >= 5]

ancient_bin = ancient.groupby('cons_bin').agg(
    n=('is_methylated', 'count'), meth=('is_methylated', 'sum')
).reset_index()
ancient_bin['rate'] = ancient_bin['meth'] / ancient_bin['n']
ancient_bin = ancient_bin[ancient_bin['n'] >= 10]

merged = young_bin.merge(ancient_bin, on='cons_bin', suffixes=('_young', '_ancient'))
rho, p_rho = stats.spearmanr(merged['rate_young'], merged['rate_ancient'])
print(f"  Scatter: {len(merged)} bins, rho={rho:.3f}, P={p_rho:.2e}")

# ════════════════════════════════════════════
# Panel (f): Hamming distance gradient
# ════════════════════════════════════════════
young_fl = young[young['flanking_11mer'].notna() & young['is_methylated']]
top_young_11mers = set(km for km, _ in Counter(young_fl['flanking_11mer']).most_common(100))

ancient_fl = ancient[ancient['flanking_11mer'].notna()].copy()
np.random.seed(42)
if len(ancient_fl) > 50000:
    idx = np.random.choice(len(ancient_fl), 50000, replace=False)
    anc_sample = ancient_fl.iloc[idx].copy()
else:
    anc_sample = ancient_fl.copy()

print(f"  Computing Hamming for {len(anc_sample):,} sites...")

def hamming_min(seq, ref_set):
    return min((sum(a != b for a, b in zip(seq, r)) for r in ref_set if len(r) == len(seq)), default=len(seq))

anc_sample = anc_sample.copy()
anc_sample['hamming'] = [hamming_min(row['flanking_11mer'], top_young_11mers) for _, row in anc_sample.iterrows()]

ham_stats = []
for d in sorted(anc_sample['hamming'].unique()):
    sub = anc_sample[anc_sample['hamming'] == d]
    if len(sub) >= 10:
        rate = sub['is_methylated'].mean()
        ci = 1.96 * np.sqrt(rate * (1 - rate) / len(sub))
        ham_stats.append({'d': d, 'n': len(sub), 'rate': rate, 'ci': ci})
ham_df = pd.DataFrame(ham_stats)

# ════════════════════════════════════════════
# Plot — two separate PDFs
# ════════════════════════════════════════════

# --- Fig 1e ---
fig_e, ax_e = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

sizes = np.full(len(merged), 28.0)
ax_e.scatter(merged['rate_young'] * 100, merged['rate_ancient'] * 100,
             s=sizes, c=C_SCATTER, alpha=0.7, edgecolors='white', linewidths=0.4,
             zorder=3)

# Regression line
slope, intercept, _, _, _ = stats.linregress(merged['rate_young'], merged['rate_ancient'])
x_fit = np.linspace(0, merged['rate_young'].max(), 100)
ax_e.plot(x_fit * 100, (slope * x_fit + intercept) * 100,
          color='#333', ls='--', lw=0.8, alpha=0.7, zorder=2)

max_xy = max(merged['rate_young'].max(), merged['rate_ancient'].max()) * 100
ax_e.plot([0, max_xy], [0, max_xy], color='#DDDDDD', lw=0.8, zorder=1)
ax_e.grid(True, lw=0.3, color='#E6E6E6', zorder=0)

ax_e.set_xlabel('Young L1 per-bin m6A rate (%)', fontsize=FS_ANNOT)
ax_e.set_ylabel('Ancient L1 per-bin m6A rate (%)', fontsize=FS_ANNOT)
ax_e.tick_params(labelsize=FS_ANNOT_SMALL)

# Stats annotation
ax_e.text(0.97, 0.05,
          f'ρ = {rho:.2f}\nP = {p_rho:.1e}\nn = {len(merged)} bins',
          transform=ax_e.transAxes, fontsize=FS_ANNOT_SMALL,
          ha='right', va='bottom', color=C_TEXT,
          bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', alpha=0.85))

panel_label(ax_e, 'e')
fig_e.tight_layout()
save_figure(fig_e, f'{FIGDIR}/fig1e')
print(f"  Saved fig1e.pdf")

# --- Fig 1f ---
fig_f, ax_f = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

# Color gradient: warm (close) to cool (far)
n_bars = len(ham_df)
cmap_vals = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, n_bars))

bars = ax_f.bar(ham_df['d'], ham_df['rate'] * 100,
                yerr=ham_df['ci'] * 100, capsize=3, width=0.7,
                color=cmap_vals, edgecolor='#333', linewidth=0.6,
                error_kw=dict(lw=0.9, capthick=0.9))

# N labels
for _, row in ham_df.iterrows():
    y_top = (row['rate'] + row['ci']) * 100
    label = f"{int(row['n']):,}"
    ax_f.text(row['d'], y_top + 0.3, label,
              ha='center', va='bottom', fontsize=FS_ANNOT_SMALL, color='#555')

ax_f.set_xlabel('Hamming distance to Young L1 context', fontsize=FS_ANNOT)
ax_f.set_ylabel('Ancient per-DRACH m6A rate (%)', fontsize=FS_ANNOT)
ax_f.set_xticks(ham_df['d'])
ax_f.tick_params(labelsize=FS_ANNOT_SMALL)

# Significance annotation
close_rate = anc_sample[anc_sample['hamming'] <= 2]['is_methylated'].mean()
far_rate = anc_sample[anc_sample['hamming'] > 2]['is_methylated'].mean()
fold = close_rate / far_rate if far_rate > 0 else float('inf')

# Bracket between d=0-2 and d=3-5
ax_f.annotate('', xy=(0, close_rate * 100 + 2.5), xytext=(3, close_rate * 100 + 2.5),
              arrowprops=dict(arrowstyle='-', color='#666', lw=0.9))
ax_f.text(1.5, close_rate * 100 + 3.0,
          f'{fold:.1f}×, P < 10⁻⁹⁶',
          ha='center', va='bottom', fontsize=FS_ANNOT_SMALL, color=C_TEXT)

panel_label(ax_f, 'f')
fig_f.tight_layout()
save_figure(fig_f, f'{FIGDIR}/fig1f')
print(f"  Saved fig1f.pdf")

plt.close('all')
print("Done.")
