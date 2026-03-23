#!/usr/bin/env python3
"""
Figure 3c: Triple stratification heatmap.
  (c) 2×2 heatmap: m6A × PAS (CpG averaged) under arsenite stress

Note: METTL3 KO poly(A) panel was removed from the paper (2026-02-24).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUTDIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════
# Panel (c): Triple Stratification 2×2 Heatmap
# ═══════════════════════════════════════════
print("=== Panel 3c: Triple Stratification ===")

df_triple = pd.read_csv(
    f'{BASE}/topic_08_sequence_features/triple_stratification_stress.tsv', sep='\t')

# Average CpG levels to make 2×2
# Rows: m6A (0=low, 1=high); Cols: PAS (0=absent, 1=present)
matrix = np.zeros((2, 2))
n_matrix = np.zeros((2, 2), dtype=int)
decay_matrix = np.zeros((2, 2))

for _, row in df_triple.iterrows():
    r = 1 if row['m6A'] == 'high' else 0
    c = 1 if row['PAS'] == 'yes' else 0
    matrix[r, c] += row['median_polya'] / 2   # average of 2 CpG levels
    n_matrix[r, c] += row['n']
    decay_matrix[r, c] += row['decay_pct'] / 2

print("2×2 matrix (CpG averaged):")
for r, m6a_label in [(1, 'high'), (0, 'low')]:
    for c, pas_label in [(0, 'absent'), (1, 'present')]:
        print(f"  m6A {m6a_label}, PAS {pas_label}: "
              f"{matrix[r,c]:.1f} nt, n={n_matrix[r,c]}, "
              f"decay={decay_matrix[r,c]*100:.1f}%")
range_nt = matrix[1, 1] - matrix[0, 0]
print(f"  Range (best - worst): {range_nt:.0f} nt")

# Now placed side-by-side with decay zone bar → use HALF_WIDTH
fig, ax = plt.subplots(figsize=(HALF_WIDTH, PANEL_HEIGHT))
panel_label(ax, 'c')  # new Fig 2c

im = ax.imshow(matrix, cmap='viridis', aspect='auto',
               vmin=50, vmax=140, origin='lower')

# Cell annotations
for r in range(2):
    for c in range(2):
        val = matrix[r, c]
        n = n_matrix[r, c]
        decay = decay_matrix[r, c] * 100
        text_color = 'white' if val < 90 else 'black'
        # Bold extremes (best and worst)
        weight = 'bold' if (r == 1 and c == 1) or (r == 0 and c == 0) else 'normal'
        ax.text(c, r, f'{val:.0f} nt\nn = {n:,}\n{decay:.0f}% decay',
                ha='center', va='center', fontsize=FS_ANNOT,
                color=text_color, fontweight=weight)

ax.set_xticks([0, 1])
ax.set_xticklabels(['PAS absent', 'PAS present'], fontsize=FS_ANNOT)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Low m6A', 'High m6A'], fontsize=FS_ANNOT)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Median poly(A) (nt)', fontsize=FS_ANNOT)
cbar.ax.tick_params(labelsize=FS_ANNOT)


save_figure(fig, f'{OUTDIR}/fig3c')
print("fig3c saved.")

print("\nFig 3c panel complete.")
