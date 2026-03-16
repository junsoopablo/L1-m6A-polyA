#!/usr/bin/env python3
"""Create 4-panel composite figure (figS_lncrna.pdf) for Supplementary Figure S15.

Panels:
  (a) Poly(A) violin: lncRNA vs L1 vs mRNA (HeLa/Ars)
  (b) Δpoly(A) bar chart
  (c) m6A/kb boxplot
  (d) m6A–poly(A) scatter (Ars, rho annotated)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

FIGDIR = Path(__file__).parent
OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')

panels = [
    ('a', FIGDIR / 'fig1_lncrna_polya_violin.png'),
    ('b', FIGDIR / 'fig2_delta_polya_bar.png'),
    ('c', FIGDIR / 'fig3_m6a_comparison.png'),
    ('d', FIGDIR / 'fig4_m6a_polya_scatter.png'),
]

for label, p in panels:
    if not p.exists():
        raise FileNotFoundError(f"Missing panel {label}: {p}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (label, png_path) in enumerate(panels):
    row, col = divmod(idx, 2)
    ax = axes[row][col]
    img = mpimg.imread(str(png_path))
    ax.imshow(img)
    ax.axis('off')
    # Panel label in top-left corner
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

fig.subplots_adjust(wspace=0.05, hspace=0.08)
outpath = OUTDIR / 'figS_lncrna.pdf'
fig.savefig(outpath, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved: {outpath}")
