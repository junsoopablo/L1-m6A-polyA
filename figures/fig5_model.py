#!/usr/bin/env python3
"""
Fig 5: Working model — m6A density and poly(A) retention under stress.

Panel a: State diagram (normal vs arsenite stress)
Panel b: Dual decay pathway + immunity feature model
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures'

plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
})

# Colors
C_YOUNG = '#1F77B4'
C_ANCIENT = '#D62728'
C_M6A_HIGH = '#2CA02C'
C_M6A_LOW = '#D62728'
C_SG = '#9467BD'
C_XRN1 = '#FF7F0E'
C_POLYA = '#E377C2'
C_IMMUNE = '#2CA02C'
C_VULN = '#D62728'
C_NEUTRAL = '#AAAAAA'
C_BG = '#F8F8F8'


def rounded_box(ax, xy, w, h, color, text, fontsize=6.5, alpha=0.85,
                text_color='black', linewidth=0.8, edgecolor=None, bold=False):
    """Draw a rounded rectangle with centered text."""
    if edgecolor is None:
        edgecolor = color
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edgecolor,
                         alpha=alpha, linewidth=linewidth,
                         transform=ax.transData)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, transform=ax.transData)


def arrow(ax, start, end, color='black', width=1.2, style='->', head_w=6, head_l=4):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                mutation_scale=10))


# ====================================================================
# Create figure: 2 panels side by side
# ====================================================================
fig = plt.figure(figsize=(7.2, 4.0))

# Panel a: left half
ax_a = fig.add_axes([0.02, 0.05, 0.42, 0.88])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 8)
ax_a.axis('off')
ax_a.text(0.0, 7.7, 'a', fontsize=11, fontweight='bold', transform=ax_a.transData)

# --- Normal conditions (left) ---
ax_a.text(2.5, 7.5, 'Normal', ha='center', fontsize=8, fontweight='bold', color='#333')

# L1 RNA schematic
# High m6A L1
rounded_box(ax_a, (0.5, 6.2), 4.0, 0.55, '#E8F5E9', '', fontsize=5.5)
ax_a.text(0.7, 6.47, 'L1 RNA', fontsize=5.5, va='center', color='#333')
# m6A marks
for xm in [1.8, 2.4, 3.0, 3.6]:
    ax_a.plot(xm, 6.47, 'o', color=C_M6A_HIGH, markersize=3.5, zorder=5)
# poly(A) tail
ax_a.plot([4.5, 4.5], [6.2, 6.75], '-', color=C_NEUTRAL, lw=0.5)
ax_a.text(4.7, 6.47, 'AAAA', fontsize=5, color='#666', va='center', family='monospace')

# Low m6A L1
rounded_box(ax_a, (0.5, 5.2), 4.0, 0.55, '#FFF3E0', '', fontsize=5.5)
ax_a.text(0.7, 5.47, 'L1 RNA', fontsize=5.5, va='center', color='#333')
ax_a.plot(2.5, 5.47, 'o', color=C_M6A_HIGH, markersize=3.5, zorder=5)
ax_a.text(4.7, 5.47, 'AAAA', fontsize=5, color='#666', va='center', family='monospace')

# Arrow and text
ax_a.text(2.5, 4.6, 'r ≈ 0', fontsize=7, ha='center', style='italic', color='#666')
ax_a.text(2.5, 4.1, 'm6A and poly(A)\nindependently distributed', fontsize=5.5,
          ha='center', color='#888', linespacing=1.3)

# XRN1 path (always active)
rounded_box(ax_a, (1.0, 2.8), 3.0, 0.6, '#FFF3E0', "5' XRN1 decay\n(modification-blind)",
            fontsize=5.5, edgecolor=C_XRN1, linewidth=1.0)
ax_a.annotate('', xy=(2.5, 3.4), xytext=(2.5, 4.0),
              arrowprops=dict(arrowstyle='->', color=C_XRN1, lw=1.0))

# Outcome: steady-state turnover
rounded_box(ax_a, (0.8, 1.8), 3.4, 0.6, '#EEEEEE', 'Steady-state L1 turnover\n(poly(A) stable)',
            fontsize=5.5, edgecolor='#999')

ax_a.annotate('', xy=(2.5, 2.4), xytext=(2.5, 2.8),
              arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

# --- Arsenite stress (right) ---
ax_a.text(7.5, 7.5, 'Arsenite stress', ha='center', fontsize=8, fontweight='bold', color=C_ANCIENT)

# Stress granule
sg_box = FancyBboxPatch((5.5, 5.0), 4.0, 2.4, boxstyle="round,pad=0.1",
                         facecolor='#F3E5F5', edgecolor=C_SG, alpha=0.5,
                         linewidth=1.2, linestyle='--')
ax_a.add_patch(sg_box)
ax_a.text(7.5, 7.15, 'Stress granule / P-body', fontsize=5.5, ha='center',
          color=C_SG, style='italic')

# High m6A → retained
rounded_box(ax_a, (5.8, 6.0), 3.4, 0.55, '#C8E6C9', '', fontsize=5.5)
ax_a.text(6.0, 6.27, 'High m6A L1', fontsize=5.5, va='center', color='#333')
for xm in [7.2, 7.7, 8.2]:
    ax_a.plot(xm, 6.27, 'o', color=C_M6A_HIGH, markersize=3.5, zorder=5)
ax_a.text(8.8, 6.27, 'AAA', fontsize=5, color=C_M6A_HIGH, va='center',
          family='monospace', fontweight='bold')

# Low m6A → shortened
rounded_box(ax_a, (5.8, 5.15), 3.4, 0.55, '#FFCDD2', '', fontsize=5.5)
ax_a.text(6.0, 5.42, 'Low m6A L1', fontsize=5.5, va='center', color='#333')
ax_a.plot(7.5, 5.42, 'o', color=C_M6A_HIGH, markersize=3.5, zorder=5)
ax_a.text(8.8, 5.42, 'A', fontsize=5, color=C_M6A_LOW, va='center',
          family='monospace', fontweight='bold')

# Divergent arrows from SG
ax_a.annotate('', xy=(8.5, 4.4), xytext=(8.0, 5.0),
              arrowprops=dict(arrowstyle='->', color=C_M6A_HIGH, lw=1.2))
ax_a.annotate('', xy=(6.5, 4.4), xytext=(7.0, 5.0),
              arrowprops=dict(arrowstyle='->', color=C_M6A_LOW, lw=1.2))

# Outcomes
rounded_box(ax_a, (7.5, 3.8), 2.2, 0.55, '#C8E6C9', 'Retained\n(poly(A) ≥ 100 nt)',
            fontsize=5, edgecolor=C_M6A_HIGH)
rounded_box(ax_a, (5.3, 3.8), 2.0, 0.55, '#FFCDD2', 'Shortened\n(poly(A) < 30 nt)',
            fontsize=5, edgecolor=C_M6A_LOW)

# ρ = 0.201
ax_a.text(7.5, 3.35, 'ρ = 0.201', fontsize=7, ha='center', style='italic',
          color='#333', fontweight='bold')
ax_a.text(7.5, 2.9, 'm6A density correlates\nwith poly(A) retention', fontsize=5.5,
          ha='center', color='#666', linespacing=1.3)

# CHX annotation
ax_a.text(5.7, 7.35, 'CHX blocks', fontsize=4.5, color=C_SG, rotation=0,
          style='italic', ha='left')

# XRN1 on right too
rounded_box(ax_a, (5.8, 1.8), 3.4, 0.6, '#FFF3E0',
            "5' XRN1 decay (converges)\nm6A-independent, OR = 0.99",
            fontsize=5, edgecolor=C_XRN1, linewidth=1.0)
ax_a.annotate('', xy=(7.5, 2.4), xytext=(7.5, 2.85),
              arrowprops=dict(arrowstyle='->', color=C_XRN1, lw=1.0))

# Divider
ax_a.plot([5.1, 5.1], [1.5, 7.8], '--', color='#CCCCCC', lw=0.8)


# ====================================================================
# Panel b: Immunity features model
# ====================================================================
ax_b = fig.add_axes([0.50, 0.05, 0.48, 0.88])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 8)
ax_b.axis('off')
ax_b.text(0.0, 7.7, 'b', fontsize=11, fontweight='bold', transform=ax_b.transData)

ax_b.text(5.0, 7.5, 'Sequence features determine L1 stress fate',
          ha='center', fontsize=8, fontweight='bold', color='#333')

# Young L1 (top, immune)
rounded_box(ax_b, (0.3, 6.0), 9.2, 1.2, '#E8F5E9', '', fontsize=6,
            edgecolor=C_IMMUNE, linewidth=1.5)
ax_b.text(0.5, 6.95, 'Young L1 (immune)', fontsize=7, fontweight='bold',
          color=C_IMMUNE, va='center')

# Draw L1 element schematic
# Full-length element with domains
y_elem = 6.35
# 5'UTR
rounded_box(ax_b, (0.6, y_elem-0.05), 0.8, 0.3, '#B3E5FC', "5'UTR", fontsize=4.5)
# ORF1
rounded_box(ax_b, (1.5, y_elem-0.05), 1.5, 0.3, '#81D4FA', 'ORF1', fontsize=5)
# EN domain
rounded_box(ax_b, (3.1, y_elem-0.05), 1.0, 0.3, '#4FC3F7', 'EN', fontsize=5, bold=True)
# RT domain
rounded_box(ax_b, (4.2, y_elem-0.05), 1.5, 0.3, '#81D4FA', 'RT', fontsize=5)
# 3'UTR
rounded_box(ax_b, (5.8, y_elem-0.05), 0.8, 0.3, '#B3E5FC', "3'UTR", fontsize=4.5)
# poly(A)
ax_b.text(6.8, y_elem+0.1, 'AAAAAA', fontsize=5, color=C_IMMUNE, va='center',
          family='monospace', fontweight='bold')
# m6A marks
for xm in [1.0, 1.8, 2.5, 3.3, 3.8, 4.5, 5.2, 6.0]:
    ax_b.plot(xm, y_elem+0.35, 'v', color=C_M6A_HIGH, markersize=3, zorder=5)

# Shield icons as text
shields = ['Full-length', 'EN domain', 'High m6A', 'ORF translation']
shield_x = [7.6, 7.6, 7.6, 7.6]
shield_y = [6.95, 6.65, 6.35, 6.05]
for i, (s, sy) in enumerate(zip(shields, shield_y)):
    ax_b.text(7.6, sy, '●', fontsize=6, color=C_IMMUNE, va='center')
    ax_b.text(7.9, sy, s, fontsize=5, va='center', color='#333')

# Arrow: Young → immune
ax_b.annotate('Δpoly(A) ≈ 0 nt', xy=(9.5, 6.5), fontsize=6,
              color=C_IMMUNE, fontweight='bold', ha='center', va='center')

# Progressive truncation arrow
ax_b.annotate('', xy=(5.0, 5.65), xytext=(5.0, 5.95),
              arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
ax_b.text(5.0, 5.78, '5\' truncation over evolutionary time', fontsize=5,
          ha='center', color='#888', style='italic')

# Ancient L1 with features (partially protected)
rounded_box(ax_b, (0.3, 4.3), 9.2, 1.2, '#FFF8E1', '', fontsize=6,
            edgecolor='#FFA000', linewidth=1.2)
ax_b.text(0.5, 5.25, 'Ancient L1 with immunity features (protected)',
          fontsize=6.5, fontweight='bold', color='#F57F17', va='center')

y_anc1 = 4.65
# Truncated but has EN domain
rounded_box(ax_b, (1.5, y_anc1-0.05), 1.0, 0.3, '#4FC3F7', 'EN', fontsize=5, bold=True)
rounded_box(ax_b, (2.6, y_anc1-0.05), 1.5, 0.3, '#81D4FA', 'RT', fontsize=5)
rounded_box(ax_b, (4.2, y_anc1-0.05), 0.8, 0.3, '#B3E5FC', "3'UTR", fontsize=4.5)
ax_b.text(5.2, y_anc1+0.1, 'AAAAA', fontsize=5, color='#F57F17', va='center',
          family='monospace', fontweight='bold')
# m6A marks (still high)
for xm in [1.8, 2.3, 3.0, 3.8, 4.5]:
    ax_b.plot(xm, y_anc1+0.35, 'v', color=C_M6A_HIGH, markersize=3, zorder=5)

# Features present
ax_b.text(6.0, 5.0, '● EN domain    ● High m6A', fontsize=5, color='#F57F17')
ax_b.text(6.0, 4.7, '→ Score 2-3: Δ ≈ 0 nt', fontsize=5.5, color='#F57F17',
          fontweight='bold')

# Arrow: further truncation
ax_b.annotate('', xy=(5.0, 3.95), xytext=(5.0, 4.25),
              arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
ax_b.text(5.0, 4.08, 'Further feature loss', fontsize=5,
          ha='center', color='#888', style='italic')

# Ancient L1 without features (vulnerable)
rounded_box(ax_b, (0.3, 2.4), 9.2, 1.4, '#FFEBEE', '', fontsize=6,
            edgecolor=C_VULN, linewidth=1.5)
ax_b.text(0.5, 3.55, 'Ancient L1 without features (vulnerable)',
          fontsize=6.5, fontweight='bold', color=C_VULN, va='center')

y_anc2 = 2.85
# Short fragment, no EN, low m6A
rounded_box(ax_b, (1.5, y_anc2-0.05), 2.0, 0.3, '#BBDEFB', 'ORF2 fragment', fontsize=5)
ax_b.text(3.7, y_anc2+0.1, 'AA', fontsize=5, color=C_VULN, va='center',
          family='monospace', fontweight='bold')
# Few m6A
ax_b.plot(2.2, y_anc2+0.35, 'v', color=C_M6A_HIGH, markersize=3, zorder=5)

# Two fate boxes
# 3' poly(A) shortening
rounded_box(ax_b, (4.5, 2.65), 2.3, 0.55, '#FFCDD2',
            "3' poly(A) shortening\n(m6A-dependent)", fontsize=5,
            edgecolor=C_VULN, text_color='#B71C1C')
# 5' XRN1 decay
rounded_box(ax_b, (7.0, 2.65), 2.3, 0.55, '#FFE0B2',
            "5' XRN1 decay\n(m6A-independent)", fontsize=5,
            edgecolor=C_XRN1, text_color='#E65100')

# Independent label
ax_b.text(6.85, 2.92, 'independent', fontsize=4.5, ha='center', va='center',
          color='#666', style='italic')

# Quantitative annotations
ax_b.text(5.65, 2.45, 'Δ = −34 to −53 nt', fontsize=5, ha='center',
          color=C_VULN, fontweight='bold')
ax_b.text(8.15, 2.45, '1.46× abundance ↑\nwhen KD', fontsize=4.5, ha='center',
          color='#E65100')

# Bottom: summary gradient
ax_b.text(5.0, 1.8, 'Composite immunity score', fontsize=7, ha='center',
          fontweight='bold', color='#333')

# Gradient bar
from matplotlib.colors import LinearSegmentedColormap
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cmap = LinearSegmentedColormap.from_list('immunity', [C_VULN, '#FFA000', C_IMMUNE])
ax_grad = fig.add_axes([0.55, 0.12, 0.35, 0.04])
ax_grad.imshow(gradient, aspect='auto', cmap=cmap)
ax_grad.set_xticks([0, 85, 170, 255])
ax_grad.set_xticklabels(['0', '1', '2', '3'], fontsize=6)
ax_grad.set_yticks([])
ax_grad.set_xlabel('Score', fontsize=6, labelpad=1)

# Score labels
ax_b.text(1.5, 1.15, '82 nt', fontsize=6, ha='center', color=C_VULN, fontweight='bold')
ax_b.text(4.0, 1.15, '108 nt', fontsize=6, ha='center', color='#FF8F00')
ax_b.text(6.5, 1.15, '116 nt', fontsize=6, ha='center', color='#558B2F')
ax_b.text(9.0, 1.15, '164 nt', fontsize=6, ha='center', color=C_IMMUNE, fontweight='bold')
ax_b.text(5.0, 0.7, r'Median poly(A) under stress ($\rho$ = 0.112, P = 1.3 $\times$ 10$^{-8}$)',
          fontsize=5.5, ha='center', color='#666', style='italic')


plt.savefig(f'{OUT}/fig5_model.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUT}/fig5_model.png', bbox_inches='tight', dpi=200)
print(f'Saved: {OUT}/fig5_model.pdf')
