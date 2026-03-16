#!/usr/bin/env python3
"""
Figure 4: Mechanistic model — m6A-dependent poly(A) protection under stress.
Panel (a): Normal vs Stress state diagram
Panel (b): Four-axis selectivity flowchart
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from fig_style import *

setup_style()

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Colors ──
C_M6A = '#CC6677'       # rose — m6A
C_POLYA = '#DDCC77'     # sand — poly(A)
C_SG = '#AA4499'        # purple — stress granule
C_PROTECT = '#44AA99'   # teal — protected/retained
C_DECAY = '#882255'     # wine — decay
C_IMMUNE = '#88CCEE'    # cyan — immune
C_BG_NORM = '#F0F4F8'   # light blue-grey — normal background
C_BG_STRESS = '#FDF0ED' # light rose — stress background
C_ARROW = '#555555'
C_CHX = '#EE8866'       # peach — CHX

def rounded_box(ax, xy, width, height, color, alpha=0.15, edgecolor=None,
                text='', fontsize=7, fontweight='normal', text_color='#333',
                lw=1.0, ls='-'):
    """Draw a rounded rectangle with centered text."""
    if edgecolor is None:
        edgecolor = color
    box = FancyBboxPatch(xy, width, height,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=edgecolor,
                         alpha=alpha, linewidth=lw, linestyle=ls,
                         zorder=2, mutation_scale=0.3)
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + width/2, xy[1] + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=3)

def draw_arrow(ax, start, end, color='#555', lw=1.2, style='->', head_width=0.012):
    """Draw an arrow."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=4)

def draw_rna(ax, x, y, length, n_m6a, polya_len, label='', stress=False):
    """Draw a stylized RNA molecule with m6A dots and poly(A) tail."""
    # RNA body
    body_color = '#666' if not stress else C_DECAY
    ax.plot([x, x + length], [y, y], color=body_color, lw=2.5, solid_capstyle='round', zorder=3)

    # m6A marks (lollipops)
    if n_m6a > 0:
        positions = np.linspace(x + length * 0.15, x + length * 0.85, n_m6a)
        for px in positions:
            ax.plot([px, px], [y, y + 0.022], color=C_M6A, lw=0.8, zorder=3)
            ax.scatter(px, y + 0.022, color=C_M6A, s=12, zorder=4, edgecolors='none')

    # Poly(A) tail
    tail_x = x + length
    tail_len = polya_len * 0.0008  # scale
    if polya_len > 0:
        # Wavy poly(A)
        t = np.linspace(0, tail_len, 30)
        wave_y = y + np.sin(t * 80) * 0.005
        ax.plot(tail_x + t, wave_y, color=C_POLYA, lw=2.0, zorder=3)
        ax.text(tail_x + tail_len/2, y - 0.025, f'{polya_len} nt',
                ha='center', fontsize=5.5, color=C_POLYA, style='italic')

    if label:
        ax.text(x - 0.01, y, label, ha='right', va='center', fontsize=6, color='#555')


# ══════════════════════════════════════════════
# Create figure
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.75))
gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0],
                      hspace=0.12, left=0.02, right=0.98,
                      top=0.96, bottom=0.02)

# ════════════════════════════════════════
# Panel (a): Normal vs Stress model
# ════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 0.55)
ax_a.axis('off')
panel_label(ax_a, 'a', x=-0.02, y=1.05)

# --- Left: NORMAL ---
# Background
rounded_box(ax_a, (0.01, 0.02), 0.45, 0.50, C_BG_NORM, alpha=0.4,
            edgecolor='#aabbcc', lw=0.8)
ax_a.text(0.235, 0.49, 'Normal conditions', ha='center', fontsize=9,
          fontweight='bold', color=C_NORMAL)

# L1 RNA with m6A marks, normal poly(A)
draw_rna(ax_a, 0.07, 0.38, 0.14, n_m6a=4, polya_len=122, label='')
ax_a.text(0.04, 0.38, 'L1', ha='right', fontsize=7, fontweight='bold', color='#444')

# m6A ↔ poly(A) uncoupled
ax_a.text(0.235, 0.28, 'r ≈ 0', ha='center', fontsize=9,
          fontweight='bold', color='#999', style='italic')
ax_a.text(0.235, 0.235, 'm6A and poly(A) independent',
          ha='center', fontsize=6.5, color='#777')

# Legend items
ax_a.scatter(0.06, 0.12, color=C_M6A, s=25, zorder=5)
ax_a.text(0.08, 0.12, 'm6A', va='center', fontsize=6.5, color=C_M6A)
ax_a.plot([0.18, 0.24], [0.12, 0.12], color=C_POLYA, lw=2.5)
ax_a.text(0.26, 0.12, 'poly(A)', va='center', fontsize=6.5, color='#999')

# Homeostasis box
rounded_box(ax_a, (0.12, 0.04), 0.23, 0.05, C_IMMUNE, alpha=0.25,
            text='Poly(A) stable',
            fontsize=6, edgecolor=C_IMMUNE, lw=0.6)

# --- Right: STRESS ---
# Background
rounded_box(ax_a, (0.54, 0.02), 0.45, 0.50, C_BG_STRESS, alpha=0.4,
            edgecolor='#cc9999', lw=0.8)
ax_a.text(0.765, 0.49, 'Arsenite stress', ha='center', fontsize=9,
          fontweight='bold', color=C_STRESS)

# SG cloud
sg_circle = plt.Circle((0.62, 0.40), 0.04, color=C_SG, alpha=0.15,
                        ec=C_SG, lw=0.8, ls='--', zorder=2)
ax_a.add_patch(sg_circle)
ax_a.text(0.62, 0.40, 'SG', ha='center', va='center', fontsize=6,
          color=C_SG, fontweight='bold')

# Arrow: SG routes to decay
draw_arrow(ax_a, (0.66, 0.40), (0.72, 0.40), color=C_ARROW, lw=1.0)

# Branch point
ax_a.scatter(0.735, 0.40, color='#333', s=20, zorder=5, marker='o')

# Upper branch: HIGH m6A → protected
draw_arrow(ax_a, (0.735, 0.40), (0.735, 0.30), color=C_PROTECT, lw=1.2)
# High m6A RNA (longer poly(A))
draw_rna(ax_a, 0.76, 0.33, 0.10, n_m6a=5, polya_len=125, stress=False)
ax_a.text(0.76, 0.36, 'High m6A', ha='left', fontsize=6.2, color=C_M6A,
          fontweight='bold')

# Protected box
rounded_box(ax_a, (0.76, 0.24), 0.18, 0.05, C_PROTECT, alpha=0.25,
            text='Poly(A) retained', fontsize=5.5,
            edgecolor=C_PROTECT, lw=0.6, text_color='#2a7a6a')

# Lower branch: LOW m6A → decay
draw_arrow(ax_a, (0.735, 0.40), (0.735, 0.18), color=C_DECAY, lw=1.2)
# Low m6A RNA (short poly(A))
draw_rna(ax_a, 0.76, 0.17, 0.10, n_m6a=1, polya_len=25, stress=True)
ax_a.text(0.76, 0.20, 'Low m6A', ha='left', fontsize=6.2, color='#888',
          fontweight='bold')

# Decay box
rounded_box(ax_a, (0.76, 0.06), 0.18, 0.05, C_DECAY, alpha=0.20,
            text='Poly(A) < 30 nt', fontsize=5.0,
            edgecolor=C_DECAY, lw=0.6, text_color=C_DECAY)

# Dose-response annotation
ax_a.text(0.71, 0.30, 'r = +0.18\np = 2e-20', ha='center', fontsize=5.2,
          color=C_STRESS, style='italic', fontweight='bold')

# CHX annotation (crosses out SG)
ax_a.plot([0.58, 0.66], [0.44, 0.36], color=C_CHX, lw=2.0, zorder=6)
ax_a.plot([0.58, 0.66], [0.36, 0.44], color=C_CHX, lw=2.0, zorder=6)
ax_a.text(0.62, 0.46, 'CHX blocks', ha='center', fontsize=5,
          color=C_CHX, fontweight='bold')

# Central arrow (Normal → Stress)
draw_arrow(ax_a, (0.47, 0.27), (0.53, 0.27), color='#888', lw=1.5, style='->')
ax_a.text(0.50, 0.30, 'NaAsO₂', ha='center', fontsize=6, color='#888',
          fontweight='bold')

# ════════════════════════════════════════
# Panel (b): Four-axis selectivity flowchart
# ════════════════════════════════════════
ax_b = fig.add_subplot(gs[1, 0])
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 0.48)
ax_b.axis('off')
panel_label(ax_b, 'b', x=-0.02, y=1.05)

ax_b.text(0.50, 0.47, 'Four-axis selectivity model', ha='center',
          fontsize=9, fontweight='bold', color='#333')

# Start: L1 RNA under stress
rounded_box(ax_b, (0.38, 0.37), 0.24, 0.06, '#ddd', alpha=0.5,
            text='L1 RNA under arsenite stress', fontsize=7,
            fontweight='bold', edgecolor='#999', text_color='#333')

# Decision node positions
decisions = [
    (0.08, 0.24, 'Translated?', 'Young L1\n(ORF1p/2p)', 'Translation'),
    (0.30, 0.24, 'In host gene\nexon?', 'Category B\n(lncRNA/pseudo)', 'Architecture'),
    (0.53, 0.24, 'Compact\nchromatin?', 'Heterochromatin\nL1', 'Chromatin'),
    (0.76, 0.24, 'Strong PAS?', 'Canonical\nAATAAA', 'PAS strength'),
]

for i, (x, y, question, immune_label, axis_name) in enumerate(decisions):
    # Decision diamond → simplified as rounded box
    rounded_box(ax_b, (x, y), 0.17, 0.07, '#f5f5f5', alpha=0.95,
                text=question, fontsize=6.6, edgecolor='#555',
                text_color='#333', lw=0.8)

    # Arrow from top
    if i == 0:
        draw_arrow(ax_b, (0.42, 0.37), (x + 0.085, y + 0.07),
                   color='#888', lw=0.8)
    elif i == 1:
        draw_arrow(ax_b, (0.50, 0.37), (x + 0.085, y + 0.07),
                   color='#888', lw=0.8)
    elif i == 2:
        draw_arrow(ax_b, (0.55, 0.37), (x + 0.085, y + 0.07),
                   color='#888', lw=0.8)
    elif i == 3:
        draw_arrow(ax_b, (0.60, 0.37), (x + 0.085, y + 0.07),
                   color='#888', lw=0.8)

    # "Yes" → Immune (upward arrow to immune box)
    # Immune box below
    rounded_box(ax_b, (x, 0.12), 0.17, 0.06, C_IMMUNE, alpha=0.4,
                text=immune_label, fontsize=6.2, edgecolor=C_IMMUNE,
                lw=0.8, text_color='#1a4f6b')
    draw_arrow(ax_b, (x + 0.085, y), (x + 0.085, 0.18),
               color=C_IMMUNE, lw=1.0)
    ax_b.text(x + 0.06, y - 0.015, 'Yes', fontsize=5.5, color=C_IMMUNE,
              fontweight='bold')

    # "IMMUNE" label
    # Axis label at bottom
    ax_b.text(x + 0.085, 0.06, axis_name, ha='center',
              fontsize=5.4, color='#555', style='italic')

    # "No" arrow → next decision (right)
    if i < 3:
        next_x = decisions[i + 1][0]
        ax_b.text(x + 0.18, y + 0.035, 'No', fontsize=5.4, color='#888',
                  va='center')

# Final outcome: Vulnerable
rounded_box(ax_b, (0.76, 0.00), 0.22, 0.05, C_DECAY, alpha=0.2,
            text='', fontsize=5.5,
            edgecolor=C_DECAY, lw=0.8, text_color=C_DECAY)
# Multi-line text for vulnerable box
ax_b.text(0.87, 0.025, 'VULNERABLE\nm6A-dependent poly(A) retention',
          ha='center', va='center', fontsize=5.5, fontweight='bold',
          color=C_DECAY)

# Arrow: No PAS → vulnerable
draw_arrow(ax_b, (0.845, 0.12), (0.87, 0.05), color=C_DECAY, lw=1.0)
ax_b.text(0.87, 0.08, 'No/weak', fontsize=5, color=C_DECAY, ha='center')

# Effect sizes annotation removed for clarity

# ── Save ──
save_figure(fig, f'{OUTDIR}/fig4')
print(f'Saved fig4.pdf to {OUTDIR}')
