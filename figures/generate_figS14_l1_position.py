#!/usr/bin/env python3
"""
Supplementary Figure S14: L1 3' Positioning and PAS-Dependent Arsenite Vulnerability.
  (a) ECDF — dist_to_3prime for Young vs Ancient L1
  (b) Grouped bar — poly(A) by L1 position × stress condition
  (c) Bar — proportion of reads at 3' end by L1 age × genomic context
Layout: [a | b | c]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import glob
from scipy import stats
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
frames = []
for f in sorted(glob.glob(f'{BASE}/results_group/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'read_length', 'overlap_length',
        'te_start', 'te_end', 'gene_id', 'dist_to_3prime',
        'polya_length', 'qc_tag', 'TE_group'
    ])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['group'] = group
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    frames.append(tmp)

df = pd.concat(frames, ignore_index=True)
df = df[df['qc_tag'] == 'PASS'].copy()
df['is_young'] = df['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')
df['overlap_frac'] = df['overlap_length'] / df['read_length']
df['l1_at_3end'] = df['dist_to_3prime'] <= 50

# ── Create figure ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.36))
gs = fig.add_gridspec(1, 3, wspace=0.45,
                      left=0.07, right=0.98, top=0.92, bottom=0.16,
                      width_ratios=[1, 1.1, 0.9])

# ────────────────────────────────────────────
# Panel (a): ECDF of dist_to_3prime
# ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, 'a')

young = df[df['is_young']]['dist_to_3prime'].values
ancient = df[~df['is_young']]['dist_to_3prime'].values

ecdf_plot(ax_a, np.clip(young, 0, 600), C_YOUNG,
          f'Young (n={len(young):,})', lw=1.5)
ecdf_plot(ax_a, np.clip(ancient, 0, 600), C_ANCIENT,
          f'Ancient (n={len(ancient):,})', lw=1.5)

# Mark 50bp threshold
ax_a.axvline(50, color=C_GREY, ls='--', lw=0.7, zorder=0)
ax_a.text(55, 0.15, '50 bp', fontsize=FS_ANNOT_SMALL, color='#777777')

# Annotations
y_frac = (young <= 50).mean()
a_frac = (ancient <= 50).mean()
ax_a.annotate(f'{y_frac:.0%}', xy=(50, y_frac), xytext=(120, y_frac - 0.05),
              fontsize=FS_ANNOT, color=C_YOUNG, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_YOUNG, lw=0.8))
ax_a.annotate(f'{a_frac:.0%}', xy=(50, a_frac), xytext=(120, a_frac + 0.05),
              fontsize=FS_ANNOT, color='#AA9933', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#AA9933', lw=0.8))

ax_a.set_xlabel('Distance to read 3\' end (bp)')
ax_a.set_ylabel('Cumulative fraction')
ax_a.set_xlim(0, 600)
ax_a.set_ylim(0, 1.02)
ax_a.legend(fontsize=FS_LEGEND, loc='lower right')

# ────────────────────────────────────────────
# Panel (b): Grouped bar — poly(A) by position × stress
# ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, 'b')

# Ancient L1 only (HeLa vs HeLa-Ars)
hela_anc = df[(df['cell_line'] == 'HeLa') & (~df['is_young'])]
ars_anc = df[(df['cell_line'] == 'HeLa-Ars') & (~df['is_young'])]

groups = [
    ('L1 at 3\' end\n(own PAS)', True),
    ('L1 upstream\n(downstr. PAS)', False),
]

x_positions = np.array([0, 1.2])
bar_w = 0.35

# Bootstrap CI function
def boot_ci(data, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    meds = [np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(meds, [2.5, 97.5])

offset = 0.15
for i, (label, at3) in enumerate(groups):
    h_data = hela_anc[hela_anc['l1_at_3end'] == at3]['polya_length'].dropna().values
    a_data = ars_anc[ars_anc['l1_at_3end'] == at3]['polya_length'].dropna().values

    h_med = np.median(h_data)
    a_med = np.median(a_data)
    h_ci = boot_ci(h_data)
    a_ci = boot_ci(a_data)

    # Connecting line
    ax_b.plot([x_positions[i] - offset, x_positions[i] + offset],
              [h_med, a_med], color='#BDC3C7', lw=1.5, zorder=1)

    # Normal dot + CI
    ax_b.scatter(x_positions[i] - offset, h_med, s=50, color=C_NORMAL,
                 edgecolors='white', linewidths=0.5, zorder=4)
    ax_b.plot([x_positions[i] - offset]*2, [h_ci[0], h_ci[1]],
              color=C_NORMAL, lw=1.0, zorder=3)

    # Arsenite dot + CI
    ax_b.scatter(x_positions[i] + offset, a_med, s=50, color=C_STRESS,
                 edgecolors='white', linewidths=0.5, zorder=4)
    ax_b.plot([x_positions[i] + offset]*2, [a_ci[0], a_ci[1]],
              color=C_STRESS, lw=1.0, zorder=3)

    # Delta annotation
    delta = a_med - h_med
    _, p_val = stats.mannwhitneyu(h_data, a_data, alternative='two-sided')
    sig = significance_text(p_val)
    y_top = max(h_ci[1], a_ci[1]) + 8
    significance_bracket(ax_b, x_positions[i] - offset, x_positions[i] + offset,
                        y_top, 3, f'$\\Delta$={delta:+.0f} nt {sig}', fontsize=FS_ANNOT_SMALL)

    # n labels below
    ax_b.text(x_positions[i], -6, f'n={len(h_data)+len(a_data):,}',
              ha='center', fontsize=5, color='#888888')

ax_b.set_xticks(x_positions)
ax_b.set_xticklabels([g[0] for g in groups], fontsize=FS_ANNOT)
ax_b.set_ylabel('Median poly(A) (nt)')
ax_b.set_ylim(0, 170)
ax_b.set_xlim(-0.6, 1.8)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_NORMAL, edgecolor='#2C3E50', lw=0.5, label='Normal'),
                   Patch(facecolor=C_STRESS, edgecolor='#2C3E50', lw=0.5, label='Arsenite')]
ax_b.legend(handles=legend_elements, fontsize=FS_LEGEND, loc='upper right')

ax_b.set_title('Ancient L1 (HeLa)', fontsize=FS_ANNOT, pad=4, color='#555555')

# ────────────────────────────────────────────
# Panel (c): Proportion at 3' end by age × context
# ────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, 'c')

categories = [
    ('Young\nIG', df[(df['is_young']) & (df['TE_group'] == 'intergenic')]),
    ('Young\nIT', df[(df['is_young']) & (df['TE_group'] == 'intronic')]),
    ('Anc.\nIG', df[(~df['is_young']) & (df['TE_group'] == 'intergenic')]),
    ('Anc.\nIT', df[(~df['is_young']) & (df['TE_group'] == 'intronic')]),
]

x_pos = np.arange(len(categories))
colors = [C_YOUNG, C_YOUNG, C_ANCIENT, C_ANCIENT]
hatches = ['', '///', '', '///']

from statsmodels.stats.proportion import proportion_confint

# Cleveland dot plot (horizontal)
cat_labels = [c[0] for c in categories]
fracs = []
ci_los = []
ci_his = []
ns = []
for i, (label, sub) in enumerate(categories):
    frac = sub['l1_at_3end'].mean() * 100
    n = len(sub)
    ci_lo_w, ci_hi_w = proportion_confint(sub['l1_at_3end'].sum(), n, method='wilson')
    fracs.append(frac)
    ci_los.append(ci_lo_w * 100)
    ci_his.append(ci_hi_w * 100)
    ns.append(n)

# Horizontal forest plot
markers = ['o', 's', 'o', 's']  # circle=IG, square=IT
for i in range(len(categories)):
    ax_c.plot([ci_los[i], ci_his[i]], [i, i], color=colors[i], lw=1.5, zorder=2)
    ax_c.scatter(fracs[i], i, s=50, color=colors[i], edgecolors='white',
                 linewidths=0.5, zorder=3, marker=markers[i])
    ax_c.text(ci_his[i] + 1.5, i, f'{fracs[i]:.0f}%',
              va='center', fontsize=FS_ANNOT, fontweight='bold', color=colors[i])
    ax_c.text(fracs[i], i + 0.3, f'n={ns[i]:,}', ha='center', fontsize=5, color='#888888')

ax_c.set_yticks(range(len(categories)))
ax_c.set_yticklabels(cat_labels, fontsize=FS_ANNOT)
ax_c.set_xlabel('Reads at 3\' end (%)')
ax_c.set_xlim(0, 85)
ax_c.set_ylim(-0.5, 3.5)
ax_c.invert_yaxis()

# Legend for marker shape
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ig_mk = Line2D([0], [0], marker='o', color='w', markerfacecolor=C_YOUNG, ms=6, label='Intergenic')
it_mk = Line2D([0], [0], marker='s', color='w', markerfacecolor=C_YOUNG, ms=6, label='Intronic')
ax_c.legend(handles=[ig_mk, it_mk], fontsize=FS_LEGEND, loc='lower right')

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS14_position')
print("Fig S14 saved")

# ── Print key stats ──
print(f"\n[KEY STATS]")
for at3, label in [(True, "L1 at 3' end"), (False, "L1 away from 3'")]:
    h = hela_anc[hela_anc['l1_at_3end'] == at3]['polya_length']
    a = ars_anc[ars_anc['l1_at_3end'] == at3]['polya_length']
    delta = a.median() - h.median()
    _, p = stats.mannwhitneyu(h, a)
    print(f"  {label}: HeLa {h.median():.1f} → Ars {a.median():.1f}, "
          f"Δ={delta:+.1f}nt (P={p:.2e}, n_h={len(h)}, n_a={len(a)})")
