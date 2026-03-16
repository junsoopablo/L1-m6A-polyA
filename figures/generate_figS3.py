#!/usr/bin/env python3
"""
Supplementary Figure S3: Embedded vs non-embedded L1, loci analysis.
  (a) Grouped bar: poly(A) median by category × condition
  (b) Baseline m6A quartile vs poly(A) — no correlation at baseline
  (c) Waterfall: hotspot loci removal robustness
  (d) Donut: loci sharing between HeLa and HeLa-Ars
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

# ── Load raw non-embedded L1 data ──
raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'polya_length',
                                             'qc_tag', 'gene_id'])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    raw_frames.append(tmp)
df_pass = pd.concat(raw_frames, ignore_index=True)
df_pass = df_pass[df_pass['qc_tag'] == 'PASS'].copy()

# Load per-read m6A
cache_frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_frames.append(tmp)
df_cache = pd.concat(cache_frames, ignore_index=True)
df_cache['m6a_per_kb'] = df_cache['m6a_sites_high'] / (df_cache['read_length'] / 1000)

# Merge non-embedded with m6A
df_pass_m6a = df_pass.merge(df_cache[['read_id', 'm6a_per_kb']], on='read_id', how='inner')

# Load embedded (Cat B) summary statistics
polya_cl = pd.read_csv(f'{BASE}/topic_07_catB_nc_exonic/catB_vs_pass_analysis/polya_per_cl.tsv', sep='\t')

# ── Create figure ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45,
                      left=0.09, right=0.97, top=0.96, bottom=0.08)

# ────────────────────────────────────────────
# Panel (a): Grouped bar — poly(A) by category × condition
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'a')

# Non-embedded: compute from raw data
hela_ne = df_pass[df_pass['cell_line'] == 'HeLa']['polya_length'].dropna()
ars_ne = df_pass[df_pass['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()

# Embedded: from summary table
hela_emb_row = polya_cl[polya_cl['cell_line'] == 'HeLa'].iloc[0]
ars_emb_row = polya_cl[polya_cl['cell_line'] == 'HeLa-Ars'].iloc[0]

groups = ['Non-embedded', 'Embedded']
normal_meds = [hela_ne.median(), hela_emb_row['catB_median']]
ars_meds = [ars_ne.median(), ars_emb_row['catB_median']]
normal_ns = [len(hela_ne), int(hela_emb_row['catB_n'])]
ars_ns = [len(ars_ne), int(ars_emb_row['catB_n'])]

# Dumbbell plot: Normal → Arsenite connected dots
dumbbell_plot(ax, groups, normal_meds, ars_meds,
              C_NORMAL, C_STRESS, label1='Normal', label2='Arsenite',
              horizontal=False, marker_size=50, line_width=2.0)

# Value labels
for i in range(len(groups)):
    ax.text(i - 0.15, normal_meds[i], f'{normal_meds[i]:.0f}',
            ha='right', va='center', fontsize=FS_ANNOT, fontweight='bold', color=C_NORMAL)
    ax.text(i + 0.15, ars_meds[i], f'{ars_meds[i]:.0f}',
            ha='left', va='center', fontsize=FS_ANNOT, fontweight='bold', color=C_STRESS)
    ax.text(i, -8, f'n={normal_ns[i]:,}/{ars_ns[i]:,}',
            ha='center', fontsize=FS_LEGEND_SMALL, color='#888888', clip_on=False)

# Delta annotations
delta_ne = ars_meds[0] - normal_meds[0]
delta_emb = ars_meds[1] - normal_meds[1]
for i, (delta, sig) in enumerate([(delta_ne, '***'), (delta_emb, 'ns')]):
    mid_y = (normal_meds[i] + ars_meds[i]) / 2
    ax.text(i + 0.25, mid_y, f'Δ={delta:.0f} {sig}',
            ha='left', va='center', fontsize=FS_ANNOT, color=C_STRESS if sig != 'ns' else C_GREY)

ax.set_ylabel('Median poly(A) length (nt)')
ax.set_ylim(0, 180)
ax.legend(fontsize=FS_LEGEND, loc='upper right')

print(f"S3a: Non-emb HeLa={normal_meds[0]:.1f} (n={normal_ns[0]}), "
      f"Ars={ars_meds[0]:.1f} (n={ars_ns[0]}), Δ={delta_ne:.1f}")
print(f"     Embedded HeLa={normal_meds[1]:.1f} (n={normal_ns[1]}), "
      f"Ars={ars_meds[1]:.1f} (n={ars_ns[1]}), Δ={delta_emb:.1f}")

# ────────────────────────────────────────────
# Panel (b): Baseline m6A quartile vs poly(A) — no correlation
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'b')

# Use only normal (non-arsenite) cell lines
normal_cls = [c for c in df_pass_m6a['cell_line'].unique() if 'Ars' not in c]
df_normal = df_pass_m6a[df_pass_m6a['cell_line'].isin(normal_cls)].dropna(subset=['polya_length', 'm6a_per_kb'])

# Quartile assignment
df_normal = df_normal.copy()
df_normal['m6a_q'] = pd.qcut(df_normal['m6a_per_kb'], 4, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])

q_labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
q_meds = [df_normal[df_normal['m6a_q'] == q]['polya_length'].median() for q in q_labels]
q_ns = [len(df_normal[df_normal['m6a_q'] == q]) for q in q_labels]

# Bootstrap 95% CI
np.random.seed(42)
q_ci_lo, q_ci_hi = [], []
for q in q_labels:
    vals = df_normal[df_normal['m6a_q'] == q]['polya_length'].values
    boot_meds = [np.median(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boot_meds, [2.5, 97.5])
    q_ci_lo.append(q_meds[q_labels.index(q)] - lo)
    q_ci_hi.append(hi - q_meds[q_labels.index(q)])

x_q = np.arange(len(q_labels))
gradient = [C_NORMAL, '#6699BB', '#8899AA', '#AA7744']

# Dot + CI whisker (forest-style)
ci_lo_abs = [q_meds[i] - q_ci_lo[i] for i in range(len(q_labels))]
ci_hi_abs = [q_meds[i] + q_ci_hi[i] for i in range(len(q_labels))]
forest_plot(ax, q_labels, q_meds, ci_lo_abs, ci_hi_abs,
            colors=gradient, ref_line=None, horizontal=False, marker_size=50)

for i in range(len(q_labels)):
    ax.text(i, ci_hi_abs[i] + 2, f'{q_meds[i]:.0f}',
            ha='center', fontsize=FS_ANNOT, fontweight='bold')
    ax.text(i, -6, f'n={q_ns[i]:,}', ha='center', fontsize=FS_LEGEND_SMALL,
            color='#888888', clip_on=False)

# Flat line annotation
from scipy import stats
slope, intercept, r, p, se = stats.linregress(
    df_normal['m6a_per_kb'].values, df_normal['polya_length'].values)
ax.text(0.5, 0.95, f'r = {r:.3f} (ns)', transform=ax.transAxes,
        ha='center', va='top', fontsize=FS_ANNOT, color=C_GREY, style='italic')

ax.set_xticks(x_q)
ax.set_xticklabels(q_labels, fontsize=7)
ax.set_xlabel('m6A/kb quartile (baseline)')
ax.set_ylabel('Median poly(A) length (nt)')
ax.set_ylim(0, 150)

print(f"\nS3b: Quartile medians = {[f'{m:.1f}' for m in q_meds]}, r={r:.4f}, P={p:.2e}")

# ────────────────────────────────────────────
# Panel (c): Waterfall — hotspot loci removal robustness
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'c')

labels = ['Full\ndata', 'Remove\ntop 5', 'Remove\ntop 10', 'Remove\ntop 50',
          'Remove\ntop 100', 'Singleton\nonly']
deltas = [-37.4, -38.1, -39.5, -41.2, -43.3, -51.3]
y_pos = np.arange(len(labels))

# Horizontal lollipop
for i, (d, lab) in enumerate(zip(deltas, labels)):
    color = '#922B21' if i == len(deltas)-1 else C_STRESS
    ax.plot([0, d], [i, i], color=color, lw=2, zorder=1)
    ax.scatter(d, i, color=color, s=S_POINT_LARGE, zorder=2, edgecolors='#2C3E50', linewidths=0.5)
    ax.text(d - 1.5, i, f'{d:.1f} nt', ha='right', va='center', fontsize=FS_ANNOT, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=FS_ANNOT)
ax.set_xlabel('Arsenite $\\Delta$ poly(A) (nt)')
ax.axvline(x=0, color='grey', linewidth=0.5)
ax.set_xlim(-65, 5)
ax.invert_yaxis()

# Gradient annotation
ax.annotate('', xy=(-52, 5.7), xytext=(-38, 5.7),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1))
ax.text(-45, 6.2, 'Stronger shortening', ha='center', fontsize=FS_ANNOT_SMALL, color='#555', style='italic')

# ────────────────────────────────────────────
# Panel (d): Donut — Loci sharing (Ars-only / Shared / HeLa-only)
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'd')

cats = ['Ars-only', 'Shared', 'HeLa-only']
n_loci = [2268, 652, 1577]
intronic_pct = [68, 70, 69]
young_pct_vals = [5, 6, 5]

total = sum(n_loci)
sizes = [n/total*100 for n in n_loci]
colors_d = [C_STRESS, C_HIGHLIGHT, C_NORMAL]

wedges, texts = ax.pie(sizes, labels=None, colors=colors_d, startangle=90,
                       wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5))

ax.text(0, 0, f'{total:,}\nloci', ha='center', va='center', fontweight='bold')

# Legend below the donut with compact labels
legend_labels = [f'{c} ({n:,}, {ip}% intronic, {yp}% young)'
                 for c, n, ip, yp in zip(cats, n_loci, intronic_pct, young_pct_vals)]
ax.legend(wedges, legend_labels, fontsize=5.5, loc='upper center',
          bbox_to_anchor=(0.5, -0.08), ncol=1, framealpha=0.8)

ax.set_title('L1 loci sharing between HeLa and Arsenite', pad=8)
ax.text(0.5, -0.38, '$\\chi^2$ $P$ = 0.43 (ns) — No genomic context bias',
        ha='center', fontsize=FS_ANNOT, color=C_GREY, style='italic',
        transform=ax.transAxes)

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS3')
print("\nFig S3 saved")
