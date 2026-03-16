#!/usr/bin/env python3
"""
Supplementary Figure S1: L1 Expression Landscape across 11 cell lines.
Redesigned: heatmap, scatter, violin, lollipop — no plain bar charts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import glob
from fig_style import *
import matplotlib.colors as mcolors

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
RESULTS = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
df_subfam = pd.read_csv(f'{BASE}/topic_05_cellline/part1_subfamily_composition.tsv', sep='\t')
df_summary = pd.read_csv(f'{BASE}/topic_05_cellline/part1_summary_table.tsv', sep='\t')
df_detect = pd.read_csv(f'{BASE}/topic_05_cellline/part1_detection_rate.tsv', sep='\t')

# Load raw per-read data for violin plots
raw_frames = []
for f in sorted(glob.glob(f'{RESULTS}/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'sample',
                                             'polya_length', 'qc_tag', 'gene_id'])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['group'] = group
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    # Fix naming
    tmp['cell_line'] = tmp['cell_line'].replace({
        'HeLa-Ars': 'HeLa-Ars', 'MCF7-EV': 'MCF7-EV',
        'SHSY5Y': 'SHSY5Y', 'Hct116': 'Hct116'
    })
    raw_frames.append(tmp)
df_raw = pd.concat(raw_frames, ignore_index=True)

# Cell line order by reads
cl_order = df_summary.sort_values('n_reads', ascending=False)['cell_line'].tolist()

# ── Create figure ──
fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.40,
                      left=0.09, right=0.97, top=0.96, bottom=0.08)

# ────────────────────────────────────────────
# Panel (a): Heatmap — Subfamily composition (clustered)
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, 'a')

subfam_cols = [c for c in df_subfam.columns if c != 'cell_line']
subfam_labels = [c.split('_', 1)[1] for c in subfam_cols]
mat = df_subfam.set_index('cell_line').loc[cl_order][subfam_cols].values

im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=35)
ax.set_xticks(range(len(subfam_labels)))
ax.set_xticklabels(subfam_labels, rotation=50, ha='right', fontsize=FS_ANNOT_SMALL)
ax.set_yticks(range(len(cl_order)))
ax.set_yticklabels(cl_order, fontsize=FS_ANNOT)

# Annotate cells with values
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        color = 'white' if v > 20 else 'black'
        if v >= 1:
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5.5, color=color)

cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cb.set_label('Proportion (%)', fontsize=FS_ANNOT)
cb.ax.tick_params(labelsize=6)

# ────────────────────────────────────────────
# Panel (b): Scatter — n_reads vs n_loci (discovery: saturation?)
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'b')

df_s = df_summary.copy()

ax.scatter(df_s['n_reads'], df_s['n_loci'], c=df_s['young_pct'],
           cmap='RdYlGn', s=S_POINT, edgecolors='#2C3E50', linewidths=0.5,
           zorder=3, vmin=0, vmax=20)

# Use adjustText to prevent label overlap
try:
    from adjustText import adjust_text
    texts_b = [ax.text(row['n_reads'], row['n_loci'], row['cell_line'],
                       fontsize=FS_LEGEND_SMALL) for _, row in df_s.iterrows()]
    adjust_text(texts_b, ax=ax, arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
                force_points=0.3, force_text=0.5, expand_points=(1.5, 1.5))
except ImportError:
    for _, row in df_s.iterrows():
        ax.annotate(row['cell_line'], (row['n_reads'], row['n_loci']),
                    fontsize=FS_LEGEND_SMALL, xytext=(4, 3), textcoords='offset points')

# Fit and show power law: loci ~ reads^alpha
from numpy.polynomial import polynomial as P
log_r = np.log10(df_s['n_reads'])
log_l = np.log10(df_s['n_loci'])
coef = np.polyfit(log_r, log_l, 1)
x_fit = np.linspace(log_r.min(), log_r.max(), 100)
ax.plot(10**x_fit, 10**np.polyval(coef, x_fit), '--', color=C_GREY, lw=0.8, zorder=1)
ax.text(0.03, 0.97, f'loci $\\propto$ reads$^{{{coef[0]:.2f}}}$',
        transform=ax.transAxes, fontsize=FS_ANNOT, va='top', color='#555')

ax.set_xlabel('Total L1 reads')
ax.set_ylabel('Unique L1 loci')
ax.set_xscale('log')
ax.set_yscale('log')

cb2 = fig.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.04)
cb2.set_label('Young L1 (%)', fontsize=FS_ANNOT)
cb2.ax.tick_params(labelsize=6)

# ────────────────────────────────────────────
# Panel (c): Violin — Read length distributions by CL
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'c')

violin_data = [df_raw[df_raw['cell_line']==cl]['read_length'].dropna().values for cl in cl_order]
# Filter empty
valid_idx = [i for i, d in enumerate(violin_data) if len(d) > 10]
valid_cls = [cl_order[i] for i in valid_idx]
valid_data = [violin_data[i] for i in valid_idx]

vp = ax.violinplot(valid_data, positions=range(len(valid_cls)),
                   showmeans=False, showmedians=True, widths=0.7)

for body in vp['bodies']:
    body.set_facecolor(C_CTRL)
    body.set_alpha(0.5)
vp['cmedians'].set_color(C_L1)
vp['cmedians'].set_linewidth(1.5)
vp['cmins'].set_linewidth(0.5)
vp['cmaxes'].set_linewidth(0.5)
vp['cbars'].set_linewidth(0.5)

ax.set_xticks(range(len(valid_cls)))
ax.set_xticklabels(valid_cls, rotation=45, ha='right', fontsize=FS_ANNOT)
ax.set_ylabel('Read length (bp)')
# Clip at 99th percentile to avoid long whiskers dominating
all_lens = np.concatenate(valid_data)
p99 = np.percentile(all_lens, 99)
ax.set_ylim(0, min(p99 * 1.1, 3000))

# Mark 3' bias cutoff
ax.axhline(y=1000, color=C_GREY, linestyle=':', linewidth=0.5)
ax.text(len(valid_cls)-0.5, 1050, "1 kb (3' bias cutoff)", ha='right', fontsize=FS_LEGEND_SMALL, color=C_GREY)

# ────────────────────────────────────────────
# Panel (d): Lollipop — Young vs Ancient detection rate
# ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'd')

df_det = df_detect.set_index('cell_line').loc[cl_order].reset_index()
y_pos = np.arange(len(cl_order))

# Lollipop: line from ancient to young
for i, row in df_det.iterrows():
    ax.plot([row['ancient_det_pct'], row['young_det_pct']], [i, i],
            color='#BDC3C7', lw=1.5, zorder=1)
    ax.scatter(row['ancient_det_pct'], i, color=C_ANCIENT, s=S_POINT, zorder=2,
              edgecolors='#2C3E50', linewidths=0.4)
    ax.scatter(row['young_det_pct'], i, color=C_YOUNG, s=S_POINT, zorder=2,
              edgecolors='#2C3E50', linewidths=0.4)

ax.set_yticks(y_pos)
ax.set_yticklabels(cl_order, fontsize=FS_ANNOT)
ax.set_xlabel('Per-locus detection rate (%)')
ax.invert_yaxis()

# Legend
ax.scatter([], [], color=C_YOUNG, s=S_POINT, label='Young L1')
ax.scatter([], [], color=C_ANCIENT, s=S_POINT, label='Ancient L1')
ax.legend(fontsize=FS_ANNOT, loc='lower right')

# Print ratio
ratios = df_det['young_det_pct'] / df_det['ancient_det_pct']
ax.text(0.97, 0.03, f'Young/Ancient: {ratios.mean():.1f}x',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        fontweight='bold', color=C_YOUNG)

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS1')
print("Fig S1 saved")
# Print discovery
print(f"\n[DISCOVERY] Loci ~ reads^{coef[0]:.3f} — sublinear scaling (saturation)")
print(f"[DISCOVERY] Read length medians: {[f'{np.median(d):.0f}' for d in valid_data[:3]]}")
