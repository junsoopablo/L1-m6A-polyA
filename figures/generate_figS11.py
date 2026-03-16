#!/usr/bin/env python3
"""
Supplementary Figure S11: LTR12C-driven chimeric L1PA7 transcript in HepG2.

4 panels:
(a) Read 5' end position histogram → defined TSS in LTR12C
(b) Read length vs 5' end position → plateau confirms TSS, not DRS truncation
(c) Cross-cell-line read count at this locus
(d) Read-length-matched m6A/kb comparison: hotspot vs rest of HepG2
"""
import sys
sys.path.insert(0, '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

from fig_style import (setup_style, panel_label, save_figure, despine,
                        lollipop_plot,
                        FULL_WIDTH, C_L1, C_CTRL, C_STRESS, C_NORMAL,
                        C_GREY, C_HIGHLIGHT, C_TEXT, C_YOUNG, C_ANCIENT, C_CATB,
                        FS_ANNOT, FS_ANNOT_SMALL, FS_LEGEND_SMALL)

setup_style()

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'

# Key coordinates (minus strand; genomic coords decrease in transcript direction)
L1PA7_FRAG1_START = 23583649  # genomic start (= transcript 3' end)
L1PA7_FRAG1_END   = 23583899  # genomic end (= transcript 5' side of frag1)
LTR12C_START       = 23583899  # = L1PA7 frag1 end
LTR12C_END         = 23585447  # = L1PA7 frag2 start
L1PA7_FRAG2_START  = 23585447
L1PA7_FRAG2_END    = 23587098

###############################################################################
# 1. Load data
###############################################################################
print("=== Loading data ===")

# HepG2 L1 summary
summary_rows = []
for f in sorted(RESULTS.glob('HepG2_*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    summary_rows.append(df)
hepg2_summary = pd.concat(summary_rows, ignore_index=True)
hepg2_summary = hepg2_summary[hepg2_summary['qc_tag'] == 'PASS'].copy()

# Identify hotspot reads
hotspot = hepg2_summary[
    (hepg2_summary['chr'] == 'chr14') &
    (hepg2_summary['start'] >= 23578000) &
    (hepg2_summary['end'] <= 23595000)
].copy()
rest = hepg2_summary[~hepg2_summary.index.isin(hotspot.index)].copy()

print(f"  Hotspot: {len(hotspot)}, Rest: {len(rest)}")

# Part3 cache for m6A
cache_rows = []
for f in sorted(CACHE_DIR.glob('HepG2_*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t')
    cache_rows.append(df)
cache = pd.concat(cache_rows, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)

hotspot_m = hotspot.merge(cache[['read_id', 'm6a_per_kb', 'read_length']],
                         on='read_id', how='inner', suffixes=('', '_cache'))
rest_m = rest.merge(cache[['read_id', 'm6a_per_kb', 'read_length']],
                    on='read_id', how='inner', suffixes=('', '_cache'))
# Use cache read_length (from BAM) — prefer *_cache if conflict
for df in [hotspot_m, rest_m]:
    if 'read_length_cache' in df.columns:
        df['read_length'] = df['read_length_cache']

# Cross-CL data
cross_cl = {}
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    locus_reads = df[(df['chr'] == 'chr14') & (df['start'] >= 23578000) & (df['end'] <= 23595000)]
    group = f.parent.parent.name
    cl = group.rsplit('_', 1)[0]
    if cl not in cross_cl:
        cross_cl[cl] = {'total': 0, 'hotspot': 0}
    cross_cl[cl]['total'] += len(df)
    cross_cl[cl]['hotspot'] += len(locus_reads)

###############################################################################
# 2. Create figure
###############################################################################
print("=== Generating figure ===")

fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.85))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                       left=0.08, right=0.97, top=0.95, bottom=0.08)

# ── Panel (a): Read 5' end position histogram ──
ax_a = fig.add_subplot(gs[0, 0])

# On minus strand, read 'end' in genomic coords = transcript 5' end
# Read 'start' in genomic coords = transcript 3' end (near poly(A))
five_prime_ends = hotspot['end'].values  # genomic end = transcript 5' for minus strand

# Convert to distance from L1PA7/LTR12C junction (positive = into LTR12C)
dist_from_junction = five_prime_ends - L1PA7_FRAG1_END  # 0 = junction, positive = LTR12C

bins = np.arange(-100, 1600, 20)
ax_a.hist(dist_from_junction, bins=bins, color=C_L1, alpha=0.85, edgecolor='white', lw=0.3)

# Mark regions
ax_a.axvspan(-100, 0, alpha=0.1, color=C_ANCIENT, zorder=0)
ax_a.axvspan(0, 1600, alpha=0.08, color=C_CATB, zorder=0)
ax_a.axvline(0, color=C_TEXT, lw=0.8, ls='--', alpha=0.5)

# Median TSS
median_dist = np.median(dist_from_junction)
ax_a.axvline(median_dist, color=C_STRESS, lw=1.2, ls='-')
ax_a.annotate(f'TSS\n{median_dist:.0f} bp', xy=(median_dist, ax_a.get_ylim()[1]*0.85),
              fontsize=FS_ANNOT_SMALL, color=C_STRESS, ha='left', va='top',
              xytext=(median_dist + 60, ax_a.get_ylim()[1]*0.85))

ax_a.set_xlabel('Distance from L1PA7/LTR12C junction (bp)')
ax_a.set_ylabel('Number of reads')
ax_a.set_xlim(-100, 1000)

# Region labels
ax_a.text(-50, ax_a.get_ylim()[1]*0.95, "L1PA7\n3'UTR", fontsize=FS_LEGEND_SMALL, ha='center',
          va='top', color=C_ANCIENT, style='italic')
ax_a.text(750, ax_a.get_ylim()[1]*0.95, "LTR12C", fontsize=FS_LEGEND_SMALL, ha='center',
          va='top', color=C_CATB, style='italic')

panel_label(ax_a, 'a')
despine(ax_a)

# ── Panel (b): Read length vs 5' end → TSS plateau ──
ax_b = fig.add_subplot(gs[0, 1])

# Use hotspot_m which already has read_length from Part3 cache merge
rl = hotspot_m['read_length'].values
fpe = hotspot_m['end'].values - L1PA7_FRAG1_END  # distance into LTR12C

ax_b.scatter(rl, fpe, s=1.5, alpha=0.2, color=C_L1, edgecolors='none', rasterized=True)

# Bin by read length and show median
rl_bins = [(200, 400), (400, 500), (500, 600), (600, 700), (700, 800),
           (800, 1000), (1000, 2000)]
bin_centers = []
bin_medians = []
bin_ns = []
for lo, hi in rl_bins:
    mask = (rl >= lo) & (rl < hi)
    if mask.sum() >= 5:
        bin_centers.append((lo + hi) / 2)
        bin_medians.append(np.median(fpe[mask]))
        bin_ns.append(mask.sum())

ax_b.plot(bin_centers, bin_medians, 'o-', color=C_STRESS, ms=4, lw=1.2, zorder=5)

# Horizontal line at TSS plateau
ax_b.axhline(median_dist, color=C_STRESS, lw=0.8, ls=':', alpha=0.6)
ax_b.annotate(f'TSS plateau ~{median_dist:.0f} bp',
              xy=(1500, median_dist), fontsize=FS_LEGEND_SMALL, color=C_STRESS,
              va='bottom', ha='right')

# DRS capacity line (45-degree: if TSS = read length, reads are DRS-limited)
x_line = np.array([0, 2000])
ax_b.plot(x_line, x_line, color=C_GREY, lw=0.7, ls='--', alpha=0.5)
ax_b.text(300, 350, 'DRS-limited', fontsize=5.5, color=C_GREY, rotation=35)

ax_b.set_xlabel('Read length (bp)')
ax_b.set_ylabel("5' end distance into LTR12C (bp)")
ax_b.set_xlim(0, 2100)
ax_b.set_ylim(-50, 1200)

panel_label(ax_b, 'b')
despine(ax_b)

# ── Panel (c): Cross-CL bar chart ──
ax_c = fig.add_subplot(gs[1, 0])

# Sort by hotspot count, exclude Ars and EV conditions
cl_order = []
for cl in sorted(cross_cl.keys()):
    if cl in ['HeLa-Ars', 'MCF7-EV']:
        continue
    cl_order.append(cl)
cl_order.sort(key=lambda x: cross_cl[x]['hotspot'], reverse=True)

# Keep only standard 9 cell lines
STANDARD_CLS = {'A549', 'H9', 'HEYA8', 'MCF7', 'HepG2', 'K562', 'Hct116', 'SHSY5Y', 'HeLa'}
cl_order = [c for c in cl_order if c in STANDARD_CLS]
# Sort: HepG2 first, then others by count (descending)
cl_order_sorted = sorted([c for c in cl_order if c != 'HepG2'],
                         key=lambda x: cross_cl[x]['hotspot'], reverse=True)
cl_order = ['HepG2'] + cl_order_sorted

x_pos = np.arange(len(cl_order))
counts = [cross_cl[cl]['hotspot'] for cl in cl_order]
pcts = [100 * cross_cl[cl]['hotspot'] / cross_cl[cl]['total'] if cross_cl[cl]['total'] > 0 else 0
        for cl in cl_order]
colors = [C_HIGHLIGHT if cl == 'HepG2' else C_GREY for cl in cl_order]

# Lollipop chart (vertical)
lollipop_plot(ax_c, cl_order, counts, colors=colors,
              horizontal=False, marker_size=45, stem_width=1.5, ref_value=0)

# Percentage label only for HepG2 and those with >0 reads
for i, (c, p) in enumerate(zip(counts, pcts)):
    if c > 0:
        ax_c.text(i, c + max(counts)*0.02, f'{p:.1f}%', ha='center', fontsize=5.5,
                  color=C_HIGHLIGHT if cl_order[i] == 'HepG2' else C_TEXT)

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=6.5)
ax_c.set_ylabel('Reads at L1PA7 locus')
ax_c.set_title('chr14:23.58 Mb (HepG2-specific)', fontsize=FS_ANNOT, style='italic')

panel_label(ax_c, 'c')
despine(ax_c)

# ── Panel (d): Read-length-matched m6A/kb ──
ax_d = fig.add_subplot(gs[1, 1])

rl_ranges = [(300, 500), (500, 750), (750, 1000)]
rl_labels = ['300-500', '500-750', '750-1000']
x_d = np.arange(len(rl_ranges))
width = 0.35

hotspot_medians = []
rest_medians = []
hotspot_ns = []
rest_ns = []
p_vals = []

for lo, hi in rl_ranges:
    h = hotspot_m[(hotspot_m['read_length'] >= lo) & (hotspot_m['read_length'] < hi)]['m6a_per_kb']
    r = rest_m[(rest_m['read_length'] >= lo) & (rest_m['read_length'] < hi)]['m6a_per_kb']
    hotspot_medians.append(h.median() if len(h) >= 5 else 0)
    rest_medians.append(r.median() if len(r) >= 5 else 0)
    hotspot_ns.append(len(h))
    rest_ns.append(len(r))
    if len(h) >= 5 and len(r) >= 5:
        _, p = stats.mannwhitneyu(h, r, alternative='two-sided')
        p_vals.append(p)
    else:
        p_vals.append(1.0)

# Paired dot + connecting lines
offset = 0.15
for i in range(len(rl_ranges)):
    if hotspot_medians[i] > 0 and rest_medians[i] > 0:
        ax_d.plot([i - offset, i + offset], [hotspot_medians[i], rest_medians[i]],
                  color='#BDC3C7', lw=1.5, zorder=1)
ax_d.scatter(x_d - offset, hotspot_medians, s=40, color=C_HIGHLIGHT,
             edgecolors='white', linewidths=0.5, zorder=3, label='L1PA7 locus')
ax_d.scatter(x_d + offset, rest_medians, s=40, color=C_GREY,
             edgecolors='white', linewidths=0.5, zorder=3, label='Rest of HepG2')

# Significance and ratio
for i, (hm, rm, p) in enumerate(zip(hotspot_medians, rest_medians, p_vals)):
    if hm > 0 and rm > 0:
        ratio = hm / rm
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        y_top = max(hm, rm) + 0.5
        ax_d.text(i, y_top, f'{ratio:.1f}x {sig}', ha='center', fontsize=5.5,
                  color=C_TEXT)

# n labels
for i in range(len(rl_ranges)):
    ax_d.text(i - width/2, -0.8, f'n={hotspot_ns[i]}', ha='center', fontsize=4.5,
              color=C_HIGHLIGHT)
    ax_d.text(i + width/2, -0.8, f'n={rest_ns[i]}', ha='center', fontsize=4.5,
              color=C_GREY)

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(rl_labels, fontsize=FS_ANNOT_SMALL)
ax_d.set_xlabel('Read length bin (bp)')
ax_d.set_ylabel('m$^6$A/kb (median)')
ax_d.legend(fontsize=FS_LEGEND_SMALL, loc='upper right')
ax_d.set_ylim(bottom=0)

panel_label(ax_d, 'd')
despine(ax_d)

###############################################################################
# 3. Save
###############################################################################
out_path = BASE / 'manuscript/figures/figS11'
save_figure(fig, str(out_path))
print(f"\n=== Saved: {out_path}.pdf ===")
