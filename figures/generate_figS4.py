#!/usr/bin/env python3
"""
Supplementary Figure S4: DDR host gene enrichment.
2 panels: (a) DDR pathway enrichment, (b) DDR gene m6A scatter
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from fig_style import *

setup_style()

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
df_enrich = pd.read_csv(f'{BASE}/topic_05_cellline/part4_host_gene_enrichment/length_controlled_enrichment.tsv', sep='\t')
df_ddr = pd.read_csv(f'{BASE}/topic_05_cellline/part4_ddr_m6a_integration/ddr_gene_m6a_stats.tsv', sep='\t')

# ── Create figure ──
fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
plt.subplots_adjust(wspace=0.45, left=0.10, right=0.96, top=0.90, bottom=0.18)

# ────────────────────────────────────────────
# Panel (a): Horizontal bar — DDR pathway enrichment
# ────────────────────────────────────────────
ax = axes[0]
panel_label(ax, 'a')

# Filter L1-specific enrichments
df_l1 = df_enrich[df_enrich['L1_specific'] == True].copy()
df_l1['neg_log10_p'] = -np.log10(df_l1['host_adjp'].astype(float))
df_l1 = df_l1.sort_values('neg_log10_p', ascending=True)

y = np.arange(len(df_l1))
colors_a = [C_L1 if p > -np.log10(0.05) else C_GREY for p in df_l1['neg_log10_p']]

# Wrap long term names
terms = []
for t in df_l1['Term']:
    if len(t) > 25:
        terms.append(t[:25] + '...')
    else:
        terms.append(t)

# Lollipop chart (horizontal)
lollipop_plot(ax, terms, df_l1['neg_log10_p'].values, colors=colors_a,
              horizontal=True, marker_size=45, stem_width=1.5, ref_value=0)
ax.set_yticklabels(terms, fontsize=FS_ANNOT)
ax.set_xlabel(r'$-\log_{10}$(adj. $P$)')
ax.axvline(x=-np.log10(0.05), color='grey', linestyle='--', linewidth=0.5)
ax.text(-np.log10(0.05) + 0.05, len(df_l1) - 0.5, '$P$ = 0.05', fontsize=FS_ANNOT_SMALL, color='grey')

# ────────────────────────────────────────────
# Panel (b): Scatter — DDR gene m6A/kb vs n_reads
# ────────────────────────────────────────────
ax = axes[1]
panel_label(ax, 'b')

ax.scatter(df_ddr['n_reads'], df_ddr['m6a_kb_median'], color=C_GREY, s=25,
           edgecolors='#2C3E50', linewidths=0.5, zorder=2)

# Highlight BRCA1
brca1 = df_ddr[df_ddr['gene'] == 'BRCA1']
if len(brca1) > 0:
    ax.scatter(brca1['n_reads'], brca1['m6a_kb_median'], color=C_L1, s=60,
               edgecolors='#2C3E50', linewidths=0.8, zorder=3, marker='*')

# Label all genes with adjustText
try:
    from adjustText import adjust_text
    texts_b = []
    for _, row in df_ddr.iterrows():
        color = C_L1 if row['gene'] == 'BRCA1' else '#555'
        fw = 'bold' if row['gene'] == 'BRCA1' else 'normal'
        fs = FS_ANNOT if row['gene'] == 'BRCA1' else FS_LEGEND_SMALL
        texts_b.append(ax.text(row['n_reads'], row['m6a_kb_median'], row['gene'],
                               fontsize=fs, color=color, fontweight=fw))
    adjust_text(texts_b, ax=ax, arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
                force_points=0.4, force_text=0.5, expand_points=(1.8, 1.8))
except ImportError:
    for _, row in df_ddr.iterrows():
        ax.annotate(row['gene'], (row['n_reads'], row['m6a_kb_median']),
                    fontsize=FS_LEGEND_SMALL, color='#555', xytext=(3, 1), textcoords='offset points')

# Dashed line at overall L1 median (compute from data)
l1_median = df_ddr['m6a_kb_median'].median()
ax.axhline(y=l1_median, color='grey', linestyle='--', linewidth=0.5)
ax.text(df_ddr['n_reads'].max() * 0.95, l1_median + 0.15, 'L1 median', ha='right', fontsize=FS_ANNOT_SMALL, color='grey')

ax.set_xlabel('Number of reads')
ax.set_ylabel('Median m6A sites per kb')
ax.set_ylim(0, df_ddr['m6a_kb_median'].max() * 1.15)

# ── Save ──
save_figure(fig, f'{OUTDIR}/figS4')
print("Fig S4 saved to", f'{OUTDIR}/figS4.pdf')
