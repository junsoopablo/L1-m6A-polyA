#!/usr/bin/env python3
"""
Generate Part 1 PDF: L1 Expression Landscape.
Research paper Results-style document with figures and tables.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fpdf import FPDF

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures'
FIGDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
MIN_READS = 200

CELL_LINES = {
    'A549':     ['A549_4', 'A549_5', 'A549_6'],
    'H9':       ['H9_2', 'H9_3', 'H9_4'],
    'Hct116':   ['Hct116_3', 'Hct116_4'],
    'HeLa':     ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':    ['HepG2_5', 'HepG2_6'],
    'HEYA8':    ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':     ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':     ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV':  ['MCF7-EV_1'],
    'SHSY5Y':   ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")
all_dfs = []
for cl, groups in CELL_LINES.items():
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        if len(df) < MIN_READS:
            continue
        df['group'] = g
        df['cell_line'] = cl
        df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

def classify_subfamily(sf):
    if sf == 'L1HS': return 'L1HS'
    elif sf in ['L1PA1','L1PA2','L1PA3']: return 'L1PA1-3'
    elif sf in ['L1PA4','L1PA5','L1PA6','L1PA7','L1PA8']: return 'L1PA4-8'
    elif sf.startswith('L1PA'): return 'L1PA9+'
    elif sf.startswith('L1PB'): return 'L1PB'
    elif sf.startswith('L1MC'): return 'L1MC'
    elif sf.startswith('L1ME'): return 'L1ME'
    elif sf.startswith('L1M'): return 'L1M (other)'
    elif sf.startswith('HAL1'): return 'HAL1'
    else: return 'Other'

data['subfam_group'] = data['gene_id'].apply(classify_subfamily)

# Reference L1
ref_l1 = pd.read_csv(PROJECT / 'reference/L1_TE_L1_family.bed', sep='\t',
                      header=None, names=['chr','start','end','subfamily','locus_id'])
ref_l1['l1_age'] = ref_l1['subfamily'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

# Cell line order by read count
cl_order = data.groupby('cell_line').size().sort_values(ascending=False).index.tolist()

# =========================================================================
# Figure 1: Overview (4 panels)
# =========================================================================
print("Generating Figure 1...")
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
fig1.subplots_adjust(hspace=0.35, wspace=0.3)

# 1A: Read counts + replicate dots
ax = axes[0, 0]
for i, cl in enumerate(cl_order):
    reps = data[data['cell_line'] == cl].groupby('group').size().values
    total = reps.sum()
    ax.bar(i, total, color='#4C72B0', alpha=0.8, edgecolor='white')
    for r in reps:
        ax.scatter(i, r, color='#C44E52', s=20, zorder=5, alpha=0.7)
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('L1 reads')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Total L1 read counts', fontsize=10, loc='center')
# Add second y-axis label note
ax.text(0.98, 0.95, 'bars=total, dots=per replicate',
        transform=ax.transAxes, fontsize=7, ha='right', va='top', style='italic')

# 1B: Read length (young vs ancient, boxplot)
ax = axes[0, 1]
positions = []
bp_data_anc = []
bp_data_yng = []
for i, cl in enumerate(cl_order):
    d = data[data['cell_line'] == cl]
    bp_data_anc.append(d[d['l1_age']=='ancient']['read_length'].values)
    bp_data_yng.append(d[d['l1_age']=='young']['read_length'].values)

x = np.arange(len(cl_order))
w = 0.35
bp1 = ax.boxplot(bp_data_anc, positions=x-w/2, widths=w*0.8, showfliers=False,
                  patch_artist=True, medianprops=dict(color='black'))
bp2 = ax.boxplot(bp_data_yng, positions=x+w/2, widths=w*0.8, showfliers=False,
                  patch_artist=True, medianprops=dict(color='black'))
for patch in bp1['boxes']:
    patch.set_facecolor('#4C72B0')
    patch.set_alpha(0.7)
for patch in bp2['boxes']:
    patch.set_facecolor('#C44E52')
    patch.set_alpha(0.7)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Read length (bp)')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Read length by L1 age', fontsize=10, loc='center')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Ancient', 'Young'], fontsize=8, loc='upper right')

# 1C: Gene context stacked bar
ax = axes[1, 0]
intronic_pct = []
intergenic_pct = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    intronic_pct.append((d['TE_group'] == 'intronic').mean() * 100)
    intergenic_pct.append((d['TE_group'] == 'intergenic').mean() * 100)
ax.bar(x, intronic_pct, label='Intronic', color='#4C72B0', alpha=0.8)
ax.bar(x, intergenic_pct, bottom=intronic_pct, label='Intergenic', color='#DD8452', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Fraction (%)')
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Genomic context', fontsize=10, loc='center')
ax.legend(fontsize=8)

# 1D: Detection rate (young vs ancient)
ax = axes[1, 1]
n_ref_young = (ref_l1['l1_age'] == 'young').sum()
n_ref_ancient = (ref_l1['l1_age'] == 'ancient').sum()
ref_young_set = set(ref_l1[ref_l1['l1_age']=='young']['locus_id'])
ref_anc_set = set(ref_l1[ref_l1['l1_age']=='ancient']['locus_id'])

young_det = []
anc_det = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    yng_expr = set(d[d['l1_age']=='young']['transcript_id'].unique())
    anc_expr = set(d[d['l1_age']=='ancient']['transcript_id'].unique())
    young_det.append(len(yng_expr & ref_young_set) / n_ref_young * 100)
    anc_det.append(len(anc_expr & ref_anc_set) / n_ref_ancient * 100)

ax.bar(x - w/2, anc_det, w, label='Ancient', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, young_det, w, label='Young', color='#C44E52', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Detection rate (%)')
ax.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 loci detection rate', fontsize=10, loc='center')
ax.legend(fontsize=8)

fig1.savefig(FIGDIR / 'fig1_overview.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  Figure 1 saved")

# =========================================================================
# Figure 2: Subfamily heatmap + stacked bar
# =========================================================================
print("Generating Figure 2...")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.5, 1]})
fig2.subplots_adjust(wspace=0.35)

# Age-grouped composition
age_groups_order = ['L1HS', 'L1PA1-3', 'L1PA4-8', 'L1PA9+', 'L1PB',
                    'L1MC', 'L1ME', 'L1M (other)', 'HAL1', 'Other']
age_cl = data.groupby(['cell_line', 'subfam_group']).size().unstack(fill_value=0)
age_pct = age_cl.div(age_cl.sum(axis=1), axis=0) * 100
# Ensure all columns exist
for col in age_groups_order:
    if col not in age_pct.columns:
        age_pct[col] = 0.0
age_pct = age_pct[age_groups_order]

# 2A: Heatmap
ax = axes2[0]
hm = age_pct.loc[cl_order]
im = ax.imshow(hm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=40)
ax.set_xticks(range(len(age_groups_order)))
ax.set_xticklabels(age_groups_order, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(cl_order)))
ax.set_yticklabels(cl_order, fontsize=9)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        val = hm.iloc[i, j]
        if val >= 0.5:
            color = 'white' if val > 30 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax, label='% of reads', shrink=0.8, pad=0.02)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 subfamily composition (%)', fontsize=10, loc='center')

# 2B: Stacked horizontal bar
ax = axes2[1]
colors_age = ['#d62728', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2',
              '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#bdbdbd']
left = np.zeros(len(cl_order))
for j, grp in enumerate(age_groups_order):
    vals = [hm.loc[cl, grp] for cl in cl_order]
    ax.barh(range(len(cl_order)), vals, left=left, color=colors_age[j],
            edgecolor='white', linewidth=0.3, label=grp)
    left += np.array(vals)
ax.set_yticks(range(len(cl_order)))
ax.set_yticklabels(cl_order, fontsize=9)
ax.set_xlabel('% of reads')
ax.invert_yaxis()
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Stacked view', fontsize=10, loc='center')
ax.legend(fontsize=6, loc='lower right', ncol=2, framealpha=0.9)

fig2.savefig(FIGDIR / 'fig2_subfamily.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  Figure 2 saved")

# =========================================================================
# Figure 3: Hotspot analysis (4 panels)
# =========================================================================
print("Generating Figure 3...")

def gini_coefficient(values):
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0: return 0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))

# Compute concentration
conc = {}
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    lc = d.groupby('transcript_id').size().sort_values(ascending=False)
    n = len(d)
    conc[cl] = {
        'gini': gini_coefficient(lc.values),
        'top5_pct': lc.head(5).sum() / n * 100,
        'top_locus': lc.index[0],
        'top_pct': lc.iloc[0] / n * 100,
    }

# Top 20 hotspots
loci_all = data.groupby('transcript_id').agg(
    total_reads=('read_id', 'count'),
    gene_id=('gene_id', 'first'),
    TE_group=('TE_group', 'first'),
    n_celllines=('cell_line', 'nunique'),
).sort_values('total_reads', ascending=False)

top20_loci = loci_all.head(20).index.tolist()
top20_spec = []
for locus in top20_loci:
    d = data[data['transcript_id'] == locus]
    cl_counts = d['cell_line'].value_counts()
    dom_pct = cl_counts.iloc[0] / len(d) * 100
    top20_spec.append('specific' if dom_pct > 80 else ('semi' if dom_pct > 50 else 'shared'))

# Loci sharing
loci_cl_count = data.groupby('transcript_id')['cell_line'].nunique()
sharing_dist = loci_cl_count.value_counts().sort_index()

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 10))
fig3.subplots_adjust(hspace=0.4, wspace=0.35)

# 3A: Gini + top5%
ax = axes3[0, 0]
conc_sorted = sorted(conc.items(), key=lambda x: x[1]['gini'])
cl_names = [c[0] for c in conc_sorted]
gini_vals = [c[1]['gini'] for c in conc_sorted]
colors_g = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cl_names)))
bars = ax.barh(range(len(cl_names)), gini_vals, color=colors_g, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(cl_names)))
ax.set_yticklabels(cl_names, fontsize=9)
ax.set_xlabel('Gini coefficient')
for i, v in enumerate(gini_vals):
    t5 = conc[cl_names[i]]['top5_pct']
    ax.text(v + 0.008, i, f'{v:.3f} (top5={t5:.0f}%)', va='center', fontsize=7)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Expression concentration', fontsize=10, loc='center')
ax.set_xlim(0, 0.75)

# 3B: Top 20 hotspots
ax = axes3[0, 1]
spec_colors = {'specific': '#C44E52', 'semi': '#E8845C', 'shared': '#4C72B0'}
reads_vals = [loci_all.loc[l, 'total_reads'] for l in top20_loci]
bar_colors = [spec_colors[s] for s in top20_spec]
ax.barh(range(len(top20_loci)), reads_vals, color=bar_colors, edgecolor='gray', linewidth=0.3)
labels = [f"{l} ({loci_all.loc[l,'gene_id']})" for l in top20_loci]
ax.set_yticks(range(len(top20_loci)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Total reads')
ax.invert_yaxis()
from matplotlib.patches import Patch
legend_el = [Patch(fc='#C44E52', label='Cell-type specific (>80%)'),
             Patch(fc='#E8845C', label='Semi-specific (50-80%)'),
             Patch(fc='#4C72B0', label='Shared (<50%)')]
ax.legend(handles=legend_el, fontsize=7, loc='lower right')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Top 20 hotspot loci', fontsize=10, loc='center')

# 3C: Top 15 hotspot x cell line heatmap
ax = axes3[1, 0]
top15 = top20_loci[:15]
hm_data = []
for locus in top15:
    d = data[data['transcript_id'] == locus]
    total = len(d)
    row = [(d['cell_line'] == cl).sum() / total * 100 for cl in cl_order]
    hm_data.append(row)
hm_arr = np.array(hm_data)
im = ax.imshow(hm_arr, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f"{l} ({loci_all.loc[l,'gene_id']})" for l in top15], fontsize=7)
for i in range(hm_arr.shape[0]):
    for j in range(hm_arr.shape[1]):
        val = hm_arr[i, j]
        if val >= 5:
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax, label='% of reads', shrink=0.8)
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Hotspot cell line distribution', fontsize=10, loc='center')

# 3D: Loci sharing
ax = axes3[1, 1]
ax.bar(sharing_dist.index, sharing_dist.values, color='#4C72B0', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of cell lines')
ax.set_ylabel('Number of loci')
# Annotate key numbers
n_unique = sharing_dist.get(1, 0)
total_loci = loci_cl_count.count()
ax.text(0.97, 0.95, f'1 CL only: {n_unique:,} ({n_unique/total_loci*100:.0f}%)\n'
        f'Shared (>=2): {total_loci-n_unique:,} ({(total_loci-n_unique)/total_loci*100:.0f}%)\n'
        f'Ubiquitous (>=8): {(loci_cl_count>=8).sum()}',
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Loci sharing across cell lines', fontsize=10, loc='center')

fig3.savefig(FIGDIR / 'fig3_hotspot.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  Figure 3 saved")

# =========================================================================
# Figure 4: 3' bias + sparsity (2 panels)
# =========================================================================
print("Generating Figure 4...")
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.subplots_adjust(wspace=0.3)

# 4A: dist_to_3prime
ax = axes4[0]
for age, color, label in [('ancient', '#4C72B0', 'Ancient'), ('young', '#C44E52', 'Young')]:
    d = data[(data['l1_age'] == age) & (data['dist_to_3prime'].notna())]
    ax.hist(d['dist_to_3prime'].clip(upper=5000), bins=60, alpha=0.5,
            color=color, label=f'{label} (n={len(d):,})', density=True)
ax.axvline(x=1000, color='gray', ls='--', lw=0.8, alpha=0.7)
ax.text(1050, ax.get_ylim()[1]*0.9, '1 kb', fontsize=8, color='gray')
ax.set_xlabel("Distance to 3' end of L1 element (bp)")
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title("3' coverage bias", fontsize=10, loc='center')

# 4B: Loci coverage distribution
ax = axes4[1]
loci_counts = data.groupby('transcript_id').size()
counts_clipped = loci_counts.clip(upper=20)
ax.hist(counts_clipped, bins=range(1, 22), color='#4C72B0', alpha=0.8, edgecolor='white')
ax.set_xlabel('Reads per locus')
ax.set_ylabel('Number of loci')
ax.set_xticks(range(1, 21))
ax.set_xticklabels([str(i) if i < 20 else '20+' for i in range(1, 21)], fontsize=7)
n_sing = (loci_counts == 1).sum()
ax.text(0.97, 0.95, f'Singletons: {n_sing:,} ({n_sing/len(loci_counts)*100:.0f}%)\n'
        f'>=5 reads: {(loci_counts>=5).sum():,}\n'
        f'>=10 reads: {(loci_counts>=10).sum():,}',
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Loci read coverage', fontsize=10, loc='center')

fig4.savefig(FIGDIR / 'fig4_bias_sparsity.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  Figure 4 saved")

# =========================================================================
# Generate PDF
# =========================================================================
print("\nGenerating PDF...")

class ResultsPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 5, 'Part 1: L1 Expression Landscape', align='L')
            self.cell(0, 5, f'Page {self.page_no()}', align='R', new_x='LMARGIN', new_y='NEXT')
            self.line(10, 12, 200, 12)
            self.ln(3)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9.5)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def figure_caption(self, text):
        self.set_font('Helvetica', 'I', 8.5)
        self.multi_cell(0, 4.5, text)
        self.ln(3)

pdf = ResultsPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --- Title Page ---
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 20)
pdf.cell(0, 12, 'Part 1: L1 Expression Landscape', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(8)
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 8, 'Direct RNA Sequencing of LINE-1 Transposons', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 8, 'across Human Cell Lines', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(15)
pdf.set_font('Helvetica', '', 10)
pdf.cell(0, 6, '11 cell lines | 29 replicates | 38,544 L1 reads | 12,125 loci', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(20)
pdf.set_font('Helvetica', 'I', 9)
pdf.cell(0, 6, 'IsoTENT L1 Project - Results Draft', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 6, 'Generated: 2026-02-09', align='C', new_x='LMARGIN', new_y='NEXT')

# --- Table 1: Summary ---
pdf.add_page()
pdf.section_title('1. L1 Expression Across Cell Lines')

pdf.body_text(
    'We analyzed L1 transposon expression using Oxford Nanopore Direct RNA '
    'Sequencing (DRS) across 11 human cell lines, comprising 29 replicates '
    '(Table 1). After quality filtering (QC=PASS, minimum 200 reads per '
    'replicate), we obtained 38,544 L1 reads mapping to 12,125 unique L1 loci '
    'from 128 subfamilies.'
)

pdf.body_text(
    'L1 read counts varied over 10-fold across cell lines, with MCF7 yielding '
    'the most reads (12,545; 4,182/replicate) and SHSY5Y the fewest (1,083; '
    '361/replicate). This variation likely reflects differences in both L1 '
    'transcriptional activity and library complexity.'
)

# Build summary table
summary = pd.read_csv(OUTDIR / 'part1_summary_table.tsv', sep='\t')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'Table 1. L1 expression summary across 11 cell lines.', new_x='LMARGIN', new_y='NEXT')
pdf.ln(1)

# Table header
pdf.set_font('Helvetica', 'B', 7.5)
col_widths = [22, 10, 18, 16, 14, 14, 16, 22, 24]
headers = ['Cell Line', 'Reps', 'Reads', 'Reads/Rep', 'Loci', 'Subfam', 'Young%', 'rdLen med', 'rdLen IQR']
for w, h in zip(col_widths, headers):
    pdf.cell(w, 5, h, border=1, align='C')
pdf.ln()

# Table body
pdf.set_font('Helvetica', '', 7.5)
for _, r in summary.iterrows():
    vals = [
        r['cell_line'],
        str(int(r['n_reps'])),
        f"{int(r['n_reads']):,}",
        f"{r['reads_per_rep']:,.0f}",
        f"{int(r['n_loci']):,}",
        str(int(r['n_subfamilies'])),
        f"{r['young_pct']:.1f}%",
        f"{r['rdlen_median']:,.0f}",
        f"{r['rdlen_Q25']:,.0f}-{r['rdlen_Q75']:,.0f}",
    ]
    for w, v in zip(col_widths, vals):
        pdf.cell(w, 4.5, v, border=1, align='C')
    pdf.ln()

pdf.ln(3)

pdf.body_text(
    'Ancient L1 elements (L1MC, L1ME, L1M families) constituted the vast '
    'majority of detected reads (mean 93.1%, range 81.8-96.9%), while young '
    'L1 elements (L1HS, L1PA1-3) comprised only 3.1-18.2% (mean 6.9%). Two '
    'notable exceptions were MCF7-EV (18.2% young) and HEYA8 (13.2% young). '
    'The median read length was 916 bp overall, with young L1 reads '
    'significantly longer than ancient (1,283 vs 900 bp; Kruskal-Wallis '
    'p < 10^-100).'
)

# --- Figure 1 ---
pdf.add_page()
pdf.section_title('Figure 1. L1 Expression Overview')
pdf.image(str(FIGDIR / 'fig1_overview.png'), x=10, w=190)
pdf.ln(2)
pdf.figure_caption(
    'Figure 1. Overview of L1 expression across 11 cell lines. '
    '(A) Total L1 read counts per cell line (bars) with individual replicate '
    'counts (red dots). MCF7 and HepG2 dominate. '
    '(B) Read length distributions for ancient (blue) and young (red) L1 '
    'elements. Young L1 reads are consistently longer. '
    '(C) Genomic context: 61% of L1 reads originate from intronic elements. '
    '(D) Detection rate: fraction of reference L1 loci with at least one read. '
    'Young L1 shows higher per-locus detection despite lower absolute counts.'
)

# --- Results text: Subfamily ---
pdf.add_page()
pdf.section_title('2. L1 Subfamily Composition')

pdf.body_text(
    'To characterize which L1 subfamilies are transcriptionally active, we '
    'grouped the 128 detected subfamilies into 10 evolutionary age categories '
    '(Figure 2). The composition was remarkably consistent across cell lines: '
    'L1MC (10.7-20.2%), L1ME (21.7-30.7%), and other ancient L1M families '
    '(20.7-35.4%) together comprised 63-79% of all L1 reads in every cell line.'
)

pdf.body_text(
    'The youngest subfamily, L1HS, was rare in all cell lines (<2.8%), '
    'consistent with the known mapping challenge for these highly similar '
    'elements (MAPQ analysis showed only 30% of L1HS reads map uniquely). '
    'The intermediate L1PA4-8 group showed the most inter-cell-line variation, '
    'with HepG2 having a notably elevated fraction (23.4%) driven by a single '
    'dominant L1PA7 locus (L1PA7_dup11216).'
)

pdf.body_text(
    'The top individual subfamilies by read count were L1PA7 (5.3%), L1MC4 '
    '(5.2%), L1M5 (4.2%), and L1MB7 (3.7%). MCF7-EV showed the highest young '
    'L1 fraction (L1PA1-3: 15.4%), consistent with previously reported '
    'enrichment of young L1 RNA in extracellular vesicles.'
)

# --- Figure 2 ---
pdf.add_page()
pdf.section_title('Figure 2. L1 Subfamily Composition')
pdf.image(str(FIGDIR / 'fig2_subfamily.png'), x=5, w=200)
pdf.ln(2)
pdf.figure_caption(
    'Figure 2. L1 subfamily composition across cell lines. '
    '(A) Heatmap showing the percentage of reads from each subfamily age '
    'group. Ancient L1M/L1ME/L1MC families dominate in all cell lines. '
    'HepG2 has elevated L1PA4-8 (23.4%). '
    '(B) Stacked bar view of the same data. Color gradient from red (youngest) '
    'to blue (most ancient). MCF7-EV and HEYA8 show elevated young L1 fractions.'
)

# --- Results text: Hotspot ---
pdf.add_page()
pdf.section_title('3. Expression Concentration and Hotspot Loci')

pdf.body_text(
    'L1 expression was not uniformly distributed across loci but concentrated '
    'at a small number of hotspots. The degree of concentration varied '
    'substantially across cell lines (Figure 3A): HepG2 showed the highest '
    'concentration (Gini coefficient = 0.581, top 5 loci = 20.1% of reads), '
    'followed by MCF7 (Gini = 0.550, top 5 = 15.0%). In contrast, SHSY5Y '
    '(Gini = 0.268) and HeLa-Ars (0.299) showed more dispersed expression.'
)

pdf.body_text(
    'We identified both cell-type specific and ubiquitous hotspot loci '
    '(Figure 3B-C). Among the top 30 hotspots ranked by total read count, '
    '8 were cell-type specific (>80% of reads from a single cell line), '
    '3 were semi-specific, and 19 were shared across multiple cell lines. '
    'The top hotspot, L1PA7_dup11216, was an intergenic L1PA7 element with '
    '1,336 reads, of which 94% originated from HepG2. Other cell-type '
    'specific hotspots included L1MC4_dup9840 and L1ME2_dup572 (both MCF7-'
    'specific) and L1MA8_dup8413 (H9-specific, 100%).'
)

pdf.body_text(
    'The most broadly expressed locus was HAL1_dup20999, detected in all '
    '11 cell lines (397 total reads), with no single cell line contributing '
    'more than 25%. Notably, all top 30 hotspots belonged to ancient L1 '
    'subfamilies (L1MC, L1ME, L1PA7, HAL1); no young L1 locus (L1HS or '
    'L1PA1-3) appeared among the top hotspots, consistent with mapping bias '
    'reducing the detectability of individual young L1 loci.'
)

# Hotspot table
pdf.ln(2)
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'Table 2. Top 15 L1 hotspot loci.', new_x='LMARGIN', new_y='NEXT')
pdf.ln(1)

hotspots = pd.read_csv(OUTDIR / 'part1_hotspots_top30.tsv', sep='\t')

pdf.set_font('Helvetica', 'B', 7)
hs_cols = [6, 42, 16, 16, 18, 10, 24, 14, 18]
hs_heads = ['#', 'Locus', 'Reads', 'Subfamily', 'Context', 'CL', 'Dominant CL', 'Dom%', 'Specificity']
for w, h in zip(hs_cols, hs_heads):
    pdf.cell(w, 5, h, border=1, align='C')
pdf.ln()

pdf.set_font('Helvetica', '', 6.5)
for _, r in hotspots.head(15).iterrows():
    vals = [
        str(int(r['rank'])),
        r['locus'],
        f"{int(r['total_reads']):,}",
        r['subfamily'],
        r['context'],
        str(int(r['n_celllines'])),
        r['dominant_cl'],
        f"{r['dominant_pct']:.0f}%",
        r['specificity'],
    ]
    for w, v in zip(hs_cols, vals):
        pdf.cell(w, 4, v, border=1, align='C')
    pdf.ln()

# --- Figure 3 ---
pdf.add_page()
pdf.section_title('Figure 3. Hotspot Analysis')
pdf.image(str(FIGDIR / 'fig3_hotspot.png'), x=5, w=200)
pdf.ln(2)
pdf.figure_caption(
    'Figure 3. L1 expression hotspot analysis. '
    '(A) Gini coefficient measuring expression concentration per cell line. '
    'Higher values indicate more reads from fewer loci. '
    '(B) Top 20 hotspot loci colored by cell-type specificity. '
    '(C) Heatmap showing the cell line distribution of top 15 hotspots. '
    'Cell-type specific loci (e.g., L1PA7_dup11216 in HepG2, L1MC4_dup9840 '
    'in MCF7) contrast with ubiquitous loci (HAL1_dup20999, L1ME4a_dup18292). '
    '(D) Loci sharing: 74% of expressed loci are detected in only one cell line.'
)

# --- Results text: 3' bias and sparsity ---
pdf.add_page()
pdf.section_title('4. Genomic Context and Technical Considerations')

pdf.body_text(
    "Consistent with the 3' polyadenylation-dependent library preparation of "
    "DRS, 90% of L1 reads mapped within 1 kb of the annotated 3' end of their "
    "respective L1 elements (Figure 4A). This 3' bias was similar for both "
    "young and ancient L1 subfamilies (90.7% vs 90.2%), indicating that the "
    "coverage pattern reflects the DRS protocol rather than biological "
    "differences in transcript structure."
)

pdf.body_text(
    'The majority of L1 reads (60.6%) originated from intronic L1 elements, '
    'with 39.4% from intergenic loci. This ratio was consistent across cell '
    'lines (range 54-69% intronic) and showed no significant difference '
    'between young and ancient L1 (chi-square p = 0.063), suggesting that '
    'transcriptional readthrough from host genes contributes substantially '
    'to the detected L1 RNA pool.'
)

pdf.body_text(
    'Of the 1,001,410 annotated L1 loci in the human genome, only 12,125 '
    '(1.2%) were detected across all cell lines combined. Young L1 elements '
    'showed a 2.4-fold higher per-locus detection rate compared to ancient '
    '(MCF7: 2.36% vs 0.45%), consistent with their higher transcriptional '
    'activity per element despite lower absolute counts.'
)

pdf.body_text(
    'A notable feature of L1 expression was its sparsity: 64% of expressed '
    'loci were singletons (supported by only 1 read), and 74% were detected '
    'in only one cell line (Figure 4B). Only 125 loci (1.0%) were expressed '
    'in 8 or more cell lines, and 15 loci were ubiquitously expressed across '
    'all 11 cell lines. This extreme sparsity should be considered when '
    'interpreting locus-level analyses.'
)

# --- Figure 4 ---
pdf.ln(3)
pdf.section_title('Figure 4. Technical Characteristics')
pdf.image(str(FIGDIR / 'fig4_bias_sparsity.png'), x=10, w=185)
pdf.ln(2)
pdf.figure_caption(
    "Figure 4. Technical characteristics of L1 DRS data. "
    "(A) Distribution of read mapping positions relative to the annotated 3' "
    "end of each L1 element. Dashed line marks 1 kb. 90% of reads fall within "
    "1 kb, reflecting the DRS 3' bias. "
    "(B) Distribution of reads per locus. 64% of expressed loci have only 1 "
    "read (singletons). Only 611 loci have 10 or more supporting reads."
)

# --- Summary page ---
pdf.add_page()
pdf.section_title('Summary of Key Findings')

findings = [
    ('Scale', '38,544 L1 reads across 11 cell lines mapping to 12,125 unique '
     'loci from 128 subfamilies. L1 read counts vary >10-fold across cell lines.'),
    ('Ancient dominance', 'Ancient L1 subfamilies (L1MC/L1ME/L1M) comprise '
     '63-79% of reads in all cell lines. L1HS (youngest, retrotransposition-'
     'competent) is <2.8%.'),
    ('Concentration', 'Expression is concentrated at hotspot loci (Gini 0.27-'
     '0.58). Both cell-type specific (L1PA7_dup11216 in HepG2, L1MC4_dup9840 '
     'in MCF7) and ubiquitous (HAL1_dup20999) hotspots exist.'),
    ('Cell-type specificity', '74% of expressed loci are unique to one cell '
     'line. All top 30 hotspots are ancient L1. Only 15 loci are expressed '
     'in all 11 cell lines.'),
    ("3' bias", '90% of reads map within 1 kb of the L1 3\' end, reflecting '
     'DRS library preparation. Young L1 reads are longer (median 1,283 vs '
     '900 bp for ancient).'),
    ('Genomic context', '61% of L1 reads come from intronic elements, '
     'suggesting substantial contribution from host gene readthrough. '
     'No significant difference between young and ancient L1.'),
    ('Detection rate', 'Only 1.2% of genomic L1 loci (12,125/1,001,410) are '
     'detected. Young L1 has 2.4x higher per-locus detection rate. 64% of '
     'expressed loci are singletons (1 read).'),
    ('MCF7-EV', 'Extracellular vesicle RNA shows enrichment of young L1 '
     '(18.2% vs 7.3% in MCF7 cellular RNA), consistent with selective '
     'packaging of young L1 transcripts.'),
]

for i, (title, text) in enumerate(findings, 1):
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.cell(0, 6, f'{i}. {title}', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 4.5, text)
    pdf.ln(2)

# Save PDF
out_path = OUTDIR / 'Part1_L1_Expression_Landscape.pdf'
pdf.output(str(out_path))
print(f"\nPDF saved: {out_path}")
print("Done!")
