#!/usr/bin/env python3
"""
Part 1 PDF v2: L1 Expression Landscape.
Revised with feedback:
  - Normalized L1 fraction (L1 reads / total mapped reads)
  - Subfamily → supplementary, body text = young% only
  - No HeLa-Ars interpretation (save for Part 4)
  - Hotspot ranking by n_celllines + within-CL fraction
  - 3' bias → supplementary
  - Intronic ≠ readthrough (neutral language)
  - Locus analyses: ≥2 reads filter
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from fpdf import FPDF

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
FIGDIR = OUTDIR / 'pdf_figures_v2'
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

# Group → individual sample mapping for normalization
GROUP_TO_SAMPLES = {
    'A549_4': ['A549_4_1'], 'A549_5': ['A549_5_1'], 'A549_6': ['A549_6_1'],
    'H9_2': ['H9_2_1','H9_2_2'], 'H9_3': ['H9_3_1','H9_3_2'], 'H9_4': ['H9_4_1','H9_4_2'],
    'Hct116_3': ['Hct116_3_1','Hct116_3_4'], 'Hct116_4': ['Hct116_4_3'],
    'HeLa_1': ['HeLa_1_1'], 'HeLa_2': ['HeLa_2_1'], 'HeLa_3': ['HeLa_3_1'],
    'HeLa-Ars_1': ['HeLa-Ars_1_1'], 'HeLa-Ars_2': ['HeLa-Ars_2_1'], 'HeLa-Ars_3': ['HeLa-Ars_3_1'],
    'HepG2_5': ['HepG2_5_1','HepG2_5_2'], 'HepG2_6': ['HepG2_6_1'],
    'HEYA8_1': ['HEYA8_1_1','HEYA8_1_2'], 'HEYA8_2': ['HEYA8_2_1','HEYA8_2_2'], 'HEYA8_3': ['HEYA8_3_1'],
    'K562_4': ['K562_4_1'], 'K562_5': ['K562_5_1'], 'K562_6': ['K562_6_1'],
    'MCF7_2': ['MCF7_2_3'], 'MCF7_3': ['MCF7_3_1'], 'MCF7_4': ['MCF7_4_1'],
    'MCF7-EV_1': ['MCF7-EV_1_1'],
    'SHSY5Y_1': ['SHSY5Y_1_1'], 'SHSY5Y_2': ['SHSY5Y_2_1'], 'SHSY5Y_3': ['SHSY5Y_3_1'],
}

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")
# Library-level read counts for normalization
lib_counts = pd.read_csv(PROJECT / 'results/l1_read_counts.tsv', sep='\t')
lib_counts = lib_counts.set_index('sample')

# Get total mapped reads per group
group_total_reads = {}
for g, samples in GROUP_TO_SAMPLES.items():
    total = sum(lib_counts.loc[s, 'mapped_reads'] for s in samples if s in lib_counts.index)
    group_total_reads[g] = total

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
        df['total_mapped'] = group_total_reads.get(g, np.nan)
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)
print(f"Total: {len(data):,} reads across {data['cell_line'].nunique()} cell lines")

# Reference L1
ref_l1 = pd.read_csv(PROJECT / 'reference/L1_TE_L1_family.bed', sep='\t',
                      header=None, names=['chr','start','end','subfamily','locus_id'])
ref_l1['l1_age'] = ref_l1['subfamily'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
n_ref = len(ref_l1)
n_ref_young = (ref_l1['l1_age'] == 'young').sum()
n_ref_ancient = (ref_l1['l1_age'] == 'ancient').sum()

# Cell line order by L1 fraction (normalized)
cl_stats = {}
for cl in CELL_LINES:
    d = data[data['cell_line'] == cl]
    if len(d) == 0:
        continue
    total_mapped = d.groupby('group')['total_mapped'].first().sum()
    l1_reads = len(d)  # QC=PASS L1 reads
    l1_fraction = l1_reads / total_mapped * 1000 if total_mapped > 0 else 0  # per mille
    young_pct = (d['l1_age'] == 'young').mean() * 100
    cl_stats[cl] = {
        'l1_reads': l1_reads, 'total_mapped': total_mapped,
        'l1_fraction_permille': l1_fraction,
        'young_pct': young_pct,
        'n_reps': d['group'].nunique(),
        'rdlen_median': d['read_length'].median(),
    }

cl_order = sorted(cl_stats.keys(), key=lambda x: cl_stats[x]['l1_fraction_permille'], reverse=True)

# ≥2 reads loci filter
loci_counts = data.groupby('transcript_id').size()
multi_loci = set(loci_counts[loci_counts >= 2].index)
data_multi = data[data['transcript_id'].isin(multi_loci)].copy()
n_multi_loci = len(multi_loci)
n_singleton = (loci_counts == 1).sum()

print(f"  Multi-read loci (≥2): {n_multi_loci:,}")
print(f"  Singleton loci (=1): {n_singleton:,}")

# =========================================================================
# Figure 1: L1 fraction + basic stats (3 panels)
# =========================================================================
print("Generating Figure 1...")
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.subplots_adjust(wspace=0.35)

# 1A: L1 fraction (normalized) per cell line, with replicate dots
ax = axes[0]
fracs = [cl_stats[cl]['l1_fraction_permille'] for cl in cl_order]
ax.bar(range(len(cl_order)), fracs, color='#4C72B0', alpha=0.8, edgecolor='white')
# Per-replicate dots
for i, cl in enumerate(cl_order):
    d = data[data['cell_line'] == cl]
    for g in d['group'].unique():
        gd = d[d['group'] == g]
        total_m = group_total_reads.get(g, 1)
        rep_frac = len(gd) / total_m * 1000
        ax.scatter(i, rep_frac, color='#C44E52', s=25, zorder=5, alpha=0.7)
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('L1 reads per 1,000 mapped reads')
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Normalized L1 expression', fontsize=10, loc='center')

# 1B: Young L1 fraction
ax = axes[1]
young_pcts = [cl_stats[cl]['young_pct'] for cl in cl_order]
colors_young = ['#C44E52' if y > 10 else '#4C72B0' for y in young_pcts]
ax.bar(range(len(cl_order)), young_pcts, color=colors_young, alpha=0.8)
ax.axhline(np.mean(young_pcts), color='gray', ls='--', lw=0.8)
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Young L1 fraction (%)')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Young L1 (L1HS + L1PA1-3)', fontsize=10, loc='center')

# 1C: Read length (young vs ancient, summary)
ax = axes[2]
x = np.arange(len(cl_order))
w = 0.35
anc_rl = [data[(data['cell_line']==cl) & (data['l1_age']=='ancient')]['read_length'].median()
          for cl in cl_order]
yng_rl = []
for cl in cl_order:
    d = data[(data['cell_line']==cl) & (data['l1_age']=='young')]
    yng_rl.append(d['read_length'].median() if len(d) > 10 else np.nan)
ax.bar(x - w/2, anc_rl, w, label='Ancient', color='#4C72B0', alpha=0.8)
ax.bar(x + w/2, yng_rl, w, label='Young', color='#C44E52', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Median read length (bp)')
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Read length by L1 age', fontsize=10, loc='center')
ax.legend(fontsize=8)

fig1.savefig(FIGDIR / 'fig1_overview.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  Figure 1 saved")

# =========================================================================
# Figure 2: Hotspot analysis (≥2 reads loci only, 4 panels)
# =========================================================================
print("Generating Figure 2...")

def gini_coefficient(values):
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0: return 0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))

# Concentration per CL (multi-read loci only)
conc = {}
for cl in cl_order:
    d = data_multi[data_multi['cell_line'] == cl]
    if len(d) == 0:
        conc[cl] = {'gini': 0, 'top5_pct': 0, 'n_loci': 0}
        continue
    lc = d.groupby('transcript_id').size().sort_values(ascending=False)
    n = len(d)
    conc[cl] = {
        'gini': gini_coefficient(lc.values),
        'top5_pct': lc.head(5).sum() / n * 100,
        'n_loci': len(lc),
    }

# Hotspots: rank by n_celllines (breadth), then by read count
loci_multi = data_multi.groupby('transcript_id').agg(
    total_reads=('read_id', 'count'),
    gene_id=('gene_id', 'first'),
    TE_group=('TE_group', 'first'),
    n_celllines=('cell_line', 'nunique'),
).sort_values(['n_celllines', 'total_reads'], ascending=[False, False])

# Per-locus: dominant CL and fraction within that CL
def get_hotspot_info(locus):
    d = data_multi[data_multi['transcript_id'] == locus]
    cl_counts = d['cell_line'].value_counts()
    dom_cl = cl_counts.index[0]
    dom_pct = cl_counts.iloc[0] / len(d) * 100
    return dom_cl, dom_pct

# Also prepare top loci per CL (within-CL fraction)
top_per_cl = {}
for cl in cl_order:
    d = data_multi[data_multi['cell_line'] == cl]
    if len(d) == 0:
        continue
    lc = d.groupby('transcript_id').size().sort_values(ascending=False)
    top_per_cl[cl] = lc.head(5)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.subplots_adjust(hspace=0.4, wspace=0.35)

# 2A: Gini coefficient
ax = axes2[0, 0]
conc_sorted = sorted(conc.items(), key=lambda x: x[1]['gini'])
cl_names = [c[0] for c in conc_sorted]
gini_vals = [c[1]['gini'] for c in conc_sorted]
colors_g = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cl_names)))
ax.barh(range(len(cl_names)), gini_vals, color=colors_g, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(cl_names)))
ax.set_yticklabels(cl_names, fontsize=9)
ax.set_xlabel('Gini coefficient')
for i, v in enumerate(gini_vals):
    n = conc[cl_names[i]]['n_loci']
    ax.text(v + 0.008, i, f'{v:.3f} ({n:,} loci)', va='center', fontsize=7)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Expression concentration (loci with >=2 reads)', fontsize=9, loc='center')
ax.set_xlim(0, 0.75)

# 2B: Top 20 hotspots (by n_celllines, then reads)
ax = axes2[0, 1]
top20 = loci_multi.head(20)
top20_loci = top20.index.tolist()
top20_spec = []
for locus in top20_loci:
    _, dom_pct = get_hotspot_info(locus)
    top20_spec.append('specific' if dom_pct > 80 else ('semi' if dom_pct > 50 else 'shared'))

spec_colors = {'specific': '#C44E52', 'semi': '#E8845C', 'shared': '#4C72B0'}
bar_colors = [spec_colors[s] for s in top20_spec]
reads_vals = [top20.loc[l, 'total_reads'] for l in top20_loci]
n_cls = [top20.loc[l, 'n_celllines'] for l in top20_loci]
labels = [f"{l} ({top20.loc[l,'gene_id']}) [{top20.loc[l,'n_celllines']}CL]"
          for l in top20_loci]
ax.barh(range(len(top20_loci)), reads_vals, color=bar_colors, edgecolor='gray', linewidth=0.3)
ax.set_yticks(range(len(top20_loci)))
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel('Total reads')
ax.invert_yaxis()
legend_el = [Patch(fc='#C44E52', label='CL-specific (>80%)'),
             Patch(fc='#E8845C', label='Semi-specific'),
             Patch(fc='#4C72B0', label='Shared (<50%)')]
ax.legend(handles=legend_el, fontsize=7, loc='lower right')
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Top 20 loci (ranked by #cell lines)', fontsize=9, loc='center')

# 2C: Hotspot CL composition heatmap (top 15)
ax = axes2[1, 0]
top15 = top20_loci[:15]
hm_data = []
for locus in top15:
    d = data_multi[data_multi['transcript_id'] == locus]
    total = len(d)
    row = [(d['cell_line'] == cl).sum() / total * 100 for cl in cl_order]
    hm_data.append(row)
hm_arr = np.array(hm_data)
im = ax.imshow(hm_arr, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
ax.set_xticks(range(len(cl_order)))
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f"{l} ({loci_multi.loc[l,'gene_id']})" for l in top15], fontsize=7)
for i in range(hm_arr.shape[0]):
    for j in range(hm_arr.shape[1]):
        val = hm_arr[i, j]
        if val >= 5:
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax, label='% of reads', shrink=0.8)
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Top 15 loci: cell line distribution', fontsize=9, loc='center')

# 2D: Loci sharing (multi-read only)
ax = axes2[1, 1]
loci_cl_count = data_multi.groupby('transcript_id')['cell_line'].nunique()
sharing_dist = loci_cl_count.value_counts().sort_index()
total_multi_loci = len(loci_cl_count)
ax.bar(sharing_dist.index, sharing_dist.values, color='#4C72B0', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of cell lines')
ax.set_ylabel('Number of loci')
n_unique_multi = sharing_dist.get(1, 0)
ax.text(0.97, 0.95,
    f'Multi-read loci: {total_multi_loci:,}\n'
    f'1 CL only: {n_unique_multi:,} ({n_unique_multi/total_multi_loci*100:.0f}%)\n'
    f'Shared (>=2 CL): {total_multi_loci-n_unique_multi:,} ({(total_multi_loci-n_unique_multi)/total_multi_loci*100:.0f}%)\n'
    f'Ubiquitous (>=8 CL): {(loci_cl_count>=8).sum()}',
    transform=ax.transAxes, fontsize=8, ha='right', va='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Loci sharing (>=2 reads per locus)', fontsize=9, loc='center')

fig2.savefig(FIGDIR / 'fig2_hotspot.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  Figure 2 saved")

# =========================================================================
# Supplementary Figure: Subfamily heatmap + 3' bias + gene context
# =========================================================================
print("Generating Supplementary Figure...")

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
age_groups_order = ['L1HS', 'L1PA1-3', 'L1PA4-8', 'L1PA9+', 'L1PB',
                    'L1MC', 'L1ME', 'L1M (other)', 'HAL1', 'Other']
age_cl = data.groupby(['cell_line', 'subfam_group']).size().unstack(fill_value=0)
age_pct = age_cl.div(age_cl.sum(axis=1), axis=0) * 100
for col in age_groups_order:
    if col not in age_pct.columns:
        age_pct[col] = 0.0
age_pct = age_pct[age_groups_order]

figS, axesS = plt.subplots(2, 2, figsize=(14, 11))
figS.subplots_adjust(hspace=0.4, wspace=0.35)

# SA: Subfamily heatmap
ax = axesS[0, 0]
hm = age_pct.loc[cl_order]
im = ax.imshow(hm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=40)
ax.set_xticks(range(len(age_groups_order)))
ax.set_xticklabels(age_groups_order, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(cl_order)))
ax.set_yticklabels(cl_order, fontsize=9)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        val = hm.iloc[i, j]
        if val >= 0.5:
            color = 'white' if val > 30 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5, color=color)
plt.colorbar(im, ax=ax, label='% of reads', shrink=0.8)
ax.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax.set_title('L1 subfamily composition', fontsize=10, loc='center')

# SB: 3' bias
ax = axesS[0, 1]
for age, color, label in [('ancient', '#4C72B0', 'Ancient'), ('young', '#C44E52', 'Young')]:
    d = data[(data['l1_age'] == age) & (data['dist_to_3prime'].notna())]
    ax.hist(d['dist_to_3prime'].clip(upper=5000), bins=60, alpha=0.5,
            color=color, label=f'{label} (n={len(d):,})', density=True)
ax.axvline(x=1000, color='gray', ls='--', lw=0.8, alpha=0.7)
ax.text(1050, ax.get_ylim()[1]*0.85, '1 kb', fontsize=8, color='gray')
med_anc = data[(data['l1_age']=='ancient') & data['dist_to_3prime'].notna()]['dist_to_3prime'].median()
med_yng = data[(data['l1_age']=='young') & data['dist_to_3prime'].notna()]['dist_to_3prime'].median()
ax.text(0.97, 0.95, f'Median dist:\nAncient={med_anc:.0f} bp\nYoung={med_yng:.0f} bp',
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel("Distance to 3' end of L1 element (bp)")
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax.set_title("3' coverage bias", fontsize=10, loc='center')

# SC: Gene context
ax = axesS[1, 0]
intronic_pct = []
intergenic_pct = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    intronic_pct.append((d['TE_group'] == 'intronic').mean() * 100)
    intergenic_pct.append((d['TE_group'] == 'intergenic').mean() * 100)
xi = np.arange(len(cl_order))
ax.bar(xi, intronic_pct, label='Intronic', color='#4C72B0', alpha=0.8)
ax.bar(xi, intergenic_pct, bottom=intronic_pct, label='Intergenic', color='#DD8452', alpha=0.8)
ax.set_xticks(xi)
ax.set_xticklabels(cl_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Fraction (%)')
ax.legend(fontsize=8)
ax.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Genomic context', fontsize=10, loc='center')

# SD: Loci coverage distribution (including singletons for context)
ax = axesS[1, 1]
counts_clipped = loci_counts.clip(upper=20)
ax.hist(counts_clipped, bins=range(1, 22), color='#4C72B0', alpha=0.8, edgecolor='white')
ax.axvline(x=1.5, color='red', ls='--', lw=1.0, alpha=0.7)
ax.text(2.0, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 5000,
        'Locus filter: >=2 reads', fontsize=8, color='red')
ax.set_xlabel('Reads per locus')
ax.set_ylabel('Number of loci')
ax.set_xticks(range(1, 21))
ax.set_xticklabels([str(i) if i < 20 else '20+' for i in range(1, 21)], fontsize=7)
ax.text(0.97, 0.95,
    f'Total loci: {len(loci_counts):,}\n'
    f'Singletons: {n_singleton:,} ({n_singleton/len(loci_counts)*100:.0f}%)\n'
    f'Multi-read: {n_multi_loci:,} ({n_multi_loci/len(loci_counts)*100:.0f}%)',
    transform=ax.transAxes, fontsize=8, ha='right', va='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax.set_title('Loci read coverage', fontsize=10, loc='center')

figS.savefig(FIGDIR / 'figS1_supplementary.png', dpi=300, bbox_inches='tight')
plt.close(figS)
print("  Supplementary Figure saved")

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
pdf.cell(0, 6, '11 cell lines | 29 replicates | 38,544 L1 reads', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(20)
pdf.set_font('Helvetica', 'I', 9)
pdf.cell(0, 6, 'IsoTENT L1 Project - Results Draft v2', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 6, 'Generated: 2026-02-09', align='C', new_x='LMARGIN', new_y='NEXT')

# --- Section 1: L1 Expression ---
pdf.add_page()
pdf.section_title('1. L1 Expression Across Cell Lines')

pdf.body_text(
    'We profiled L1 transposon expression using Oxford Nanopore Direct RNA '
    'Sequencing (DRS) across 11 human cell lines (29 replicates total). '
    'After quality filtering, we obtained 38,544 L1 reads mapping to 12,125 '
    'unique L1 loci from 128 subfamilies (Table 1).'
)

pdf.body_text(
    'To account for differences in sequencing depth, we normalized L1 read '
    'counts by total mapped reads per library. The normalized L1 expression '
    f'fraction ranged from {min(cl_stats[cl]["l1_fraction_permille"] for cl in cl_order):.2f} '
    f'to {max(cl_stats[cl]["l1_fraction_permille"] for cl in cl_order):.2f} '
    'per thousand mapped reads (Figure 1A), indicating '
    'that cell-line differences in L1 expression persist after controlling for '
    'library size.'
)

pdf.body_text(
    'Ancient L1 elements (L1MC, L1ME, L1M families) constituted the majority '
    'of detected reads (mean 93%), while young, retrotransposition-competent '
    'elements (L1HS, L1PA1-3) comprised 3-18% (Figure 1B). MCF7-EV '
    '(extracellular vesicle RNA) and HEYA8 showed notably elevated young L1 '
    'fractions (18.2% and 13.2%, respectively). Young L1 reads were '
    'consistently longer than ancient L1 reads within each cell line '
    '(Figure 1C), reflecting the longer intact open reading frames of '
    'young elements. Note that cross-cell-line variation in median read '
    'length may partly reflect library preparation batch effects rather '
    'than biological differences.'
)

# Table 1
pdf.ln(1)
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'Table 1. L1 expression summary across 11 cell lines.', new_x='LMARGIN', new_y='NEXT')
pdf.ln(1)

pdf.set_font('Helvetica', 'B', 7)
col_w = [20, 9, 15, 22, 14, 14, 17, 25]
headers = ['Cell Line', 'Reps', 'L1 reads', 'L1/1000 mapped', 'Loci', 'Young%', 'rdLen med', 'rdLen IQR']
for w, h in zip(col_w, headers):
    pdf.cell(w, 5, h, border=1, align='C')
pdf.ln()

summary = pd.read_csv(OUTDIR / 'part1_summary_table.tsv', sep='\t')
pdf.set_font('Helvetica', '', 7)
for cl in cl_order:
    r = summary[summary['cell_line'] == cl].iloc[0]
    frac = cl_stats[cl]['l1_fraction_permille']
    vals = [
        cl,
        str(int(r['n_reps'])),
        f"{int(r['n_reads']):,}",
        f"{frac:.2f}",
        f"{int(r['n_loci']):,}",
        f"{r['young_pct']:.1f}%",
        f"{r['rdlen_median']:,.0f}",
        f"{r['rdlen_Q25']:,.0f}-{r['rdlen_Q75']:,.0f}",
    ]
    for w, v in zip(col_w, vals):
        pdf.cell(w, 4.5, v, border=1, align='C')
    pdf.ln()

# --- Figure 1 ---
pdf.add_page()
pdf.section_title('Figure 1')
pdf.image(str(FIGDIR / 'fig1_overview.png'), x=5, w=200)
pdf.ln(2)
pdf.figure_caption(
    'Figure 1. L1 expression across cell lines. '
    '(A) Normalized L1 expression: L1 reads per 1,000 total mapped reads. '
    'Red dots show individual replicates. '
    '(B) Fraction of young L1 elements (L1HS + L1PA1-3) per cell line. '
    'MCF7-EV and HEYA8 are highlighted (>10%). '
    '(C) Median read length for ancient (blue) and young (red) L1 elements. '
    'Young L1 reads are consistently longer within each cell line. '
    'Cross-cell-line variation in read length may reflect library batch effects.'
)

# --- Section 2: Hotspot ---
pdf.add_page()
pdf.section_title('2. L1 Expression Hotspot Loci')

pdf.body_text(
    f'To identify robust L1 expression loci, we excluded singleton loci '
    f'(supported by only 1 read, n={n_singleton:,}; {n_singleton/len(loci_counts)*100:.0f}% '
    f'of all expressed loci) to minimize mapping errors and stochastic noise. '
    f'The remaining {n_multi_loci:,} multi-read loci were used for all '
    f'locus-level analyses.'
)

pdf.body_text(
    'L1 expression was concentrated at a small number of hotspot loci, with '
    'the degree of concentration varying across cell lines (Figure 2A). '
    f'HepG2 showed the highest concentration (Gini = {conc["HepG2"]["gini"]:.3f}), '
    f'while SHSY5Y showed the most dispersed expression (Gini = {conc["SHSY5Y"]["gini"]:.3f}).'
)

pdf.body_text(
    'We ranked hotspot loci by the number of cell lines in which they were '
    'detected, to avoid bias toward cell lines with higher sequencing depth '
    '(Figure 2B). The most broadly expressed locus was HAL1_dup20999 '
    '(detected in all 11 cell lines), followed by L1MEd_dup7990 and '
    'L1ME4a_dup18292. These ubiquitous loci showed dispersed expression '
    'across cell lines (no single cell line >30%), suggesting they represent '
    'constitutively active L1 elements.'
)

pdf.body_text(
    'In contrast, several hotspots were highly cell-type specific '
    '(Figure 2C): L1MC4_dup9840 and L1ME2_dup572 were almost exclusively '
    'expressed in MCF7 (>90%), L1MA8_dup8413 was unique to H9 (100%), and '
    'L1PA7_dup11216 was dominated by HepG2 (94%). All top hotspots belonged '
    'to ancient L1 subfamilies; no young L1 locus appeared among the top-'
    'ranked loci, likely reflecting the mapping challenge for these highly '
    'similar elements.'
)

pdf.body_text(
    f'Among the {n_multi_loci:,} multi-read loci, '
    f'{n_unique_multi:,} ({n_unique_multi/total_multi_loci*100:.0f}%) were detected '
    f'in only one cell line, while {total_multi_loci-n_unique_multi:,} '
    f'({(total_multi_loci-n_unique_multi)/total_multi_loci*100:.0f}%) were shared '
    f'across two or more cell lines (Figure 2D). Only {(loci_cl_count>=8).sum()} '
    f'loci were ubiquitously expressed across 8 or more cell lines.'
)

# Hotspot table
pdf.ln(1)
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'Table 2. Top 15 L1 hotspot loci (ranked by number of cell lines).', new_x='LMARGIN', new_y='NEXT')
pdf.ln(1)

pdf.set_font('Helvetica', 'B', 6.5)
hs_cols = [5, 38, 13, 14, 16, 8, 20, 12, 14, 18]
hs_heads = ['#', 'Locus', 'Reads', 'Subfamily', 'Context', 'CL', 'Dominant CL', 'Dom%', 'N CL', 'Type']
for w, h in zip(hs_cols, hs_heads):
    pdf.cell(w, 5, h, border=1, align='C')
pdf.ln()

pdf.set_font('Helvetica', '', 6)
for i, locus in enumerate(loci_multi.head(15).index):
    row = loci_multi.loc[locus]
    dom_cl, dom_pct = get_hotspot_info(locus)
    spec = 'specific' if dom_pct > 80 else ('semi' if dom_pct > 50 else 'shared')
    vals = [
        str(i+1), locus, f"{int(row['total_reads']):,}", row['gene_id'],
        row['TE_group'], str(int(row['n_celllines'])), dom_cl,
        f"{dom_pct:.0f}%", str(int(row['n_celllines'])), spec,
    ]
    for w, v in zip(hs_cols, vals):
        pdf.cell(w, 4, v, border=1, align='C')
    pdf.ln()

# --- Figure 2 ---
pdf.add_page()
pdf.section_title('Figure 2')
pdf.image(str(FIGDIR / 'fig2_hotspot.png'), x=5, w=200)
pdf.ln(2)
pdf.figure_caption(
    'Figure 2. L1 expression hotspot analysis (loci with >=2 reads). '
    '(A) Gini coefficient measuring expression concentration. '
    '(B) Top 20 loci ranked by number of cell lines detected; colored by '
    'cell-type specificity. '
    '(C) Heatmap of top 15 loci showing read distribution across cell lines. '
    '(D) Loci sharing: number of cell lines in which each multi-read locus '
    'is detected.'
)

# --- Section 3: Dimensionality Reduction ---
pdf.add_page()
pdf.section_title('3. Cell-Line-Specific L1 Loci Expression Patterns')

# Load dim reduction stats
dimreduc_stats = pd.read_csv(OUTDIR / 'part1_dimreduc_stats.tsv', sep='\t')
dr = dimreduc_stats.set_index('metric')['value']

pdf.body_text(
    'To assess whether L1 expression profiles are cell-line-specific, we '
    'constructed a binary loci x sample matrix (presence/absence of each L1 '
    f'locus per replicate) and applied dimensionality reduction. To control for '
    f'sequencing depth differences (range: 325-8,765 reads per replicate), we '
    f'subsampled each replicate to {int(dr["subsample_depth"])} reads (the minimum '
    f'observed), yielding {int(dr["n_loci_binary"]):,} loci detected in >= 2 samples.'
)

pdf.body_text(
    f'PCA on the subsampled binary matrix showed that replicates from the same '
    f'cell line tended to cluster together (Figure 3A). Critically, PCA structure '
    f'was not driven by sequencing depth: the correlation between original read count '
    f'and PC1 was non-significant (Spearman r = {dr["depth_vs_PC1_spearman_r"]:.3f}, '
    f'p = {dr["depth_vs_PC1_spearman_p"]:.3f}; Figure 3B). '
    f'Individual PCs each explained ~6% of variance (Figure 3C), characteristic of '
    f'high-dimensional sparse binary data where signal is distributed across many axes.'
)

pdf.body_text(
    f'UMAP with Jaccard distance revealed cell-line-specific clustering, '
    f'with replicates grouping by cell line rather than by sequencing batch '
    f'(Figure 3D). Quantitatively, within-cell-line Jaccard similarity was '
    f'{dr["within_CL_jaccard_mean"]:.3f} vs {dr["between_CL_jaccard_mean"]:.3f} '
    f'between cell lines - a {dr["ratio_mean"]:.2f}-fold enrichment '
    f'(95% CI: [{dr["ratio_ci_lo"]:.2f}, {dr["ratio_ci_hi"]:.2f}]; '
    f'{int(dr["bootstrap_n"])}x bootstrap, Figure 3E). '
    'This demonstrates that L1 loci expression patterns carry a genuine '
    'cell-type-specific signature, independent of sequencing depth.'
)

# --- Figure 3 ---
pdf.add_page()
pdf.section_title('Figure 3')
pdf.image(str(FIGDIR / 'fig_dimreduc.png'), x=3, w=204)
pdf.ln(2)
pdf.figure_caption(
    'Figure 3. Cell-line-specific L1 loci expression patterns. '
    f'All analyses use depth-matched subsampling ({int(dr["subsample_depth"])} reads/replicate). '
    '(A) PCA of binary loci presence/absence. Spider lines connect replicates '
    'to their cell-line centroid. '
    '(B) Depth independence: no significant correlation between original L1 '
    f'read count and PC1 (Spearman r = {dr["depth_vs_PC1_spearman_r"]:.3f}, '
    f'p = {dr["depth_vs_PC1_spearman_p"]:.2f}). '
    '(C) Scree plot: variance distributed across many PCs (sparse data). '
    '(D) UMAP (Jaccard distance) with sample labels. '
    '(E) Sample-sample Jaccard similarity heatmap (hierarchical clustering). '
    f'Within-cell-line similarity is {dr["ratio_mean"]:.1f}x higher than '
    'between-cell-line (bootstrap 95% CI in inset).'
)

# --- Section 4: Detection rate (renumbered) ---
pdf.add_page()
pdf.section_title('4. L1 Detection Rate')

ref_young_set = set(ref_l1[ref_l1['l1_age']=='young']['locus_id'])
ref_anc_set = set(ref_l1[ref_l1['l1_age']=='ancient']['locus_id'])
all_expr = set(data['transcript_id'].unique())
all_det = all_expr & set(ref_l1['locus_id'])

pdf.body_text(
    f'The human genome contains {n_ref:,} annotated L1 elements, of which '
    f'{n_ref_young:,} ({n_ref_young/n_ref*100:.1f}%) belong to young subfamilies '
    f'(L1HS, L1PA1-3) and {n_ref_ancient:,} ({n_ref_ancient/n_ref*100:.1f}%) are '
    f'ancient. Across all 11 cell lines, we detected expression from {len(all_det):,} '
    f'loci ({len(all_det)/n_ref*100:.2f}% of genomic L1 elements).'
)

# Compute mean detection rates
young_det_rates = []
anc_det_rates = []
for cl in cl_order:
    d = data[data['cell_line'] == cl]
    yng_expr = set(d[d['l1_age']=='young']['transcript_id'].unique())
    anc_expr = set(d[d['l1_age']=='ancient']['transcript_id'].unique())
    young_det_rates.append(len(yng_expr & ref_young_set) / n_ref_young * 100)
    anc_det_rates.append(len(anc_expr & ref_anc_set) / n_ref_ancient * 100)

pdf.body_text(
    f'Young L1 elements showed a higher per-locus detection rate than ancient '
    f'elements (mean {np.mean(young_det_rates):.2f}% vs {np.mean(anc_det_rates):.2f}% '
    f'per cell line), consistent with higher transcriptional activity per element '
    f'despite their lower absolute numbers. The highest young L1 detection was in '
    f'MCF7 ({max(young_det_rates):.2f}%).'
)

# --- Supplementary ---
pdf.add_page()
pdf.section_title('Supplementary Figure S1')
pdf.image(str(FIGDIR / 'figS1_supplementary.png'), x=5, w=200)
pdf.ln(2)
pdf.figure_caption(
    'Supplementary Figure S1. Technical characterization. '
    '(A) L1 subfamily composition by cell line. Ancient L1M/L1ME/L1MC families '
    'dominate. Note that this distribution largely reflects the genomic abundance '
    'of each subfamily and should be interpreted with caution given DRS 3\' bias '
    'and multi-mapping limitations. '
    "(B) 3' coverage bias: most reads map near the 3' end of L1 elements "
    "(median distance: ancient=149 bp, young=29 bp), consistent with DRS "
    "library preparation from the poly(A) tail. "
    '(C) Genomic context: 61% of L1 reads originate from intronic elements. '
    '(D) Per-locus read coverage distribution. 64% of expressed loci are '
    'singletons (1 read), excluded from locus-level analyses.'
)

# --- Summary ---
pdf.add_page()
pdf.section_title('Summary of Key Findings')

findings = [
    ('Normalized L1 expression',
     'After normalizing for sequencing depth, L1 expression fraction varies '
     'across cell lines, with differences persisting after library size correction.'),
    ('Young L1 enrichment',
     'Young, retrotransposition-competent L1 elements (L1HS, L1PA1-3) comprise '
     '3-18% of L1 reads. MCF7-EV (extracellular vesicles) shows the highest '
     'young L1 fraction (18.2%), consistent with selective EV packaging.'),
    ('Hotspot concentration',
     'L1 expression is concentrated at hotspot loci. HepG2 and MCF7 show the '
     'highest concentration (Gini >0.5), while other cell lines show more '
     'dispersed expression.'),
    ('Cell-type specific and ubiquitous loci',
     'Both cell-type specific hotspots (L1PA7_dup11216 in HepG2, L1MC4_dup9840 '
     'in MCF7) and ubiquitous loci (HAL1_dup20999 in all 11 cell lines) coexist. '
     'All top hotspots are ancient L1 subfamilies.'),
    ('Cell-line-specific loci patterns',
     f'Depth-matched subsampling confirms that L1 loci expression patterns are '
     f'cell-line-specific (within-CL Jaccard {dr["within_CL_jaccard_mean"]:.3f} vs '
     f'between-CL {dr["between_CL_jaccard_mean"]:.3f}; {dr["ratio_mean"]:.1f}x enrichment, '
     f'95% CI [{dr["ratio_ci_lo"]:.1f}-{dr["ratio_ci_hi"]:.1f}]). This is not '
     'driven by sequencing depth (PC1-depth correlation ns).'),
    ('Detection rate',
     f'Only {len(all_det)/n_ref*100:.1f}% of genomic L1 loci are detected. '
     'Young L1 has higher per-locus detection rate, reflecting greater '
     'transcriptional activity per element.'),
    ('Locus sharing',
     f'{n_unique_multi:,}/{total_multi_loci:,} multi-read loci '
     f'({n_unique_multi/total_multi_loci*100:.0f}%) are unique to one cell line. '
     'Only 125 loci are expressed in 8+ cell lines.'),
]

for i, (title, text) in enumerate(findings, 1):
    pdf.set_font('Helvetica', 'B', 9.5)
    pdf.cell(0, 6, f'{i}. {title}', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 4.5, text)
    pdf.ln(2)

# Save
out_path = OUTDIR / 'Part1_L1_Expression_Landscape_v2.pdf'
pdf.output(str(out_path))
print(f"\nPDF saved: {out_path}")
print("Done!")
