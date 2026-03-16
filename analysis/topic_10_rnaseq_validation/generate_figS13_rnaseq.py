#!/usr/bin/env python3
"""
Supplementary Figure S13: Illumina RNA-seq validation of L1 expression under arsenite.
GSE278916 (Liu et al. Cell 2025): HeLa ribo-depleted RNA-seq.

3 panels (no DRS comparison — different library preps are not directly comparable):
(a) Gene-normalized L1 fold change by age group (featureCounts)
(b) TEtranscripts DESeq2 L1 subfamily volcano
(c) Stress response gene validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
BAM_DIR = Path('/scratch1/junsoopablo/GSE278916_alignment')
OUTPUT = TOPIC / 'rnaseq_validation_results'
FIG_DIR = PROJECT / 'manuscript/figures'

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

def classify_l1_age(sf):
    if sf in YOUNG_L1:
        return 'young'
    if sf.startswith('L1') or sf.startswith('HAL1'):
        return 'ancient'
    return 'other'

samples = ['HeLa_UN_rep1', 'HeLa_UN_rep2', 'HeLa_SA_rep1', 'HeLa_SA_rep2']
un_cols = [s for s in samples if 'UN' in s]
sa_cols = [s for s in samples if 'SA' in s]

# ====================================================================
# 1. Load STAR stats + gene size factors
# ====================================================================
star_stats = {}
for sample in samples:
    log_file = BAM_DIR / f'{sample}_Log.final.out'
    with open(log_file) as f:
        d = {}
        for line in f:
            if '|' in line:
                k, v = line.split('|', 1)
                d[k.strip()] = v.strip()
    star_stats[sample] = {
        'total_mapped': int(d['Uniquely mapped reads number']) + int(d['Number of reads mapped to multiple loci']),
    }

gene_counts = {}
for s in samples:
    rpg = pd.read_csv(BAM_DIR / f'{s}_ReadsPerGene.out.tab', sep='\t', header=None,
                      names=['gene', 'unstranded', 'sense', 'antisense']).iloc[4:]
    gene_counts[s] = rpg.set_index('gene')['unstranded']

gdf = pd.DataFrame(gene_counts)
gdf_nz = gdf[(gdf > 0).all(axis=1)]
geo_mean = gdf_nz.apply(np.log).mean(axis=1).apply(np.exp)
size_factors = gdf_nz.div(geo_mean, axis=0).median()

# ====================================================================
# 2. Load featureCounts
# ====================================================================
fc = pd.read_csv(OUTPUT / 'featurecounts_L1.txt', sep='\t', comment='#')
bam_cols = [c for c in fc.columns if 'Aligned' in c]
col_map = {col: name for col in bam_cols for name in samples if name in col}
fc = fc.rename(columns=col_map)
fc['age'] = fc['Geneid'].apply(classify_l1_age)
fc = fc[fc['age'].isin(['young', 'ancient'])]

for s in samples:
    fc[f'{s}_norm'] = fc[s] / size_factors[s]

norm_results = {}
for age in ['young', 'ancient', 'all']:
    sub = fc if age == 'all' else fc[fc['age'] == age]
    fc_per_rep = []
    for un_s, sa_s in zip(un_cols, sa_cols):
        un_total = sub[f'{un_s}_norm'].sum()
        sa_total = sub[f'{sa_s}_norm'].sum()
        fc_per_rep.append(sa_total / un_total if un_total > 0 else np.nan)
    norm_results[age] = fc_per_rep

# ====================================================================
# 3. Load TEtranscripts DESeq2 results
# ====================================================================
te_results = pd.read_csv(TOPIC / 'tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
                         sep='\t', index_col=0)
te_l1 = te_results[te_results.index.str.contains(':L1:LINE')].copy()
te_l1 = te_l1.dropna(subset=['padj'])
te_l1['subfamily'] = te_l1.index.str.split(':').str[0]
te_l1['age'] = te_l1['subfamily'].apply(classify_l1_age)
te_l1['sig'] = te_l1['padj'] < 0.05
te_l1['neg_log10_padj'] = -np.log10(te_l1['padj'].clip(lower=1e-300))

n_down = ((te_l1['sig']) & (te_l1['log2FoldChange'] < 0)).sum()
n_up = ((te_l1['sig']) & (te_l1['log2FoldChange'] > 0)).sum()

# ====================================================================
# 4. Load gene name mapping for stress genes
# ====================================================================
gtf_file = PROJECT / 'reference/Human.gtf'
gene_name_map = {}
with open(gtf_file) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attrs = parts[8]
        if 'gene_id' in attrs and 'gene_name' in attrs:
            gid = attrs.split('gene_id "')[1].split('"')[0]
            gname = attrs.split('gene_name "')[1].split('"')[0]
            gene_name_map[gid] = gname

name_to_id = {}
for gid, gname in gene_name_map.items():
    name_to_id.setdefault(gname, []).append(gid)

gene_norm = gdf.div(size_factors)

# ====================================================================
# 5. Create 3-panel figure (1 row × 3 columns)
# ====================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
plt.subplots_adjust(wspace=0.38)

# ---------- Panel (a): Gene-normalized FC ----------
ax = axes[0]

# Young shown as grey/hatched to indicate unreliable quantification
colors_age = {'young': '#BDBDBD', 'ancient': '#8D6E63', 'all': '#607D8B'}
labels_age = {'young': 'Young L1\n(unreliable)', 'ancient': 'Ancient L1', 'all': 'All L1'}
hatches = {'young': '///', 'ancient': '', 'all': ''}
alphas = {'young': 0.5, 'ancient': 0.85, 'all': 0.85}
for i, age in enumerate(['young', 'ancient', 'all']):
    vals = norm_results[age]
    mean_val = np.mean(vals)
    ax.bar(i, mean_val, 0.55, color=colors_age[age], alpha=alphas[age],
           edgecolor='black', linewidth=0.5, hatch=hatches[age])
    ax.scatter([i]*len(vals), vals, color='black', s=30, zorder=5,
              edgecolors='white', linewidths=0.5)
    ax.text(i, mean_val - 0.015, f'{mean_val:.3f}', ha='center', va='top',
            fontsize=8, fontweight='bold',
            color='#999999' if age == 'young' else 'black')

ax.axhline(1.0, color='grey', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([labels_age[a] for a in ['young', 'ancient', 'all']], fontsize=8)
ax.set_ylabel('Fold change (Arsenite / Untreated)', fontsize=9)
ax.set_ylim(0.83, 1.05)
ax.set_title('Gene-normalized L1\nexpression change', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Note about young L1
ax.annotate('>97% seq. identity\n→ multi-mapping', xy=(0, 0.89), fontsize=6,
            ha='center', va='bottom', color='#777777', style='italic')
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

# ---------- Panel (b): TEtranscripts volcano ----------
ax = axes[1]

# Non-significant
ns = te_l1[~te_l1['sig']]
ax.scatter(ns['log2FoldChange'], ns['neg_log10_padj'],
           c='#BDBDBD', s=20, alpha=0.5, edgecolors='none', label='ns')

# Significant ancient
sig_anc = te_l1[(te_l1['sig']) & (te_l1['age'] == 'ancient')]
ax.scatter(sig_anc['log2FoldChange'], sig_anc['neg_log10_padj'],
           c='#8D6E63', s=30, alpha=0.7, edgecolors='white', linewidths=0.3,
           label='Ancient (sig)')

# Significant young
sig_young = te_l1[(te_l1['sig']) & (te_l1['age'] == 'young')]
ax.scatter(sig_young['log2FoldChange'], sig_young['neg_log10_padj'],
           c='#4CAF50', s=60, alpha=0.9, edgecolors='black', linewidths=0.5,
           marker='^', label='Young (sig)', zorder=6)

# Label key subfamilies: young L1 + top significant ancient only
try:
    from adjustText import adjust_text
    texts_vol = []
    # Always label young L1 and highly significant ancient (top 8 by -log10 padj)
    top_ancient = te_l1[(te_l1['sig']) & (te_l1['age'] == 'ancient')].nlargest(8, 'neg_log10_padj')
    label_sfs = set(YOUNG_L1) | set(top_ancient['subfamily'])
    for _, row in te_l1.iterrows():
        if row['subfamily'] in label_sfs:
            color = '#4CAF50' if row['age'] == 'young' else '#5D4037'
            texts_vol.append(ax.text(row['log2FoldChange'], row['neg_log10_padj'],
                                     row['subfamily'], fontsize=6, color=color))
    adjust_text(texts_vol, ax=ax, arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
                force_points=0.4, force_text=0.6, expand_points=(2.0, 2.0))
except ImportError:
    for _, row in te_l1.iterrows():
        if row['subfamily'] in YOUNG_L1 or row['neg_log10_padj'] > 15:
            ax.annotate(row['subfamily'],
                        (row['log2FoldChange'], row['neg_log10_padj']),
                        fontsize=6, ha='left', va='bottom',
                        color='#4CAF50' if row['age'] == 'young' else '#5D4037',
                        xytext=(3, 3), textcoords='offset points')

ax.axhline(-np.log10(0.05), color='red', linestyle=':', alpha=0.4, linewidth=0.8)
ax.axvline(0, color='grey', linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_xlabel('log₂(Fold Change)', fontsize=9)
ax.set_ylabel('-log₁₀(adjusted P)', fontsize=9)
ax.set_title('TEtranscripts DESeq2\nL1 subfamily DE', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.03, 0.97, f'{n_down} ↓  {n_up} ↑\n(padj < 0.05)',
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

# ---------- Panel (c): Stress response genes ----------
ax = axes[2]

stress_genes = ['HSPA6', 'HSPA1A', 'ATF3', 'HMOX1', 'DDIT3', 'BAG3', 'HSPH1', 'MT2A']
stress_fc_list = []
for gene in stress_genes:
    gids = name_to_id.get(gene, [])
    for gid in gids:
        if gid in gene_norm.index:
            un_mean = gene_norm.loc[gid, un_cols].mean()
            sa_mean = gene_norm.loc[gid, sa_cols].mean()
            if un_mean > 10:
                stress_fc_list.append({'gene': gene, 'log2FC': np.log2(sa_mean / un_mean),
                                       'FC': sa_mean / un_mean})
            break

sdf = pd.DataFrame(stress_fc_list).sort_values('log2FC')
colors_bar = ['#FF5722' if x > 1 else '#42A5F5' for x in sdf['FC']]
ax.barh(range(len(sdf)), sdf['log2FC'], color=colors_bar, alpha=0.85,
        edgecolor='black', linewidth=0.5)

for i, (_, row) in enumerate(sdf.iterrows()):
    if row['FC'] > 10:
        ax.text(row['log2FC'] - 0.5, i, f'{row["FC"]:.0f}x', va='center', ha='right',
                fontsize=7, fontweight='bold', color='white')
    else:
        ax.text(row['log2FC'] + 0.2, i, f'{row["FC"]:.1f}x', va='center', ha='left',
                fontsize=7, fontweight='bold')

ax.set_yticks(range(len(sdf)))
ax.set_yticklabels(sdf['gene'], fontsize=8)
ax.set_xlabel('log₂(Arsenite / Untreated)', fontsize=9)
ax.set_title('Stress response\ngene validation', fontsize=10, fontweight='bold')
ax.axvline(0, color='grey', linestyle='--', alpha=0.3, linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.15, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

# ====================================================================
# Save
# ====================================================================
for fmt in ['pdf', 'png']:
    out = FIG_DIR / f'figS13.{fmt}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")

plt.close()

# Print summary
print(f"\n(a) featureCounts: Young {np.mean(norm_results['young']):.3f}x, "
      f"Ancient {np.mean(norm_results['ancient']):.3f}x, All {np.mean(norm_results['all']):.3f}x")
print(f"(b) TEtranscripts: {n_down} DOWN, {n_up} UP (of {te_l1['sig'].sum()} sig)")
print(f"(c) Stress: HSPA6 {sdf[sdf['gene']=='HSPA6']['FC'].values[0]:.0f}x")
