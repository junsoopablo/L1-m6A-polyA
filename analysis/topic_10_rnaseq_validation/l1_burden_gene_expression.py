#!/usr/bin/env python3
"""
Genome-wide analysis: Does intronic L1 burden correlate with gene expression
change under arsenite?

Approach: RepeatMasker L1 annotation (all ~1M elements) intersected with
GENCODE gene bodies → L1 burden per gene → RNA-seq DESeq2 log2FC correlation.
No DRS dependency → n = thousands of genes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC = PROJECT / 'analysis/01_exploration/topic_10_rnaseq_validation'
OUTPUT = TOPIC / 'l1_burden_analysis'
OUTPUT.mkdir(exist_ok=True)

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ====================================================================
# 1. Load L1 elements from RepeatMasker TE GTF
# ====================================================================
print("Loading L1 elements from RepeatMasker...")
te_gtf = PROJECT / 'reference/hg38_rmsk_TE.gtf'

l1_elements = []
with open(te_gtf) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        chrom, _, feat, start, end, _, strand, _, attrs = parts
        if 'family_id "L1"' not in attrs:
            continue
        # Extract subfamily
        sf = attrs.split('gene_id "')[1].split('"')[0]
        l1_elements.append({
            'chrom': chrom, 'start': int(start), 'end': int(end),
            'strand': strand, 'subfamily': sf,
            'length': int(end) - int(start),
            'age': 'young' if sf in YOUNG_L1 else 'ancient'
        })

l1_df = pd.DataFrame(l1_elements)
print(f"Total L1 elements: {len(l1_df):,}")
print(f"  Young: {(l1_df['age'] == 'young').sum():,}")
print(f"  Ancient: {(l1_df['age'] == 'ancient').sum():,}")

# ====================================================================
# 2. Load GENCODE gene annotations
# ====================================================================
print("\nLoading GENCODE gene annotations...")
gtf_file = PROJECT / 'reference/Human.gtf'

genes = []
gene_name_map = {}
with open(gtf_file) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attrs = parts[8]
        if 'gene_id' not in attrs or 'gene_name' not in attrs:
            continue
        gid = attrs.split('gene_id "')[1].split('"')[0]
        gname = attrs.split('gene_name "')[1].split('"')[0]
        gtype = attrs.split('gene_type "')[1].split('"')[0] if 'gene_type' in attrs else ''
        genes.append({
            'chrom': parts[0], 'start': int(parts[3]), 'end': int(parts[4]),
            'strand': parts[6], 'gene_id': gid, 'gene_name': gname,
            'gene_type': gtype, 'gene_length': int(parts[4]) - int(parts[3])
        })
        gene_name_map[gid] = gname

gene_df = pd.DataFrame(genes)
# Focus on protein-coding genes
pc_genes = gene_df[gene_df['gene_type'] == 'protein_coding'].copy()
print(f"Protein-coding genes: {len(pc_genes):,}")

# ====================================================================
# 3. Count L1 elements per gene (sorted sweep-line, fast)
# ====================================================================
print("\nCounting intronic L1 per gene (sorted sweep-line)...")

# Build sorted arrays per chromosome for L1
l1_by_chrom = {}
for chrom, grp in l1_df.groupby('chrom'):
    starts = grp['start'].values
    ends = grp['end'].values
    ages = (grp['age'] == 'ancient').values
    order = np.argsort(starts)
    l1_by_chrom[chrom] = (starts[order], ends[order], ages[order])

gene_l1_counts = []
for _, gene in pc_genes.iterrows():
    chrom = gene['chrom']
    g_start, g_end = gene['start'], gene['end']

    total_l1 = 0
    ancient_l1 = 0
    young_l1 = 0
    total_l1_bp = 0
    ancient_l1_bp = 0

    if chrom in l1_by_chrom:
        starts, ends, is_ancient = l1_by_chrom[chrom]
        # Binary search for first L1 that could overlap
        lo = np.searchsorted(ends, g_start, side='right')
        hi = np.searchsorted(starts, g_end, side='left')
        for i in range(lo, hi):
            overlap_bp = min(g_end, ends[i]) - max(g_start, starts[i])
            if overlap_bp > 0:
                total_l1 += 1
                total_l1_bp += overlap_bp
                if is_ancient[i]:
                    ancient_l1 += 1
                    ancient_l1_bp += overlap_bp
                else:
                    young_l1 += 1

    gene_l1_counts.append({
        'gene_id': gene['gene_id'],
        'gene_name': gene['gene_name'],
        'gene_length': gene['gene_length'],
        'total_l1': total_l1,
        'ancient_l1': ancient_l1,
        'young_l1': young_l1,
        'total_l1_bp': total_l1_bp,
        'ancient_l1_bp': ancient_l1_bp,
        'l1_density': total_l1_bp / gene['gene_length'] if gene['gene_length'] > 0 else 0,
        'ancient_l1_density': ancient_l1_bp / gene['gene_length'] if gene['gene_length'] > 0 else 0,
    })

burden = pd.DataFrame(gene_l1_counts)
print(f"Genes with ≥1 L1: {(burden['total_l1'] > 0).sum():,} / {len(burden):,}")
print(f"Genes with ≥1 ancient L1: {(burden['ancient_l1'] > 0).sum():,}")
print(f"Median L1 count per gene: {burden['total_l1'].median():.0f}")
print(f"Median L1 density: {burden['l1_density'].median():.3f}")

# ====================================================================
# 4. Load RNA-seq DESeq2 results
# ====================================================================
print("\nLoading RNA-seq DESeq2 results...")
deseq = pd.read_csv(TOPIC / 'tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
                     sep='\t', index_col=0)
deseq_genes = deseq[~deseq.index.str.contains(':')].copy()
deseq_genes = deseq_genes.dropna(subset=['log2FoldChange', 'padj'])
print(f"DESeq2 genes: {len(deseq_genes):,}")

# ====================================================================
# 5. Merge
# ====================================================================
burden = burden.set_index('gene_id')
merged = burden.join(deseq_genes[['baseMean', 'log2FoldChange', 'padj']], how='inner')
merged = merged[merged['baseMean'] > 50].copy()  # Filter low-expression
merged['abs_log2FC'] = merged['log2FoldChange'].abs()
print(f"\nMerged genes (baseMean > 50): {len(merged):,}")
print(f"  With ≥1 L1: {(merged['total_l1'] > 0).sum():,}")
print(f"  With ≥1 ancient L1: {(merged['ancient_l1'] > 0).sum():,}")

# ====================================================================
# 6. Correlation analyses
# ====================================================================
print("\n" + "=" * 70)
print("GENOME-WIDE: L1 burden ↔ Gene expression change under arsenite")
print("=" * 70)

# 6a. L1 count vs log2FC
has_l1 = merged[merged['total_l1'] > 0]
no_l1 = merged[merged['total_l1'] == 0]

print(f"\n--- Genes with vs without L1 ---")
print(f"  With L1 (n={len(has_l1):,}): median log2FC = {has_l1['log2FoldChange'].median():.4f}")
print(f"  No L1   (n={len(no_l1):,}): median log2FC = {no_l1['log2FoldChange'].median():.4f}")
u, p = stats.mannwhitneyu(has_l1['log2FoldChange'], no_l1['log2FoldChange'])
print(f"  Mann-Whitney P = {p:.2e}")

# |log2FC|
print(f"  With L1: median |log2FC| = {has_l1['abs_log2FC'].median():.4f}")
print(f"  No L1:   median |log2FC| = {no_l1['abs_log2FC'].median():.4f}")
u2, p2 = stats.mannwhitneyu(has_l1['abs_log2FC'], no_l1['abs_log2FC'])
print(f"  |log2FC| Mann-Whitney P = {p2:.2e}")

# 6b. Ancient L1 burden quantiles
print(f"\n--- Ancient L1 count quintiles ---")
has_ancient = merged[merged['ancient_l1'] > 0].copy()
has_ancient['burden_q'] = pd.qcut(has_ancient['ancient_l1'], 5, labels=['Q1(few)', 'Q2', 'Q3', 'Q4', 'Q5(many)'],
                                   duplicates='drop')
for q in has_ancient['burden_q'].cat.categories:
    sub = has_ancient[has_ancient['burden_q'] == q]
    print(f"  {q} (n={len(sub):,}, L1 count={sub['ancient_l1'].median():.0f}): "
          f"log2FC={sub['log2FoldChange'].median():.4f}, |log2FC|={sub['abs_log2FC'].median():.4f}")

r_count, p_count = stats.spearmanr(has_ancient['ancient_l1'], has_ancient['log2FoldChange'])
print(f"  Ancient L1 count ↔ log2FC: ρ = {r_count:.4f}, P = {p_count:.2e}")
r_count_abs, p_count_abs = stats.spearmanr(has_ancient['ancient_l1'], has_ancient['abs_log2FC'])
print(f"  Ancient L1 count ↔ |log2FC|: ρ = {r_count_abs:.4f}, P = {p_count_abs:.2e}")

# 6c. L1 density
print(f"\n--- L1 density (L1 bp / gene length) ---")
has_l1_d = merged[merged['l1_density'] > 0].copy()
r_dens, p_dens = stats.spearmanr(has_l1_d['l1_density'], has_l1_d['log2FoldChange'])
print(f"  L1 density ↔ log2FC: ρ = {r_dens:.4f}, P = {p_dens:.2e}")
r_dens_abs, p_dens_abs = stats.spearmanr(has_l1_d['l1_density'], has_l1_d['abs_log2FC'])
print(f"  L1 density ↔ |log2FC|: ρ = {r_dens_abs:.4f}, P = {p_dens_abs:.2e}")

# Quintile by density
has_l1_d['dens_q'] = pd.qcut(has_l1_d['l1_density'], 5,
                               labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                               duplicates='drop')
for q in has_l1_d['dens_q'].cat.categories:
    sub = has_l1_d[has_l1_d['dens_q'] == q]
    print(f"  {q} (n={len(sub):,}, density={sub['l1_density'].median():.3f}): "
          f"log2FC={sub['log2FoldChange'].median():.4f}, |log2FC|={sub['abs_log2FC'].median():.4f}")

# 6d. Gene length confound check
print(f"\n--- Gene length confound ---")
r_len, p_len = stats.spearmanr(merged['gene_length'], merged['log2FoldChange'])
print(f"  Gene length ↔ log2FC: ρ = {r_len:.4f}, P = {p_len:.2e}")
r_len_l1, p_len_l1 = stats.spearmanr(merged['gene_length'], merged['total_l1'])
print(f"  Gene length ↔ L1 count: ρ = {r_len_l1:.4f}, P = {p_len_l1:.2e}")

# Partial correlation: L1 density after controlling for gene length
# Use rank-based partial correlation
from scipy.stats import rankdata
def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y controlling for z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom == 0:
        return 0, 1
    r_partial = (r_xy - r_xz * r_yz) / denom
    n = len(x)
    t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2))
    from scipy.stats import t as t_dist
    p_val = 2 * t_dist.sf(abs(t_stat), n - 3)
    return r_partial, p_val

rp, pp = partial_spearman(has_l1_d['l1_density'].values,
                           has_l1_d['log2FoldChange'].values,
                           has_l1_d['gene_length'].values)
print(f"  Partial ρ (L1 density ↔ log2FC | gene length): {rp:.4f}, P = {pp:.2e}")

rp_abs, pp_abs = partial_spearman(has_l1_d['l1_density'].values,
                                    has_l1_d['abs_log2FC'].values,
                                    has_l1_d['gene_length'].values)
print(f"  Partial ρ (L1 density ↔ |log2FC| | gene length): {rp_abs:.4f}, P = {pp_abs:.2e}")

# 6e. Downregulated vs upregulated genes
print(f"\n--- L1 burden in up vs down-regulated genes ---")
sig = merged[merged['padj'] < 0.05]
sig_up = sig[sig['log2FoldChange'] > 0]
sig_down = sig[sig['log2FoldChange'] < 0]
print(f"  Sig UP genes (n={len(sig_up):,}): median L1 count = {sig_up['total_l1'].median():.0f}, "
      f"L1 density = {sig_up['l1_density'].median():.3f}")
print(f"  Sig DOWN genes (n={len(sig_down):,}): median L1 count = {sig_down['total_l1'].median():.0f}, "
      f"L1 density = {sig_down['l1_density'].median():.3f}")
u3, p3 = stats.mannwhitneyu(sig_up['l1_density'], sig_down['l1_density'])
print(f"  L1 density UP vs DOWN: MW P = {p3:.2e}")
u4, p4 = stats.mannwhitneyu(sig_up['ancient_l1'], sig_down['ancient_l1'])
print(f"  Ancient L1 count UP vs DOWN: MW P = {p4:.2e}")

# Fraction with L1
print(f"  Sig UP: {(sig_up['total_l1'] > 0).mean():.1%} have L1")
print(f"  Sig DOWN: {(sig_down['total_l1'] > 0).mean():.1%} have L1")

# ====================================================================
# 7. Figure
# ====================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.38, wspace=0.35)

# (a) With L1 vs without L1 — log2FC distribution
ax = axes[0, 0]
bins = np.linspace(-3, 3, 80)
ax.hist(no_l1['log2FoldChange'], bins=bins, alpha=0.5, color='#BDBDBD',
        density=True, label=f'No L1 (n={len(no_l1):,})')
ax.hist(has_l1['log2FoldChange'], bins=bins, alpha=0.5, color='#8D6E63',
        density=True, label=f'With L1 (n={len(has_l1):,})')
ax.axvline(no_l1['log2FoldChange'].median(), color='#757575', linestyle='--', linewidth=1.5)
ax.axvline(has_l1['log2FoldChange'].median(), color='#5D4037', linestyle='--', linewidth=1.5)
ax.legend(fontsize=8)
ax.set_xlabel('log₂FC (Ars/UN)', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.set_title('Expression change distribution', fontsize=10, fontweight='bold')
ax.text(0.05, 0.95, f'MW P = {p:.2e}', transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (b) Ancient L1 count quintile boxplot
ax = axes[0, 1]
q_order = has_ancient['burden_q'].cat.categories.tolist()
bp_data = [has_ancient[has_ancient['burden_q'] == q]['log2FoldChange'].values for q in q_order]
bp = ax.boxplot(bp_data, tick_labels=[f'{q}\n(n={len(has_ancient[has_ancient["burden_q"]==q])})' for q in q_order],
                patch_artist=True, widths=0.6, showfliers=False)
cmap = plt.cm.YlOrBr(np.linspace(0.2, 0.8, len(q_order)))
for patch, c in zip(bp['boxes'], cmap):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_ylabel('log₂FC (Ars/UN)', fontsize=9)
ax.set_title(f'Expression by ancient L1 count\n(ρ={r_count:.3f}, P={p_count:.1e})', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (c) L1 density quintile boxplot
ax = axes[0, 2]
dq_order = has_l1_d['dens_q'].cat.categories.tolist()
bp_data2 = [has_l1_d[has_l1_d['dens_q'] == q]['log2FoldChange'].values for q in dq_order]
bp2 = ax.boxplot(bp_data2, tick_labels=[f'{q}\n(n={len(has_l1_d[has_l1_d["dens_q"]==q])})' for q in dq_order],
                 patch_artist=True, widths=0.6, showfliers=False)
cmap2 = plt.cm.RdPu(np.linspace(0.2, 0.8, len(dq_order)))
for patch, c in zip(bp2['boxes'], cmap2):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_ylabel('log₂FC (Ars/UN)', fontsize=9)
ax.set_title(f'Expression by L1 density\n(ρ={r_dens:.3f}, P={p_dens:.1e})', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (d) L1 density scatter
ax = axes[1, 0]
ax.scatter(has_l1_d['l1_density'], has_l1_d['log2FoldChange'],
           c='#8D6E63', alpha=0.05, s=5, edgecolors='none')
# Binned means
bins_d = pd.qcut(has_l1_d['l1_density'], 20)
binned = has_l1_d.groupby(bins_d)['log2FoldChange'].agg(['mean', 'median', 'count'])
x_mid = [(b.left + b.right) / 2 for b in binned.index]
ax.plot(x_mid, binned['median'], 'r-o', linewidth=2, markersize=4, zorder=5)
ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
ax.set_xlabel('L1 density (L1 bp / gene bp)', fontsize=9)
ax.set_ylabel('log₂FC (Ars/UN)', fontsize=9)
ax.set_title(f'L1 density vs expression\n(partial ρ={rp:.3f}, P={pp:.1e})',
             fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (e) |log2FC| by L1 density quintile
ax = axes[1, 1]
bp_data3 = [has_l1_d[has_l1_d['dens_q'] == q]['abs_log2FC'].values for q in dq_order]
bp3 = ax.boxplot(bp_data3, tick_labels=[q for q in dq_order],
                 patch_artist=True, widths=0.6, showfliers=False)
for patch, c in zip(bp3['boxes'], cmap2):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel('|log₂FC|', fontsize=9)
ax.set_title(f'Expression variability by L1 density\n(ρ={r_dens_abs:.3f}, P={p_dens_abs:.1e})',
             fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold')

# (f) L1 density in sig UP vs DOWN genes
ax = axes[1, 2]
bp_data4 = [sig_down['l1_density'].values, sig_up['l1_density'].values]
bp4 = ax.boxplot(bp_data4, tick_labels=[f'DOWN\n(n={len(sig_down):,})', f'UP\n(n={len(sig_up):,})'],
                 patch_artist=True, widths=0.5, showfliers=False)
bp4['boxes'][0].set_facecolor('#1565C0')
bp4['boxes'][0].set_alpha(0.7)
bp4['boxes'][1].set_facecolor('#C62828')
bp4['boxes'][1].set_alpha(0.7)
ax.set_ylabel('L1 density', fontsize=9)
ax.set_title(f'L1 density: UP vs DOWN genes\n(MW P={p3:.1e})', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.05, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold')

fig.savefig(OUTPUT / 'l1_burden_gene_expression.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT / 'l1_burden_gene_expression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUTPUT / 'l1_burden_gene_expression.pdf'}")

# Save data
merged.to_csv(OUTPUT / 'gene_l1_burden_rnaseq.tsv', sep='\t')
print(f"Data saved: {OUTPUT / 'gene_l1_burden_rnaseq.tsv'}")
