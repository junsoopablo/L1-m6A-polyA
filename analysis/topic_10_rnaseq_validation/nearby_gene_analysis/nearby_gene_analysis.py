#!/usr/bin/env python3
"""
Nearby gene analysis: Do genes near regulatory ancient L1 elements
show differential expression under arsenite stress?
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import re
import os

OUT_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/nearby_gene_analysis'

# ============================================================
# 1. Load ChromHMM-annotated L1 data
# ============================================================
print("=== Loading ChromHMM L1 data ===")
l1 = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv',
    sep='\t'
)
print(f"Total L1 reads: {len(l1)}")

# Filter: regulatory ancient L1 in HeLa
mask = (
    l1['chromhmm_group'].isin(['Enhancer', 'Promoter']) &
    (l1['l1_age'] == 'ancient') &
    (l1['cellline'] == 'HeLa')
)
reg_l1 = l1[mask].copy()
print(f"Regulatory ancient L1 reads (HeLa): {len(reg_l1)}")
print(f"  Conditions: {reg_l1['condition'].value_counts().to_dict()}")
print(f"  ChromHMM groups: {reg_l1['chromhmm_group'].value_counts().to_dict()}")

# Get unique L1 loci (deduplicate by genomic position)
# Use chr + rounded start (within 500bp) to cluster reads from same locus
reg_l1['locus_key'] = reg_l1['chr'] + ':' + (reg_l1['start'] // 500 * 500).astype(str)
loci = reg_l1.groupby('locus_key').agg(
    chr=('chr', 'first'),
    start=('start', 'min'),
    end=('end', 'max'),
    n_reads=('read_id', 'nunique'),
    mean_polya_normal=('polya_length', lambda x: reg_l1.loc[x.index[reg_l1.loc[x.index, 'condition'] == 'normal'], 'polya_length'].mean()),
    mean_polya_stress=('polya_length', lambda x: reg_l1.loc[x.index[reg_l1.loc[x.index, 'condition'] == 'stress'], 'polya_length'].mean()),
).reset_index()

# Compute poly(A) change per locus
loci['delta_polya'] = loci['mean_polya_stress'] - loci['mean_polya_normal']
loci['shortened'] = loci['delta_polya'] < 0

# Also get loci with reads in both conditions
loci_both = loci.dropna(subset=['mean_polya_normal', 'mean_polya_stress'])
print(f"\nUnique regulatory ancient L1 loci: {len(loci)}")
print(f"Loci with reads in both conditions: {len(loci_both)}")

# ============================================================
# 2. Parse GTF for gene coordinates (TSS)
# ============================================================
print("\n=== Parsing GTF for gene coordinates ===")
gtf_path = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.gtf'

genes = []
with open(gtf_path) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if fields[2] != 'gene':
            continue
        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]

        # Parse gene_id
        m = re.search(r'gene_id "([^"]+)"', fields[8])
        if not m:
            continue
        gene_id = m.group(1)

        # Parse gene_type
        m2 = re.search(r'gene_type "([^"]+)"', fields[8])
        gene_type = m2.group(1) if m2 else 'unknown'

        # Parse gene_name
        m3 = re.search(r'gene_name "([^"]+)"', fields[8])
        gene_name = m3.group(1) if m3 else gene_id

        # TSS
        tss = start if strand == '+' else end

        genes.append({
            'gene_id': gene_id,
            'gene_name': gene_name,
            'gene_type': gene_type,
            'chr': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'tss': tss,
        })

genes_df = pd.DataFrame(genes)
print(f"Total genes in GTF: {len(genes_df)}")

# ============================================================
# 3. Load DESeq2 results
# ============================================================
print("\n=== Loading DESeq2 results ===")
deseq = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
    sep='\t', index_col=0
)
# Keep only ENSG entries (genes, not TEs)
deseq = deseq[deseq.index.str.startswith('ENSG')]
print(f"DESeq2 gene results: {len(deseq)}")
print(f"  Significant (padj<0.05): {(deseq['padj'] < 0.05).sum()}")
print(f"  Mean log2FC: {deseq['log2FoldChange'].mean():.4f}")

# Strip version from gene_id for matching
deseq['gene_id_base'] = deseq.index.str.split('.').str[0]
genes_df['gene_id_base'] = genes_df['gene_id'].str.split('.').str[0]

# Merge DESeq2 with gene coordinates
gene_de = genes_df.merge(deseq[['gene_id_base', 'baseMean', 'log2FoldChange', 'lfcSE', 'pvalue', 'padj']],
                         on='gene_id_base', how='inner')
gene_de = gene_de.dropna(subset=['log2FoldChange', 'padj'])
print(f"Genes with both coordinates and DE results: {len(gene_de)}")

# ============================================================
# 4. Find nearby genes for each L1 locus
# ============================================================
print("\n=== Finding nearby genes ===")

# Build chromosome-indexed gene lookup
chr_genes = defaultdict(list)
for _, g in gene_de.iterrows():
    chr_genes[g['chr']].append(g)

def find_nearby_genes(l1_chr, l1_start, l1_end, window_kb):
    """Find genes within window_kb of an L1 element."""
    window = window_kb * 1000
    l1_mid = (l1_start + l1_end) / 2
    nearby = []
    for g in chr_genes.get(l1_chr, []):
        # Distance from L1 midpoint to gene TSS
        dist = abs(g['tss'] - l1_mid)
        if dist <= window:
            nearby.append(g['gene_id_base'])
    return nearby

# Find nearby genes for 50kb and 100kb windows
for window_kb in [50, 100]:
    nearby_genes = set()
    for _, loc in loci.iterrows():
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
        nearby_genes.update(genes_near)
    print(f"  {window_kb}kb window: {len(nearby_genes)} unique nearby genes")

# Also find nearby genes stratified by poly(A) shortening
nearby_shortened = {50: set(), 100: set()}
nearby_lengthened = {50: set(), 100: set()}

for _, loc in loci_both.iterrows():
    for wk in [50, 100]:
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], wk)
        if loc['shortened']:
            nearby_shortened[wk].update(genes_near)
        else:
            nearby_lengthened[wk].update(genes_near)

for wk in [50, 100]:
    print(f"  {wk}kb - near shortened L1: {len(nearby_shortened[wk])}, near lengthened L1: {len(nearby_lengthened[wk])}")

# ============================================================
# 5. Statistical comparisons
# ============================================================
print("\n" + "="*70)
print("=== RESULTS ===")
print("="*70)

results = []

for window_kb in [50, 100]:
    print(f"\n--- {window_kb}kb window ---")

    # Nearby gene set
    nearby_genes = set()
    for _, loc in loci.iterrows():
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
        nearby_genes.update(genes_near)

    nearby_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes)]
    bg_de = gene_de[~gene_de['gene_id_base'].isin(nearby_genes)]

    print(f"\nNearby genes (n={len(nearby_de)}) vs Background (n={len(bg_de)})")

    # 5a. log2FC distribution comparison
    nearby_lfc = nearby_de['log2FoldChange'].values
    bg_lfc = bg_de['log2FoldChange'].values

    mwu_stat, mwu_p = stats.mannwhitneyu(nearby_lfc, bg_lfc, alternative='two-sided')

    print(f"  Nearby mean log2FC: {nearby_lfc.mean():.4f} (median: {np.median(nearby_lfc):.4f})")
    print(f"  Background mean log2FC: {bg_lfc.mean():.4f} (median: {np.median(bg_lfc):.4f})")
    print(f"  Mann-Whitney U P = {mwu_p:.4e}")

    # Effect size (rank-biserial)
    n1, n2 = len(nearby_lfc), len(bg_lfc)
    r_rb = 1 - (2 * mwu_stat) / (n1 * n2)
    print(f"  Rank-biserial r = {r_rb:.4f}")

    # 5b. Proportion of significantly changed genes
    nearby_sig = (nearby_de['padj'] < 0.05).sum()
    nearby_total = len(nearby_de)
    bg_sig = (bg_de['padj'] < 0.05).sum()
    bg_total = len(bg_de)

    # Fisher's exact test
    table = [[nearby_sig, nearby_total - nearby_sig],
             [bg_sig, bg_total - bg_sig]]
    fisher_or, fisher_p = stats.fisher_exact(table)

    nearby_pct = nearby_sig / nearby_total * 100 if nearby_total > 0 else 0
    bg_pct = bg_sig / bg_total * 100 if bg_total > 0 else 0

    print(f"\n  Proportion significant (padj<0.05):")
    print(f"    Nearby: {nearby_sig}/{nearby_total} ({nearby_pct:.1f}%)")
    print(f"    Background: {bg_sig}/{bg_total} ({bg_pct:.1f}%)")
    print(f"    Fisher's OR = {fisher_or:.3f}, P = {fisher_p:.4e}")

    # Direction breakdown
    nearby_up = ((nearby_de['padj'] < 0.05) & (nearby_de['log2FoldChange'] > 0)).sum()
    nearby_down = ((nearby_de['padj'] < 0.05) & (nearby_de['log2FoldChange'] < 0)).sum()
    bg_up = ((bg_de['padj'] < 0.05) & (bg_de['log2FoldChange'] > 0)).sum()
    bg_down = ((bg_de['padj'] < 0.05) & (bg_de['log2FoldChange'] < 0)).sum()

    print(f"    Nearby UP/DOWN: {nearby_up}/{nearby_down}")
    print(f"    Background UP/DOWN: {bg_up}/{bg_down}")

    results.append({
        'window_kb': window_kb,
        'comparison': 'nearby_vs_background',
        'n_nearby': len(nearby_de),
        'n_background': len(bg_de),
        'nearby_mean_lfc': nearby_lfc.mean(),
        'bg_mean_lfc': bg_lfc.mean(),
        'nearby_median_lfc': np.median(nearby_lfc),
        'bg_median_lfc': np.median(bg_lfc),
        'mwu_p': mwu_p,
        'rank_biserial_r': r_rb,
        'nearby_sig_pct': nearby_pct,
        'bg_sig_pct': bg_pct,
        'fisher_or': fisher_or,
        'fisher_p': fisher_p,
    })

    # 5c. Stratify by L1 poly(A) shortening
    print(f"\n  --- Stratified by L1 poly(A) change ---")
    for label, gene_set in [('near_shortened_L1', nearby_shortened[window_kb]),
                             ('near_lengthened_L1', nearby_lengthened[window_kb])]:
        sub_de = gene_de[gene_de['gene_id_base'].isin(gene_set)]
        if len(sub_de) < 5:
            print(f"  {label}: too few genes ({len(sub_de)}), skipping")
            continue

        sub_lfc = sub_de['log2FoldChange'].values
        mwu2_stat, mwu2_p = stats.mannwhitneyu(sub_lfc, bg_lfc, alternative='two-sided')

        sub_sig = (sub_de['padj'] < 0.05).sum()
        sub_total = len(sub_de)
        sub_pct = sub_sig / sub_total * 100

        table2 = [[sub_sig, sub_total - sub_sig],
                   [bg_sig, bg_total - bg_sig]]
        fisher2_or, fisher2_p = stats.fisher_exact(table2)

        print(f"\n  {label} (n={len(sub_de)}):")
        print(f"    Mean log2FC: {sub_lfc.mean():.4f} (median: {np.median(sub_lfc):.4f})")
        print(f"    MWU vs background: P = {mwu2_p:.4e}")
        print(f"    Sig (padj<0.05): {sub_sig}/{sub_total} ({sub_pct:.1f}%)")
        print(f"    Fisher's OR = {fisher2_or:.3f}, P = {fisher2_p:.4e}")

        results.append({
            'window_kb': window_kb,
            'comparison': label,
            'n_nearby': len(sub_de),
            'n_background': len(bg_de),
            'nearby_mean_lfc': sub_lfc.mean(),
            'bg_mean_lfc': bg_lfc.mean(),
            'nearby_median_lfc': np.median(sub_lfc),
            'bg_median_lfc': np.median(bg_lfc),
            'mwu_p': mwu2_p,
            'rank_biserial_r': 1 - (2 * mwu2_stat) / (len(sub_lfc) * len(bg_lfc)),
            'nearby_sig_pct': sub_pct,
            'bg_sig_pct': bg_pct,
            'fisher_or': fisher2_or,
            'fisher_p': fisher2_p,
        })

    # Also compare shortened vs lengthened directly
    short_genes = nearby_shortened[window_kb]
    long_genes = nearby_lengthened[window_kb]
    short_de = gene_de[gene_de['gene_id_base'].isin(short_genes)]
    long_de = gene_de[gene_de['gene_id_base'].isin(long_genes)]

    if len(short_de) >= 5 and len(long_de) >= 5:
        mwu3_stat, mwu3_p = stats.mannwhitneyu(
            short_de['log2FoldChange'].values,
            long_de['log2FoldChange'].values,
            alternative='two-sided'
        )
        print(f"\n  Shortened vs Lengthened L1 nearby genes:")
        print(f"    Shortened (n={len(short_de)}): mean lfc = {short_de['log2FoldChange'].mean():.4f}")
        print(f"    Lengthened (n={len(long_de)}): mean lfc = {long_de['log2FoldChange'].mean():.4f}")
        print(f"    MWU P = {mwu3_p:.4e}")

# ============================================================
# 6. Permutation test (random L1 loci)
# ============================================================
print("\n" + "="*70)
print("=== PERMUTATION TEST (1000 iterations) ===")
print("="*70)

np.random.seed(42)
n_perm = 1000

for window_kb in [50, 100]:
    # Observed
    nearby_genes_obs = set()
    for _, loc in loci.iterrows():
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
        nearby_genes_obs.update(genes_near)

    obs_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes_obs)]
    obs_mean_lfc = obs_de['log2FoldChange'].mean()
    obs_abs_mean_lfc = obs_de['log2FoldChange'].abs().mean()
    obs_sig_pct = (obs_de['padj'] < 0.05).mean()

    # Permutation: randomly sample same number of genes
    n_nearby = len(nearby_genes_obs)
    perm_mean_lfc = []
    perm_abs_mean_lfc = []
    perm_sig_pct = []

    all_gene_ids = gene_de['gene_id_base'].values

    for i in range(n_perm):
        rand_idx = np.random.choice(len(all_gene_ids), size=n_nearby, replace=False)
        rand_de = gene_de.iloc[rand_idx]
        perm_mean_lfc.append(rand_de['log2FoldChange'].mean())
        perm_abs_mean_lfc.append(rand_de['log2FoldChange'].abs().mean())
        perm_sig_pct.append((rand_de['padj'] < 0.05).mean())

    perm_mean_lfc = np.array(perm_mean_lfc)
    perm_abs_mean_lfc = np.array(perm_abs_mean_lfc)
    perm_sig_pct = np.array(perm_sig_pct)

    # Two-sided p-values
    p_mean = np.mean(np.abs(perm_mean_lfc - perm_mean_lfc.mean()) >= np.abs(obs_mean_lfc - perm_mean_lfc.mean()))
    p_abs = np.mean(perm_abs_mean_lfc >= obs_abs_mean_lfc)
    p_sig = np.mean(perm_sig_pct >= obs_sig_pct)

    print(f"\n--- {window_kb}kb window (n_nearby={n_nearby}) ---")
    print(f"  Mean log2FC: observed={obs_mean_lfc:.4f}, perm mean={perm_mean_lfc.mean():.4f} +/- {perm_mean_lfc.std():.4f}, P={p_mean:.4f}")
    print(f"  Mean |log2FC|: observed={obs_abs_mean_lfc:.4f}, perm mean={perm_abs_mean_lfc.mean():.4f}, P={p_abs:.4f}")
    print(f"  Sig proportion: observed={obs_sig_pct:.4f}, perm mean={perm_sig_pct.mean():.4f}, P={p_sig:.4f}")

# ============================================================
# 7. Save results
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, 'nearby_gene_results.tsv'), sep='\t', index=False)
print(f"\nResults saved to {OUT_DIR}/nearby_gene_results.tsv")

# Also save nearby gene lists
for window_kb in [50, 100]:
    nearby_genes = set()
    for _, loc in loci.iterrows():
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
        nearby_genes.update(genes_near)

    nearby_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes)].copy()
    nearby_de.to_csv(os.path.join(OUT_DIR, f'nearby_genes_{window_kb}kb.tsv'), sep='\t', index=False)

print("\n=== CONCLUSION ===")
print("See above results. If all P-values are non-significant (>0.05),")
print("then genes near regulatory ancient L1 do NOT show differential")
print("expression patterns distinct from genome-wide background.")
