#!/usr/bin/env python3
"""
Nearby gene analysis v2: Include HeLa-Ars for poly(A) stratification.
HeLa = normal, HeLa-Ars = stress (separate cellline entries).
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

# Filter: regulatory ancient L1 in HeLa or HeLa-Ars (E117 ChromHMM applies to both)
mask = (
    l1['chromhmm_group'].isin(['Enhancer', 'Promoter']) &
    (l1['l1_age'] == 'ancient') &
    (l1['cellline'].isin(['HeLa', 'HeLa-Ars']))
)
reg_l1 = l1[mask].copy()

# Unify condition labels
reg_l1['cond'] = reg_l1['cellline'].map({'HeLa': 'normal', 'HeLa-Ars': 'stress'})
print(f"Regulatory ancient L1 reads (HeLa+HeLa-Ars): {len(reg_l1)}")
print(f"  Conditions: {reg_l1['cond'].value_counts().to_dict()}")
print(f"  ChromHMM groups: {reg_l1['chromhmm_group'].value_counts().to_dict()}")

# Cluster reads into loci (500bp bins)
reg_l1['locus_key'] = reg_l1['chr'] + ':' + (reg_l1['start'] // 500 * 500).astype(str)

# Per-locus aggregation
locus_stats = []
for lk, grp in reg_l1.groupby('locus_key'):
    normal = grp[grp['cond'] == 'normal']
    stress = grp[grp['cond'] == 'stress']
    entry = {
        'locus_key': lk,
        'chr': grp['chr'].iloc[0],
        'start': grp['start'].min(),
        'end': grp['end'].max(),
        'n_reads': len(grp),
        'n_normal': len(normal),
        'n_stress': len(stress),
        'mean_polya_normal': normal['polya_length'].mean() if len(normal) > 0 else np.nan,
        'mean_polya_stress': stress['polya_length'].mean() if len(stress) > 0 else np.nan,
    }
    locus_stats.append(entry)

loci = pd.DataFrame(locus_stats)
loci['delta_polya'] = loci['mean_polya_stress'] - loci['mean_polya_normal']
loci_both = loci.dropna(subset=['mean_polya_normal', 'mean_polya_stress'])
loci_both['shortened'] = loci_both['delta_polya'] < 0

print(f"\nUnique regulatory ancient L1 loci: {len(loci)}")
print(f"Loci with reads in both conditions: {len(loci_both)}")
if len(loci_both) > 0:
    print(f"  Shortened: {loci_both['shortened'].sum()}, Lengthened: {(~loci_both['shortened']).sum()}")
    print(f"  Mean delta poly(A): {loci_both['delta_polya'].mean():.1f} nt")

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
        m = re.search(r'gene_id "([^"]+)"', fields[8])
        if not m:
            continue
        gene_id = m.group(1)
        m2 = re.search(r'gene_type "([^"]+)"', fields[8])
        gene_type = m2.group(1) if m2 else 'unknown'
        m3 = re.search(r'gene_name "([^"]+)"', fields[8])
        gene_name = m3.group(1) if m3 else gene_id
        tss = start if strand == '+' else end
        genes.append({'gene_id': gene_id, 'gene_name': gene_name, 'gene_type': gene_type,
                      'chr': chrom, 'start': start, 'end': end, 'strand': strand, 'tss': tss})

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
deseq = deseq[deseq.index.str.startswith('ENSG')]
deseq['gene_id_base'] = deseq.index.str.split('.').str[0]
genes_df['gene_id_base'] = genes_df['gene_id'].str.split('.').str[0]

gene_de = genes_df.merge(deseq[['gene_id_base', 'baseMean', 'log2FoldChange', 'lfcSE', 'pvalue', 'padj']],
                         on='gene_id_base', how='inner')
gene_de = gene_de.dropna(subset=['log2FoldChange', 'padj'])
print(f"Genes with coordinates + DE results: {len(gene_de)}")

# ============================================================
# 4. Build spatial index for nearby gene lookup
# ============================================================
chr_genes = defaultdict(list)
for _, g in gene_de.iterrows():
    chr_genes[g['chr']].append(g)

def find_nearby_genes(l1_chr, l1_start, l1_end, window_kb):
    window = window_kb * 1000
    l1_mid = (l1_start + l1_end) / 2
    nearby = []
    for g in chr_genes.get(l1_chr, []):
        dist = abs(g['tss'] - l1_mid)
        if dist <= window:
            nearby.append(g['gene_id_base'])
    return nearby

# ============================================================
# 5. Main analysis
# ============================================================
print("\n" + "="*70)
print("=== RESULTS ===")
print("="*70)

results = []

for window_kb in [50, 100]:
    print(f"\n{'='*50}")
    print(f"--- {window_kb}kb window ---")
    print(f"{'='*50}")

    # ALL nearby genes
    nearby_genes = set()
    for _, loc in loci.iterrows():
        genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
        nearby_genes.update(genes_near)

    nearby_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes)]
    bg_de = gene_de[~gene_de['gene_id_base'].isin(nearby_genes)]

    print(f"\n[A] All nearby genes (n={len(nearby_de)}) vs Background (n={len(bg_de)})")

    nearby_lfc = nearby_de['log2FoldChange'].values
    bg_lfc = bg_de['log2FoldChange'].values
    mwu_stat, mwu_p = stats.mannwhitneyu(nearby_lfc, bg_lfc, alternative='two-sided')
    r_rb = 1 - (2 * mwu_stat) / (len(nearby_lfc) * len(bg_lfc))

    print(f"  Nearby mean log2FC: {nearby_lfc.mean():.4f} (median: {np.median(nearby_lfc):.4f})")
    print(f"  Background mean log2FC: {bg_lfc.mean():.4f} (median: {np.median(bg_lfc):.4f})")
    print(f"  Mann-Whitney U P = {mwu_p:.4e}, rank-biserial r = {r_rb:.4f}")

    # Proportion significant
    nearby_sig = (nearby_de['padj'] < 0.05).sum()
    bg_sig = (bg_de['padj'] < 0.05).sum()
    table = [[nearby_sig, len(nearby_de) - nearby_sig],
             [bg_sig, len(bg_de) - bg_sig]]
    fisher_or, fisher_p = stats.fisher_exact(table)
    nearby_pct = nearby_sig / len(nearby_de) * 100
    bg_pct = bg_sig / len(bg_de) * 100

    print(f"  Sig (padj<0.05): Nearby {nearby_sig}/{len(nearby_de)} ({nearby_pct:.1f}%) vs BG {bg_sig}/{len(bg_de)} ({bg_pct:.1f}%)")
    print(f"  Fisher's OR = {fisher_or:.3f}, P = {fisher_p:.4e}")

    # Direction
    nearby_up = ((nearby_de['padj'] < 0.05) & (nearby_de['log2FoldChange'] > 0)).sum()
    nearby_down = ((nearby_de['padj'] < 0.05) & (nearby_de['log2FoldChange'] < 0)).sum()
    bg_up = ((bg_de['padj'] < 0.05) & (bg_de['log2FoldChange'] > 0)).sum()
    bg_down = ((bg_de['padj'] < 0.05) & (bg_de['log2FoldChange'] < 0)).sum()
    print(f"  Direction - Nearby UP/DOWN: {nearby_up}/{nearby_down}, BG UP/DOWN: {bg_up}/{bg_down}")
    print(f"  Nearby UP ratio: {nearby_up/(nearby_up+nearby_down):.3f}, BG UP ratio: {bg_up/(bg_up+bg_down):.3f}")

    results.append({
        'window_kb': window_kb, 'comparison': 'nearby_vs_background',
        'n_test': len(nearby_de), 'n_background': len(bg_de),
        'test_mean_lfc': nearby_lfc.mean(), 'bg_mean_lfc': bg_lfc.mean(),
        'test_median_lfc': np.median(nearby_lfc), 'bg_median_lfc': np.median(bg_lfc),
        'mwu_p': mwu_p, 'rank_biserial_r': r_rb,
        'test_sig_pct': nearby_pct, 'bg_sig_pct': bg_pct,
        'fisher_or': fisher_or, 'fisher_p': fisher_p,
    })

    # [B] Stratified by L1 poly(A) change
    if len(loci_both) > 0:
        print(f"\n[B] Stratified by L1 poly(A) change")

        shortened_loci = loci_both[loci_both['shortened']]
        lengthened_loci = loci_both[~loci_both['shortened']]

        for label, sub_loci in [('near_shortened_L1', shortened_loci),
                                 ('near_lengthened_L1', lengthened_loci)]:
            sub_genes = set()
            for _, loc in sub_loci.iterrows():
                genes_near = find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb)
                sub_genes.update(genes_near)

            sub_de = gene_de[gene_de['gene_id_base'].isin(sub_genes)]

            if len(sub_de) < 5:
                print(f"  {label}: too few genes ({len(sub_de)}), skipping")
                continue

            sub_lfc = sub_de['log2FoldChange'].values
            mwu2_stat, mwu2_p = stats.mannwhitneyu(sub_lfc, bg_lfc, alternative='two-sided')
            r2 = 1 - (2 * mwu2_stat) / (len(sub_lfc) * len(bg_lfc))

            sub_sig = (sub_de['padj'] < 0.05).sum()
            sub_pct = sub_sig / len(sub_de) * 100
            table2 = [[sub_sig, len(sub_de) - sub_sig], [bg_sig, len(bg_de) - bg_sig]]
            fisher2_or, fisher2_p = stats.fisher_exact(table2)

            print(f"\n  {label} (n={len(sub_de)}, from {len(sub_loci)} loci):")
            print(f"    Mean log2FC: {sub_lfc.mean():.4f} (median: {np.median(sub_lfc):.4f})")
            print(f"    MWU vs BG: P = {mwu2_p:.4e}, r = {r2:.4f}")
            print(f"    Sig: {sub_sig}/{len(sub_de)} ({sub_pct:.1f}%), Fisher OR={fisher2_or:.3f}, P={fisher2_p:.4e}")

            results.append({
                'window_kb': window_kb, 'comparison': label,
                'n_test': len(sub_de), 'n_background': len(bg_de),
                'test_mean_lfc': sub_lfc.mean(), 'bg_mean_lfc': bg_lfc.mean(),
                'test_median_lfc': np.median(sub_lfc), 'bg_median_lfc': np.median(bg_lfc),
                'mwu_p': mwu2_p, 'rank_biserial_r': r2,
                'test_sig_pct': sub_pct, 'bg_sig_pct': bg_pct,
                'fisher_or': fisher2_or, 'fisher_p': fisher2_p,
            })

        # Direct shortened vs lengthened comparison
        short_genes = set()
        for _, loc in shortened_loci.iterrows():
            short_genes.update(find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb))
        long_genes = set()
        for _, loc in lengthened_loci.iterrows():
            long_genes.update(find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb))

        short_de = gene_de[gene_de['gene_id_base'].isin(short_genes)]
        long_de = gene_de[gene_de['gene_id_base'].isin(long_genes)]

        if len(short_de) >= 5 and len(long_de) >= 5:
            mwu3_stat, mwu3_p = stats.mannwhitneyu(
                short_de['log2FoldChange'].values, long_de['log2FoldChange'].values, alternative='two-sided')
            print(f"\n  Shortened vs Lengthened direct comparison:")
            print(f"    Shortened (n={len(short_de)}): mean lfc={short_de['log2FoldChange'].mean():.4f}")
            print(f"    Lengthened (n={len(long_de)}): mean lfc={long_de['log2FoldChange'].mean():.4f}")
            print(f"    MWU P = {mwu3_p:.4e}")

    # [C] Enhancer vs Promoter separately
    print(f"\n[C] Enhancer vs Promoter L1 nearby genes")
    for chrom_grp in ['Enhancer', 'Promoter']:
        sub_loci = loci.merge(
            reg_l1[reg_l1['chromhmm_group'] == chrom_grp][['locus_key']].drop_duplicates(),
            on='locus_key'
        )
        sub_genes = set()
        for _, loc in sub_loci.iterrows():
            sub_genes.update(find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb))

        sub_de = gene_de[gene_de['gene_id_base'].isin(sub_genes)]
        if len(sub_de) < 5:
            continue

        sub_lfc = sub_de['log2FoldChange'].values
        mwu_s, mwu_p2 = stats.mannwhitneyu(sub_lfc, bg_lfc, alternative='two-sided')
        sub_sig = (sub_de['padj'] < 0.05).sum()
        sub_pct = sub_sig / len(sub_de) * 100

        print(f"  {chrom_grp} (n={len(sub_de)}):")
        print(f"    Mean log2FC: {sub_lfc.mean():.4f}, MWU P={mwu_p2:.4e}")
        print(f"    Sig: {sub_sig}/{len(sub_de)} ({sub_pct:.1f}%)")

# ============================================================
# 6. Permutation test
# ============================================================
print("\n" + "="*70)
print("=== PERMUTATION TEST (1000 iterations) ===")
print("="*70)

np.random.seed(42)
n_perm = 1000

for window_kb in [50, 100]:
    nearby_genes = set()
    for _, loc in loci.iterrows():
        nearby_genes.update(find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb))

    obs_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes)]
    obs_mean_lfc = obs_de['log2FoldChange'].mean()
    obs_sig_pct = (obs_de['padj'] < 0.05).mean()
    n_nearby = len(obs_de)

    perm_mean_lfc = np.zeros(n_perm)
    perm_sig_pct = np.zeros(n_perm)

    for i in range(n_perm):
        idx = np.random.choice(len(gene_de), size=n_nearby, replace=False)
        rand_de = gene_de.iloc[idx]
        perm_mean_lfc[i] = rand_de['log2FoldChange'].mean()
        perm_sig_pct[i] = (rand_de['padj'] < 0.05).mean()

    p_mean = np.mean(np.abs(perm_mean_lfc - perm_mean_lfc.mean()) >= np.abs(obs_mean_lfc - perm_mean_lfc.mean()))
    p_sig = np.mean(perm_sig_pct >= obs_sig_pct)

    print(f"\n--- {window_kb}kb (n={n_nearby}) ---")
    print(f"  Mean log2FC: obs={obs_mean_lfc:.4f}, perm={perm_mean_lfc.mean():.4f}+/-{perm_mean_lfc.std():.4f}, P={p_mean:.4f}")
    print(f"  Sig %: obs={obs_sig_pct:.4f}, perm={perm_sig_pct.mean():.4f}+/-{perm_sig_pct.std():.4f}, P={p_sig:.4f}")
    print(f"  Obs sig% is {(obs_sig_pct - perm_sig_pct.mean()) / perm_sig_pct.std():.1f} SD above permutation mean")

# ============================================================
# 7. Save
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, 'nearby_gene_results.tsv'), sep='\t', index=False)

for window_kb in [50, 100]:
    nearby_genes = set()
    for _, loc in loci.iterrows():
        nearby_genes.update(find_nearby_genes(loc['chr'], loc['start'], loc['end'], window_kb))
    nearby_de = gene_de[gene_de['gene_id_base'].isin(nearby_genes)].copy()
    nearby_de.to_csv(os.path.join(OUT_DIR, f'nearby_genes_{window_kb}kb.tsv'), sep='\t', index=False)

print(f"\nResults saved to {OUT_DIR}/")
