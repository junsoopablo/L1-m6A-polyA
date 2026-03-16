#!/usr/bin/env python3
"""
ChromHMM subtype analysis: Split regulatory L1 host genes by Enhancer vs Promoter
and test host gene expression changes separately.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

OUTDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/chromhmm_subtype_analysis'

# ── 1. Load data ──────────────────────────────────────────────────────
# Regulatory L1 host gene RNA-seq data
rnaseq = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/'
    'topic_10_rnaseq_validation/regulatory_l1_rnaseq_tests/regulatory_l1_host_gene_rnaseq.tsv',
    sep='\t'
)

# Regulatory L1 host gene ChromHMM annotation (has pct_enhancer)
gene_annot = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/'
    'topic_08_regulatory_chromatin/regulatory_stress_response/gene_response_annotated.tsv',
    sep='\t'
)

# Full DESeq2 results (genome-wide background)
deseq2 = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/'
    'topic_10_rnaseq_validation/tetranscripts_output/HeLa_SA_vs_UN_gene_TE_analysis.txt',
    sep='\t', index_col=0
)

# Full per-read ChromHMM data (for 15-state analysis)
chromhmm_reads = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/'
    'topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv',
    sep='\t'
)

print(f"RNA-seq host gene rows: {len(rnaseq)}")
print(f"Gene annotation rows: {len(gene_annot)}")
print(f"DESeq2 genome-wide genes: {len(deseq2)}")
print(f"ChromHMM per-read rows: {len(chromhmm_reads)}")

# ── 2. Merge rnaseq with gene_annot to get pct_enhancer ──────────────
# gene_annot has host_gene + pct_enhancer; rnaseq has gene_symbol + rnaseq data
# They share host_gene column
merged = rnaseq.merge(
    gene_annot[['host_gene', 'n_enhancer', 'n_promoter', 'pct_enhancer']],
    on='host_gene', how='left'
)
# Drop rows without ChromHMM annotation or RNA-seq data
merged = merged.dropna(subset=['pct_enhancer', 'rnaseq_log2FC', 'rnaseq_padj'])
print(f"\nMerged rows with both ChromHMM + RNA-seq: {len(merged)}")
print(f"Unique host genes: {merged['host_gene'].nunique()}")
print(f"Unique gene symbols: {merged['gene_symbol'].nunique()}")

# ── 3. Classify Enhancer vs Promoter ─────────────────────────────────
merged['chromhmm_class'] = np.where(merged['pct_enhancer'] > 50, 'Enhancer', 'Promoter')
print(f"\nEnhancer genes: {(merged['chromhmm_class']=='Enhancer').sum()}")
print(f"Promoter genes: {(merged['chromhmm_class']=='Promoter').sum()}")

# ── 4. Genome-wide background ────────────────────────────────────────
bg = deseq2.dropna(subset=['log2FoldChange', 'padj'])
bg_log2fc = bg['log2FoldChange'].values
bg_sig_up = (bg['padj'] < 0.05) & (bg['log2FoldChange'] > 0)
bg_sig_down = (bg['padj'] < 0.05) & (bg['log2FoldChange'] < 0)
bg_sig = bg['padj'] < 0.05
print(f"\nGenome-wide background: {len(bg)} genes")
print(f"  Sig up: {bg_sig_up.sum()} ({bg_sig_up.mean()*100:.1f}%)")
print(f"  Sig down: {bg_sig_down.sum()} ({bg_sig_down.mean()*100:.1f}%)")

# ── 5. Per-subtype analysis ──────────────────────────────────────────
results = []
print("\n" + "="*80)
print("BINARY CLASSIFICATION: Enhancer (pct_enh>50) vs Promoter (pct_enh<=50)")
print("="*80)

for cls in ['Enhancer', 'Promoter']:
    sub = merged[merged['chromhmm_class'] == cls].copy()
    # De-duplicate by gene_symbol (take first)
    sub_dedup = sub.drop_duplicates(subset='gene_symbol')
    n = len(sub_dedup)
    if n < 3:
        print(f"\n{cls}: only {n} genes, skipping")
        continue

    fc = sub_dedup['rnaseq_log2FC'].values
    padj = sub_dedup['rnaseq_padj'].values

    # a. Mann-Whitney vs genome-wide
    mwu_stat, mwu_p = stats.mannwhitneyu(fc, bg_log2fc, alternative='two-sided')

    # b. Proportion sig changed
    n_sig = (padj < 0.05).sum()
    pct_sig = n_sig / n * 100
    n_sig_down = ((padj < 0.05) & (fc < 0)).sum()
    n_sig_up = ((padj < 0.05) & (fc > 0)).sum()

    # Fisher's exact: sig proportion vs genome-wide
    # [[sig_subset, nonsig_subset], [sig_bg, nonsig_bg]]
    fisher_table = [[n_sig, n - n_sig],
                    [bg_sig.sum(), (~bg_sig).sum()]]
    fisher_or, fisher_p = stats.fisher_exact(fisher_table)

    # c. Correlation: L1 poly(A) delta vs host gene log2FC
    delta = sub_dedup['delta'].values
    rho, rho_p = stats.spearmanr(delta, fc)

    median_fc = np.median(fc)
    mean_fc = np.mean(fc)

    print(f"\n--- {cls} (n={n} genes) ---")
    print(f"  Median log2FC: {median_fc:.4f}, Mean: {mean_fc:.4f}")
    print(f"  MWU vs genome-wide: U={mwu_stat:.0f}, P={mwu_p:.4g}")
    print(f"  Sig genes: {n_sig}/{n} ({pct_sig:.1f}%) [up:{n_sig_up}, down:{n_sig_down}]")
    print(f"  Genome-wide sig: {bg_sig.mean()*100:.1f}%")
    print(f"  Fisher OR={fisher_or:.3f}, P={fisher_p:.4g}")
    print(f"  L1 delta vs log2FC: rho={rho:.4f}, P={rho_p:.4g}")

    results.append({
        'subtype': cls,
        'n_genes': n,
        'median_log2FC': median_fc,
        'mean_log2FC': mean_fc,
        'MWU_P': mwu_p,
        'n_sig': n_sig,
        'pct_sig': pct_sig,
        'n_sig_up': n_sig_up,
        'n_sig_down': n_sig_down,
        'Fisher_OR': fisher_or,
        'Fisher_P': fisher_p,
        'delta_vs_fc_rho': rho,
        'delta_vs_fc_P': rho_p,
    })

# ── 6. 15-state ChromHMM: identify host genes per state ─────────────
# From per-read data, get HeLa reads in regulatory states, map to host genes
print("\n" + "="*80)
print("15-STATE CHROMHMM ANALYSIS (HeLa only)")
print("="*80)

# Filter HeLa reads in regulatory states
hela_reads = chromhmm_reads[chromhmm_reads['cellline'] == 'HeLa'].copy()
print(f"HeLa reads total: {len(hela_reads)}")

# Get unique chromhmm_state values
print(f"ChromHMM states: {hela_reads['chromhmm_state'].unique()}")

# Regulatory states: Enhancer (6_EnhG, 7_Enh) and Promoter (1_TssA, 2_TssAFlnk)
reg_states = ['1_TssA', '2_TssAFlnk', '6_EnhG', '7_Enh']

# For each state, find host genes (intronic reads → gene_id approximation)
# Actually, we need to map back to host genes. The per-read file has gene_id (L1 family).
# We need a different approach: use the gene_annot file which already has per-host-gene
# enhancer/promoter counts.

# Alternative: from chromhmm_reads, get HeLa intronic reads per state,
# then overlap with rnaseq host genes
# But chromhmm_reads gene_id is L1 family, not host gene.
# Let's use the merged dataset more carefully.

# For 15-state, we need to go back to the per-read level and link to host genes.
# Since we don't have direct host gene in chromhmm_reads, let's use a different approach:
# Use gene_annot which has n_enhancer, n_promoter and create pure subsets.

# Pure enhancer: n_enhancer > 0 AND n_promoter == 0
# Pure promoter: n_promoter > 0 AND n_enhancer == 0
# Mixed: both > 0

print("\n--- PURE vs MIXED classification ---")
merged['pure_class'] = 'Mixed'
merged.loc[(merged['n_enhancer'] > 0) & (merged['n_promoter'] == 0), 'pure_class'] = 'Pure_Enhancer'
merged.loc[(merged['n_promoter'] > 0) & (merged['n_enhancer'] == 0), 'pure_class'] = 'Pure_Promoter'

for cls in ['Pure_Enhancer', 'Pure_Promoter', 'Mixed']:
    sub = merged[merged['pure_class'] == cls].drop_duplicates(subset='gene_symbol')
    n = len(sub)
    if n < 3:
        print(f"\n{cls}: only {n} genes, skipping")
        continue

    fc = sub['rnaseq_log2FC'].values
    padj = sub['rnaseq_padj'].values

    mwu_stat, mwu_p = stats.mannwhitneyu(fc, bg_log2fc, alternative='two-sided')
    n_sig = (padj < 0.05).sum()
    pct_sig = n_sig / n * 100

    delta = sub['delta'].values
    if len(delta) >= 3:
        rho, rho_p = stats.spearmanr(delta, fc)
    else:
        rho, rho_p = np.nan, np.nan

    print(f"\n{cls} (n={n}): median log2FC={np.median(fc):.4f}, MWU P={mwu_p:.4g}, "
          f"sig={n_sig}/{n} ({pct_sig:.1f}%), delta-FC rho={rho:.4f} P={rho_p:.4g}")

    results.append({
        'subtype': cls,
        'n_genes': n,
        'median_log2FC': np.median(fc),
        'mean_log2FC': np.mean(fc),
        'MWU_P': mwu_p,
        'n_sig': n_sig,
        'pct_sig': pct_sig,
        'n_sig_up': ((padj < 0.05) & (fc > 0)).sum(),
        'n_sig_down': ((padj < 0.05) & (fc < 0)).sum(),
        'Fisher_OR': np.nan,
        'Fisher_P': np.nan,
        'delta_vs_fc_rho': rho,
        'delta_vs_fc_P': rho_p,
    })

# ── 7. Per-state host gene identification via genomic overlap ────────
# Since chromhmm_reads doesn't have host gene, let's use a proxy:
# HeLa reads that are intronic → they are in a host gene.
# We can try to find which host genes overlap each chromhmm state by
# looking at reads from merged dataset and their state.

# Actually, let's build this from gene_annot more carefully.
# gene_annot has n_enhancer and n_promoter counts per host_gene.
# We can compute a "dominant state" more granularly.

# For 15-state, let's try something different: build per-host-gene state profile
# from the per-read chromhmm data by matching genomic coordinates.

# Since we can't directly link chromhmm_reads to host genes (gene_id = L1 family),
# let's use the HeLa intronic reads approach. Filter HeLa intronic reads,
# get their chr:start-end, and intersect with host gene coords.
# This is complex. Instead, let's just report the binary + pure results clearly.

# ── 8. Additional test: stratify by L1 poly(A) response ─────────────
print("\n" + "="*80)
print("STRATIFIED BY POLY(A) RESPONSE × CHROMHMM CLASS")
print("="*80)

for resp in ['shortened', 'lengthened', 'stable']:
    for cls in ['Enhancer', 'Promoter']:
        sub = merged[(merged['chromhmm_class'] == cls) &
                     (merged['response'] == resp)].drop_duplicates(subset='gene_symbol')
        n = len(sub)
        if n < 3:
            continue
        fc = sub['rnaseq_log2FC'].values
        median_fc = np.median(fc)
        n_sig_down = ((sub['rnaseq_padj'] < 0.05) & (sub['rnaseq_log2FC'] < 0)).sum()
        n_sig_up = ((sub['rnaseq_padj'] < 0.05) & (sub['rnaseq_log2FC'] > 0)).sum()
        print(f"  {resp} × {cls}: n={n}, med_log2FC={median_fc:.3f}, "
              f"sig_up={n_sig_up}, sig_down={n_sig_down}")

# ── 9. Save results ──────────────────────────────────────────────────
results_df = pd.DataFrame(results)
outpath = os.path.join(OUTDIR, 'chromhmm_subtype_results.tsv')
results_df.to_csv(outpath, sep='\t', index=False)
print(f"\nResults saved to: {outpath}")

# Save merged data for reference
merged_path = os.path.join(OUTDIR, 'merged_host_gene_chromhmm_rnaseq.tsv')
merged.to_csv(merged_path, sep='\t', index=False)
print(f"Merged data saved to: {merged_path}")

# ── 10. Summary ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for _, r in results_df.iterrows():
    sig_label = "***" if r['MWU_P'] < 0.001 else "**" if r['MWU_P'] < 0.01 else "*" if r['MWU_P'] < 0.05 else "ns"
    print(f"  {r['subtype']:20s}  n={r['n_genes']:3.0f}  med_log2FC={r['median_log2FC']:+.4f}  "
          f"MWU_P={r['MWU_P']:.4g} {sig_label}  rho={r['delta_vs_fc_rho']:+.4f}")

print("\nConclusion: Check if any subtype shows P<0.05 for MWU or delta-FC correlation.")
