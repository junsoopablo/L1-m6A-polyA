#!/usr/bin/env python3
"""
METTL3 KO: Select better control transcripts for thr=128 comparison.

Strategy:
  1. Parse ctrl MAFIA BAMs at thr=128 with gene info
  2. Compute per-gene WT m6A/kb
  3. Select genes with high basal m6A (top quartile in WT)
  4. Show KO effect in high-m6A controls

Also explore: exclude ribosomal proteins (RPL/RPS),
housekeeping exclusion, and gene-level paired comparison.
"""
import os, sys
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

VAULT = Path('/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore')
MAFIA_DIR = VAULT / 'mafia_guppy'
OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_m6a_signal')
OUTDIR.mkdir(exist_ok=True)

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

###############################################################################
# 1. Load gene annotations for ctrl reads
###############################################################################
ctrl_gene = pd.read_csv(VAULT / 'analysis/matched_guppy/ctrl_per_read_gene.tsv', sep='\t')
# Only protein-coding
ctrl_gene = ctrl_gene[ctrl_gene['gene_biotype'] == 'protein_coding'].copy()
print(f"Protein-coding ctrl reads: {len(ctrl_gene)}")

# Flag ribosomal proteins
ctrl_gene['is_ribo'] = ctrl_gene['gene_name'].str.match(r'^(RPL|RPS|MRPL|MRPS)')
ribo_n = ctrl_gene['is_ribo'].sum()
print(f"Ribosomal protein reads: {ribo_n} ({100*ribo_n/len(ctrl_gene):.1f}%)")

###############################################################################
# 2. Parse ctrl MAFIA BAMs at multiple thresholds
###############################################################################
def parse_bam_multi_thr(bam_path, thresholds=[128, 204]):
    if not os.path.exists(bam_path):
        return []
    bam = pysam.AlignmentFile(bam_path, 'rb')
    rows = []
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rlen = read.query_alignment_length
        if rlen is None or rlen < 100:
            continue
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break
        row = {'read_id': read.query_name, 'read_length': rlen}
        if mm_tag is None or ml_tag is None:
            for thr in thresholds:
                row[f'sites_{thr}'] = 0
            rows.append(row)
            continue
        ml_list = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        idx = 0
        m6a_mls = []
        for entry in entries:
            parts = entry.strip().split(',')
            base_mod = parts[0]
            skips = [int(x) for x in parts[1:]] if len(parts) > 1 else []
            n_sites = len(skips)
            if '21891' in base_mod:
                for i in range(n_sites):
                    if idx + i < len(ml_list):
                        m6a_mls.append(ml_list[idx + i])
            idx += n_sites
        for thr in thresholds:
            row[f'sites_{thr}'] = sum(1 for v in m6a_mls if v >= thr)
        rows.append(row)
    bam.close()
    return rows

print("\nParsing ctrl MAFIA BAMs at thr=128 and 204...")
ctrl_mafia = []
for sample, condition in SAMPLES.items():
    bam_path = MAFIA_DIR / f'{sample}_ctrl' / 'mAFiA.reads.bam'
    print(f"  {sample}_ctrl...", end=' ', flush=True)
    rows = parse_bam_multi_thr(str(bam_path))
    for r in rows:
        r['sample'] = sample
        r['condition'] = condition
    ctrl_mafia.extend(rows)
    print(f"{len(rows)} reads")

mafia_df = pd.DataFrame(ctrl_mafia)
mafia_df['m6a_kb_128'] = mafia_df['sites_128'] / (mafia_df['read_length'] / 1000)
mafia_df['m6a_kb_204'] = mafia_df['sites_204'] / (mafia_df['read_length'] / 1000)

###############################################################################
# 3. Merge with gene info
###############################################################################
merged = mafia_df.merge(
    ctrl_gene[['read_id', 'gene_name', 'gene_biotype', 'is_ribo']],
    on='read_id', how='inner'
)
print(f"\nMerged ctrl reads: {len(merged)}")

###############################################################################
# 4. Compute per-gene WT m6A level
###############################################################################
wt_per_gene = merged[merged['condition'] == 'WT'].groupby('gene_name').agg(
    n_reads=('read_id', 'count'),
    mean_m6a_128=('m6a_kb_128', 'mean'),
    mean_m6a_204=('m6a_kb_204', 'mean'),
).reset_index()

# Only genes with >= 3 reads in WT for reliable estimate
wt_per_gene = wt_per_gene[wt_per_gene['n_reads'] >= 3]
print(f"Genes with >=3 WT reads: {len(wt_per_gene)}")

# Quartiles of WT m6A/kb at thr=128
q25, q50, q75 = wt_per_gene['mean_m6a_128'].quantile([0.25, 0.5, 0.75])
print(f"WT m6A/kb (thr=128) quartiles: Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")

high_m6a_genes = set(wt_per_gene[wt_per_gene['mean_m6a_128'] >= q50]['gene_name'])
top_quartile_genes = set(wt_per_gene[wt_per_gene['mean_m6a_128'] >= q75]['gene_name'])
print(f"High m6A genes (>=median): {len(high_m6a_genes)}")
print(f"Top quartile genes: {len(top_quartile_genes)}")

# Show top m6A genes
print("\nTop 20 WT m6A/kb genes (thr=128):")
for _, row in wt_per_gene.nlargest(20, 'mean_m6a_128').iterrows():
    is_ribo = 'ribo' if row['gene_name'].startswith(('RPL', 'RPS', 'MRPL', 'MRPS')) else ''
    print(f"  {row['gene_name']:15s} m6A/kb={row['mean_m6a_128']:.2f} (n={row['n_reads']}) {is_ribo}")

###############################################################################
# 5. Compare WT vs KO across different control subsets
###############################################################################
print("\n" + "=" * 70)
print("WT vs KO comparison at thr=128 — different control subsets")
print("=" * 70)

subsets = {
    'all_protein_coding': merged,
    'non-ribosomal': merged[~merged['is_ribo']],
    'ribosomal_only': merged[merged['is_ribo']],
    'high_m6a (>=median)': merged[merged['gene_name'].isin(high_m6a_genes)],
    'top_quartile_m6a': merged[merged['gene_name'].isin(top_quartile_genes)],
    'low_m6a (<median)': merged[~merged['gene_name'].isin(high_m6a_genes)],
}

print(f"\n  {'Subset':>25s} {'N_WT':>6s} {'N_KO':>6s} {'WT med':>8s} {'KO med':>8s} {'KO/WT':>8s} {'P':>12s} {'sig':>4s}")
print("  " + "-" * 80)

results = []
for label, subset in subsets.items():
    for thr in [128, 204]:
        col = f'm6a_kb_{thr}'
        wt = subset[subset['condition'] == 'WT'][col]
        ko = subset[subset['condition'] == 'KO'][col]
        if len(wt) < 10 or len(ko) < 10:
            continue
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:>25s} {len(wt):6d} {len(ko):6d} {wt_med:8.3f} {ko_med:8.3f} {ratio:7.3f}x {p:12.2e} {sig:>4s}  (thr={thr})")
        results.append({
            'subset': label, 'threshold': thr,
            'n_wt': len(wt), 'n_ko': len(ko),
            'wt_median': round(wt_med, 3), 'ko_median': round(ko_med, 3),
            'ratio': round(ratio, 4) if not np.isnan(ratio) else 'nan',
            'p': p, 'sig': sig
        })

###############################################################################
# 6. Gene-level paired comparison (WT mean vs KO mean per gene)
###############################################################################
print("\n" + "=" * 70)
print("Gene-level paired comparison")
print("=" * 70)

for thr in [128, 204]:
    col = f'm6a_kb_{thr}'
    gene_means = merged.groupby(['gene_name', 'condition'])[col].mean().reset_index()
    gene_pivot = gene_means.pivot(index='gene_name', columns='condition', values=col).dropna()
    if 'WT' in gene_pivot.columns and 'KO' in gene_pivot.columns:
        # Filter genes present in both
        gene_pivot = gene_pivot[(gene_pivot['WT'] > 0) | (gene_pivot['KO'] > 0)]
        # Paired test
        if len(gene_pivot) >= 5:
            t_stat, t_p = stats.wilcoxon(gene_pivot['WT'], gene_pivot['KO'])
            ratio = gene_pivot['KO'].median() / gene_pivot['WT'].median() if gene_pivot['WT'].median() > 0 else float('nan')
            print(f"\n  thr={thr}: {len(gene_pivot)} genes")
            print(f"    WT median gene m6A/kb: {gene_pivot['WT'].median():.3f}")
            print(f"    KO median gene m6A/kb: {gene_pivot['KO'].median():.3f}")
            print(f"    KO/WT: {ratio:.3f}x")
            print(f"    Wilcoxon signed-rank P: {t_p:.2e}")

            # Also for high-m6a genes only
            high_pivot = gene_pivot[gene_pivot.index.isin(high_m6a_genes)]
            if len(high_pivot) >= 5:
                t2, p2 = stats.wilcoxon(high_pivot['WT'], high_pivot['KO'])
                r2 = high_pivot['KO'].median() / high_pivot['WT'].median() if high_pivot['WT'].median() > 0 else float('nan')
                print(f"    High-m6A genes ({len(high_pivot)}): KO/WT={r2:.3f}x, P={p2:.2e}")

###############################################################################
# 7. Also check transcriptome MAFIA (larger dataset)
###############################################################################
print("\n" + "=" * 70)
print("Transcriptome MAFIA — protein-coding reads (larger N)")
print("=" * 70)

tx_dir = VAULT / 'mafia_transcriptome'
tx_rows = []
for sample in ['WT_rep1', 'WT_rep2', 'WT_rep3', 'KO_rep1', 'KO_rep2', 'KO_rep3']:
    condition = 'WT' if sample.startswith('WT') else 'KO'
    bam_path = tx_dir / f'{sample}_protein_coding' / 'mAFiA.reads.bam'
    if not bam_path.exists():
        print(f"  SKIP: {bam_path}")
        continue
    print(f"  {sample}_protein_coding...", end=' ', flush=True)
    rows = parse_bam_multi_thr(str(bam_path))
    for r in rows:
        r['sample'] = sample
        r['condition'] = condition
    tx_rows.extend(rows)
    print(f"{len(rows)} reads")

if tx_rows:
    tx_df = pd.DataFrame(tx_rows)
    tx_df['m6a_kb_128'] = tx_df['sites_128'] / (tx_df['read_length'] / 1000)
    tx_df['m6a_kb_204'] = tx_df['sites_204'] / (tx_df['read_length'] / 1000)

    print(f"\n  Total transcriptome reads: {len(tx_df):,}")
    for thr in [128, 204]:
        col = f'm6a_kb_{thr}'
        wt = tx_df[tx_df['condition'] == 'WT'][col]
        ko = tx_df[tx_df['condition'] == 'KO'][col]
        wt_med = wt.median()
        ko_med = ko.median()
        ratio = ko_med / wt_med if wt_med > 0 else float('nan')
        _, p = stats.mannwhitneyu(wt, ko, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  thr={thr}: WT {wt_med:.3f} vs KO {ko_med:.3f} = {ratio:.3f}x P={p:.2e} {sig} (n_wt={len(wt)}, n_ko={len(ko)})")

###############################################################################
# Save
###############################################################################
pd.DataFrame(results).to_csv(OUTDIR / 'ctrl_subset_comparison.tsv', sep='\t', index=False)
print(f"\nSaved to {OUTDIR}/")
print("\nDONE.")
