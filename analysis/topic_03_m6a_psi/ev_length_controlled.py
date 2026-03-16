#!/usr/bin/env python3
"""
Controlled read length comparison: MCF7 vs MCF7-EV L1.

The EV library may have globally longer reads. To determine if L1 reads
are *disproportionately* longer in EV, we:
  1. Match common genes between MCF7 and MCF7-EV whole transcriptome
  2. Compute per-gene median read length ratio (EV / MCF7)
  3. Compare L1 read length ratio vs matched-gene background ratio
"""

import pysam
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
EXON_GTF = f'{PROJECT}/reference/Human.gtf'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# We use MCF7_2 (largest rep) vs MCF7-EV_1
MCF7_BAM = f'{PROJECT}/results/MCF7_2_3/a_hg38_mapping_LRS/MCF7_2_3_hg38_mapped.sorted_position.bam'
EV_BAM = f'{PROJECT}/results/MCF7-EV_1_1/a_hg38_mapping_LRS/MCF7-EV_1_1_hg38_mapped.sorted_position.bam'

# =========================================================================
# 1. Build gene annotation from GTF (gene regions)
# =========================================================================
print("Loading gene annotation...")

gene_coords = {}  # gene_name -> (chr, start, end, strand)
with open(EXON_GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        if cols[2] != 'gene':
            continue
        attrs = cols[8]
        gene_name = None
        gene_type = None
        for attr in attrs.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_name'):
                gene_name = attr.split('"')[1]
            if attr.startswith('gene_type') or attr.startswith('gene_biotype'):
                gene_type = attr.split('"')[1]
        if gene_name and gene_type == 'protein_coding':
            gene_coords[gene_name] = (cols[0], int(cols[3]), int(cols[4]), cols[6])

print(f"  Protein-coding genes: {len(gene_coords):,}")

# =========================================================================
# 2. Extract read lengths per gene from BAMs (sampling approach)
# =========================================================================
def get_gene_read_lengths(bam_path, gene_coords, max_reads_per_gene=500, max_genes=5000):
    """For each gene, collect read lengths from reads overlapping the gene."""
    gene_lengths = defaultdict(list)
    all_lengths = []

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        genes_done = 0
        for gene_name, (chrom, start, end, strand) in gene_coords.items():
            if genes_done >= max_genes:
                break
            try:
                count = 0
                for read in bam.fetch(chrom, start, end):
                    if read.query_length and read.query_length > 0:
                        gene_lengths[gene_name].append(read.query_length)
                        all_lengths.append(read.query_length)
                        count += 1
                        if count >= max_reads_per_gene:
                            break
                if count > 0:
                    genes_done += 1
            except ValueError:
                continue

    return gene_lengths, np.array(all_lengths)


print("\nExtracting gene-level read lengths...")

print("  MCF7_2...")
mcf7_gene_lens, mcf7_all_lens = get_gene_read_lengths(MCF7_BAM, gene_coords)
print(f"    Genes with reads: {len(mcf7_gene_lens):,}, total reads: {len(mcf7_all_lens):,}")

print("  MCF7-EV_1...")
ev_gene_lens, ev_all_lens = get_gene_read_lengths(EV_BAM, gene_coords)
print(f"    Genes with reads: {len(ev_gene_lens):,}, total reads: {len(ev_all_lens):,}")

# =========================================================================
# 3. Global read length ratio
# =========================================================================
print(f"\n{'='*80}")
print("Global Transcriptome Read Length")
print(f"{'='*80}")
print(f"  MCF7:    mean={mcf7_all_lens.mean():.0f}  median={np.median(mcf7_all_lens):.0f}")
print(f"  MCF7-EV: mean={ev_all_lens.mean():.0f}  median={np.median(ev_all_lens):.0f}")
global_ratio = np.median(ev_all_lens) / np.median(mcf7_all_lens)
print(f"  Global median ratio (EV/MCF7): {global_ratio:.3f}")

# =========================================================================
# 4. Per-gene matched comparison
# =========================================================================
print(f"\n{'='*80}")
print("Gene-Matched Read Length Ratio (EV / MCF7)")
print(f"{'='*80}")

# Only genes with reads in both
common_genes = set(mcf7_gene_lens.keys()) & set(ev_gene_lens.keys())
# Filter for genes with >= 10 reads in both
MIN_READS = 10
matched_genes = [g for g in common_genes
                 if len(mcf7_gene_lens[g]) >= MIN_READS and len(ev_gene_lens[g]) >= MIN_READS]

print(f"  Common genes (>={MIN_READS} reads each): {len(matched_genes):,}")

gene_ratios = []
for g in matched_genes:
    m_med = np.median(mcf7_gene_lens[g])
    e_med = np.median(ev_gene_lens[g])
    if m_med > 0:
        gene_ratios.append({
            'gene': g,
            'mcf7_median': m_med,
            'ev_median': e_med,
            'ratio': e_med / m_med,
            'mcf7_n': len(mcf7_gene_lens[g]),
            'ev_n': len(ev_gene_lens[g]),
        })

gene_ratio_df = pd.DataFrame(gene_ratios)
bg_ratio_median = gene_ratio_df['ratio'].median()
bg_ratio_mean = gene_ratio_df['ratio'].mean()

print(f"  Background gene ratio (EV/MCF7):")
print(f"    Median of per-gene ratios: {bg_ratio_median:.3f}")
print(f"    Mean of per-gene ratios:   {bg_ratio_mean:.3f}")
print(f"    25th percentile:           {gene_ratio_df['ratio'].quantile(0.25):.3f}")
print(f"    75th percentile:           {gene_ratio_df['ratio'].quantile(0.75):.3f}")

# =========================================================================
# 5. L1 read length ratio vs background
# =========================================================================
print(f"\n{'='*80}")
print("L1 Read Length Ratio vs Gene Background")
print(f"{'='*80}")

# Load L1 data
mcf7_l1_dfs = []
for g in ['MCF7_2', 'MCF7_3', 'MCF7_4']:
    df = pd.read_csv(f'{PROJECT}/results_group/{g}/g_summary/{g}_L1_summary.tsv', sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    mcf7_l1_dfs.append(df)
mcf7_l1 = pd.concat(mcf7_l1_dfs)

ev_l1 = pd.read_csv(f'{PROJECT}/results_group/MCF7-EV_1/g_summary/MCF7-EV_1_L1_summary.tsv', sep='\t')
ev_l1 = ev_l1[ev_l1['qc_tag'] == 'PASS']

mcf7_l1['l1_age'] = mcf7_l1['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
ev_l1['l1_age'] = ev_l1['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

categories = [
    ('L1 all',     mcf7_l1,                          ev_l1),
    ('L1 young',   mcf7_l1[mcf7_l1['l1_age']=='young'],   ev_l1[ev_l1['l1_age']=='young']),
    ('L1 ancient', mcf7_l1[mcf7_l1['l1_age']=='ancient'], ev_l1[ev_l1['l1_age']=='ancient']),
]

print(f"\n{'Category':<15} {'MCF7 med':>9} {'EV med':>9} {'Ratio':>7} {'BG ratio':>9} {'Excess':>8} {'Interpretation'}")
print("-" * 85)

for label, m_df, e_df in categories:
    m_med = m_df['read_length'].median()
    e_med = e_df['read_length'].median()
    l1_ratio = e_med / m_med if m_med > 0 else float('inf')
    excess = l1_ratio / bg_ratio_median
    interp = "L1-specific" if excess > 1.2 else "global artifact" if excess < 0.8 else "~background"
    print(f"{label:<15} {m_med:>9.0f} {e_med:>9.0f} {l1_ratio:>7.3f} {bg_ratio_median:>9.3f} {excess:>7.2f}x  {interp}")

# =========================================================================
# 6. Percentile rank of L1 ratio in gene background distribution
# =========================================================================
print(f"\n{'='*80}")
print("L1 Ratio Percentile in Gene Background Distribution")
print(f"{'='*80}")

bg_ratios = gene_ratio_df['ratio'].values

for label, m_df, e_df in categories:
    m_med = m_df['read_length'].median()
    e_med = e_df['read_length'].median()
    l1_ratio = e_med / m_med
    pctile = (bg_ratios < l1_ratio).mean() * 100
    print(f"  {label:<15} ratio={l1_ratio:.3f}  percentile in BG: {pctile:.1f}%")

# =========================================================================
# 7. Read-length normalized by background: are L1 reads disproportionately long?
# =========================================================================
print(f"\n{'='*80}")
print("Background-Corrected Read Length Comparison")
print(f"(Divide EV read lengths by global ratio to correct for library effect)")
print(f"{'='*80}")

correction_factor = bg_ratio_median
print(f"  Correction factor (BG median ratio): {correction_factor:.3f}")
print(f"  = EV reads are on average {correction_factor:.3f}x longer globally\n")

for label, m_df, e_df in categories:
    # Correct EV lengths
    e_corrected = e_df['read_length'] / correction_factor

    m_med = m_df['read_length'].median()
    e_med_raw = e_df['read_length'].median()
    e_med_corr = e_corrected.median()

    _, p_raw = stats.mannwhitneyu(m_df['read_length'], e_df['read_length'], alternative='two-sided')
    _, p_corr = stats.mannwhitneyu(m_df['read_length'], e_corrected, alternative='two-sided')
    sig_raw = "***" if p_raw < 0.001 else "**" if p_raw < 0.01 else "*" if p_raw < 0.05 else "ns"
    sig_corr = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"

    print(f"  {label}:")
    print(f"    MCF7 median:          {m_med:.0f}")
    print(f"    EV median (raw):      {e_med_raw:.0f}  p={p_raw:.2e} ({sig_raw})")
    print(f"    EV median (corrected):{e_med_corr:.0f}  p={p_corr:.2e} ({sig_corr})")
    print()

# =========================================================================
# 8. Young L1 bins after correction
# =========================================================================
print(f"{'='*80}")
print("Young L1 Read Length Bins: After Background Correction")
print(f"{'='*80}")

m_young = mcf7_l1[mcf7_l1['l1_age']=='young']['read_length']
e_young = ev_l1[ev_l1['l1_age']=='young']['read_length']
e_young_corr = e_young / correction_factor

bins = [0, 500, 1000, 2000, 3000, 5000, 10000]
bin_labels = ['<0.5kb', '0.5-1kb', '1-2kb', '2-3kb', '3-5kb', '>=5kb']

m_hist = pd.cut(m_young, bins=bins, labels=bin_labels).value_counts().sort_index()
e_hist_raw = pd.cut(e_young, bins=bins, labels=bin_labels).value_counts().sort_index()
e_hist_corr = pd.cut(e_young_corr, bins=bins, labels=bin_labels).value_counts().sort_index()

m_pct = m_hist / len(m_young) * 100
e_pct_raw = e_hist_raw / len(e_young) * 100
e_pct_corr = e_hist_corr / len(e_young_corr) * 100

print(f"\n{'Bin':<12} {'MCF7 %':>8} {'EV raw%':>8} {'EV corr%':>9} {'raw/MCF7':>9} {'corr/MCF7':>10}")
print("-" * 60)
for b in bin_labels:
    mp = m_pct.get(b, 0)
    ep_raw = e_pct_raw.get(b, 0)
    ep_corr = e_pct_corr.get(b, 0)
    r_raw = ep_raw / mp if mp > 0 else float('inf')
    r_corr = ep_corr / mp if mp > 0 else float('inf')
    print(f"{b:<12} {mp:>7.1f}% {ep_raw:>7.1f}% {ep_corr:>8.1f}% {r_raw:>8.2f}x {r_corr:>9.2f}x")

# Save gene ratio data
gene_ratio_df.to_csv(f'{PROJECT}/analysis/01_exploration/topic_03_m6a_psi/gene_matched_ratios.tsv',
                      sep='\t', index=False)
print(f"\nSaved: gene_matched_ratios.tsv")
