#!/usr/bin/env python3
"""
Quick check: does arsenite change the L1 sense/antisense ratio?

For each HeLa and HeLa-Ars sample:
1. Load L1_reads.tsv (PASS reads) → sense reads
2. Load stage 1 read IDs (before strand filter)
3. Compare strand concordance rate between conditions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results'

samples = {
    'HeLa': ['HeLa_1_1', 'HeLa_2_1', 'HeLa_3_1'],
    'HeLa-Ars': ['HeLa-Ars_1_1', 'HeLa-Ars_2_1', 'HeLa-Ars_3_1'],
}

print('='*70)
print('L1 SENSE / ANTISENSE RATIO: HeLa vs HeLa-Ars')
print('='*70)

# First, check strand info in L1_reads.tsv
sample0 = 'HeLa_1_1'
tsv = RESULTS / sample0 / 'd_LINE_quantification' / f'{sample0}_L1_reads.tsv'
df0 = pd.read_csv(tsv, sep='\t')
print(f'\nL1_reads.tsv columns: {list(df0.columns)}')
print(f'te_strand values: {df0["te_strand"].value_counts().to_dict()}')
print(f'read_strand values: {df0["read_strand"].value_counts().to_dict()}')
print(f'Strand concordance: {(df0["te_strand"] == df0["read_strand"]).sum()} / {len(df0)}')

# But L1_reads.tsv only has PASS reads (already strand-filtered)
# We need the FULL stage 2 output including rejected reads
# The filter script only outputs PASS reads, so strand-discordant are lost

# Alternative: check the BAM directly for stage 1 reads
# Let's look at the exon_overlap data which has all stage 1 reads
ov = pd.read_csv(PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/exon_overlap_validation/exon_overlap_all_stage1.tsv', sep='\t')
print(f'\nExon overlap data has {len(ov)} stage 1 reads')
print(f'But no strand information in this file.')

# We need to extract strand info from BAM for stage 1 reads
# Let's do a quick check using samtools
import subprocess

print('\n' + '='*70)
print('EXTRACTING STRAND INFO FROM BAM FOR STAGE 1 READS')
print('='*70)

results = []

for cond, sample_list in samples.items():
    for sample in sample_list:
        # Get stage 1 read IDs
        rid_file = RESULTS / sample / 'b_l1_te_filter' / f'{sample}_L1_readIDs.txt'
        with open(rid_file) as f:
            stage1_ids = set(line.strip() for line in f if line.strip())

        # Get L1 reads TSV for strand info on PASS reads
        tsv = RESULTS / sample / 'd_LINE_quantification' / f'{sample}_L1_reads.tsv'
        df = pd.read_csv(tsv, sep='\t')

        # PASS reads: all have strand concordance (by definition)
        n_pass = len(df)
        pass_ids = set(df['read_id'])

        # Rejected at stage 2
        n_stage1 = len(stage1_ids)
        n_rejected = n_stage1 - n_pass

        # For PASS reads, check te_strand distribution
        n_plus = (df['te_strand'] == '+').sum()
        n_minus = (df['te_strand'] == '-').sum()

        print(f'\n{sample}:')
        print(f'  Stage 1: {n_stage1}, PASS: {n_pass} ({100*n_pass/n_stage1:.1f}%), Rejected: {n_rejected}')
        print(f'  PASS L1 strand: + = {n_plus}, - = {n_minus} (ratio = {n_plus/n_minus:.2f})')

        results.append({
            'sample': sample,
            'condition': cond,
            'stage1': n_stage1,
            'pass': n_pass,
            'rejected': n_rejected,
            'pass_rate': n_pass / n_stage1,
            'plus': n_plus,
            'minus': n_minus,
        })

# Now extract read strand from BAM for stage 1 reads to find antisense reads
print('\n' + '='*70)
print('BAM STRAND ANALYSIS: ALL STAGE 1 READS')
print('='*70)

# For efficiency, use samtools to get flag field
for cond, sample_list in samples.items():
    for sample in sample_list:
        rid_file = RESULTS / sample / 'b_l1_te_filter' / f'{sample}_L1_readIDs.txt'
        bam = RESULTS / sample / 'a_hg38_mapping_LRS' / f'{sample}_hg38_mapped.sorted_position.bam'

        # Extract flag and qname for stage 1 reads
        cmd = f'samtools view {bam} | grep -F -f {rid_file} | cut -f1,2'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        strand_counts = {'+': 0, '-': 0}
        read_strands = {}
        for line in proc.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            qname = parts[0]
            flag = int(parts[1])
            # In DRS, flag 16 = reverse complement = maps to minus strand
            strand = '-' if (flag & 16) else '+'
            if qname not in read_strands:  # first (primary) alignment
                read_strands[qname] = strand
                strand_counts[strand] += 1

        n_total = sum(strand_counts.values())
        pct_plus = 100 * strand_counts['+'] / n_total if n_total > 0 else 0
        print(f'{sample}: {n_total} reads, + = {strand_counts["+"]} ({pct_plus:.1f}%), '
              f'- = {strand_counts["-"]} ({100-pct_plus:.1f}%)')

# Now the key: compare with L1 TE annotations
print('\n' + '='*70)
print('STRAND CONCORDANCE ANALYSIS (READ vs L1 TE ANNOTATION)')
print('='*70)

# Load L1 TE annotations to determine expected strand
# We need to check each stage 1 read's overlap with L1 TEs and compare strands
# This is already done in the L1_reads.tsv (PASS only) but we need rejected reads too

# Simpler approach: load L1_reads.bed (PASS only) which has coordinates
# and the exon_overlap_all_stage1.tsv to identify PASS vs rejected
# Then check PASS reads strand in L1_reads.tsv

# Key question: what fraction of REJECTED reads are rejected BECAUSE of strand mismatch?
# We know from the filter script that strand concordance is one of 4 stage 2 filters
# Let's estimate by looking at the PASS rate across conditions

print('\nPASS rate comparison (proxy for strand concordance):')
df_res = pd.DataFrame(results)
for cond in ['HeLa', 'HeLa-Ars']:
    sub = df_res[df_res['condition'] == cond]
    mean_rate = sub['pass_rate'].mean()
    print(f'  {cond}: mean PASS rate = {mean_rate:.4f} ({100*mean_rate:.2f}%)')

hela_rates = df_res[df_res['condition'] == 'HeLa']['pass_rate'].values
ars_rates = df_res[df_res['condition'] == 'HeLa-Ars']['pass_rate'].values
if len(hela_rates) >= 2 and len(ars_rates) >= 2:
    _, p = stats.mannwhitneyu(hela_rates, ars_rates, alternative='two-sided')
    print(f'  MWU p = {p:.4f}')

# Check +/- ratio in PASS reads
print('\nPASS reads +/- strand ratio:')
for cond in ['HeLa', 'HeLa-Ars']:
    sub = df_res[df_res['condition'] == cond]
    total_plus = sub['plus'].sum()
    total_minus = sub['minus'].sum()
    ratio = total_plus / total_minus if total_minus > 0 else float('inf')
    print(f'  {cond}: + = {total_plus}, - = {total_minus}, ratio = {ratio:.3f}')

print('\nDone!')
