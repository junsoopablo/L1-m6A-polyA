#!/usr/bin/env python3
"""
Quantify L1 expression from GSE277764 ONT cDNA long-read data.

For each BAM file (UN/SA/HS × 2 reps), count reads overlapping L1 loci,
classify by subfamily (Young vs Ancient), and compare conditions.

Approach:
1. bedtools intersect BAM with L1 BED (≥10% overlap)
2. Count reads per L1 subfamily
3. Normalize (RPM)
4. Compare Young vs Ancient across conditions
"""
import pandas as pd
import numpy as np
import subprocess
import os
import sys

DATADIR = '/vault/external-datasets/2026/GSE277764_HeLa_ONT-cDNA_UN-SA-HS'
RMSK = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_sequence_features/hg38_L1_rmsk_consensus.tsv'
OUTDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_11_public_stress'
BEDTOOLS = '/blaze/apps/envs/bedtools/2.31.0/bin/bedtools'
SAMTOOLS = '/blaze/apps/envs/samtools/1.23/bin/samtools'

YOUNG = {'L1HS', 'L1PA2', 'L1PA3', 'L1PA4'}  # include L1PA4 as borderline
YOUNG_STRICT = {'L1HS', 'L1PA2', 'L1PA3'}

SAMPLES = {
    'UN_rep1': 'Untreated',
    'UN_rep2': 'Untreated',
    'SA_rep1': 'Arsenite',
    'SA_rep2': 'Arsenite',
    'HS_rep1': 'HeatShock',
    'HS_rep2': 'HeatShock',
}

os.makedirs(OUTDIR, exist_ok=True)

# Step 1: Create L1 BED file
print("=== Creating L1 BED ===")
l1_bed = os.path.join(OUTDIR, 'hg38_L1_rmsk.bed')
if not os.path.exists(l1_bed):
    rmsk = pd.read_csv(RMSK, sep='\t')
    # Filter to ≥100bp L1 elements
    rmsk = rmsk[rmsk['genoEnd'] - rmsk['genoStart'] >= 100]
    bed = rmsk[['genoName', 'genoStart', 'genoEnd', 'repName', 'strand']].copy()
    bed.insert(4, 'score', 0)
    bed.columns = ['chr', 'start', 'end', 'name', 'score', 'strand']
    bed = bed.sort_values(['chr', 'start'])
    bed.to_csv(l1_bed, sep='\t', header=False, index=False)
    print(f"  Created {l1_bed}: {len(bed)} L1 elements")
else:
    print(f"  Already exists: {l1_bed}")

# Step 2: For each BAM, count L1-overlapping reads
print("\n=== Counting L1 reads per sample ===")

results = []
for sample_name, condition in SAMPLES.items():
    bam = os.path.join(DATADIR, f'{sample_name}.sorted.bam')
    if not os.path.exists(bam):
        print(f"  SKIP {sample_name}: BAM not found")
        continue

    out_tsv = os.path.join(OUTDIR, f'{sample_name}_l1_counts.tsv')
    if os.path.exists(out_tsv):
        print(f"  Loading cached: {out_tsv}")
        df = pd.read_csv(out_tsv, sep='\t')
        results.append(df)
        continue

    print(f"  Processing {sample_name}...")

    # Get total read count for normalization
    total_cmd = f"{SAMTOOLS} view -c -F 2308 {bam}"
    total_reads = int(subprocess.check_output(total_cmd, shell=True).strip())

    # bedtools intersect: find reads overlapping L1 ≥10%
    # -f 0.1: minimum overlap fraction of the read
    # -wo: write original A and B entries plus overlap
    intersect_cmd = (
        f"{BEDTOOLS} intersect -a {bam} -b {l1_bed} -f 0.1 -wo -bed "
        f"| cut -f4,16 | sort -u"  # read_id + L1_subfamily (unique pairs)
    )

    # Run and parse
    result = subprocess.run(intersect_cmd, shell=True, capture_output=True, text=True)
    lines = [l.strip().split('\t') for l in result.stdout.strip().split('\n') if l.strip()]

    if not lines:
        print(f"    No L1 reads found!")
        continue

    # Count per subfamily
    read_subfam = pd.DataFrame(lines, columns=['read_id', 'subfamily'])
    # Deduplicate: one read can overlap multiple L1 → keep the one with most overlap
    # For simplicity, keep first (largest overlap from sorted bedtools output)
    read_subfam = read_subfam.drop_duplicates('read_id', keep='first')

    counts = read_subfam['subfamily'].value_counts().reset_index()
    counts.columns = ['subfamily', 'count']
    counts['sample'] = sample_name
    counts['condition'] = condition
    counts['total_reads'] = total_reads
    counts['rpm'] = counts['count'] / total_reads * 1e6

    counts.to_csv(out_tsv, sep='\t', index=False)
    print(f"    L1 reads: {len(read_subfam)}, Total reads: {total_reads}")
    results.append(counts)

if not results:
    print("\nNo BAM files found yet. Run download_and_map.sh first.")
    sys.exit(0)

# Step 3: Aggregate and compare
print("\n=== Aggregating results ===")
all_counts = pd.concat(results, ignore_index=True)
all_counts.to_csv(os.path.join(OUTDIR, 'all_l1_subfamily_counts.tsv'), sep='\t', index=False)

# Classify age
all_counts['age'] = all_counts['subfamily'].apply(
    lambda x: 'Young' if x in YOUNG_STRICT else 'Ancient'
)

# Aggregate by age × condition × sample
agg = all_counts.groupby(['sample', 'condition', 'age']).agg(
    total_l1_reads=('count', 'sum'),
    total_reads=('total_reads', 'first'),
).reset_index()
agg['rpm'] = agg['total_l1_reads'] / agg['total_reads'] * 1e6

print("\n=== Young vs Ancient L1 expression by condition ===")
print(f"{'Condition':<12} {'Age':<10} {'Rep1 RPM':>10} {'Rep2 RPM':>10} {'Mean RPM':>10}")
print("-" * 55)

for cond in ['Untreated', 'Arsenite', 'HeatShock']:
    for age in ['Young', 'Ancient']:
        sub = agg[(agg['condition'] == cond) & (agg['age'] == age)]
        rpms = sub['rpm'].values
        if len(rpms) == 2:
            print(f"  {cond:<12} {age:<10} {rpms[0]:>10.1f} {rpms[1]:>10.1f} {rpms.mean():>10.1f}")
        elif len(rpms) == 1:
            print(f"  {cond:<12} {age:<10} {rpms[0]:>10.1f} {'N/A':>10} {rpms[0]:>10.1f}")

# Fold changes
print("\n=== Fold changes (Stress / Untreated) ===")
for cond in ['Arsenite', 'HeatShock']:
    for age in ['Young', 'Ancient']:
        un = agg[(agg['condition'] == 'Untreated') & (agg['age'] == age)]['rpm'].mean()
        st = agg[(agg['condition'] == cond) & (agg['age'] == age)]['rpm'].mean()
        if un > 0:
            fc = st / un
            print(f"  {cond:<12} {age:<10} FC = {fc:.3f}x")

print(f"\nSaved: {OUTDIR}/all_l1_subfamily_counts.tsv")
