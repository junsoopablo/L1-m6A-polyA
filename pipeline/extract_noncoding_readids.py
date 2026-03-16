#!/usr/bin/env python3
"""Extract intronic and intergenic non-L1 read IDs from full BAM files.

For each HeLa/HeLa-Ars group:
1. Convert BAM → BED (primary aligned reads only, -F 260)
2. Classify reads via bedtools intersect chain:
   - Intronic: overlaps gene body, no exon overlap, no L1 overlap
   - Intergenic: no gene body overlap, no L1 overlap
3. Subsample: 3000 intronic + 2000 intergenic = 5000 per group
4. Convert POD5 → FAST5 for the subsampled reads

Output (results_group/{group}/k_noncoding_ctrl/):
  - {group}_intronic_readIDs.txt
  - {group}_intergenic_readIDs.txt
  - {group}_noncoding_readIDs.txt (combined)
  - {group}_read_classification.tsv (read_id → category)
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import sys

# =============================================================================
# Configuration
# =============================================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'

GENE_BED = TOPIC_05 / 'gencode_genes_merged.bed'
EXON_BED = TOPIC_05 / 'gencode_exons_merged.bed'
L1_BED = PROJECT / 'reference/L1_TE_L1_family.bed'

SAMTOOLS = '/blaze/junsoopablo/conda/envs/research/bin/samtools'
BEDTOOLS = '/blaze/junsoopablo/conda/envs/research/bin/bedtools'

# Sample → Group mapping (HeLa + HeLa-Ars only)
GROUPS = {
    'HeLa_1':     ['HeLa_1_1'],
    'HeLa_2':     ['HeLa_2_1'],
    'HeLa_3':     ['HeLa_3_1'],
    'HeLa-Ars_1': ['HeLa-Ars_1_1'],
    'HeLa-Ars_2': ['HeLa-Ars_2_1'],
    'HeLa-Ars_3': ['HeLa-Ars_3_1'],
}

INTRONIC_N = 3000
INTERGENIC_N = 2000
SEED = 42

FAST5_OUT_ROOT = Path('/scratch1/junsoopablo/IsoTENT_002_L1_noncoding')
POD5_THREADS = 8


def run_cmd(cmd, desc=""):
    """Run a shell command, print it, and raise on failure."""
    display = cmd[:140] + '...' if len(cmd) > 140 else cmd
    print(f"  [{desc}] {display}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd[:200]}")
    return result.stdout


def count_lines(filepath):
    """Count lines in a file without loading it entirely."""
    out = subprocess.run(f"wc -l < {filepath}", shell=True, capture_output=True, text=True)
    return int(out.stdout.strip()) if out.returncode == 0 else 0


def read_bed_readnames(bed_path):
    """Read unique read names (column 4) from a BED file."""
    if not bed_path.exists() or bed_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[3],
                     dtype={3: str}, low_memory=False)
    return set(df[3].unique())


def process_group(group, samples):
    print(f"\n{'='*60}")
    print(f"Processing {group} (samples: {', '.join(samples)})")
    print(f"{'='*60}")

    out_dir = PROJECT / 'results_group' / group / 'k_noncoding_ctrl'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / '_tmp'
    tmp_dir.mkdir(exist_ok=True)

    # --- Step 1: BAM → BED (primary aligned only) ---
    bam_paths = []
    for sample in samples:
        bam = PROJECT / f'results/{sample}/a_hg38_mapping_LRS/{sample}_hg38_mapped.sorted_position.bam'
        if bam.exists():
            bam_paths.append(str(bam))
        else:
            print(f"  WARNING: BAM not found: {bam}")
    if not bam_paths:
        print(f"  No BAMs found for {group}, skipping")
        return None

    all_bed = tmp_dir / 'all_reads.bed'
    if len(bam_paths) == 1:
        bam_input = bam_paths[0]
    else:
        merged_bam = tmp_dir / 'merged.bam'
        run_cmd(f"{SAMTOOLS} merge -f {merged_bam} {' '.join(bam_paths)}", "merge BAMs")
        bam_input = str(merged_bam)

    run_cmd(
        f"{SAMTOOLS} view -F 260 -b {bam_input} | {BEDTOOLS} bamtobed -i stdin > {all_bed}",
        "BAM→BED"
    )

    total = count_lines(all_bed)
    print(f"  Total primary aligned intervals: {total:,}")

    # --- Step 2: Classify reads via bedtools intersect chain ---
    genic_bed = tmp_dir / 'genic.bed'
    intronic_all_bed = tmp_dir / 'intronic_all.bed'
    intronic_nonL1_bed = tmp_dir / 'intronic_nonL1.bed'
    not_genic_bed = tmp_dir / 'not_genic.bed'
    intergenic_nonL1_bed = tmp_dir / 'intergenic_nonL1.bed'

    # Genic = overlaps any gene body
    run_cmd(f"{BEDTOOLS} intersect -a {all_bed} -b {GENE_BED} -u > {genic_bed}", "genic")
    # Intronic = genic but NOT overlapping any exon
    run_cmd(f"{BEDTOOLS} intersect -a {genic_bed} -b {EXON_BED} -v > {intronic_all_bed}", "intronic(all)")
    # Remove L1 overlaps
    run_cmd(f"{BEDTOOLS} intersect -a {intronic_all_bed} -b {L1_BED} -v > {intronic_nonL1_bed}", "intronic-nonL1")
    # Not genic = no gene body overlap
    run_cmd(f"{BEDTOOLS} intersect -a {all_bed} -b {GENE_BED} -v > {not_genic_bed}", "not-genic")
    # Remove L1 overlaps
    run_cmd(f"{BEDTOOLS} intersect -a {not_genic_bed} -b {L1_BED} -v > {intergenic_nonL1_bed}", "intergenic-nonL1")

    intronic_ids = read_bed_readnames(intronic_nonL1_bed)
    intergenic_ids = read_bed_readnames(intergenic_nonL1_bed)

    # Remove reads appearing in both categories (keep in intronic)
    overlap = intronic_ids & intergenic_ids
    if overlap:
        print(f"  {len(overlap)} reads in both categories → assigned to intronic")
        intergenic_ids -= overlap

    print(f"  Intronic non-L1:    {len(intronic_ids):,} reads")
    print(f"  Intergenic non-L1:  {len(intergenic_ids):,} reads")

    # --- Step 3: Subsample ---
    rng = np.random.RandomState(SEED)

    intronic_sorted = sorted(intronic_ids)
    intergenic_sorted = sorted(intergenic_ids)

    n_intr = min(INTRONIC_N, len(intronic_sorted))
    n_inter = min(INTERGENIC_N, len(intergenic_sorted))

    if n_intr < INTRONIC_N:
        print(f"  WARNING: Only {n_intr} intronic available (requested {INTRONIC_N})")
    if n_inter < INTERGENIC_N:
        print(f"  WARNING: Only {n_inter} intergenic available (requested {INTERGENIC_N})")

    sampled_intr = list(rng.choice(intronic_sorted, size=n_intr, replace=False))
    sampled_inter = list(rng.choice(intergenic_sorted, size=n_inter, replace=False))

    # Write output files
    intronic_file = out_dir / f'{group}_intronic_readIDs.txt'
    intergenic_file = out_dir / f'{group}_intergenic_readIDs.txt'
    combined_file = out_dir / f'{group}_noncoding_readIDs.txt'
    classification_file = out_dir / f'{group}_read_classification.tsv'

    with open(intronic_file, 'w') as f:
        f.write('\n'.join(sorted(sampled_intr)) + '\n')
    with open(intergenic_file, 'w') as f:
        f.write('\n'.join(sorted(sampled_inter)) + '\n')

    all_sampled = sorted(set(sampled_intr) | set(sampled_inter))
    with open(combined_file, 'w') as f:
        f.write('\n'.join(all_sampled) + '\n')

    records = [{'read_id': rid, 'category': 'intronic'} for rid in sampled_intr]
    records += [{'read_id': rid, 'category': 'intergenic'} for rid in sampled_inter]
    pd.DataFrame(records).to_csv(classification_file, sep='\t', index=False)

    print(f"  Sampled: {n_intr} intronic + {n_inter} intergenic = {len(all_sampled)} total")

    # --- Step 4: POD5 → FAST5 ---
    sample = samples[0]
    pod5_path = PROJECT / f'data_pod5/{sample}/{sample}.pod5'
    fast5_out = FAST5_OUT_ROOT / group

    if not pod5_path.exists():
        # Fallback: try directory
        pod5_path = PROJECT / f'data_pod5/{sample}'
        if not pod5_path.exists():
            print(f"  ERROR: POD5 not found for {sample}")
            return None

    print(f"  POD5: {pod5_path}")
    print(f"  FAST5 output: {fast5_out}")

    pod5_script = PROJECT / 'scripts/pod5_subset_from_pod5.sh'
    run_cmd(
        f"conda run -n research bash {pod5_script} "
        f"{pod5_path} {combined_file} {fast5_out} {POD5_THREADS}",
        "POD5→FAST5"
    )

    fast5_count = len(list(fast5_out.glob('*.fast5')))
    print(f"  FAST5 files created: {fast5_count}")

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        'group': group,
        'intronic_pool': len(intronic_ids),
        'intergenic_pool': len(intergenic_ids),
        'sampled_intronic': n_intr,
        'sampled_intergenic': n_inter,
        'total_sampled': len(all_sampled),
        'fast5_count': fast5_count,
    }


if __name__ == '__main__':
    # Verify reference BED files
    for bed, name in [(GENE_BED, 'gene'), (EXON_BED, 'exon'), (L1_BED, 'L1')]:
        if not bed.exists():
            print(f"ERROR: {name} BED not found: {bed}")
            sys.exit(1)

    results = []
    for group, samples in GROUPS.items():
        try:
            r = process_group(group, samples)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("\nNo groups processed successfully.")
