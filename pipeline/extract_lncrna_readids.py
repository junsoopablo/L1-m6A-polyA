#!/usr/bin/env python3
"""Extract exclusive lncRNA read IDs from full BAM files.

For each HeLa/HeLa-Ars group:
1. Generate strand-specific lncRNA exon BED + PC exon BED from GENCODE v38 GTF
2. Convert BAM → BED (primary aligned reads only, -F 260)
3. Classify reads via bedtools intersect chain:
   - lncRNA exon overlap (strand-specific, -s -f 0.1)
   - Remove reads overlapping protein-coding exons
   - Remove reads overlapping L1 annotations
4. Annotate each read with lncRNA gene_name and transcript subtype
5. Subsample: 5000 per group (seed=42)
6. Convert POD5 → FAST5 for the subsampled reads

Output (results_group/{group}/l_lncrna_ctrl/):
  - {group}_lncrna_readIDs.txt
  - {group}_lncrna_classification.tsv (read_id, gene_name, gene_id, lncrna_subtype, read_length)
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import sys
import re
import gzip

# =============================================================================
# Configuration
# =============================================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
REFERENCE = PROJECT / 'reference'

GTF = REFERENCE / 'gencode.v38.annotation.gtf'
L1_BED = REFERENCE / 'L1_TE_L1_family.bed'

# Generated BED files (will be created if missing)
LNCRNA_EXON_BED = TOPIC_05 / 'lncrna_exons_stranded.bed'
PC_EXON_BED = TOPIC_05 / 'pc_exons_merged.bed'

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

SUBSAMPLE_N = 5000
SEED = 42

FAST5_OUT_ROOT = Path('/scratch1/junsoopablo/IsoTENT_002_L1_lncrna')
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


def parse_gtf_attribute(attr_str, key):
    """Extract a specific attribute value from GTF attribute string."""
    match = re.search(rf'{key}\s+"([^"]+)"', attr_str)
    return match.group(1) if match else None


# =============================================================================
# BED file generation from GENCODE GTF
# =============================================================================
def generate_bed_files():
    """Generate lncRNA exon BED and PC exon BED from GENCODE v38 GTF."""

    if LNCRNA_EXON_BED.exists() and PC_EXON_BED.exists():
        lnc_n = count_lines(LNCRNA_EXON_BED)
        pc_n = count_lines(PC_EXON_BED)
        print(f"BED files already exist: lncRNA exons={lnc_n:,}, PC exons={pc_n:,}")
        return

    print("Generating BED files from GENCODE v38 GTF...")

    lncrna_exons = []  # chr, start, end, strand, gene_name, gene_id, transcript_type
    pc_exons = []      # chr, start, end

    opener = gzip.open if str(GTF).endswith('.gz') else open
    with opener(GTF, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue

            chrom = fields[0]
            feature = fields[2]
            start = int(fields[3]) - 1  # GTF 1-based → BED 0-based
            end = int(fields[4])
            strand = fields[6]
            attr = fields[8]

            if feature != 'exon':
                continue

            gene_type = parse_gtf_attribute(attr, 'gene_type')

            if gene_type == 'lncRNA':
                gene_name = parse_gtf_attribute(attr, 'gene_name') or '.'
                gene_id = parse_gtf_attribute(attr, 'gene_id') or '.'
                transcript_type = parse_gtf_attribute(attr, 'transcript_type') or 'lncRNA'
                lncrna_exons.append(f"{chrom}\t{start}\t{end}\t{gene_name}\t{gene_id}\t{strand}\t{transcript_type}")
            elif gene_type == 'protein_coding':
                pc_exons.append(f"{chrom}\t{start}\t{end}")

    # Write lncRNA exon BED (7-col: chr, start, end, gene_name, gene_id, strand, transcript_type)
    # Sort and merge by strand
    tmp_lnc = TOPIC_05 / '_tmp_lncrna_exons_raw.bed'
    tmp_lnc_sorted = TOPIC_05 / '_tmp_lncrna_exons_sorted.bed'

    with open(tmp_lnc, 'w') as f:
        f.write('\n'.join(lncrna_exons) + '\n')
    print(f"  Raw lncRNA exon intervals: {len(lncrna_exons):,}")

    # Sort by chrom, start, strand
    run_cmd(f"sort -k1,1 -k2,2n -k6,6 {tmp_lnc} > {tmp_lnc_sorted}", "sort lncRNA")

    # Merge overlapping intervals per strand, keeping gene_name info
    # Use bedtools merge -s (strand-specific) -c 4,5,6,7 -o distinct
    run_cmd(
        f"{BEDTOOLS} merge -i {tmp_lnc_sorted} -s "
        f"-c 4,5,6,7 -o distinct,distinct,distinct,distinct > {LNCRNA_EXON_BED}",
        "merge lncRNA exons"
    )
    lnc_n = count_lines(LNCRNA_EXON_BED)
    print(f"  Merged lncRNA exon intervals: {lnc_n:,}")

    # Write PC exon BED (3-col) — sort and merge
    tmp_pc = TOPIC_05 / '_tmp_pc_exons_raw.bed'
    with open(tmp_pc, 'w') as f:
        f.write('\n'.join(pc_exons) + '\n')
    print(f"  Raw PC exon intervals: {len(pc_exons):,}")

    run_cmd(f"sort -k1,1 -k2,2n {tmp_pc} | {BEDTOOLS} merge -i - > {PC_EXON_BED}", "merge PC exons")
    pc_n = count_lines(PC_EXON_BED)
    print(f"  Merged PC exon intervals: {pc_n:,}")

    # Cleanup tmp files
    for f in [tmp_lnc, tmp_lnc_sorted, tmp_pc]:
        f.unlink(missing_ok=True)


# =============================================================================
# Build lncRNA annotation lookup from GTF
# =============================================================================
def build_lncrna_annotation():
    """Build a lookup: gene_name → (gene_id, transcript_type) from GENCODE GTF.
    For per-read annotation, we map read → overlapping lncRNA gene_name via BED,
    then look up subtype info here.
    """
    gene_info = {}  # gene_name → {gene_id, subtypes}

    opener = gzip.open if str(GTF).endswith('.gz') else open
    with opener(GTF, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'gene':
                continue

            attr = fields[8]
            gene_type = parse_gtf_attribute(attr, 'gene_type')
            if gene_type != 'lncRNA':
                continue

            gene_name = parse_gtf_attribute(attr, 'gene_name') or '.'
            gene_id = parse_gtf_attribute(attr, 'gene_id') or '.'
            transcript_type = parse_gtf_attribute(attr, 'transcript_type') or 'lncRNA'

            gene_info[gene_name] = {
                'gene_id': gene_id,
                'transcript_type': transcript_type,
            }

    print(f"  lncRNA gene annotation: {len(gene_info):,} genes")
    return gene_info


# =============================================================================
# Per-read lncRNA annotation via BED overlap
# =============================================================================
def annotate_reads_with_lncrna(read_bed_path, lncrna_bed, tmp_dir):
    """For each read in read_bed_path, find which lncRNA gene(s) it overlaps.
    Returns DataFrame: read_id → gene_name, gene_id, lncrna_subtype.
    """
    # bedtools intersect -a reads.bed -b lncrna_exons.bed -s -wo
    # Output: read columns + lncRNA columns + overlap_bp
    overlap_file = tmp_dir / 'lncrna_read_annotation.bed'
    run_cmd(
        f"{BEDTOOLS} intersect -a {read_bed_path} -b {lncrna_bed} -s -wo > {overlap_file}",
        "annotate reads"
    )

    if not overlap_file.exists() or overlap_file.stat().st_size == 0:
        return pd.DataFrame(columns=['read_id', 'gene_name', 'gene_id', 'lncrna_subtype'])

    # lncRNA BED is 7-col merged: chr, start, end, gene_name(s), gene_id(s), strand, transcript_type(s)
    # Read BED is 6-col: chr, start, end, name, score, strand
    # Overlap output: 6 read cols + 7 lncRNA cols + 1 overlap_bp = 14 cols
    df = pd.read_csv(overlap_file, sep='\t', header=None, low_memory=False,
                     dtype={3: str, 9: str, 10: str, 12: str})

    # Column indices: 3=read_id, 9=gene_name(s), 10=gene_id(s), 12=transcript_type(s)
    records = []
    for _, row in df.iterrows():
        read_id = str(row[3])
        gene_names = str(row[9]).split(',')
        gene_ids = str(row[10]).split(',')
        subtypes = str(row[12]).split(',')

        # Take first gene info (most common overlap)
        records.append({
            'read_id': read_id,
            'gene_name': gene_names[0],
            'gene_id': gene_ids[0],
            'lncrna_subtype': subtypes[0],
        })

    anno_df = pd.DataFrame(records)
    # Deduplicate: keep first annotation per read
    anno_df = anno_df.drop_duplicates(subset='read_id', keep='first')
    return anno_df


# =============================================================================
# Process one group
# =============================================================================
def process_group(group, samples, gene_info):
    print(f"\n{'='*60}")
    print(f"Processing {group} (samples: {', '.join(samples)})")
    print(f"{'='*60}")

    out_dir = PROJECT / 'results_group' / group / 'l_lncrna_ctrl'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / '_tmp'
    tmp_dir.mkdir(exist_ok=True)

    # --- Step 1: BAM → BED (primary aligned only, 6-col with strand) ---
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

    # 6-column BED: chr, start, end, name(read_id), score, strand
    run_cmd(
        f"{SAMTOOLS} view -F 260 -b {bam_input} | {BEDTOOLS} bamtobed -i stdin > {all_bed}",
        "BAM→BED"
    )

    total = count_lines(all_bed)
    print(f"  Total primary aligned intervals: {total:,}")

    # --- Step 2: Classify reads via bedtools intersect chain ---
    # Step 2a: lncRNA exon overlap (strand-specific, 10% overlap)
    lncrna_overlap_bed = tmp_dir / 'lncrna_exon_overlap.bed'
    run_cmd(
        f"{BEDTOOLS} intersect -a {all_bed} -b {LNCRNA_EXON_BED} -u -s -f 0.1 > {lncrna_overlap_bed}",
        "lncRNA exon overlap"
    )
    lncrna_overlap_n = count_lines(lncrna_overlap_bed)
    print(f"  lncRNA exon overlap: {lncrna_overlap_n:,}")

    # Step 2b: Remove reads overlapping protein-coding exons
    lncrna_no_pc_bed = tmp_dir / 'lncrna_no_pc.bed'
    run_cmd(
        f"{BEDTOOLS} intersect -a {lncrna_overlap_bed} -b {PC_EXON_BED} -v > {lncrna_no_pc_bed}",
        "remove PC exon overlap"
    )
    no_pc_n = count_lines(lncrna_no_pc_bed)
    print(f"  After removing PC exon overlap: {no_pc_n:,}")

    # Step 2c: Remove reads overlapping L1 annotations
    lncrna_exclusive_bed = tmp_dir / 'lncrna_exclusive.bed'
    run_cmd(
        f"{BEDTOOLS} intersect -a {lncrna_no_pc_bed} -b {L1_BED} -v > {lncrna_exclusive_bed}",
        "remove L1 overlap"
    )
    exclusive_n = count_lines(lncrna_exclusive_bed)
    print(f"  Exclusive lncRNA reads: {exclusive_n:,}")

    exclusive_ids = read_bed_readnames(lncrna_exclusive_bed)
    print(f"  Unique exclusive lncRNA read IDs: {len(exclusive_ids):,}")

    if len(exclusive_ids) == 0:
        print(f"  WARNING: No exclusive lncRNA reads for {group}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # --- Step 3: Annotate reads with lncRNA gene info ---
    anno_df = annotate_reads_with_lncrna(lncrna_exclusive_bed, LNCRNA_EXON_BED, tmp_dir)
    print(f"  Annotated reads: {len(anno_df):,}")

    # Add read_length from BED (end - start)
    bed_df = pd.read_csv(lncrna_exclusive_bed, sep='\t', header=None,
                         names=['chrom', 'start', 'end', 'read_id', 'score', 'strand'],
                         dtype={'read_id': str})
    bed_df['read_length'] = bed_df['end'] - bed_df['start']
    bed_df = bed_df[['read_id', 'read_length']].drop_duplicates(subset='read_id', keep='first')

    anno_df = anno_df.merge(bed_df, on='read_id', how='left')

    # --- Step 4: Subsample ---
    rng = np.random.RandomState(SEED)

    # Only keep reads with annotation
    annotated_ids = set(anno_df['read_id'].unique()) & exclusive_ids
    annotated_sorted = sorted(annotated_ids)

    n_sample = min(SUBSAMPLE_N, len(annotated_sorted))
    if n_sample < SUBSAMPLE_N:
        print(f"  WARNING: Only {n_sample} annotated lncRNA available (requested {SUBSAMPLE_N})")

    sampled_ids = list(rng.choice(annotated_sorted, size=n_sample, replace=False))
    sampled_set = set(sampled_ids)

    # Write output files
    readid_file = out_dir / f'{group}_lncrna_readIDs.txt'
    classification_file = out_dir / f'{group}_lncrna_classification.tsv'

    with open(readid_file, 'w') as f:
        f.write('\n'.join(sorted(sampled_ids)) + '\n')

    # Classification TSV for sampled reads
    cls_df = anno_df[anno_df['read_id'].isin(sampled_set)].copy()
    cls_df.to_csv(classification_file, sep='\t', index=False)

    # Print subtype distribution
    if 'lncrna_subtype' in cls_df.columns:
        subtype_counts = cls_df['lncrna_subtype'].value_counts()
        print(f"  Subtype distribution:")
        for st, cnt in subtype_counts.items():
            print(f"    {st}: {cnt}")

    print(f"  Sampled: {n_sample} lncRNA reads")

    # --- Step 5: POD5 → FAST5 ---
    sample = samples[0]
    pod5_path = PROJECT / f'data_pod5/{sample}/{sample}.pod5'
    fast5_out = FAST5_OUT_ROOT / group

    if not pod5_path.exists():
        pod5_path = PROJECT / f'data_pod5/{sample}'
        if not pod5_path.exists():
            print(f"  ERROR: POD5 not found for {sample}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    print(f"  POD5: {pod5_path}")
    print(f"  FAST5 output: {fast5_out}")

    pod5_script = PROJECT / 'scripts/pod5_subset_from_pod5.sh'
    run_cmd(
        f"conda run -n research bash {pod5_script} "
        f"{pod5_path} {readid_file} {fast5_out} {POD5_THREADS}",
        "POD5→FAST5"
    )

    fast5_count = len(list(fast5_out.glob('*.fast5')))
    print(f"  FAST5 files created: {fast5_count}")

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        'group': group,
        'lncrna_pool': len(exclusive_ids),
        'annotated': len(annotated_ids),
        'sampled': n_sample,
        'fast5_count': fast5_count,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # Verify reference files
    for f, name in [(GTF, 'GENCODE GTF'), (L1_BED, 'L1 BED')]:
        if not f.exists():
            print(f"ERROR: {name} not found: {f}")
            sys.exit(1)

    # Phase 0: Generate BED files from GTF
    generate_bed_files()

    # Build gene annotation lookup
    gene_info = build_lncrna_annotation()

    # Process each group
    results = []
    for group, samples in GROUPS.items():
        try:
            r = process_group(group, samples, gene_info)
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
