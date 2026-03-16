#!/usr/bin/env python3
"""
Create L1 summary TSV combining L1_reads.tsv with ninetails reclassified results.
Adds TE_group column (intronic/intergenic), overlapping_genes column, and m6A/psi columns from mAFiA.
"""

import os
import glob
import re
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
import pysam


def sample_to_group(sample):
    """Extract group name from sample name (e.g., HeLa_1_1 -> HeLa_1)."""
    match = re.match(r'^(.+)_\d+$', sample)
    if match:
        return match.group(1)
    return sample


def parse_mm_tag(mm_tag):
    """
    Parse MM:Z tag to detect m6A and psi modifications.

    MM tag format: N+17802,pos1,pos2;N+21891,pos1,pos2;
    - N+17802 or A+a = m6A
    - N+21891 = psi (pseudouridine)

    Returns: (has_m6A, has_psi)
    """
    has_m6A = False
    has_psi = False

    if mm_tag is None:
        return has_m6A, has_psi

    if 'N+17802' in mm_tag or 'A+a' in mm_tag or 'A+m' in mm_tag:
        has_m6A = True
    if 'N+21891' in mm_tag or 'U+p' in mm_tag or 'T+p' in mm_tag:
        has_psi = True

    return has_m6A, has_psi


def extract_modifications_from_bam(bam_path):
    """
    Extract m6A and psi status for each read from mAFiA.reads.bam.

    Returns: dict {read_id: (has_m6A, has_psi)}
    """
    modifications = {}

    if not os.path.exists(bam_path) or os.path.getsize(bam_path) == 0:
        return modifications

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                read_id = read.query_name
                mm_tag = read.get_tag("MM") if read.has_tag("MM") else None
                has_m6A, has_psi = parse_mm_tag(mm_tag)
                modifications[read_id] = (has_m6A, has_psi)
    except Exception as e:
        print(f"Error reading BAM: {e}")

    return modifications


def load_gene_regions(gtf_path):
    """Load gene regions from GTF file into a BED-like dataframe."""
    genes = []
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'gene':
                continue

            chrom = fields[0]
            start = int(fields[3]) - 1  # Convert to 0-based
            end = int(fields[4])
            strand = fields[6]

            # Extract gene_name from attributes
            attrs = fields[8]
            gene_name = None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name'):
                    match = re.search(r'gene_name "([^"]+)"', attr)
                    if match:
                        gene_name = match.group(1)
                        break

            if gene_name:
                genes.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'gene_name': gene_name,
                    'strand': strand
                })

    return pd.DataFrame(genes)


def find_overlapping_genes_bedtools(reads_df, gtf_path):
    """Use bedtools to find overlapping genes for each read."""

    # Create temporary BED file for reads
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as reads_bed:
        for idx, row in reads_df.iterrows():
            # BED format: chrom, start, end, name, score, strand
            reads_bed.write(f"{row['chr']}\t{row['start']}\t{row['end']}\t{row['read_id']}\t0\t{row['read_strand']}\n")
        reads_bed_path = reads_bed.name

    # Create temporary BED file for genes from GTF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as genes_bed:
        with open(gtf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                if fields[2] != 'gene':
                    continue

                chrom = fields[0]
                start = int(fields[3]) - 1  # Convert to 0-based
                end = int(fields[4])
                strand = fields[6]

                # Extract gene_name
                attrs = fields[8]
                gene_name = "unknown"
                for attr in attrs.split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_name'):
                        match = re.search(r'gene_name "([^"]+)"', attr)
                        if match:
                            gene_name = match.group(1)
                            break

                genes_bed.write(f"{chrom}\t{start}\t{end}\t{gene_name}\t0\t{strand}\n")
        genes_bed_path = genes_bed.name

    # Run bedtools intersect
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as out_file:
        out_path = out_file.name

    try:
        # Use bedtools intersect with -wao to get all overlaps including no-overlap cases
        cmd = f"bedtools intersect -a {reads_bed_path} -b {genes_bed_path} -wao"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

        # Parse result
        overlaps = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            fields = line.split('\t')
            read_id = fields[3]
            gene_name = fields[9] if len(fields) > 9 else '.'
            overlap_bp = int(fields[-1]) if fields[-1] != '.' else 0

            if read_id not in overlaps:
                overlaps[read_id] = []

            if overlap_bp > 0 and gene_name != '.':
                overlaps[read_id].append(gene_name)

        # Create result
        result_dict = {}
        for read_id, genes in overlaps.items():
            if genes:
                result_dict[read_id] = {
                    'TE_group': 'intronic',
                    'overlapping_genes': ';'.join(sorted(set(genes)))
                }
            else:
                result_dict[read_id] = {
                    'TE_group': 'intergenic',
                    'overlapping_genes': ''
                }

        return result_dict

    finally:
        # Cleanup temp files
        os.unlink(reads_bed_path)
        os.unlink(genes_bed_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def create_group_summary(group, results_dir, group_results_dir, gtf_path, mafia_bam=None, output_dir=None):
    """Create summary TSV for a single group."""

    # Find all samples in this group
    sample_dirs = []
    for sample_name in os.listdir(results_dir):
        sample_path = os.path.join(results_dir, sample_name)
        if os.path.isdir(sample_path) and sample_to_group(sample_name) == group:
            sample_dirs.append((sample_name, sample_path))

    if not sample_dirs:
        print(f"No samples found for group {group}")
        return None

    # Collect L1_reads.tsv from all samples
    all_l1_reads = []
    for sample_name, sample_path in sample_dirs:
        l1_reads_path = os.path.join(sample_path, 'd_LINE_quantification', f'{sample_name}_L1_reads.tsv')
        if os.path.exists(l1_reads_path) and os.path.getsize(l1_reads_path) > 0:
            df = pd.read_csv(l1_reads_path, sep='\t')
            df['sample'] = sample_name
            all_l1_reads.append(df)

    if not all_l1_reads:
        print(f"No L1_reads.tsv files found for group {group}")
        return None

    l1_df = pd.concat(all_l1_reads, ignore_index=True)
    print(f"Group {group}: {len(l1_df)} L1 reads from {len(sample_dirs)} samples")

    # Load reclassified ninetails results
    reclassified_path = os.path.join(group_results_dir, group, 'f_ninetails', f'{group}_read_classes_reclassified.tsv')
    if os.path.exists(reclassified_path) and os.path.getsize(reclassified_path) > 0:
        ninetails_df = pd.read_csv(reclassified_path, sep='\t')
        ninetails_df = ninetails_df.rename(columns={'readname': 'read_id'})
    else:
        print(f"Warning: No reclassified ninetails file for group {group}")
        ninetails_df = pd.DataFrame(columns=['read_id', 'contig', 'polya_length', 'qc_tag', 'class', 'comments'])

    # Merge L1 reads with ninetails results
    merged_df = l1_df.merge(ninetails_df[['read_id', 'polya_length', 'qc_tag', 'class', 'comments']],
                            on='read_id', how='left')

    # Find overlapping genes using bedtools
    print(f"Finding overlapping genes for {len(merged_df)} reads...")
    overlap_dict = find_overlapping_genes_bedtools(merged_df, gtf_path)

    # Add TE_group and overlapping_genes columns
    merged_df['TE_group'] = merged_df['read_id'].apply(
        lambda x: overlap_dict.get(x, {'TE_group': 'intergenic'})['TE_group']
    )
    merged_df['overlapping_genes'] = merged_df['read_id'].apply(
        lambda x: overlap_dict.get(x, {'overlapping_genes': ''})['overlapping_genes']
    )

    # Add m6A and psi columns from mAFiA
    if mafia_bam is None:
        mafia_bam = os.path.join(group_results_dir, group, 'h_mafia', f'{group}.mAFiA.reads.bam')

    if os.path.exists(mafia_bam) and os.path.getsize(mafia_bam) > 0:
        print(f"Loading mAFiA modifications from {mafia_bam}...")
        modifications = extract_modifications_from_bam(mafia_bam)
        print(f"Found {len(modifications)} reads with modification data")

        # NA = not processed by mAFiA (no callable sites)
        # 0 = processed but no modification detected
        # 1 = modification detected
        def get_m6a(read_id):
            if read_id not in modifications:
                return np.nan
            return 1 if modifications[read_id][0] else 0

        def get_psi(read_id):
            if read_id not in modifications:
                return np.nan
            return 1 if modifications[read_id][1] else 0

        merged_df['m6A'] = merged_df['read_id'].apply(get_m6a)
        merged_df['psi'] = merged_df['read_id'].apply(get_psi)
    else:
        print(f"Warning: mAFiA BAM not found: {mafia_bam}")
        merged_df['m6A'] = np.nan
        merged_df['psi'] = np.nan

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(group_results_dir, group, 'g_summary')
    os.makedirs(output_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(output_dir, f'{group}_L1_summary.tsv')
    merged_df.to_csv(output_path, sep='\t', index=False)

    # Print stats
    print(f"\n=== {group} ===")
    print(f"Output: {output_path}")
    print(f"Total reads: {len(merged_df)}")
    print(f"With ninetails data: {merged_df['polya_length'].notna().sum()}")
    print(f"\nTE_group distribution:")
    print(merged_df['TE_group'].value_counts().to_string())
    print(f"\nClass distribution (reclassified):")
    print(merged_df['class'].value_counts(dropna=False).to_string())

    # mAFiA stats
    m6a_called = merged_df['m6A'].notna().sum()
    psi_called = merged_df['psi'].notna().sum()
    if m6a_called > 0:
        m6a_positive = (merged_df['m6A'] == 1).sum()
        psi_positive = (merged_df['psi'] == 1).sum()
        print(f"\nmAFiA modification stats:")
        print(f"  Reads processed by mAFiA: {m6a_called} ({100*m6a_called/len(merged_df):.1f}%)")
        print(f"  Reads not processed (NA): {len(merged_df) - m6a_called} ({100*(len(merged_df) - m6a_called)/len(merged_df):.1f}%)")
        print(f"  m6A positive (of processed): {m6a_positive} ({100*m6a_positive/m6a_called:.1f}%)")
        print(f"  psi positive (of processed): {psi_positive} ({100*psi_positive/psi_called:.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create L1 summary TSV")
    parser.add_argument("--results-dir",
                        default="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results",
                        help="Path to results directory (per-sample)")
    parser.add_argument("--results-group-dir",
                        default="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group",
                        help="Path to results_group directory")
    parser.add_argument("--gtf",
                        default="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.gtf",
                        help="Path to gene annotation GTF")
    parser.add_argument("--group", help="Process specific group only")
    parser.add_argument("--mafia-bam", help="Path to mAFiA.reads.bam (optional, auto-detected if not specified)")
    args = parser.parse_args()

    if args.group:
        create_group_summary(args.group, args.results_dir, args.results_group_dir, args.gtf, args.mafia_bam)
    else:
        # Find all groups from results_group directory
        for group_name in sorted(os.listdir(args.results_group_dir)):
            group_dir = os.path.join(args.results_group_dir, group_name)
            if os.path.isdir(group_dir):
                create_group_summary(group_name, args.results_dir, args.results_group_dir, args.gtf)


if __name__ == "__main__":
    main()
