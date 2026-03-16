#!/usr/bin/env python3
"""
Add m6A and psi columns to L1_summary.tsv based on mAFiA.reads.bam modifications.
"""

import argparse
import os
import numpy as np
import pandas as pd
import pysam


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

    # Check for m6A markers
    if 'N+17802' in mm_tag or 'A+a' in mm_tag or 'A+m' in mm_tag:
        has_m6A = True

    # Check for psi markers
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
        print(f"Warning: BAM file not found or empty: {bam_path}")
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


def add_mafia_columns(summary_path, bam_path, output_path):
    """
    Add m6A and psi columns to L1_summary.tsv.
    """
    # Read summary TSV
    df = pd.read_csv(summary_path, sep='\t')
    print(f"Loaded {len(df)} reads from {summary_path}")

    # Extract modifications from BAM
    print(f"Extracting modifications from {bam_path}...")
    modifications = extract_modifications_from_bam(bam_path)
    print(f"Found {len(modifications)} reads with modification data")

    # Add m6A and psi columns
    # NA = not processed by mAFiA (no callable sites)
    # 0 = processed but no modification detected
    # 1 = modification detected
    def get_m6a(read_id):
        if read_id not in modifications:
            return np.nan  # Not processed by mAFiA
        return 1 if modifications[read_id][0] else 0

    def get_psi(read_id):
        if read_id not in modifications:
            return np.nan  # Not processed by mAFiA
        return 1 if modifications[read_id][1] else 0

    df['m6A'] = df['read_id'].apply(get_m6a)
    df['psi'] = df['read_id'].apply(get_psi)

    # Count statistics
    total_reads = len(df)
    matched_reads = df['m6A'].notna().sum()
    m6a_reads = (df['m6A'] == 1).sum()
    psi_reads = (df['psi'] == 1).sum()
    na_reads = df['m6A'].isna().sum()

    print(f"\nStatistics:")
    print(f"  Total reads in summary: {total_reads}")
    print(f"  Reads processed by mAFiA: {matched_reads} ({100*matched_reads/total_reads:.1f}%)")
    print(f"  Reads not processed (NA): {na_reads} ({100*na_reads/total_reads:.1f}%)")
    print(f"  Reads with m6A (of processed): {m6a_reads} ({100*m6a_reads/matched_reads:.1f}%)" if matched_reads > 0 else "  Reads with m6A: 0")
    print(f"  Reads with psi (of processed): {psi_reads} ({100*psi_reads/matched_reads:.1f}%)" if matched_reads > 0 else "  Reads with psi: 0")

    # Save output
    df.to_csv(output_path, sep='\t', index=False)
    print(f"\nOutput saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Add mAFiA m6A/psi columns to L1_summary.tsv")
    parser.add_argument("--summary", required=True, help="Input L1_summary.tsv path")
    parser.add_argument("--bam", required=True, help="mAFiA.reads.bam path")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    add_mafia_columns(args.summary, args.bam, args.output)


if __name__ == "__main__":
    main()
