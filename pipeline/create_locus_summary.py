#!/usr/bin/env python3
"""
Create locus-level summary table from L1_summary.tsv.
Groups reads by TE locus and computes aggregate statistics.
"""

import argparse
import pandas as pd
import numpy as np


def create_locus_summary(summary_path, output_path, group_name):
    """
    Create locus summary from L1_summary.tsv.
    """
    df = pd.read_csv(summary_path, sep='\t')
    print(f"Loaded {len(df)} reads from {summary_path}")

    # Create te_pos column
    df['te_pos'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)

    # Group by locus
    groupby_cols = ['te_pos', 'transcript_id', 'gene_id', 'te_strand', 'TE_group', 'overlapping_genes']

    def agg_func(x):
        # For m6A and psi, only consider non-NA values
        m6a_valid = x['m6A'].dropna()
        psi_valid = x['psi'].dropna()

        return pd.Series({
            'sample': x['sample'].iloc[0],
            'group': group_name,
            'read_count': len(x),
            'median_polya_length': x['polya_length'].median(),
            'mixed_tail_ratio': (x['class'] == 'decorated').sum() / len(x) if len(x) > 0 else 0,
            'n_m6A_called': len(m6a_valid),
            'mean_m6A': m6a_valid.mean() if len(m6a_valid) > 0 else np.nan,
            'n_psi_called': len(psi_valid),
            'mean_psi': psi_valid.mean() if len(psi_valid) > 0 else np.nan,
        })

    locus_df = df.groupby(groupby_cols, dropna=False).apply(agg_func, include_groups=False).reset_index()

    # Reorder columns
    col_order = [
        'sample', 'group', 'te_pos', 'transcript_id', 'gene_id', 'te_strand',
        'read_count', 'median_polya_length', 'mixed_tail_ratio',
        'n_m6A_called', 'mean_m6A', 'n_psi_called', 'mean_psi',
        'TE_group', 'overlapping_genes'
    ]
    locus_df = locus_df[col_order]

    # Sort by read_count descending to show hotspots first
    locus_df = locus_df.sort_values('read_count', ascending=False)

    # Save
    locus_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nLocus summary statistics:")
    print(f"  Total loci: {len(locus_df)}")
    print(f"  Total reads: {locus_df['read_count'].sum()}")
    print(f"  Top 10 loci account for: {locus_df['read_count'].head(10).sum()} reads ({100*locus_df['read_count'].head(10).sum()/locus_df['read_count'].sum():.1f}%)")
    print(f"  Top 20 loci account for: {locus_df['read_count'].head(20).sum()} reads ({100*locus_df['read_count'].head(20).sum()/locus_df['read_count'].sum():.1f}%)")
    print(f"\nOutput saved to: {output_path}")

    return locus_df


def main():
    parser = argparse.ArgumentParser(description="Create locus-level summary from L1_summary.tsv")
    parser.add_argument("--summary", required=True, help="Input L1_summary.tsv path")
    parser.add_argument("--output", required=True, help="Output locus summary TSV path")
    parser.add_argument("--group", required=True, help="Group name")
    args = parser.parse_args()

    create_locus_summary(args.summary, args.output, args.group)


if __name__ == "__main__":
    main()
