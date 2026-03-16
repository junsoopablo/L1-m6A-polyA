#!/usr/bin/env python3
"""
Housekeeping Gene Control Analysis

Compare poly(A) tail and decorated rate between L1 and housekeeping genes.
This serves as a control to distinguish L1-specific vs global effects.

Housekeeping genes used:
- ACTB (chr7:5526409-5563902)
- GAPDH (chr12:6534512-6538374)
- B2M (chr15:44711487-44718851)
- PPIA (chr7:44796680-44824564)
- RPLP0 (chr12:120196699-120201235)
- RPL13A (chr19:49487510-49493057)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import gzip
import warnings
warnings.filterwarnings('ignore')

# Housekeeping gene coordinates (hg38)
HOUSEKEEPING_GENES = {
    'ACTB': ('chr7', 5526409, 5563902),
    'GAPDH': ('chr12', 6534512, 6538374),
    'B2M': ('chr15', 44711487, 44718851),
    'PPIA': ('chr7', 44796680, 44824564),
    'RPLP0': ('chr12', 120196699, 120201235),
    'RPL13A': ('chr19', 49487510, 49493057),
    'TBP': ('chr6', 170554302, 170572870),
    'HPRT1': ('chrX', 134460165, 134520513),
}

def parse_cell_line(group):
    """Extract cell line from group name"""
    parts = group.replace("-", "_").split("_")
    cell_line = parts[0]
    if len(parts) > 1 and parts[1] in ["Ars", "EV", "Kasumi3", "HFF"]:
        cell_line = f"{parts[0]}-{parts[1]}"
    return cell_line

def find_gene(chrom, pos):
    """Find which housekeeping gene a position belongs to"""
    for gene, (g_chr, g_start, g_end) in HOUSEKEEPING_GENES.items():
        if chrom == g_chr and g_start <= pos <= g_end:
            return gene
    return None

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    print("=" * 70)
    print("HOUSEKEEPING GENE CONTROL ANALYSIS")
    print("=" * 70)

    # Process groups of interest
    groups_of_interest = ['HeLa_1', 'HeLa_2', 'HeLa_3',
                          'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

    all_hk_data = []

    for group in groups_of_interest:
        print(f"\nProcessing {group}...")

        # Load nanopolish polya data
        polya_file = base / group / "e_nanopolish" / f"{group}.nanopolish.polya.tsv.gz"
        if not polya_file.exists():
            print(f"  Warning: {polya_file} not found")
            continue

        # Load ninetails class data
        class_file = base / group / "f_ninetails" / f"{group}_read_classes.tsv"
        if not class_file.exists():
            # Try the other filename pattern
            class_files = list((base / group / "f_ninetails").glob("*_read_classes.tsv"))
            if class_files:
                class_file = class_files[0]
            else:
                print(f"  Warning: ninetails class file not found")
                continue

        # Read ninetails classes
        class_df = pd.read_csv(class_file, sep='\t')
        class_dict = dict(zip(class_df['readname'], class_df['class']))

        # Read nanopolish polya and filter for housekeeping genes
        hk_reads = []
        with gzip.open(polya_file, 'rt') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue

                read_id = fields[0]
                chrom = fields[1]
                pos = int(fields[2])
                polya_length = float(fields[8])
                qc_tag = fields[9]

                gene = find_gene(chrom, pos)
                if gene and qc_tag == 'PASS':
                    read_class = class_dict.get(read_id, 'unknown')
                    hk_reads.append({
                        'read_id': read_id,
                        'gene': gene,
                        'polya_length': polya_length,
                        'class': read_class,
                        'group': group,
                        'cell_line': parse_cell_line(group)
                    })

        if hk_reads:
            all_hk_data.extend(hk_reads)
            print(f"  Found {len(hk_reads)} housekeeping gene reads")

    if not all_hk_data:
        print("\nNo housekeeping gene reads found!")
        return

    hk_df = pd.DataFrame(all_hk_data)

    print("\n" + "=" * 70)
    print("1. HOUSEKEEPING GENE READ COUNTS")
    print("=" * 70)

    # Count by gene and cell line
    gene_counts = hk_df.groupby(['cell_line', 'gene']).size().unstack(fill_value=0)
    print("\nReads per gene and cell line:")
    print(gene_counts.to_string())

    # =========================================================
    # 2. Compare HeLa vs HeLa-Ars
    # =========================================================
    print("\n" + "=" * 70)
    print("2. HeLa vs HeLa-Ars COMPARISON (Housekeeping Genes)")
    print("=" * 70)

    hela_hk = hk_df[hk_df['cell_line'] == 'HeLa']
    hela_ars_hk = hk_df[hk_df['cell_line'] == 'HeLa-Ars']

    if len(hela_hk) > 0 and len(hela_ars_hk) > 0:
        print(f"\nHeLa (housekeeping genes):")
        print(f"  N = {len(hela_hk)}")
        print(f"  Median poly(A) = {hela_hk['polya_length'].median():.1f}")
        print(f"  Mean poly(A) = {hela_hk['polya_length'].mean():.1f}")

        print(f"\nHeLa-Ars (housekeeping genes):")
        print(f"  N = {len(hela_ars_hk)}")
        print(f"  Median poly(A) = {hela_ars_hk['polya_length'].median():.1f}")
        print(f"  Mean poly(A) = {hela_ars_hk['polya_length'].mean():.1f}")

        # Statistical test
        if len(hela_hk) >= 10 and len(hela_ars_hk) >= 10:
            mw_stat, mw_pval = stats.mannwhitneyu(
                hela_hk['polya_length'], hela_ars_hk['polya_length'],
                alternative='two-sided')
            diff_median = hela_ars_hk['polya_length'].median() - hela_hk['polya_length'].median()

            print(f"\n  Mann-Whitney U: p = {mw_pval:.2e}")
            print(f"  Δ Median = {diff_median:+.1f} (Ars - Ctrl)")

    # =========================================================
    # 3. Decorated rate comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("3. DECORATED RATE COMPARISON")
    print("=" * 70)

    for cl in ['HeLa', 'HeLa-Ars']:
        cl_data = hk_df[hk_df['cell_line'] == cl]
        if len(cl_data) > 0:
            dec_rate = (cl_data['class'] == 'decorated').mean() * 100
            print(f"\n{cl} (housekeeping):")
            print(f"  N = {len(cl_data)}, Decorated rate = {dec_rate:.1f}%")

            # By poly(A) bin
            bins = [0, 50, 100, 150, 200, 500]
            labels = ['0-50', '51-100', '101-150', '151-200', '200+']
            cl_data = cl_data.copy()
            cl_data['polya_bin'] = pd.cut(cl_data['polya_length'], bins=bins, labels=labels)

            print("  By poly(A) bin:")
            for bin_label in labels:
                bin_data = cl_data[cl_data['polya_bin'] == bin_label]
                if len(bin_data) >= 5:
                    bin_dec_rate = (bin_data['class'] == 'decorated').mean() * 100
                    print(f"    {bin_label}: n={len(bin_data)}, decorated={bin_dec_rate:.1f}%")

    # =========================================================
    # 4. Summary comparison with L1
    # =========================================================
    print("\n" + "=" * 70)
    print("4. COMPARISON WITH L1 (from previous analysis)")
    print("=" * 70)

    # Load L1 data for comparison
    l1_summary = {
        'HeLa': {'n': 2319, 'median': 121.4, 'delta': -31.0},
        'HeLa-Ars': {'n': 2844, 'median': 90.3}
    }

    print("\n{:<20} {:>15} {:>15} {:>12}".format(
        "Comparison", "L1 Median", "HK Median", "Δ (Ars-Ctrl)"))
    print("-" * 65)

    if len(hela_hk) > 0 and len(hela_ars_hk) > 0:
        l1_delta = l1_summary['HeLa-Ars']['median'] - l1_summary['HeLa']['median']
        hk_delta = hela_ars_hk['polya_length'].median() - hela_hk['polya_length'].median()

        print("{:<20} {:>15.1f} {:>15.1f} {:>12}".format(
            "HeLa (control)",
            l1_summary['HeLa']['median'],
            hela_hk['polya_length'].median(),
            ""))
        print("{:<20} {:>15.1f} {:>15.1f} {:>12}".format(
            "HeLa-Ars",
            l1_summary['HeLa-Ars']['median'],
            hela_ars_hk['polya_length'].median(),
            ""))
        print("-" * 65)
        print("{:<20} {:>15.1f} {:>15.1f}".format(
            "Δ (Ars - Ctrl)", l1_delta, hk_delta))

    # Save results
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_02_polya")
    hk_df.to_csv(out_dir / "housekeeping_polya.tsv", sep='\t', index=False)

    print(f"\n\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
