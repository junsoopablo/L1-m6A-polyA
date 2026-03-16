#!/usr/bin/env python3
"""
Analyze relationship between poly(A) length and decorated rate.

Key question: Is the higher decorated rate in longer poly(A) a technical artifact?
- Longer tails have more chances to be detected as modified (probability)
- Need to normalize by poly(A) length to distinguish biological vs technical effect

Analysis:
1. Bin reads by poly(A) length
2. Calculate decorated rate per bin
3. Compare pattern in L1 across conditions (HeLa vs HeLa-Ars)
4. If possible, compare with housekeeping genes as control
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def parse_cell_line(group):
    """Extract cell line from group name"""
    parts = group.replace("-", "_").split("_")
    cell_line = parts[0]
    if len(parts) > 1 and parts[1] in ["Ars", "EV", "Kasumi3", "HFF"]:
        cell_line = f"{parts[0]}-{parts[1]}"
    return cell_line

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Load all L1 data with PASS qc_tag
    all_data = []

    for summary_file in base.glob("*/g_summary/*_L1_summary.tsv"):
        group = summary_file.stem.replace("_L1_summary", "")
        if "THP1" in group:
            continue

        df = pd.read_csv(summary_file, sep='\t')
        df['group'] = group
        df['cell_line'] = parse_cell_line(group)
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    pass_df = full_df[full_df['qc_tag'] == 'PASS'].copy()

    print("=" * 70)
    print("DECORATED RATE vs POLY(A) LENGTH ANALYSIS")
    print("=" * 70)
    print(f"\nTotal PASS L1 reads: {len(pass_df):,}")

    # =========================================================
    # 1. Overall relationship: Bin by poly(A) and calc decorated rate
    # =========================================================
    print("\n" + "=" * 70)
    print("1. DECORATED RATE BY POLY(A) LENGTH BIN (All L1)")
    print("=" * 70)

    # Create bins
    bins = [0, 50, 100, 150, 200, 250, 300, 500]
    labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '300+']
    pass_df['polya_bin'] = pd.cut(pass_df['polya_length'], bins=bins, labels=labels)

    # Calculate decorated rate per bin
    bin_stats = pass_df.groupby('polya_bin', observed=True).apply(
        lambda x: pd.Series({
            'n': len(x),
            'n_decorated': (x['class'] == 'decorated').sum(),
            'decorated_rate': (x['class'] == 'decorated').mean() * 100
        })
    )

    print("\n{:<12} {:>10} {:>12} {:>15}".format(
        "Poly(A) Bin", "N reads", "N decorated", "Decorated Rate"))
    print("-" * 55)
    for bin_label, row in bin_stats.iterrows():
        print("{:<12} {:>10} {:>12} {:>14.1f}%".format(
            bin_label, int(row['n']), int(row['n_decorated']), row['decorated_rate']))

    # =========================================================
    # 2. HeLa vs HeLa-Ars: Decorated rate normalized by poly(A)
    # =========================================================
    print("\n" + "=" * 70)
    print("2. HeLa vs HeLa-Ars: DECORATED RATE BY POLY(A) BIN")
    print("=" * 70)

    for cl in ['HeLa', 'HeLa-Ars']:
        print(f"\n{cl}:")
        cl_data = pass_df[pass_df['cell_line'] == cl]

        cl_bin_stats = cl_data.groupby('polya_bin', observed=True).apply(
            lambda x: pd.Series({
                'n': len(x),
                'decorated_rate': (x['class'] == 'decorated').mean() * 100 if len(x) > 0 else 0
            })
        )

        print("{:<12} {:>8} {:>15}".format("Poly(A) Bin", "N", "Decorated Rate"))
        print("-" * 40)
        for bin_label, row in cl_bin_stats.iterrows():
            print("{:<12} {:>8} {:>14.1f}%".format(
                bin_label, int(row['n']), row['decorated_rate']))

    # =========================================================
    # 3. Compare decorated rate at same poly(A) range
    # =========================================================
    print("\n" + "=" * 70)
    print("3. DECORATED RATE COMPARISON (Same Poly(A) Range)")
    print("=" * 70)

    print("\nComparing decorated rate at each poly(A) bin:")
    print("-" * 60)
    print("{:<12} {:>15} {:>15} {:>10}".format(
        "Poly(A) Bin", "HeLa", "HeLa-Ars", "Difference"))
    print("-" * 60)

    hela_data = pass_df[pass_df['cell_line'] == 'HeLa']
    hela_ars_data = pass_df[pass_df['cell_line'] == 'HeLa-Ars']

    for bin_label in labels:
        hela_bin = hela_data[hela_data['polya_bin'] == bin_label]
        ars_bin = hela_ars_data[hela_ars_data['polya_bin'] == bin_label]

        if len(hela_bin) >= 10 and len(ars_bin) >= 10:
            hela_rate = (hela_bin['class'] == 'decorated').mean() * 100
            ars_rate = (ars_bin['class'] == 'decorated').mean() * 100
            diff = ars_rate - hela_rate
            print("{:<12} {:>14.1f}% {:>14.1f}% {:>+9.1f}%".format(
                bin_label, hela_rate, ars_rate, diff))
        else:
            print("{:<12} {:>15} {:>15} {:>10}".format(
                bin_label, f"n={len(hela_bin)}", f"n={len(ars_bin)}", "N/A"))

    # =========================================================
    # 4. MCF7 vs MCF7-EV comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("4. MCF7 vs MCF7-EV: DECORATED RATE BY POLY(A) BIN")
    print("=" * 70)

    for cl in ['MCF7', 'MCF7-EV']:
        print(f"\n{cl}:")
        cl_data = pass_df[pass_df['cell_line'] == cl]

        cl_bin_stats = cl_data.groupby('polya_bin', observed=True).apply(
            lambda x: pd.Series({
                'n': len(x),
                'decorated_rate': (x['class'] == 'decorated').mean() * 100 if len(x) > 0 else 0
            })
        )

        print("{:<12} {:>8} {:>15}".format("Poly(A) Bin", "N", "Decorated Rate"))
        print("-" * 40)
        for bin_label, row in cl_bin_stats.iterrows():
            print("{:<12} {:>8} {:>14.1f}%".format(
                bin_label, int(row['n']), row['decorated_rate']))

    # =========================================================
    # 5. Statistical summary
    # =========================================================
    print("\n" + "=" * 70)
    print("5. SUMMARY: Is Decorated Rate Simply a Function of Poly(A) Length?")
    print("=" * 70)

    # Calculate overall correlation
    pass_df['is_decorated'] = (pass_df['class'] == 'decorated').astype(int)
    corr, pval = stats.pointbiserialr(pass_df['polya_length'], pass_df['is_decorated'])

    print(f"\nOverall correlation (poly(A) length vs decorated):")
    print(f"  Point-biserial r = {corr:.3f}, p = {pval:.2e}")

    # Per cell line correlation
    print("\nPer cell line:")
    print("-" * 50)
    for cl in ['HeLa', 'HeLa-Ars', 'MCF7', 'MCF7-EV']:
        cl_data = pass_df[pass_df['cell_line'] == cl]
        if len(cl_data) > 30:
            cl_corr, cl_pval = stats.pointbiserialr(
                cl_data['polya_length'], cl_data['is_decorated'])
            print(f"  {cl:<15}: r = {cl_corr:.3f}, p = {cl_pval:.2e}")

    # =========================================================
    # Save results
    # =========================================================
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_02_polya")
    bin_stats.to_csv(out_dir / "decorated_by_polya_bin.tsv", sep='\t')

    print(f"\n\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
