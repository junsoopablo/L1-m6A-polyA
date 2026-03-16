#!/usr/bin/env python3
"""
Topic 2: Poly(A) Tail Length Analysis

1. Cell line landscape of L1 poly(A) tail distribution
2. HeLa vs HeLa-Ars comparison (arsenite effect)
3. Locus-specific poly(A) patterns
4. Young vs Ancient L1 poly(A) comparison
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

def categorize_l1(gene_id):
    """Categorize L1 as Young or Ancient"""
    if gene_id.startswith('L1HS'):
        return 'young', 'L1HS'
    elif gene_id.startswith('L1PA'):
        rest = gene_id[4:].split('_')[0]
        if rest.isdigit():
            num = int(rest)
            if num <= 4:
                return 'young', f'L1PA1-4'
            else:
                return 'ancient', 'L1PA5+'
        return 'ancient', 'L1PA_other'
    elif gene_id.startswith('L1M'):
        return 'ancient', 'L1M'
    elif gene_id.startswith('HAL1'):
        return 'ancient', 'HAL1'
    return 'ancient', 'Other'

def main():
    base = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group")

    # Load all data with PASS qc_tag only
    all_data = []

    for summary_file in base.glob("*/g_summary/*_L1_summary.tsv"):
        group = summary_file.stem.replace("_L1_summary", "")
        if "THP1" in group:
            continue

        df = pd.read_csv(summary_file, sep='\t')
        df['group'] = group
        df['cell_line'] = parse_cell_line(group)
        df[['age_category', 'subfamily_cat']] = df['gene_id'].apply(
            lambda x: pd.Series(categorize_l1(x))
        )
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # Filter for PASS only (valid poly(A) measurements)
    pass_df = full_df[full_df['qc_tag'] == 'PASS'].copy()

    print("=" * 70)
    print("TOPIC 2: POLY(A) TAIL LENGTH ANALYSIS")
    print("=" * 70)
    print(f"\nTotal L1 reads: {len(full_df):,}")
    print(f"PASS reads (valid poly(A)): {len(pass_df):,} ({len(pass_df)/len(full_df)*100:.1f}%)")

    # =========================================================
    # 1. Cell Line Landscape
    # =========================================================
    print("\n" + "=" * 70)
    print("1. POLY(A) LENGTH BY CELL LINE")
    print("=" * 70)

    cellline_stats = pass_df.groupby('cell_line')['polya_length'].agg([
        'count', 'mean', 'median', 'std',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75)
    ])
    cellline_stats.columns = ['n', 'mean', 'median', 'std', 'Q1', 'Q3']
    cellline_stats = cellline_stats.sort_values('median', ascending=False)

    print("\n{:<15} {:>8} {:>10} {:>10} {:>10} {:>15}".format(
        "Cell Line", "N", "Mean", "Median", "Std", "IQR (Q1-Q3)"))
    print("-" * 70)
    for cl, row in cellline_stats.iterrows():
        print("{:<15} {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>7.1f}-{:.1f}".format(
            cl, int(row['n']), row['mean'], row['median'], row['std'],
            row['Q1'], row['Q3']))

    # =========================================================
    # 2. HeLa vs HeLa-Ars Comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("2. HeLa vs HeLa-Ars (Arsenite 60min) COMPARISON")
    print("=" * 70)

    hela_ctrl = pass_df[pass_df['cell_line'] == 'HeLa']['polya_length']
    hela_ars = pass_df[pass_df['cell_line'] == 'HeLa-Ars']['polya_length']

    print(f"\nHeLa (control):")
    print(f"  N = {len(hela_ctrl):,}")
    print(f"  Mean = {hela_ctrl.mean():.1f}, Median = {hela_ctrl.median():.1f}")
    print(f"  Q1-Q3 = {np.percentile(hela_ctrl, 25):.1f} - {np.percentile(hela_ctrl, 75):.1f}")

    print(f"\nHeLa-Ars (arsenite):")
    print(f"  N = {len(hela_ars):,}")
    print(f"  Mean = {hela_ars.mean():.1f}, Median = {hela_ars.median():.1f}")
    print(f"  Q1-Q3 = {np.percentile(hela_ars, 25):.1f} - {np.percentile(hela_ars, 75):.1f}")

    # Statistical test
    ks_stat, ks_pval = stats.ks_2samp(hela_ctrl, hela_ars)
    mw_stat, mw_pval = stats.mannwhitneyu(hela_ctrl, hela_ars, alternative='two-sided')

    print(f"\nStatistical Tests:")
    print(f"  KS test: D = {ks_stat:.3f}, p = {ks_pval:.2e}")
    print(f"  Mann-Whitney U: p = {mw_pval:.2e}")

    # Effect size
    diff_median = hela_ars.median() - hela_ctrl.median()
    print(f"\n  Δ Median = {diff_median:+.1f} (Ars - Ctrl)")

    # =========================================================
    # 3. Per-replicate consistency
    # =========================================================
    print("\n" + "=" * 70)
    print("3. REPLICATE CONSISTENCY (HeLa groups)")
    print("=" * 70)

    for cl in ['HeLa', 'HeLa-Ars']:
        print(f"\n{cl}:")
        cl_data = pass_df[pass_df['cell_line'] == cl]
        for group in sorted(cl_data['group'].unique()):
            g_data = cl_data[cl_data['group'] == group]['polya_length']
            print(f"  {group}: N={len(g_data):,}, median={g_data.median():.1f}, mean={g_data.mean():.1f}")

    # =========================================================
    # 4. Young vs Ancient L1 Poly(A)
    # =========================================================
    print("\n" + "=" * 70)
    print("4. YOUNG vs ANCIENT L1 POLY(A) LENGTH")
    print("=" * 70)

    young_polya = pass_df[pass_df['age_category'] == 'young']['polya_length']
    ancient_polya = pass_df[pass_df['age_category'] == 'ancient']['polya_length']

    print(f"\nYoung L1 (L1HS, L1PA1-4):")
    print(f"  N = {len(young_polya):,}")
    print(f"  Median = {young_polya.median():.1f}, Mean = {young_polya.mean():.1f}")

    print(f"\nAncient L1:")
    print(f"  N = {len(ancient_polya):,}")
    print(f"  Median = {ancient_polya.median():.1f}, Mean = {ancient_polya.mean():.1f}")

    mw_stat2, mw_pval2 = stats.mannwhitneyu(young_polya, ancient_polya, alternative='two-sided')
    print(f"\n  Mann-Whitney U: p = {mw_pval2:.2e}")
    print(f"  Δ Median = {young_polya.median() - ancient_polya.median():+.1f} (Young - Ancient)")

    # =========================================================
    # 5. Top Hotspot Poly(A) Analysis
    # =========================================================
    print("\n" + "=" * 70)
    print("5. POLY(A) BY TOP HOTSPOTS")
    print("=" * 70)

    hotspot_polya = pass_df.groupby('transcript_id').agg({
        'polya_length': ['count', 'median', 'mean', 'std'],
        'gene_id': 'first'
    })
    hotspot_polya.columns = ['n', 'median', 'mean', 'std', 'subfamily']
    hotspot_polya = hotspot_polya[hotspot_polya['n'] >= 20]  # Minimum reads
    hotspot_polya = hotspot_polya.sort_values('n', ascending=False)

    print("\nTop 15 hotspots by read count (n >= 20):")
    print("-" * 75)
    print("{:<25} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "Locus", "N", "Median", "Mean", "Std", "Subfamily"))
    print("-" * 75)
    for tid, row in hotspot_polya.head(15).iterrows():
        print("{:<25} {:>6} {:>10.1f} {:>10.1f} {:>10.1f} {:>10}".format(
            tid[:24], int(row['n']), row['median'], row['mean'], row['std'], row['subfamily'][:10]))

    # =========================================================
    # 6. Poly(A) by ninetails class
    # =========================================================
    print("\n" + "=" * 70)
    print("6. POLY(A) BY NINETAILS CLASS")
    print("=" * 70)

    class_polya = pass_df.groupby('class')['polya_length'].agg(['count', 'median', 'mean'])
    class_polya = class_polya.sort_values('count', ascending=False)

    print("\n{:<15} {:>10} {:>12} {:>12}".format("Class", "Count", "Median", "Mean"))
    print("-" * 55)
    for cls, row in class_polya.iterrows():
        print("{:<15} {:>10} {:>12.1f} {:>12.1f}".format(cls, int(row['count']), row['median'], row['mean']))

    # =========================================================
    # 7. MCF7 vs MCF7-EV (Cellular vs Exosome) Comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("7. MCF7 vs MCF7-EV (Cellular vs Exosome) COMPARISON")
    print("=" * 70)

    mcf7_cell = pass_df[pass_df['cell_line'] == 'MCF7']['polya_length']
    mcf7_ev = pass_df[pass_df['cell_line'] == 'MCF7-EV']['polya_length']

    print(f"\nMCF7 (Cellular RNA):")
    print(f"  N = {len(mcf7_cell):,}")
    print(f"  Mean = {mcf7_cell.mean():.1f}, Median = {mcf7_cell.median():.1f}")
    print(f"  Q1-Q3 = {np.percentile(mcf7_cell, 25):.1f} - {np.percentile(mcf7_cell, 75):.1f}")

    print(f"\nMCF7-EV (Exosome RNA):")
    print(f"  N = {len(mcf7_ev):,}")
    print(f"  Mean = {mcf7_ev.mean():.1f}, Median = {mcf7_ev.median():.1f}")
    print(f"  Q1-Q3 = {np.percentile(mcf7_ev, 25):.1f} - {np.percentile(mcf7_ev, 75):.1f}")

    # Statistical test
    ks_stat_mcf7, ks_pval_mcf7 = stats.ks_2samp(mcf7_cell, mcf7_ev)
    mw_stat_mcf7, mw_pval_mcf7 = stats.mannwhitneyu(mcf7_cell, mcf7_ev, alternative='two-sided')

    print(f"\nStatistical Tests:")
    print(f"  KS test: D = {ks_stat_mcf7:.3f}, p = {ks_pval_mcf7:.2e}")
    print(f"  Mann-Whitney U: p = {mw_pval_mcf7:.2e}")

    # Effect size
    diff_median_mcf7 = mcf7_ev.median() - mcf7_cell.median()
    print(f"\n  Δ Median = {diff_median_mcf7:+.1f} (EV - Cellular)")

    # Compare by L1 age category
    print("\n  By L1 Age Category:")
    for age in ['young', 'ancient']:
        mcf7_cell_age = pass_df[(pass_df['cell_line'] == 'MCF7') & (pass_df['age_category'] == age)]['polya_length']
        mcf7_ev_age = pass_df[(pass_df['cell_line'] == 'MCF7-EV') & (pass_df['age_category'] == age)]['polya_length']
        if len(mcf7_cell_age) > 0 and len(mcf7_ev_age) > 0:
            _, mw_p = stats.mannwhitneyu(mcf7_cell_age, mcf7_ev_age, alternative='two-sided')
            print(f"    {age.capitalize():8} L1: Cell={mcf7_cell_age.median():.1f} (n={len(mcf7_cell_age)}), "
                  f"EV={mcf7_ev_age.median():.1f} (n={len(mcf7_ev_age)}), Δ={mcf7_ev_age.median()-mcf7_cell_age.median():+.1f}, p={mw_p:.2e}")

    # =========================================================
    # Save results
    # =========================================================
    out_dir = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_02_polya")

    cellline_stats.to_csv(out_dir / "polya_by_cellline.tsv", sep='\t')
    hotspot_polya.to_csv(out_dir / "polya_by_hotspot.tsv", sep='\t')

    # Save HeLa comparison data for plotting
    hela_comparison = pd.DataFrame({
        'cell_line': ['HeLa'] * len(hela_ctrl) + ['HeLa-Ars'] * len(hela_ars),
        'polya_length': list(hela_ctrl) + list(hela_ars)
    })
    hela_comparison.to_csv(out_dir / "hela_vs_ars_polya.tsv", sep='\t', index=False)

    # Save MCF7 comparison data for plotting
    mcf7_comparison = pd.DataFrame({
        'cell_line': ['MCF7'] * len(mcf7_cell) + ['MCF7-EV'] * len(mcf7_ev),
        'polya_length': list(mcf7_cell) + list(mcf7_ev)
    })
    mcf7_comparison.to_csv(out_dir / "mcf7_vs_ev_polya.tsv", sep='\t', index=False)

    print(f"\n\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
