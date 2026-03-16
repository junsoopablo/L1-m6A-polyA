#!/usr/bin/env python3
"""
Analyze U-tailing in L1 transcripts.

Focuses only on U-tailing (uridylation) as C/G tailing lacks literature support for L1.
Applies filtering: est_nonA_pos > 30 AND ratio > 0.3 to exclude technical artifacts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import glob

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_02_polya"

# Filter criteria
MIN_POS = 30
MIN_RATIO = 0.3

# Groups
BASE_GROUPS = [
    "HeLa_1", "HeLa_2", "HeLa_3",
    "MCF7_2", "MCF7_3", "MCF7_4",
    "A549_4", "A549_5", "A549_6",
    "K562_4", "K562_5", "K562_6",
    "H9_2", "H9_3", "H9_4",
    "HepG2_5", "HepG2_6",
    "HEYA8_1", "HEYA8_2", "HEYA8_3",
    "Hct116_3", "Hct116_4",
    "Hek293T_3", "Hek293T_4",
    "SHSY5Y_1", "SHSY5Y_2", "SHSY5Y_3",
]

VARIANT_GROUPS = [
    "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3",
    "MCF7-EV_1",
]


def load_l1_reads(group):
    """Load L1 read IDs for a group."""
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        return None
    df = pd.read_csv(l1_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    return set(df['read_id'].values)


def load_nonadenosine_residues(group, is_control=False):
    """Load nonadenosine residues data."""
    if is_control:
        search_dir = RESULTS_GROUP_DIR / group / "i_control"
        pattern = f"*{group}_control_nonadenosine_residues.tsv"
    else:
        search_dir = RESULTS_GROUP_DIR / group / "f_ninetails"
        pattern = f"*{group}_nonadenosine_residues.tsv"

    files = list(search_dir.glob(pattern))
    if not files:
        return None

    # Use the most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    return df


def filter_reliable_u_tails(df):
    """
    Filter for reliable U-tailing events.
    - Only U predictions
    - est_nonA_pos > 30 (exclude technical artifacts near 3'UTR junction)
    - ratio > 0.3 (position should be >30% into the poly(A) tail)
    """
    df = df[df['prediction'] == 'U'].copy()
    df['ratio'] = df['est_nonA_pos'] / df['polya_length']
    df = df[(df['est_nonA_pos'] > MIN_POS) & (df['ratio'] > MIN_RATIO)].copy()
    return df


def get_u_tailing_stats(group, l1_reads=None):
    """Get U-tailing statistics for a group."""
    # Load all nonadenosine residues
    nonA_df = load_nonadenosine_residues(group, is_control=False)
    if nonA_df is None:
        return None

    # If L1 reads provided, filter for L1 only
    if l1_reads is not None:
        nonA_df = nonA_df[nonA_df['readname'].isin(l1_reads)].copy()

    # Get total reads with any non-A (before filtering)
    total_reads_with_nonA = nonA_df['readname'].nunique()

    # Filter for reliable U-tailing
    u_tail_df = filter_reliable_u_tails(nonA_df)

    # Count unique reads with U-tailing
    u_tail_reads = u_tail_df['readname'].nunique()

    return {
        'total_nonA_reads': total_reads_with_nonA,
        'u_tail_reads': u_tail_reads,
        'u_tail_events': len(u_tail_df),
        'u_tail_df': u_tail_df
    }


def main():
    print("=" * 70)
    print("U-tailing Analysis in L1 Transcripts")
    print("=" * 70)
    print(f"\nFilter criteria: est_nonA_pos > {MIN_POS} AND ratio > {MIN_RATIO}")

    # =========================================================================
    # Part 1: U-tailing rates across all groups
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: U-tailing Rates by Group")
    print("=" * 70)

    results = []
    for group in BASE_GROUPS + VARIANT_GROUPS:
        l1_reads = load_l1_reads(group)
        if l1_reads is None:
            continue

        total_l1 = len(l1_reads)
        stats_result = get_u_tailing_stats(group, l1_reads)

        if stats_result is None:
            continue

        u_rate = stats_result['u_tail_reads'] / total_l1 * 100 if total_l1 > 0 else 0

        results.append({
            'group': group,
            'total_l1': total_l1,
            'u_tail_reads': stats_result['u_tail_reads'],
            'u_tail_rate': u_rate,
            'u_tail_events': stats_result['u_tail_events'],
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'Group':<15} {'Total L1':>10} {'U-tail reads':>12} {'U-tail rate':>12} {'U events':>10}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        print(f"{row['group']:<15} {row['total_l1']:>10,} {row['u_tail_reads']:>12,} {row['u_tail_rate']:>11.2f}% {row['u_tail_events']:>10,}")

    # =========================================================================
    # Part 2: HeLa vs HeLa-Ars Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: HeLa vs HeLa-Ars (Arsenic Treatment) Comparison")
    print("=" * 70)

    # Aggregate HeLa
    hela_groups = ["HeLa_1", "HeLa_2", "HeLa_3"]
    hela_ars_groups = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]

    hela_total = 0
    hela_u_reads = 0
    hela_u_events = 0

    for group in hela_groups:
        row = results_df[results_df['group'] == group]
        if len(row) > 0:
            hela_total += row['total_l1'].values[0]
            hela_u_reads += row['u_tail_reads'].values[0]
            hela_u_events += row['u_tail_events'].values[0]

    hela_ars_total = 0
    hela_ars_u_reads = 0
    hela_ars_u_events = 0

    for group in hela_ars_groups:
        row = results_df[results_df['group'] == group]
        if len(row) > 0:
            hela_ars_total += row['total_l1'].values[0]
            hela_ars_u_reads += row['u_tail_reads'].values[0]
            hela_ars_u_events += row['u_tail_events'].values[0]

    hela_rate = hela_u_reads / hela_total * 100 if hela_total > 0 else 0
    hela_ars_rate = hela_ars_u_reads / hela_ars_total * 100 if hela_ars_total > 0 else 0

    print(f"\nHeLa (control):")
    print(f"  Total L1 reads: {hela_total:,}")
    print(f"  U-tailed reads: {hela_u_reads:,}")
    print(f"  U-tailing rate: {hela_rate:.2f}%")
    print(f"  U-tail events:  {hela_u_events:,}")

    print(f"\nHeLa-Ars (arsenic):")
    print(f"  Total L1 reads: {hela_ars_total:,}")
    print(f"  U-tailed reads: {hela_ars_u_reads:,}")
    print(f"  U-tailing rate: {hela_ars_rate:.2f}%")
    print(f"  U-tail events:  {hela_ars_u_events:,}")

    print(f"\nDifference: {hela_ars_rate - hela_rate:+.2f}%")

    # Fisher's exact test
    table = [
        [hela_u_reads, hela_total - hela_u_reads],
        [hela_ars_u_reads, hela_ars_total - hela_ars_u_reads]
    ]
    odds_ratio, p_value = stats.fisher_exact(table)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"Fisher's exact test: OR={odds_ratio:.2f}, p={p_value:.2e} ({sig})")

    # =========================================================================
    # Part 3: Per-replicate comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 3: Per-Replicate Comparison")
    print("=" * 70)

    print(f"\n{'Replicate':<12} {'HeLa U-rate':>12} {'HeLa-Ars U-rate':>16} {'Diff':>10} {'p-value':>12}")
    print("-" * 65)

    for i in range(1, 4):
        hela_row = results_df[results_df['group'] == f"HeLa_{i}"]
        ars_row = results_df[results_df['group'] == f"HeLa-Ars_{i}"]

        if len(hela_row) > 0 and len(ars_row) > 0:
            h_rate = hela_row['u_tail_rate'].values[0]
            a_rate = ars_row['u_tail_rate'].values[0]
            diff = a_rate - h_rate

            # Fisher's test per replicate
            h_total = hela_row['total_l1'].values[0]
            h_u = hela_row['u_tail_reads'].values[0]
            a_total = ars_row['total_l1'].values[0]
            a_u = ars_row['u_tail_reads'].values[0]

            table = [[h_u, h_total - h_u], [a_u, a_total - a_u]]
            _, p = stats.fisher_exact(table)
            p_str = f"{p:.2e}" if p < 0.05 else "ns"

            print(f"Rep {i:<8} {h_rate:>11.2f}% {a_rate:>15.2f}% {diff:>+9.2f}% {p_str:>12}")

    # =========================================================================
    # Part 4: U-tail position analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 4: U-tail Position Distribution")
    print("=" * 70)

    # Collect all U-tail events for HeLa and HeLa-Ars
    hela_u_positions = []
    hela_ars_u_positions = []

    for group in hela_groups:
        l1_reads = load_l1_reads(group)
        if l1_reads is None:
            continue
        stats_result = get_u_tailing_stats(group, l1_reads)
        if stats_result and 'u_tail_df' in stats_result:
            hela_u_positions.extend(stats_result['u_tail_df']['est_nonA_pos'].values)

    for group in hela_ars_groups:
        l1_reads = load_l1_reads(group)
        if l1_reads is None:
            continue
        stats_result = get_u_tailing_stats(group, l1_reads)
        if stats_result and 'u_tail_df' in stats_result:
            hela_ars_u_positions.extend(stats_result['u_tail_df']['est_nonA_pos'].values)

    if hela_u_positions and hela_ars_u_positions:
        hela_u_positions = np.array(hela_u_positions)
        hela_ars_u_positions = np.array(hela_ars_u_positions)

        print(f"\nHeLa U-tail position:")
        print(f"  Mean: {np.mean(hela_u_positions):.1f}")
        print(f"  Median: {np.median(hela_u_positions):.1f}")
        print(f"  Std: {np.std(hela_u_positions):.1f}")

        print(f"\nHeLa-Ars U-tail position:")
        print(f"  Mean: {np.mean(hela_ars_u_positions):.1f}")
        print(f"  Median: {np.median(hela_ars_u_positions):.1f}")
        print(f"  Std: {np.std(hela_ars_u_positions):.1f}")

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(hela_u_positions, hela_ars_u_positions, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\nMann-Whitney U test: p={p:.2e} ({sig})")

    # =========================================================================
    # Part 5: Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'u_tailing_by_group.tsv', sep='\t', index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'u_tailing_by_group.tsv'}")

    # Overall summary across all base groups
    base_results = results_df[results_df['group'].isin(BASE_GROUPS)]
    if len(base_results) > 0:
        total_l1 = base_results['total_l1'].sum()
        total_u_reads = base_results['u_tail_reads'].sum()
        overall_rate = total_u_reads / total_l1 * 100 if total_l1 > 0 else 0

        print(f"\nOverall U-tailing in L1 (base groups):")
        print(f"  Total L1 reads: {total_l1:,}")
        print(f"  U-tailed reads: {total_u_reads:,}")
        print(f"  U-tailing rate: {overall_rate:.2f}%")


if __name__ == "__main__":
    main()
