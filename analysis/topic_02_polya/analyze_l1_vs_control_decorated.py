#!/usr/bin/env python3
"""
Analyze L1 vs Control decorated rates normalized by poly(A) length.

This script compares decorated tail detection rates between L1 transcripts
and control (non-L1) transcripts, binned by poly(A) tail length to control
for the technical artifact where longer tails have higher detection rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_02_polya"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Groups to analyze (base groups only)
GROUPS = [
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

# Poly(A) length bins
POLYA_BINS = [0, 25, 50, 75, 100, 150, 200, 300]
POLYA_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300+"]


def load_l1_data(group):
    """Load L1 summary data for a group."""
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        return None

    df = pd.read_csv(l1_file, sep='\t')
    # Filter for PASS QC only
    df = df[df['qc_tag'] == 'PASS'].copy()
    df['type'] = 'L1'
    df['decorated'] = df['class'] == 'decorated'
    df = df[['read_id', 'polya_length', 'class', 'decorated', 'type']].copy()
    df['group'] = group
    return df


def load_control_data(group):
    """Load control summary data for a group."""
    ctrl_file = RESULTS_GROUP_DIR / group / "i_control" / f"{group}_control_summary.tsv"
    if not ctrl_file.exists():
        return None

    df = pd.read_csv(ctrl_file, sep='\t')
    df['type'] = 'control'
    df['decorated'] = df['class'] == 'decorated'
    df = df[['read_id', 'polya_length', 'class', 'decorated', 'type']].copy()
    df['group'] = group
    return df


def calculate_decorated_rate_by_bin(df, bins, labels):
    """Calculate decorated rate for each poly(A) length bin."""
    df = df.copy()
    # bins includes inf as last element
    full_bins = bins + [np.inf]
    df['polya_bin'] = pd.cut(df['polya_length'], bins=full_bins, labels=labels, right=False)

    result = df.groupby('polya_bin', observed=True).agg(
        total=('decorated', 'count'),
        decorated=('decorated', 'sum')
    ).reset_index()

    result['rate'] = result['decorated'] / result['total'] * 100
    return result


def main():
    print("=" * 70)
    print("L1 vs Control Decorated Rate Analysis (Normalized by Poly(A) Length)")
    print("=" * 70)

    # Load all data
    all_l1 = []
    all_ctrl = []

    for group in GROUPS:
        l1_df = load_l1_data(group)
        ctrl_df = load_control_data(group)

        if l1_df is not None:
            all_l1.append(l1_df)
        if ctrl_df is not None:
            all_ctrl.append(ctrl_df)

    l1_combined = pd.concat(all_l1, ignore_index=True)
    ctrl_combined = pd.concat(all_ctrl, ignore_index=True)

    print(f"\nTotal L1 reads (PASS): {len(l1_combined):,}")
    print(f"Total Control reads: {len(ctrl_combined):,}")

    # Calculate decorated rate by bin for each type
    l1_by_bin = calculate_decorated_rate_by_bin(l1_combined, POLYA_BINS, POLYA_LABELS)
    ctrl_by_bin = calculate_decorated_rate_by_bin(ctrl_combined, POLYA_BINS, POLYA_LABELS)

    print("\n" + "=" * 70)
    print("Decorated Rate by Poly(A) Length Bin")
    print("=" * 70)
    print(f"\n{'Bin':<12} {'L1_n':>8} {'L1_dec':>8} {'L1_rate':>10} {'Ctrl_n':>8} {'Ctrl_dec':>8} {'Ctrl_rate':>10} {'Diff':>10}")
    print("-" * 85)

    comparison_data = []
    for i, label in enumerate(POLYA_LABELS):
        l1_row = l1_by_bin[l1_by_bin['polya_bin'] == label]
        ctrl_row = ctrl_by_bin[ctrl_by_bin['polya_bin'] == label]

        l1_n = l1_row['total'].values[0] if len(l1_row) > 0 else 0
        l1_dec = l1_row['decorated'].values[0] if len(l1_row) > 0 else 0
        l1_rate = l1_row['rate'].values[0] if len(l1_row) > 0 else 0

        ctrl_n = ctrl_row['total'].values[0] if len(ctrl_row) > 0 else 0
        ctrl_dec = ctrl_row['decorated'].values[0] if len(ctrl_row) > 0 else 0
        ctrl_rate = ctrl_row['rate'].values[0] if len(ctrl_row) > 0 else 0

        diff = l1_rate - ctrl_rate

        print(f"{label:<12} {l1_n:>8} {l1_dec:>8} {l1_rate:>9.1f}% {ctrl_n:>8} {ctrl_dec:>8} {ctrl_rate:>9.1f}% {diff:>+9.1f}%")

        comparison_data.append({
            'polya_bin': label,
            'l1_total': l1_n,
            'l1_decorated': l1_dec,
            'l1_rate': l1_rate,
            'ctrl_total': ctrl_n,
            'ctrl_decorated': ctrl_dec,
            'ctrl_rate': ctrl_rate,
            'diff': diff
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Statistical test for each bin
    print("\n" + "=" * 70)
    print("Statistical Tests (Fisher's Exact Test)")
    print("=" * 70)
    print(f"\n{'Bin':<12} {'Odds Ratio':>12} {'p-value':>15} {'Significant':>12}")
    print("-" * 55)

    for _, row in comparison_df.iterrows():
        if row['l1_total'] > 0 and row['ctrl_total'] > 0:
            # 2x2 contingency table
            table = [
                [row['l1_decorated'], row['l1_total'] - row['l1_decorated']],
                [row['ctrl_decorated'], row['ctrl_total'] - row['ctrl_decorated']]
            ]
            odds_ratio, p_value = stats.fisher_exact(table)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{row['polya_bin']:<12} {odds_ratio:>12.2f} {p_value:>15.2e} {sig:>12}")

    # Overall comparison
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    l1_total = len(l1_combined)
    l1_decorated = l1_combined['decorated'].sum()
    l1_rate = l1_decorated / l1_total * 100

    ctrl_total = len(ctrl_combined)
    ctrl_decorated = ctrl_combined['decorated'].sum()
    ctrl_rate = ctrl_decorated / ctrl_total * 100

    print(f"\nL1:      {l1_decorated:,} / {l1_total:,} = {l1_rate:.1f}% decorated")
    print(f"Control: {ctrl_decorated:,} / {ctrl_total:,} = {ctrl_rate:.1f}% decorated")
    print(f"Difference: {l1_rate - ctrl_rate:+.1f}%")

    # Overall Fisher's exact test
    table = [[l1_decorated, l1_total - l1_decorated],
             [ctrl_decorated, ctrl_total - ctrl_decorated]]
    odds_ratio, p_value = stats.fisher_exact(table)
    print(f"\nFisher's Exact Test: OR = {odds_ratio:.2f}, p = {p_value:.2e}")

    # Calculate weighted average difference (accounting for poly(A) distribution)
    print("\n" + "=" * 70)
    print("Weighted Analysis (Adjusting for Poly(A) Length Distribution)")
    print("=" * 70)

    # For each L1 read, calculate expected decorated rate based on control at same poly(A) bin
    l1_combined['polya_bin'] = pd.cut(l1_combined['polya_length'],
                                       bins=POLYA_BINS + [np.inf],
                                       labels=POLYA_LABELS, right=False)

    ctrl_rate_map = dict(zip(comparison_df['polya_bin'], comparison_df['ctrl_rate']))
    l1_combined['expected_rate'] = l1_combined['polya_bin'].astype(str).map(ctrl_rate_map)

    # Calculate expected vs observed
    observed_decorated = l1_combined['decorated'].sum()
    expected_decorated = (l1_combined['expected_rate'].fillna(0) / 100).sum()

    print(f"\nObserved L1 decorated: {observed_decorated:,.0f}")
    print(f"Expected (based on control): {expected_decorated:,.1f}")
    print(f"Observed / Expected ratio: {observed_decorated / expected_decorated:.2f}")

    if observed_decorated > expected_decorated:
        print(f"\n=> L1 has {(observed_decorated / expected_decorated - 1) * 100:.1f}% MORE decorated reads than expected")
    else:
        print(f"\n=> L1 has {(1 - observed_decorated / expected_decorated) * 100:.1f}% FEWER decorated reads than expected")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Decorated rate by poly(A) bin
    ax1 = axes[0]
    x = np.arange(len(POLYA_LABELS))
    width = 0.35

    l1_rates = [comparison_df[comparison_df['polya_bin'] == label]['l1_rate'].values[0]
                if len(comparison_df[comparison_df['polya_bin'] == label]) > 0 else 0
                for label in POLYA_LABELS]
    ctrl_rates = [comparison_df[comparison_df['polya_bin'] == label]['ctrl_rate'].values[0]
                  if len(comparison_df[comparison_df['polya_bin'] == label]) > 0 else 0
                  for label in POLYA_LABELS]

    bars1 = ax1.bar(x - width/2, l1_rates, width, label='L1', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, ctrl_rates, width, label='Control', color='#3498db', alpha=0.8)

    ax1.set_xlabel('Poly(A) Length Bin')
    ax1.set_ylabel('Decorated Rate (%)')
    ax1.set_title('Decorated Rate by Poly(A) Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(POLYA_LABELS, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Difference (L1 - Control)
    ax2 = axes[1]
    diffs = [l1_rates[i] - ctrl_rates[i] for i in range(len(POLYA_LABELS))]
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diffs]
    ax2.bar(x, diffs, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Poly(A) Length Bin')
    ax2.set_ylabel('Difference (L1 - Control) (%)')
    ax2.set_title('L1 vs Control Difference')
    ax2.set_xticks(x)
    ax2.set_xticklabels(POLYA_LABELS, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Poly(A) length distribution
    ax3 = axes[2]
    ax3.hist(l1_combined['polya_length'], bins=50, alpha=0.6, label='L1', color='#e74c3c', density=True)
    ax3.hist(ctrl_combined['polya_length'], bins=50, alpha=0.6, label='Control', color='#3498db', density=True)
    ax3.set_xlabel('Poly(A) Length')
    ax3.set_ylabel('Density')
    ax3.set_title('Poly(A) Length Distribution')
    ax3.legend()
    ax3.set_xlim(0, 400)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'l1_vs_control_decorated_by_polya.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'l1_vs_control_decorated_by_polya.pdf', bbox_inches='tight')
    print(f"\nPlot saved: {OUTPUT_DIR / 'l1_vs_control_decorated_by_polya.png'}")

    # Save comparison data
    comparison_df.to_csv(OUTPUT_DIR / 'l1_vs_control_decorated_comparison.tsv', sep='\t', index=False)
    print(f"Data saved: {OUTPUT_DIR / 'l1_vs_control_decorated_comparison.tsv'}")

    # Per-group analysis
    print("\n" + "=" * 70)
    print("Per-Group Analysis")
    print("=" * 70)

    group_results = []
    for group in GROUPS:
        l1_df = load_l1_data(group)
        ctrl_df = load_control_data(group)

        if l1_df is None or ctrl_df is None:
            continue

        l1_n = len(l1_df)
        l1_dec = l1_df['decorated'].sum()
        ctrl_n = len(ctrl_df)
        ctrl_dec = ctrl_df['decorated'].sum()

        if l1_n > 0 and ctrl_n > 0:
            l1_rate = l1_dec / l1_n * 100
            ctrl_rate = ctrl_dec / ctrl_n * 100

            # Calculate expected based on poly(A) bins
            l1_df['polya_bin'] = pd.cut(l1_df['polya_length'],
                                        bins=POLYA_BINS + [np.inf],
                                        labels=POLYA_LABELS, right=False)
            ctrl_by_bin_group = calculate_decorated_rate_by_bin(ctrl_df, POLYA_BINS, POLYA_LABELS)
            ctrl_rate_map_group = dict(zip(ctrl_by_bin_group['polya_bin'].astype(str), ctrl_by_bin_group['rate']))
            l1_df['expected_rate'] = l1_df['polya_bin'].astype(str).map(ctrl_rate_map_group).fillna(ctrl_rate)

            expected = (l1_df['expected_rate'] / 100).sum()
            observed = l1_df['decorated'].sum()
            oe_ratio = observed / expected if expected > 0 else np.nan

            group_results.append({
                'group': group,
                'l1_n': l1_n,
                'l1_rate': l1_rate,
                'ctrl_n': ctrl_n,
                'ctrl_rate': ctrl_rate,
                'raw_diff': l1_rate - ctrl_rate,
                'expected': expected,
                'observed': observed,
                'oe_ratio': oe_ratio
            })

    group_results_df = pd.DataFrame(group_results)
    print(f"\n{'Group':<12} {'L1_rate':>8} {'Ctrl_rate':>10} {'Raw_Diff':>10} {'O/E_Ratio':>10}")
    print("-" * 55)
    for _, row in group_results_df.iterrows():
        print(f"{row['group']:<12} {row['l1_rate']:>7.1f}% {row['ctrl_rate']:>9.1f}% {row['raw_diff']:>+9.1f}% {row['oe_ratio']:>10.2f}")

    print(f"\nMean O/E Ratio: {group_results_df['oe_ratio'].mean():.2f}")
    print(f"Median O/E Ratio: {group_results_df['oe_ratio'].median():.2f}")

    # Save group results
    group_results_df.to_csv(OUTPUT_DIR / 'l1_vs_control_by_group.tsv', sep='\t', index=False)
    print(f"\nGroup data saved: {OUTPUT_DIR / 'l1_vs_control_by_group.tsv'}")


if __name__ == "__main__":
    main()
