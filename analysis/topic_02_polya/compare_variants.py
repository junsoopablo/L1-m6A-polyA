#!/usr/bin/env python3
"""Compare L1 decorated rates between base and variant groups."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"

POLYA_BINS = [0, 25, 50, 75, 100, 150, 200, 300]
POLYA_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300+"]

def load_l1_data(group):
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        print(f"  [WARNING] File not found: {l1_file}")
        return None
    df = pd.read_csv(l1_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    df['decorated'] = df['class'] == 'decorated'
    df['polya_bin'] = pd.cut(df['polya_length'], bins=POLYA_BINS + [np.inf], labels=POLYA_LABELS, right=False)
    return df

# Comparisons
comparisons = [
    ("MCF7_2", "MCF7-EV_1", "MCF7 vs MCF7-EV (Cellular vs Exosome)"),
    ("HeLa_1", "HeLa-Ars_1", "HeLa vs HeLa-Ars (Normal vs Arsenic) Rep1"),
    ("HeLa_2", "HeLa-Ars_2", "HeLa vs HeLa-Ars Rep2"),
    ("HeLa_3", "HeLa-Ars_3", "HeLa vs HeLa-Ars Rep3"),
]

print("=" * 80)
print("L1 Decorated Rate Comparison: Base vs Variant Groups")
print("=" * 80)

for base, variant, title in comparisons:
    print(f"\nLoading {base} and {variant}...")
    base_df = load_l1_data(base)
    var_df = load_l1_data(variant)

    if base_df is None or var_df is None:
        print(f"{title}: Data not available")
        continue

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Overall stats
    base_n = len(base_df)
    base_dec = base_df['decorated'].sum()
    base_rate = base_dec / base_n * 100 if base_n > 0 else 0

    var_n = len(var_df)
    var_dec = var_df['decorated'].sum()
    var_rate = var_dec / var_n * 100 if var_n > 0 else 0

    print(f"\nOverall:")
    print(f"  {base}: {base_dec:,} / {base_n:,} = {base_rate:.1f}% decorated")
    print(f"  {variant}: {var_dec:,} / {var_n:,} = {var_rate:.1f}% decorated")
    print(f"  Difference: {var_rate - base_rate:+.1f}%")

    # Fisher's exact test
    table = [[base_dec, base_n - base_dec], [var_dec, var_n - var_dec]]
    odds_ratio, p_value = stats.fisher_exact(table)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"  Fisher's exact: OR={odds_ratio:.2f}, p={p_value:.2e} ({sig})")

    # By poly(A) bin
    print(f"\nBy Poly(A) Length Bin:")
    print(f"{'Bin':<12} {base:>12} {variant:>12} {'Diff':>10} {'p-value':>12}")
    print("-" * 60)

    for label in POLYA_LABELS:
        b = base_df[base_df['polya_bin'] == label]
        v = var_df[var_df['polya_bin'] == label]

        b_n = len(b)
        b_dec = b['decorated'].sum()
        b_rate = b_dec / b_n * 100 if b_n > 0 else 0

        v_n = len(v)
        v_dec = v['decorated'].sum()
        v_rate = v_dec / v_n * 100 if v_n > 0 else 0

        diff = v_rate - b_rate

        if b_n > 0 and v_n > 0:
            table = [[b_dec, b_n - b_dec], [v_dec, v_n - v_dec]]
            _, p = stats.fisher_exact(table)
            p_str = f"{p:.2e}" if p < 0.05 else "ns"
        else:
            p_str = "N/A"

        print(f"{label:<12} {b_rate:>11.1f}% {v_rate:>11.1f}% {diff:>+9.1f}% {p_str:>12}")

# Poly(A) length distribution comparison
print("\n" + "=" * 80)
print("Poly(A) Length Distribution Comparison")
print("=" * 80)

for base, variant, title in comparisons:
    base_df = load_l1_data(base)
    var_df = load_l1_data(variant)

    if base_df is None or var_df is None:
        continue

    base_median = base_df['polya_length'].median()
    var_median = var_df['polya_length'].median()

    # Mann-Whitney U test
    stat, p = stats.mannwhitneyu(base_df['polya_length'], var_df['polya_length'], alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"\n{title}:")
    print(f"  {base} median poly(A): {base_median:.1f}")
    print(f"  {variant} median poly(A): {var_median:.1f}")
    print(f"  Difference: {var_median - base_median:+.1f}")
    print(f"  Mann-Whitney U p={p:.2e} ({sig})")
