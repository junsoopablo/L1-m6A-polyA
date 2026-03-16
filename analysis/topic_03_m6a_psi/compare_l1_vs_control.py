#!/usr/bin/env python3
"""
Compare m6A and pseudouridine between L1 and Control transcripts.

Uses MAFIA output (MM/ML tags) with per-site probability analysis.
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_03_m6a_psi"

# Probability threshold (0-255 scale)
PROB_THRESHOLD = 128  # 50%

# Groups to analyze
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


def parse_mm_ml_tags(mm_tag, ml_tag):
    """Parse MM and ML tags to extract modification positions and probabilities."""
    result = {'m6A': [], 'psi': []}

    if mm_tag is None or ml_tag is None:
        return result

    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0

    for block in mod_blocks:
        if not block:
            continue

        parts = block.split(',')
        mod_type = parts[0]

        if 'N+17802' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        elif 'N+21891' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        else:
            ml_idx += len(parts) - 1
            continue

        current_pos = 0
        for pos_str in parts[1:]:
            if pos_str:
                delta = int(pos_str)
                current_pos += delta
                if ml_idx < len(ml_tag):
                    prob = ml_tag[ml_idx]
                    result[mod_key].append((current_pos, prob))
                ml_idx += 1

    return result


def analyze_bam_modifications(bam_path, read_ids=None):
    """Extract modification data from MAFIA BAM."""
    records = []

    if not Path(bam_path).exists() or Path(bam_path).stat().st_size == 0:
        return pd.DataFrame()

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                read_id = read.query_name

                if read_ids is not None and read_id not in read_ids:
                    continue

                mm_tag = read.get_tag("MM") if read.has_tag("MM") else None
                ml_tag = read.get_tag("ML") if read.has_tag("ML") else None

                mods = parse_mm_ml_tags(mm_tag, ml_tag)

                m6a_all = mods['m6A']
                m6a_high = [(p, prob) for p, prob in m6a_all if prob >= PROB_THRESHOLD]

                psi_all = mods['psi']
                psi_high = [(p, prob) for p, prob in psi_all if prob >= PROB_THRESHOLD]

                records.append({
                    'read_id': read_id,
                    'm6a_sites_total': len(m6a_all),
                    'm6a_sites_high': len(m6a_high),
                    'm6a_mean_prob': np.mean([prob for _, prob in m6a_all]) if m6a_all else np.nan,
                    'psi_sites_total': len(psi_all),
                    'psi_sites_high': len(psi_high),
                    'psi_mean_prob': np.mean([prob for _, prob in psi_all]) if psi_all else np.nan,
                })

    except Exception as e:
        print(f"  Error reading BAM: {e}")
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_l1_reads(group):
    """Load L1 read IDs."""
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        return None
    df = pd.read_csv(l1_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    return set(df['read_id'].values)


def load_control_reads(group):
    """Load control read IDs."""
    ctrl_file = RESULTS_GROUP_DIR / group / "i_control" / f"{group}_control_summary.tsv"
    if not ctrl_file.exists():
        return None
    df = pd.read_csv(ctrl_file, sep='\t')
    return set(df['read_id'].values)


def main():
    print("=" * 70)
    print("L1 vs Control: m6A and Pseudouridine Comparison")
    print("=" * 70)
    print(f"\nProbability threshold: {PROB_THRESHOLD}/255 ({PROB_THRESHOLD/255*100:.0f}%)")

    l1_results = []
    ctrl_results = []

    for group in BASE_GROUPS:
        print(f"\nProcessing: {group}")

        # L1 data
        l1_read_ids = load_l1_reads(group)
        l1_bam = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.reads.bam"

        if l1_read_ids and l1_bam.exists():
            l1_df = analyze_bam_modifications(l1_bam, l1_read_ids)
            if len(l1_df) > 0:
                l1_results.append({
                    'group': group,
                    'type': 'L1',
                    'total_reads': len(l1_df),
                    'm6a_reads': (l1_df['m6a_sites_high'] > 0).sum(),
                    'm6a_sites': l1_df['m6a_sites_high'].sum(),
                    'psi_reads': (l1_df['psi_sites_high'] > 0).sum(),
                    'psi_sites': l1_df['psi_sites_high'].sum(),
                })
                print(f"  L1: {len(l1_df)} reads")

        # Control data
        ctrl_read_ids = load_control_reads(group)
        ctrl_bam = RESULTS_GROUP_DIR / group / "i_control" / "mafia" / f"{group}.control.mAFiA.reads.bam"

        if ctrl_read_ids and ctrl_bam.exists():
            ctrl_df = analyze_bam_modifications(ctrl_bam, ctrl_read_ids)
            if len(ctrl_df) > 0:
                ctrl_results.append({
                    'group': group,
                    'type': 'Control',
                    'total_reads': len(ctrl_df),
                    'm6a_reads': (ctrl_df['m6a_sites_high'] > 0).sum(),
                    'm6a_sites': ctrl_df['m6a_sites_high'].sum(),
                    'psi_reads': (ctrl_df['psi_sites_high'] > 0).sum(),
                    'psi_sites': ctrl_df['psi_sites_high'].sum(),
                })
                print(f"  Control: {len(ctrl_df)} reads")

    # Combine results
    l1_df = pd.DataFrame(l1_results)
    ctrl_df = pd.DataFrame(ctrl_results)

    if len(l1_df) == 0 or len(ctrl_df) == 0:
        print("\nInsufficient data for comparison")
        return

    # Calculate rates
    l1_df['m6a_rate'] = l1_df['m6a_reads'] / l1_df['total_reads'] * 100
    l1_df['psi_rate'] = l1_df['psi_reads'] / l1_df['total_reads'] * 100
    l1_df['m6a_per_read'] = l1_df['m6a_sites'] / l1_df['total_reads']
    l1_df['psi_per_read'] = l1_df['psi_sites'] / l1_df['total_reads']

    ctrl_df['m6a_rate'] = ctrl_df['m6a_reads'] / ctrl_df['total_reads'] * 100
    ctrl_df['psi_rate'] = ctrl_df['psi_reads'] / ctrl_df['total_reads'] * 100
    ctrl_df['m6a_per_read'] = ctrl_df['m6a_sites'] / ctrl_df['total_reads']
    ctrl_df['psi_per_read'] = ctrl_df['psi_sites'] / ctrl_df['total_reads']

    # =========================================================================
    # Per-group comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Per-Group Comparison")
    print("=" * 70)

    print(f"\n{'Group':<12} {'L1 m6A%':>10} {'Ctrl m6A%':>10} {'L1 psi%':>10} {'Ctrl psi%':>10}")
    print("-" * 55)

    for group in BASE_GROUPS:
        l1_row = l1_df[l1_df['group'] == group]
        ctrl_row = ctrl_df[ctrl_df['group'] == group]

        if len(l1_row) > 0 and len(ctrl_row) > 0:
            print(f"{group:<12} {l1_row['m6a_rate'].values[0]:>9.1f}% {ctrl_row['m6a_rate'].values[0]:>9.1f}% "
                  f"{l1_row['psi_rate'].values[0]:>9.1f}% {ctrl_row['psi_rate'].values[0]:>9.1f}%")

    # =========================================================================
    # Overall comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Overall Comparison (All Groups Combined)")
    print("=" * 70)

    # Aggregate
    l1_total = l1_df['total_reads'].sum()
    l1_m6a = l1_df['m6a_reads'].sum()
    l1_m6a_sites = l1_df['m6a_sites'].sum()
    l1_psi = l1_df['psi_reads'].sum()
    l1_psi_sites = l1_df['psi_sites'].sum()

    ctrl_total = ctrl_df['total_reads'].sum()
    ctrl_m6a = ctrl_df['m6a_reads'].sum()
    ctrl_m6a_sites = ctrl_df['m6a_sites'].sum()
    ctrl_psi = ctrl_df['psi_reads'].sum()
    ctrl_psi_sites = ctrl_df['psi_sites'].sum()

    l1_m6a_rate = l1_m6a / l1_total * 100
    l1_psi_rate = l1_psi / l1_total * 100
    ctrl_m6a_rate = ctrl_m6a / ctrl_total * 100
    ctrl_psi_rate = ctrl_psi / ctrl_total * 100

    print(f"\n{'Metric':<25} {'L1':>15} {'Control':>15} {'Diff':>12}")
    print("-" * 70)
    print(f"{'Total reads':<25} {l1_total:>15,} {ctrl_total:>15,}")
    print(f"{'m6A reads':<25} {l1_m6a:>15,} {ctrl_m6a:>15,}")
    print(f"{'m6A rate':<25} {l1_m6a_rate:>14.2f}% {ctrl_m6a_rate:>14.2f}% {l1_m6a_rate - ctrl_m6a_rate:>+11.2f}%")
    print(f"{'m6A sites/read':<25} {l1_m6a_sites/l1_total:>15.3f} {ctrl_m6a_sites/ctrl_total:>15.3f}")
    print(f"{'psi reads':<25} {l1_psi:>15,} {ctrl_psi:>15,}")
    print(f"{'psi rate':<25} {l1_psi_rate:>14.2f}% {ctrl_psi_rate:>14.2f}% {l1_psi_rate - ctrl_psi_rate:>+11.2f}%")
    print(f"{'psi sites/read':<25} {l1_psi_sites/l1_total:>15.3f} {ctrl_psi_sites/ctrl_total:>15.3f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("Statistical Tests (Fisher's Exact)")
    print("=" * 70)

    # m6A
    table_m6a = [[l1_m6a, l1_total - l1_m6a], [ctrl_m6a, ctrl_total - ctrl_m6a]]
    or_m6a, p_m6a = stats.fisher_exact(table_m6a)
    sig_m6a = "***" if p_m6a < 0.001 else "**" if p_m6a < 0.01 else "*" if p_m6a < 0.05 else "ns"

    print(f"\nm6A: OR={or_m6a:.3f}, p={p_m6a:.2e} ({sig_m6a})")
    if l1_m6a_rate > ctrl_m6a_rate:
        print(f"  → L1 has {(l1_m6a_rate/ctrl_m6a_rate - 1)*100:.1f}% MORE m6A than Control")
    else:
        print(f"  → L1 has {(1 - l1_m6a_rate/ctrl_m6a_rate)*100:.1f}% LESS m6A than Control")

    # psi
    table_psi = [[l1_psi, l1_total - l1_psi], [ctrl_psi, ctrl_total - ctrl_psi]]
    or_psi, p_psi = stats.fisher_exact(table_psi)
    sig_psi = "***" if p_psi < 0.001 else "**" if p_psi < 0.01 else "*" if p_psi < 0.05 else "ns"

    print(f"\npsi: OR={or_psi:.3f}, p={p_psi:.2e} ({sig_psi})")
    if l1_psi_rate > ctrl_psi_rate:
        print(f"  → L1 has {(l1_psi_rate/ctrl_psi_rate - 1)*100:.1f}% MORE psi than Control")
    else:
        print(f"  → L1 has {(1 - l1_psi_rate/ctrl_psi_rate)*100:.1f}% LESS psi than Control")

    # =========================================================================
    # Save results
    # =========================================================================
    combined = pd.concat([l1_df, ctrl_df], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "l1_vs_control_m6a_psi.tsv", sep='\t', index=False)

    summary = pd.DataFrame([
        {'type': 'L1', 'total_reads': l1_total, 'm6a_reads': l1_m6a, 'm6a_rate': l1_m6a_rate,
         'm6a_sites': l1_m6a_sites, 'psi_reads': l1_psi, 'psi_rate': l1_psi_rate, 'psi_sites': l1_psi_sites},
        {'type': 'Control', 'total_reads': ctrl_total, 'm6a_reads': ctrl_m6a, 'm6a_rate': ctrl_m6a_rate,
         'm6a_sites': ctrl_m6a_sites, 'psi_reads': ctrl_psi, 'psi_rate': ctrl_psi_rate, 'psi_sites': ctrl_psi_sites},
    ])
    summary.to_csv(OUTPUT_DIR / "l1_vs_control_m6a_psi_summary.tsv", sep='\t', index=False)

    print(f"\n\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
