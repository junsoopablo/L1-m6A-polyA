#!/usr/bin/env python3
"""
Comprehensive m6A and pseudouridine analysis for L1 transcripts.

Parses MAFIA output (MM/ML tags) to get:
- Per-site modification probability
- Modification density per read
- Position-aware analysis within L1
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from collections import defaultdict
from scipy import stats
import re

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_03_m6a_psi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Probability threshold (0-255 scale, 128 = 50%)
PROB_THRESHOLD = 128  # 50% probability

# Groups to analyze
BASE_GROUPS = [
    "HeLa_1", "HeLa_2", "HeLa_3",
    "MCF7_2", "MCF7_3", "MCF7_4",
    "A549_4", "A549_5", "A549_6",
    "K562_4", "K562_5", "K562_6",
]

VARIANT_GROUPS = [
    "HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3",
]


def parse_mm_ml_tags(mm_tag, ml_tag):
    """
    Parse MM and ML tags to extract modification positions and probabilities.

    MM format: 'N+17802,pos1,pos2,...;N+21891,pos1,pos2,...;'
    - Positions are delta-encoded (offset from previous)
    - N+17802 = m6A
    - N+21891 = psi

    ML format: array of probabilities (0-255) corresponding to MM positions

    Returns: dict with 'm6A' and 'psi' lists of (position, probability) tuples
    """
    result = {'m6A': [], 'psi': []}

    if mm_tag is None or ml_tag is None:
        return result

    # Parse MM tag
    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0

    for block in mod_blocks:
        if not block:
            continue

        parts = block.split(',')
        mod_type = parts[0]

        # Determine modification type
        if 'N+17802' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        elif 'N+21891' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        else:
            # Skip unknown modification types
            ml_idx += len(parts) - 1
            continue

        # Parse positions (delta-encoded)
        current_pos = 0
        for i, pos_str in enumerate(parts[1:]):
            if pos_str:
                delta = int(pos_str)
                current_pos += delta

                # Get probability from ML tag
                if ml_idx < len(ml_tag):
                    prob = ml_tag[ml_idx]
                    result[mod_key].append((current_pos, prob))
                ml_idx += 1

    return result


def analyze_read_modifications(bam_path, l1_read_ids):
    """
    Extract detailed modification data for L1 reads from MAFIA BAM.

    Returns DataFrame with per-read modification summary.
    """
    records = []

    if not Path(bam_path).exists() or Path(bam_path).stat().st_size == 0:
        return pd.DataFrame()

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                read_id = read.query_name

                # Only process L1 reads
                if read_id not in l1_read_ids:
                    continue

                mm_tag = read.get_tag("MM") if read.has_tag("MM") else None
                ml_tag = read.get_tag("ML") if read.has_tag("ML") else None

                mods = parse_mm_ml_tags(mm_tag, ml_tag)

                # m6A stats
                m6a_all = mods['m6A']
                m6a_high = [(p, prob) for p, prob in m6a_all if prob >= PROB_THRESHOLD]

                # psi stats
                psi_all = mods['psi']
                psi_high = [(p, prob) for p, prob in psi_all if prob >= PROB_THRESHOLD]

                # Read length
                read_len = read.query_length if read.query_length else 0

                records.append({
                    'read_id': read_id,
                    'read_length': read_len,
                    'chr': read.reference_name,
                    'start': read.reference_start,
                    'end': read.reference_end,
                    # m6A
                    'm6a_sites_total': len(m6a_all),
                    'm6a_sites_high': len(m6a_high),
                    'm6a_probs': [prob for _, prob in m6a_all] if m6a_all else [],
                    'm6a_mean_prob': np.mean([prob for _, prob in m6a_all]) if m6a_all else np.nan,
                    'm6a_max_prob': max([prob for _, prob in m6a_all]) if m6a_all else 0,
                    'm6a_positions': [p for p, _ in m6a_all] if m6a_all else [],
                    # psi
                    'psi_sites_total': len(psi_all),
                    'psi_sites_high': len(psi_high),
                    'psi_probs': [prob for _, prob in psi_all] if psi_all else [],
                    'psi_mean_prob': np.mean([prob for _, prob in psi_all]) if psi_all else np.nan,
                    'psi_max_prob': max([prob for _, prob in psi_all]) if psi_all else 0,
                    'psi_positions': [p for p, _ in psi_all] if psi_all else [],
                })

    except Exception as e:
        print(f"  Error reading BAM: {e}")
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_l1_reads(group):
    """Load L1 read IDs and metadata for a group."""
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        return None

    df = pd.read_csv(l1_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    return df


def main():
    print("=" * 70)
    print("m6A and Pseudouridine Analysis in L1 Transcripts")
    print("=" * 70)
    print(f"\nProbability threshold: {PROB_THRESHOLD}/255 ({PROB_THRESHOLD/255*100:.0f}%)")

    all_results = []

    for group in BASE_GROUPS + VARIANT_GROUPS:
        print(f"\n{'='*50}")
        print(f"Processing: {group}")
        print(f"{'='*50}")

        # Load L1 reads
        l1_df = load_l1_reads(group)
        if l1_df is None:
            print(f"  [SKIP] No L1 summary found")
            continue

        l1_read_ids = set(l1_df['read_id'].values)
        print(f"  Total L1 reads: {len(l1_read_ids):,}")

        # Load MAFIA BAM
        bam_path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.reads.bam"
        if not bam_path.exists():
            print(f"  [SKIP] No MAFIA BAM found")
            continue

        # Analyze modifications
        mod_df = analyze_read_modifications(bam_path, l1_read_ids)

        if len(mod_df) == 0:
            print(f"  [SKIP] No modification data")
            continue

        print(f"  Reads with MAFIA data: {len(mod_df):,}")

        # Summary statistics
        # m6A
        reads_with_m6a = (mod_df['m6a_sites_high'] > 0).sum()
        total_m6a_sites = mod_df['m6a_sites_high'].sum()
        m6a_rate = reads_with_m6a / len(mod_df) * 100

        # psi
        reads_with_psi = (mod_df['psi_sites_high'] > 0).sum()
        total_psi_sites = mod_df['psi_sites_high'].sum()
        psi_rate = reads_with_psi / len(mod_df) * 100

        print(f"\n  m6A (prob >= {PROB_THRESHOLD}/255):")
        print(f"    Reads with m6A: {reads_with_m6a:,} / {len(mod_df):,} = {m6a_rate:.2f}%")
        print(f"    Total m6A sites: {total_m6a_sites:,}")
        print(f"    Mean sites/read: {mod_df['m6a_sites_high'].mean():.2f}")

        print(f"\n  Pseudouridine (prob >= {PROB_THRESHOLD}/255):")
        print(f"    Reads with psi: {reads_with_psi:,} / {len(mod_df):,} = {psi_rate:.2f}%")
        print(f"    Total psi sites: {total_psi_sites:,}")
        print(f"    Mean sites/read: {mod_df['psi_sites_high'].mean():.2f}")

        # Store results
        all_results.append({
            'group': group,
            'total_l1': len(l1_read_ids),
            'mafia_reads': len(mod_df),
            'm6a_reads': reads_with_m6a,
            'm6a_rate': m6a_rate,
            'm6a_sites': total_m6a_sites,
            'm6a_sites_per_read': mod_df['m6a_sites_high'].mean(),
            'psi_reads': reads_with_psi,
            'psi_rate': psi_rate,
            'psi_sites': total_psi_sites,
            'psi_sites_per_read': mod_df['psi_sites_high'].mean(),
        })

        # Save per-read data
        mod_df['group'] = group
        mod_df.to_csv(OUTPUT_DIR / f"{group}_mafia_per_read.tsv", sep='\t', index=False)

    # =========================================================================
    # Summary across groups
    # =========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "m6a_psi_summary_by_group.tsv", sep='\t', index=False)

    print("\n" + "=" * 70)
    print("Summary by Group")
    print("=" * 70)
    print(f"\n{'Group':<15} {'MAFIA reads':>12} {'m6A rate':>10} {'m6A/read':>10} {'psi rate':>10} {'psi/read':>10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['group']:<15} {row['mafia_reads']:>12,} {row['m6a_rate']:>9.1f}% {row['m6a_sites_per_read']:>10.2f} {row['psi_rate']:>9.1f}% {row['psi_sites_per_read']:>10.2f}")

    # =========================================================================
    # HeLa vs HeLa-Ars Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("HeLa vs HeLa-Ars Comparison")
    print("=" * 70)

    hela_df = results_df[results_df['group'].str.startswith('HeLa_')]
    hela_ars_df = results_df[results_df['group'].str.startswith('HeLa-Ars_')]

    if len(hela_df) > 0 and len(hela_ars_df) > 0:
        # m6A comparison
        hela_m6a = hela_df['m6a_reads'].sum()
        hela_total = hela_df['mafia_reads'].sum()
        hela_m6a_rate = hela_m6a / hela_total * 100

        ars_m6a = hela_ars_df['m6a_reads'].sum()
        ars_total = hela_ars_df['mafia_reads'].sum()
        ars_m6a_rate = ars_m6a / ars_total * 100

        table = [[hela_m6a, hela_total - hela_m6a], [ars_m6a, ars_total - ars_m6a]]
        or_m6a, p_m6a = stats.fisher_exact(table)
        sig_m6a = "***" if p_m6a < 0.001 else "**" if p_m6a < 0.01 else "*" if p_m6a < 0.05 else "ns"

        print(f"\nm6A:")
        print(f"  HeLa:     {hela_m6a:,} / {hela_total:,} = {hela_m6a_rate:.2f}%")
        print(f"  HeLa-Ars: {ars_m6a:,} / {ars_total:,} = {ars_m6a_rate:.2f}%")
        print(f"  Diff: {ars_m6a_rate - hela_m6a_rate:+.2f}%")
        print(f"  Fisher's exact: OR={or_m6a:.2f}, p={p_m6a:.2e} ({sig_m6a})")

        # psi comparison
        hela_psi = hela_df['psi_reads'].sum()
        ars_psi = hela_ars_df['psi_reads'].sum()
        hela_psi_rate = hela_psi / hela_total * 100
        ars_psi_rate = ars_psi / ars_total * 100

        table = [[hela_psi, hela_total - hela_psi], [ars_psi, ars_total - ars_psi]]
        or_psi, p_psi = stats.fisher_exact(table)
        sig_psi = "***" if p_psi < 0.001 else "**" if p_psi < 0.01 else "*" if p_psi < 0.05 else "ns"

        print(f"\nPseudouridine:")
        print(f"  HeLa:     {hela_psi:,} / {hela_total:,} = {hela_psi_rate:.2f}%")
        print(f"  HeLa-Ars: {ars_psi:,} / {ars_total:,} = {ars_psi_rate:.2f}%")
        print(f"  Diff: {ars_psi_rate - hela_psi_rate:+.2f}%")
        print(f"  Fisher's exact: OR={or_psi:.2f}, p={p_psi:.2e} ({sig_psi})")

        # Sites per read comparison
        print(f"\nModification density (sites per read):")
        hela_m6a_density = hela_df['m6a_sites'].sum() / hela_total
        ars_m6a_density = hela_ars_df['m6a_sites'].sum() / ars_total
        hela_psi_density = hela_df['psi_sites'].sum() / hela_total
        ars_psi_density = hela_ars_df['psi_sites'].sum() / ars_total

        print(f"  m6A:  HeLa={hela_m6a_density:.3f}, HeLa-Ars={ars_m6a_density:.3f} (diff: {ars_m6a_density - hela_m6a_density:+.3f})")
        print(f"  psi:  HeLa={hela_psi_density:.3f}, HeLa-Ars={ars_psi_density:.3f} (diff: {ars_psi_density - hela_psi_density:+.3f})")

    print(f"\n\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
