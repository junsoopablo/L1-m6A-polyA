#!/usr/bin/env python3
"""
Compare m6A and pseudouridine between MCF7 L1, MCF7-EV L1, and MCF7 Control.

MCF7-EV = extracellular vesicle RNA from MCF7 cells.
Control = non-L1 transcripts from MCF7 (already processed via control MAFIA pipeline).

Comparisons:
  1. MCF7 L1 vs MCF7 Control  -> L1-specific modification in cells
  2. MCF7-EV L1 vs MCF7 Control -> EV L1 modification vs cell control
  3. MCF7 L1 vs MCF7-EV L1 -> cell vs EV L1 modification
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_03_m6a_psi"

PROB_THRESHOLD = 128  # 50% on 0-255 scale

MCF7_GROUPS = ["MCF7_2", "MCF7_3", "MCF7_4"]
MCF7_EV_GROUPS = ["MCF7-EV_1"]


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
    """Extract per-read modification data from MAFIA BAM."""
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

                read_len = read.query_length if read.query_length else 0

                records.append({
                    'read_id': read_id,
                    'read_length': read_len,
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


def load_l1_read_ids(group):
    """Load L1 PASS read IDs."""
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        return None
    df = pd.read_csv(l1_file, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    return set(df['read_id'].values)


def load_control_read_ids(group):
    """Load control read IDs."""
    ctrl_file = RESULTS_GROUP_DIR / group / "i_control" / f"{group}_control_summary.tsv"
    if not ctrl_file.exists():
        return None
    df = pd.read_csv(ctrl_file, sep='\t')
    return set(df['read_id'].values)


def collect_group_data(groups, data_type, label):
    """Collect per-read modification data across groups.
    data_type: 'l1' or 'control'
    """
    all_dfs = []
    for group in groups:
        print(f"  {group} ({data_type})...")

        if data_type == 'l1':
            read_ids = load_l1_read_ids(group)
            bam_path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.reads.bam"
        else:
            read_ids = load_control_read_ids(group)
            bam_path = RESULTS_GROUP_DIR / group / "i_control" / "mafia" / f"{group}.control.mAFiA.reads.bam"

        if read_ids is None:
            print(f"    [SKIP] No read IDs")
            continue

        df = analyze_bam_modifications(bam_path, read_ids)
        if len(df) == 0:
            print(f"    [SKIP] No modification data")
            continue

        df['group'] = group
        df['label'] = label
        all_dfs.append(df)
        print(f"    {len(df):,} reads")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def summarize(df, label):
    """Compute summary statistics for a set of reads."""
    n = len(df)
    if n == 0:
        return {}
    m6a_reads = (df['m6a_sites_high'] > 0).sum()
    psi_reads = (df['psi_sites_high'] > 0).sum()
    return {
        'label': label,
        'total_reads': n,
        'm6a_reads': int(m6a_reads),
        'm6a_rate': m6a_reads / n * 100,
        'm6a_sites': int(df['m6a_sites_high'].sum()),
        'm6a_per_read': df['m6a_sites_high'].mean(),
        'psi_reads': int(psi_reads),
        'psi_rate': psi_reads / n * 100,
        'psi_sites': int(df['psi_sites_high'].sum()),
        'psi_per_read': df['psi_sites_high'].mean(),
    }


def fisher_test(n1_pos, n1_total, n2_pos, n2_total):
    """Fisher's exact test for two proportions."""
    table = [[n1_pos, n1_total - n1_pos], [n2_pos, n2_total - n2_pos]]
    odds_ratio, p_value = stats.fisher_exact(table)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    return odds_ratio, p_value, sig


def print_comparison(name, s1, s2, mod_type):
    """Print comparison between two groups for a modification type."""
    key_reads = f'{mod_type}_reads'
    key_rate = f'{mod_type}_rate'
    key_per_read = f'{mod_type}_per_read'
    mod_label = 'm6A' if mod_type == 'm6a' else 'Pseudouridine'

    rate1, rate2 = s1[key_rate], s2[key_rate]
    or_val, p_val, sig = fisher_test(
        s1[key_reads], s1['total_reads'],
        s2[key_reads], s2['total_reads'],
    )

    print(f"\n  {mod_label}:")
    print(f"    {s1['label']:<20} {s1[key_reads]:>6,} / {s1['total_reads']:>6,} = {rate1:>6.2f}%  ({s1[key_per_read]:.2f} sites/read)")
    print(f"    {s2['label']:<20} {s2[key_reads]:>6,} / {s2['total_reads']:>6,} = {rate2:>6.2f}%  ({s2[key_per_read]:.2f} sites/read)")
    print(f"    Diff: {rate1 - rate2:+.2f}%  OR={or_val:.3f}  p={p_val:.2e} ({sig})")


def main():
    print("=" * 70)
    print("MCF7 vs MCF7-EV: m6A and Pseudouridine Comparison")
    print(f"Probability threshold: {PROB_THRESHOLD}/255 ({PROB_THRESHOLD/255*100:.0f}%)")
    print("=" * 70)

    # =========================================================================
    # 1. Collect data
    # =========================================================================
    print("\nLoading data...")

    mcf7_l1 = collect_group_data(MCF7_GROUPS, 'l1', 'MCF7_L1')
    mcf7_ev_l1 = collect_group_data(MCF7_EV_GROUPS, 'l1', 'MCF7-EV_L1')
    mcf7_ctrl = collect_group_data(MCF7_GROUPS, 'control', 'MCF7_Control')

    print(f"\nData summary:")
    print(f"  MCF7 L1:      {len(mcf7_l1):>6,} reads")
    print(f"  MCF7-EV L1:   {len(mcf7_ev_l1):>6,} reads")
    print(f"  MCF7 Control: {len(mcf7_ctrl):>6,} reads")

    if len(mcf7_l1) == 0 or len(mcf7_ev_l1) == 0 or len(mcf7_ctrl) == 0:
        print("\nInsufficient data for comparison.")
        return

    # =========================================================================
    # 2. Summary statistics
    # =========================================================================
    s_mcf7_l1 = summarize(mcf7_l1, 'MCF7_L1')
    s_mcf7_ev = summarize(mcf7_ev_l1, 'MCF7-EV_L1')
    s_ctrl = summarize(mcf7_ctrl, 'MCF7_Control')

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{'Label':<20} {'Reads':>7} {'m6A%':>8} {'m6A/read':>10} {'psi%':>8} {'psi/read':>10}")
    print("-" * 65)
    for s in [s_mcf7_l1, s_mcf7_ev, s_ctrl]:
        print(f"{s['label']:<20} {s['total_reads']:>7,} {s['m6a_rate']:>7.2f}% {s['m6a_per_read']:>10.2f} {s['psi_rate']:>7.2f}% {s['psi_per_read']:>10.2f}")

    # =========================================================================
    # 3. Pairwise comparisons
    # =========================================================================

    # --- Comparison 1: MCF7 L1 vs MCF7 Control ---
    print("\n" + "=" * 70)
    print("Comparison 1: MCF7 L1 vs MCF7 Control")
    print("  (Is L1 modification different from non-L1 transcripts?)")
    print("=" * 70)
    print_comparison("MCF7 L1 vs Ctrl", s_mcf7_l1, s_ctrl, 'm6a')
    print_comparison("MCF7 L1 vs Ctrl", s_mcf7_l1, s_ctrl, 'psi')

    # --- Comparison 2: MCF7-EV L1 vs MCF7 Control ---
    print("\n" + "=" * 70)
    print("Comparison 2: MCF7-EV L1 vs MCF7 Control")
    print("  (Is EV L1 modification different from cell non-L1?)")
    print("=" * 70)
    print_comparison("MCF7-EV L1 vs Ctrl", s_mcf7_ev, s_ctrl, 'm6a')
    print_comparison("MCF7-EV L1 vs Ctrl", s_mcf7_ev, s_ctrl, 'psi')

    # --- Comparison 3: MCF7 L1 vs MCF7-EV L1 ---
    print("\n" + "=" * 70)
    print("Comparison 3: MCF7 L1 vs MCF7-EV L1")
    print("  (Does EV sorting change L1 modification pattern?)")
    print("=" * 70)
    print_comparison("MCF7 vs EV L1", s_mcf7_l1, s_mcf7_ev, 'm6a')
    print_comparison("MCF7 vs EV L1", s_mcf7_l1, s_mcf7_ev, 'psi')

    # =========================================================================
    # 4. Per-group breakdown
    # =========================================================================
    print("\n" + "=" * 70)
    print("Per-Group Breakdown")
    print("=" * 70)

    per_group_rows = []
    for grp_name, grp_df in [('MCF7_L1', mcf7_l1), ('MCF7-EV_L1', mcf7_ev_l1), ('MCF7_Control', mcf7_ctrl)]:
        for g, gdf in grp_df.groupby('group'):
            s = summarize(gdf, grp_name)
            s['group'] = g
            per_group_rows.append(s)

    print(f"\n{'Group':<15} {'Label':<15} {'Reads':>7} {'m6A%':>8} {'m6A/read':>10} {'psi%':>8} {'psi/read':>10}")
    print("-" * 75)
    for r in per_group_rows:
        print(f"{r['group']:<15} {r['label']:<15} {r['total_reads']:>7,} {r['m6a_rate']:>7.2f}% {r['m6a_per_read']:>10.2f} {r['psi_rate']:>7.2f}% {r['psi_per_read']:>10.2f}")

    # =========================================================================
    # 5. Save results
    # =========================================================================
    # Per-read data
    all_reads = pd.concat([mcf7_l1, mcf7_ev_l1, mcf7_ctrl], ignore_index=True)
    all_reads.to_csv(OUTPUT_DIR / "mcf7_ev_comparison_per_read.tsv", sep='\t', index=False)

    # Summary
    summary_df = pd.DataFrame([s_mcf7_l1, s_mcf7_ev, s_ctrl])
    summary_df.to_csv(OUTPUT_DIR / "mcf7_ev_comparison_summary.tsv", sep='\t', index=False)

    # Per-group
    per_group_df = pd.DataFrame(per_group_rows)
    per_group_df.to_csv(OUTPUT_DIR / "mcf7_ev_comparison_per_group.tsv", sep='\t', index=False)

    print(f"\n\nResults saved to: {OUTPUT_DIR}")
    print("  - mcf7_ev_comparison_per_read.tsv")
    print("  - mcf7_ev_comparison_summary.tsv")
    print("  - mcf7_ev_comparison_per_group.tsv")


if __name__ == "__main__":
    main()
