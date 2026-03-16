#!/usr/bin/env python3
"""
Compare m6A/psi between MCF7 L1, MCF7-EV L1, and MCF7 Control,
stratified by L1 age (young vs ancient).

Young: L1HS, L1PA1, L1PA2, L1PA3
Ancient: all other L1 subfamilies
Control: non-L1 transcripts (no age split)
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

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}


def parse_mm_ml_tags(mm_tag, ml_tag):
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


def load_l1_summary(group):
    """Load L1 summary with subfamily info."""
    path = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG_SUBFAMILIES else 'ancient')
    return df


def load_control_read_ids(group):
    ctrl_file = RESULTS_GROUP_DIR / group / "i_control" / f"{group}_control_summary.tsv"
    if not ctrl_file.exists():
        return None
    df = pd.read_csv(ctrl_file, sep='\t')
    return set(df['read_id'].values)


def summarize(df, label):
    n = len(df)
    if n == 0:
        return {'label': label, 'total_reads': 0, 'm6a_reads': 0, 'm6a_rate': np.nan,
                'm6a_sites': 0, 'm6a_per_read': np.nan, 'psi_reads': 0, 'psi_rate': np.nan,
                'psi_sites': 0, 'psi_per_read': np.nan, 'mean_read_length': np.nan}
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
        'mean_read_length': df['read_length'].mean(),
    }


def fisher_test(n1_pos, n1_total, n2_pos, n2_total):
    if n1_total == 0 or n2_total == 0:
        return np.nan, np.nan, 'n/a'
    table = [[n1_pos, n1_total - n1_pos], [n2_pos, n2_total - n2_pos]]
    odds_ratio, p_value = stats.fisher_exact(table)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    return odds_ratio, p_value, sig


def mann_whitney_test(vals1, vals2):
    """Mann-Whitney U test for sites/read."""
    if len(vals1) == 0 or len(vals2) == 0:
        return np.nan, np.nan, 'n/a'
    stat, p_value = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    return stat, p_value, sig


def print_comparison(s1, s2):
    """Print pairwise comparison."""
    for mod_type, mod_label in [('m6a', 'm6A'), ('psi', 'Pseudouridine')]:
        key_reads = f'{mod_type}_reads'
        key_rate = f'{mod_type}_rate'
        key_per_read = f'{mod_type}_per_read'

        r1 = s1[key_rate] if not np.isnan(s1.get(key_rate, np.nan)) else 0
        r2 = s2[key_rate] if not np.isnan(s2.get(key_rate, np.nan)) else 0

        or_val, p_val, sig = fisher_test(
            s1[key_reads], s1['total_reads'],
            s2[key_reads], s2['total_reads'],
        )

        p_str = f"p={p_val:.2e}" if not np.isnan(p_val) else "p=n/a"

        print(f"\n  {mod_label}:")
        print(f"    {s1['label']:<25} {s1[key_reads]:>6,} / {s1['total_reads']:>6,} = {r1:>6.2f}%  ({s1[key_per_read]:.2f} sites/read)")
        print(f"    {s2['label']:<25} {s2[key_reads]:>6,} / {s2['total_reads']:>6,} = {r2:>6.2f}%  ({s2[key_per_read]:.2f} sites/read)")
        print(f"    Diff: {r1 - r2:+.2f}%  OR={or_val:.3f}  {p_str} ({sig})")


def main():
    print("=" * 70)
    print("MCF7 vs MCF7-EV: m6A/psi by L1 Age (Young vs Ancient)")
    print(f"Young: {', '.join(sorted(YOUNG_SUBFAMILIES))}")
    print(f"Probability threshold: {PROB_THRESHOLD}/255 ({PROB_THRESHOLD/255*100:.0f}%)")
    print("=" * 70)

    # =========================================================================
    # 1. Load L1 summaries (for read_id -> age mapping)
    # =========================================================================
    print("\nLoading L1 summaries...")

    mcf7_l1_meta = {}   # read_id -> l1_age
    for g in MCF7_GROUPS:
        df = load_l1_summary(g)
        if df is not None:
            for _, row in df.iterrows():
                mcf7_l1_meta[row['read_id']] = row['l1_age']
            print(f"  {g}: {len(df)} reads (young={sum(df['l1_age']=='young')}, ancient={sum(df['l1_age']=='ancient')})")

    ev_l1_meta = {}
    for g in MCF7_EV_GROUPS:
        df = load_l1_summary(g)
        if df is not None:
            for _, row in df.iterrows():
                ev_l1_meta[row['read_id']] = row['l1_age']
            print(f"  {g}: {len(df)} reads (young={sum(df['l1_age']=='young')}, ancient={sum(df['l1_age']=='ancient')})")

    # =========================================================================
    # 2. Extract modification data from MAFIA BAMs
    # =========================================================================
    print("\nExtracting MAFIA data...")

    # MCF7 L1
    mcf7_l1_dfs = []
    for g in MCF7_GROUPS:
        bam_path = RESULTS_GROUP_DIR / g / "h_mafia" / f"{g}.mAFiA.reads.bam"
        read_ids = set(k for k, v in mcf7_l1_meta.items())
        df = analyze_bam_modifications(bam_path, read_ids)
        if len(df) > 0:
            df['group'] = g
            mcf7_l1_dfs.append(df)
            print(f"  MCF7 L1 {g}: {len(df)} reads")
    mcf7_l1_all = pd.concat(mcf7_l1_dfs, ignore_index=True) if mcf7_l1_dfs else pd.DataFrame()

    # MCF7-EV L1
    ev_l1_dfs = []
    for g in MCF7_EV_GROUPS:
        bam_path = RESULTS_GROUP_DIR / g / "h_mafia" / f"{g}.mAFiA.reads.bam"
        read_ids = set(k for k, v in ev_l1_meta.items())
        df = analyze_bam_modifications(bam_path, read_ids)
        if len(df) > 0:
            df['group'] = g
            ev_l1_dfs.append(df)
            print(f"  MCF7-EV L1 {g}: {len(df)} reads")
    ev_l1_all = pd.concat(ev_l1_dfs, ignore_index=True) if ev_l1_dfs else pd.DataFrame()

    # MCF7 Control
    ctrl_dfs = []
    for g in MCF7_GROUPS:
        read_ids = load_control_read_ids(g)
        bam_path = RESULTS_GROUP_DIR / g / "i_control" / "mafia" / f"{g}.control.mAFiA.reads.bam"
        df = analyze_bam_modifications(bam_path, read_ids)
        if len(df) > 0:
            df['group'] = g
            ctrl_dfs.append(df)
            print(f"  MCF7 Ctrl {g}: {len(df)} reads")
    ctrl_all = pd.concat(ctrl_dfs, ignore_index=True) if ctrl_dfs else pd.DataFrame()

    # =========================================================================
    # 3. Annotate L1 reads with age
    # =========================================================================
    mcf7_l1_all['l1_age'] = mcf7_l1_all['read_id'].map(mcf7_l1_meta)
    ev_l1_all['l1_age'] = ev_l1_all['read_id'].map(ev_l1_meta)

    # Drop reads without age annotation (not in L1 summary)
    mcf7_l1_all = mcf7_l1_all.dropna(subset=['l1_age'])
    ev_l1_all = ev_l1_all.dropna(subset=['l1_age'])

    mcf7_young = mcf7_l1_all[mcf7_l1_all['l1_age'] == 'young']
    mcf7_ancient = mcf7_l1_all[mcf7_l1_all['l1_age'] == 'ancient']
    ev_young = ev_l1_all[ev_l1_all['l1_age'] == 'young']
    ev_ancient = ev_l1_all[ev_l1_all['l1_age'] == 'ancient']

    print(f"\n{'='*70}")
    print("Read counts by category")
    print(f"{'='*70}")
    print(f"  MCF7 L1 young:    {len(mcf7_young):>6,}")
    print(f"  MCF7 L1 ancient:  {len(mcf7_ancient):>6,}")
    print(f"  MCF7-EV L1 young: {len(ev_young):>6,}")
    print(f"  MCF7-EV L1 ancient:{len(ev_ancient):>5,}")
    print(f"  MCF7 Control:     {len(ctrl_all):>6,}")

    # =========================================================================
    # 4. Summary table
    # =========================================================================
    summaries = {
        'MCF7_L1_young': summarize(mcf7_young, 'MCF7_L1_young'),
        'MCF7_L1_ancient': summarize(mcf7_ancient, 'MCF7_L1_ancient'),
        'EV_L1_young': summarize(ev_young, 'MCF7-EV_L1_young'),
        'EV_L1_ancient': summarize(ev_ancient, 'MCF7-EV_L1_ancient'),
        'MCF7_Control': summarize(ctrl_all, 'MCF7_Control'),
    }

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Label':<25} {'Reads':>7} {'m6A%':>8} {'m6A/read':>10} {'psi%':>8} {'psi/read':>10} {'AvgLen':>8}")
    print("-" * 80)
    for s in summaries.values():
        if s['total_reads'] == 0:
            continue
        print(f"{s['label']:<25} {s['total_reads']:>7,} {s['m6a_rate']:>7.2f}% {s['m6a_per_read']:>10.2f} "
              f"{s['psi_rate']:>7.2f}% {s['psi_per_read']:>10.2f} {s['mean_read_length']:>8.0f}")

    # =========================================================================
    # 5. Pairwise comparisons
    # =========================================================================

    comparisons = [
        ("YOUNG: MCF7 L1 vs MCF7 Control",
         "Is young L1 modification different from non-L1?",
         summaries['MCF7_L1_young'], summaries['MCF7_Control']),
        ("YOUNG: MCF7-EV L1 vs MCF7 Control",
         "Is EV young L1 modification different from non-L1?",
         summaries['EV_L1_young'], summaries['MCF7_Control']),
        ("YOUNG: MCF7 L1 vs MCF7-EV L1",
         "Does EV sorting change young L1 modification?",
         summaries['MCF7_L1_young'], summaries['EV_L1_young']),
        ("ANCIENT: MCF7 L1 vs MCF7 Control",
         "Is ancient L1 modification different from non-L1?",
         summaries['MCF7_L1_ancient'], summaries['MCF7_Control']),
        ("ANCIENT: MCF7-EV L1 vs MCF7 Control",
         "Is EV ancient L1 modification different from non-L1?",
         summaries['EV_L1_ancient'], summaries['MCF7_Control']),
        ("ANCIENT: MCF7 L1 vs MCF7-EV L1",
         "Does EV sorting change ancient L1 modification?",
         summaries['MCF7_L1_ancient'], summaries['EV_L1_ancient']),
        ("YOUNG vs ANCIENT within MCF7",
         "Are young and ancient L1 differently modified in cells?",
         summaries['MCF7_L1_young'], summaries['MCF7_L1_ancient']),
        ("YOUNG vs ANCIENT within MCF7-EV",
         "Are young and ancient L1 differently modified in EV?",
         summaries['EV_L1_young'], summaries['EV_L1_ancient']),
    ]

    for title, desc, s1, s2 in comparisons:
        if s1['total_reads'] == 0 or s2['total_reads'] == 0:
            continue
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"  ({desc})")
        print(f"{'='*70}")
        print_comparison(s1, s2)

    # =========================================================================
    # 6. Mann-Whitney for sites/read (density comparison)
    # =========================================================================
    print(f"\n{'='*70}")
    print("Sites/read density tests (Mann-Whitney U)")
    print(f"{'='*70}")

    density_tests = [
        ("YOUNG MCF7 vs EV", mcf7_young, ev_young),
        ("ANCIENT MCF7 vs EV", mcf7_ancient, ev_ancient),
        ("YOUNG vs ANCIENT (MCF7)", mcf7_young, mcf7_ancient),
        ("YOUNG vs ANCIENT (EV)", ev_young, ev_ancient),
        ("MCF7 L1 all vs Control", mcf7_l1_all, ctrl_all),
    ]

    print(f"\n{'Comparison':<30} {'mod':>5} {'median1':>8} {'median2':>8} {'p':>12} {'sig':>4}")
    print("-" * 72)
    for name, df1, df2 in density_tests:
        if len(df1) == 0 or len(df2) == 0:
            continue
        for mod, col in [('m6A', 'm6a_sites_high'), ('psi', 'psi_sites_high')]:
            _, p_val, sig = mann_whitney_test(df1[col].values, df2[col].values)
            p_str = f"{p_val:.2e}" if not np.isnan(p_val) else "n/a"
            print(f"{name:<30} {mod:>5} {df1[col].median():>8.1f} {df2[col].median():>8.1f} {p_str:>12} {sig:>4}")

    # =========================================================================
    # 7. Save results
    # =========================================================================
    # Per-read with age annotation
    mcf7_l1_all['source'] = 'MCF7_L1'
    ev_l1_all['source'] = 'MCF7-EV_L1'
    ctrl_all['source'] = 'MCF7_Control'
    ctrl_all['l1_age'] = 'control'

    all_reads = pd.concat([mcf7_l1_all, ev_l1_all, ctrl_all], ignore_index=True)
    all_reads.to_csv(OUTPUT_DIR / "mcf7_ev_by_age_per_read.tsv", sep='\t', index=False)

    # Summary
    summary_df = pd.DataFrame(list(summaries.values()))
    summary_df.to_csv(OUTPUT_DIR / "mcf7_ev_by_age_summary.tsv", sep='\t', index=False)

    print(f"\n\nResults saved to: {OUTPUT_DIR}")
    print("  - mcf7_ev_by_age_per_read.tsv")
    print("  - mcf7_ev_by_age_summary.tsv")


if __name__ == "__main__":
    main()
