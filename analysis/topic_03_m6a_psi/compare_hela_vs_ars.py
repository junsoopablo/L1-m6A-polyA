#!/usr/bin/env python3
"""
HeLa vs HeLa-Ars: m6A/psi comparison (L1 + Control).

Stratified by L1 age (young vs ancient).
All modification density reported as sites/kb (read-length normalized).

Comparisons:
  1. HeLa L1 vs HeLa Control
  2. HeLa-Ars L1 vs HeLa-Ars Control
  3. HeLa L1 vs HeLa-Ars L1  (arsenite effect on L1)
  4. HeLa Control vs HeLa-Ars Control  (arsenite effect on non-L1, baseline)
  5. All above stratified by young/ancient
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_03_m6a_psi"

PROB_THRESHOLD = 128
YOUNG = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

HELA_GROUPS = ["HeLa_1", "HeLa_2", "HeLa_3"]
ARS_GROUPS = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]


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
                current_pos += int(pos_str)
                if ml_idx < len(ml_tag):
                    result[mod_key].append((current_pos, ml_tag[ml_idx]))
                ml_idx += 1
    return result


def analyze_bam(bam_path, read_ids=None):
    records = []
    if not Path(bam_path).exists() or Path(bam_path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for read in bam:
                rid = read.query_name
                if read_ids is not None and rid not in read_ids:
                    continue
                mm = read.get_tag("MM") if read.has_tag("MM") else None
                ml = read.get_tag("ML") if read.has_tag("ML") else None
                mods = parse_mm_ml_tags(mm, ml)
                m6a_high = sum(1 for _, p in mods['m6A'] if p >= PROB_THRESHOLD)
                psi_high = sum(1 for _, p in mods['psi'] if p >= PROB_THRESHOLD)
                rlen = read.query_length or 0
                records.append({
                    'read_id': rid, 'read_length': rlen,
                    'm6a_sites_high': m6a_high, 'psi_sites_high': psi_high,
                    'm6a_sites_total': len(mods['m6A']), 'psi_sites_total': len(mods['psi']),
                })
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_l1_summary(group):
    path = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep='\t')
    df = df[df['qc_tag'] == 'PASS'].copy()
    df['l1_age'] = df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
    return df


def load_control_ids(group):
    path = RESULTS_GROUP_DIR / group / "i_control" / f"{group}_control_summary.tsv"
    if not path.exists():
        return None
    return set(pd.read_csv(path, sep='\t')['read_id'].values)


def collect_l1(groups, label):
    meta = {}  # read_id -> l1_age
    all_dfs = []
    for g in groups:
        summary = load_l1_summary(g)
        if summary is None:
            continue
        for _, row in summary.iterrows():
            meta[row['read_id']] = row['l1_age']
        bam = RESULTS_GROUP_DIR / g / "h_mafia" / f"{g}.mAFiA.reads.bam"
        df = analyze_bam(bam, set(meta.keys()))
        if len(df) > 0:
            df['group'] = g
            all_dfs.append(df)
            print(f"  {g} L1: {len(df):,} reads")
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result['l1_age'] = result['read_id'].map(meta)
    result['source'] = label
    return result.dropna(subset=['l1_age'])


def collect_ctrl(groups, label):
    all_dfs = []
    for g in groups:
        ids = load_control_ids(g)
        if ids is None:
            continue
        bam = RESULTS_GROUP_DIR / g / "i_control" / "mafia" / f"{g}.control.mAFiA.reads.bam"
        df = analyze_bam(bam, ids)
        if len(df) > 0:
            df['group'] = g
            all_dfs.append(df)
            print(f"  {g} Ctrl: {len(df):,} reads")
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result['l1_age'] = 'control'
    result['source'] = label
    return result


def add_per_kb(df):
    df = df.copy()
    df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)
    df['psi_per_kb'] = df['psi_sites_high'] / (df['read_length'] / 1000)
    return df


def summarize(df, label):
    n = len(df)
    if n == 0:
        return {'label': label, 'n': 0}
    m6a_pos = (df['m6a_sites_high'] > 0).sum()
    psi_pos = (df['psi_sites_high'] > 0).sum()
    return {
        'label': label, 'n': n,
        'm6a_rate': m6a_pos / n * 100, 'm6a_per_kb': df['m6a_per_kb'].mean(),
        'psi_rate': psi_pos / n * 100, 'psi_per_kb': df['psi_per_kb'].mean(),
        'avg_len': df['read_length'].mean(),
    }


def mw(v1, v2):
    if len(v1) == 0 or len(v2) == 0:
        return np.nan, 'n/a'
    _, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return p, sig


def fisher(n1, t1, n2, t2):
    if t1 == 0 or t2 == 0:
        return np.nan, np.nan, 'n/a'
    oddsratio, p = stats.fisher_exact([[n1, t1 - n1], [n2, t2 - n2]])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return oddsratio, p, sig


def print_pair(s1, s2):
    for mod, rate_key, kb_key, label in [('m6A', 'm6a_rate', 'm6a_per_kb', 'm6A'),
                                          ('psi', 'psi_rate', 'psi_per_kb', 'Pseudouridine')]:
        r1, r2 = s1[rate_key], s2[rate_key]
        kb1, kb2 = s1[kb_key], s2[kb_key]
        _, fp, fsig = fisher(
            int(r1 / 100 * s1['n']), s1['n'],
            int(r2 / 100 * s2['n']), s2['n'])
        p_kb, sig_kb = mw(
            globals().get('_tmp1_' + mod, np.array([])),
            globals().get('_tmp2_' + mod, np.array([])))
        print(f"\n  {label}:")
        print(f"    {s1['label']:<25} rate={r1:>6.2f}%  /kb={kb1:.3f}  (n={s1['n']:,})")
        print(f"    {s2['label']:<25} rate={r2:>6.2f}%  /kb={kb2:.3f}  (n={s2['n']:,})")
        print(f"    Rate diff: {r1 - r2:+.2f}%  Fisher p={fp:.2e}({fsig})")


def compare(label, df1, s1, df2, s2):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    for mod, col_rate, col_kb, mod_label in [
        ('m6a', 'm6a_rate', 'm6a_per_kb', 'm6A'),
        ('psi', 'psi_rate', 'psi_per_kb', 'Pseudouridine')]:

        r1, r2 = s1[col_rate], s2[col_rate]
        kb1, kb2 = s1[col_kb], s2[col_kb]

        n1_pos = int(r1 / 100 * s1['n'])
        n2_pos = int(r2 / 100 * s2['n'])
        _, fp, fsig = fisher(n1_pos, s1['n'], n2_pos, s2['n'])

        p_kb, sig_kb = mw(df1[f'{mod}_per_kb'].values, df2[f'{mod}_per_kb'].values)
        fp_str = f"{fp:.2e}" if not np.isnan(fp) else "n/a"
        pk_str = f"{p_kb:.2e}" if not np.isnan(p_kb) else "n/a"

        print(f"\n  {mod_label}:")
        print(f"    {s1['label']:<25} rate={r1:>6.2f}%  /kb={kb1:.3f}  (n={s1['n']:,}, avgLen={s1['avg_len']:.0f})")
        print(f"    {s2['label']:<25} rate={r2:>6.2f}%  /kb={kb2:.3f}  (n={s2['n']:,}, avgLen={s2['avg_len']:.0f})")
        print(f"    Rate diff: {r1 - r2:+.2f}%  Fisher p={fp_str}({fsig})  |  /kb MW p={pk_str}({sig_kb})")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 80)
    print("HeLa vs HeLa-Ars: m6A/psi Comparison")
    print(f"Young: {', '.join(sorted(YOUNG))}")
    print(f"Prob threshold: {PROB_THRESHOLD}/255 ({PROB_THRESHOLD/255*100:.0f}%)")
    print("=" * 80)

    # Collect data
    print("\nLoading data...")
    hela_l1 = collect_l1(HELA_GROUPS, 'HeLa_L1')
    ars_l1 = collect_l1(ARS_GROUPS, 'HeLa-Ars_L1')
    hela_ctrl = collect_ctrl(HELA_GROUPS, 'HeLa_Ctrl')
    ars_ctrl = collect_ctrl(ARS_GROUPS, 'HeLa-Ars_Ctrl')

    # Add /kb
    hela_l1 = add_per_kb(hela_l1)
    ars_l1 = add_per_kb(ars_l1)
    hela_ctrl = add_per_kb(hela_ctrl)
    ars_ctrl = add_per_kb(ars_ctrl)

    # Subsets
    hela_young = hela_l1[hela_l1['l1_age'] == 'young']
    hela_ancient = hela_l1[hela_l1['l1_age'] == 'ancient']
    ars_young = ars_l1[ars_l1['l1_age'] == 'young']
    ars_ancient = ars_l1[ars_l1['l1_age'] == 'ancient']

    # Summaries
    sums = {
        'HeLa_L1_all': summarize(hela_l1, 'HeLa_L1_all'),
        'HeLa_L1_young': summarize(hela_young, 'HeLa_L1_young'),
        'HeLa_L1_ancient': summarize(hela_ancient, 'HeLa_L1_ancient'),
        'HeLa-Ars_L1_all': summarize(ars_l1, 'HeLa-Ars_L1_all'),
        'HeLa-Ars_L1_young': summarize(ars_young, 'HeLa-Ars_L1_young'),
        'HeLa-Ars_L1_ancient': summarize(ars_ancient, 'HeLa-Ars_L1_ancient'),
        'HeLa_Ctrl': summarize(hela_ctrl, 'HeLa_Ctrl'),
        'HeLa-Ars_Ctrl': summarize(ars_ctrl, 'HeLa-Ars_Ctrl'),
    }

    # Summary table
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    print(f"{'Label':<25} {'N':>7} {'AvgLen':>7} {'m6A%':>7} {'m6A/kb':>8} {'psi%':>7} {'psi/kb':>8}")
    print("-" * 75)
    for s in sums.values():
        if s['n'] == 0:
            continue
        print(f"{s['label']:<25} {s['n']:>7,} {s['avg_len']:>7.0f} "
              f"{s['m6a_rate']:>6.2f}% {s['m6a_per_kb']:>8.3f} "
              f"{s['psi_rate']:>6.2f}% {s['psi_per_kb']:>8.3f}")

    # =====================================================================
    # Pairwise comparisons
    # =====================================================================

    comparisons = [
        # Arsenite effect on L1
        ("Arsenite effect on L1 (all)", hela_l1, sums['HeLa_L1_all'], ars_l1, sums['HeLa-Ars_L1_all']),
        ("Arsenite effect on L1 (young)", hela_young, sums['HeLa_L1_young'], ars_young, sums['HeLa-Ars_L1_young']),
        ("Arsenite effect on L1 (ancient)", hela_ancient, sums['HeLa_L1_ancient'], ars_ancient, sums['HeLa-Ars_L1_ancient']),

        # Arsenite effect on Control (baseline)
        ("Arsenite effect on Control (non-L1)", hela_ctrl, sums['HeLa_Ctrl'], ars_ctrl, sums['HeLa-Ars_Ctrl']),

        # L1 vs Control within each condition
        ("HeLa: L1 vs Control", hela_l1, sums['HeLa_L1_all'], hela_ctrl, sums['HeLa_Ctrl']),
        ("HeLa-Ars: L1 vs Control", ars_l1, sums['HeLa-Ars_L1_all'], ars_ctrl, sums['HeLa-Ars_Ctrl']),

        # Young vs Ancient within each condition
        ("HeLa: Young vs Ancient L1", hela_young, sums['HeLa_L1_young'], hela_ancient, sums['HeLa_L1_ancient']),
        ("HeLa-Ars: Young vs Ancient L1", ars_young, sums['HeLa-Ars_L1_young'], ars_ancient, sums['HeLa-Ars_L1_ancient']),
    ]

    for label, df1, s1, df2, s2 in comparisons:
        if s1['n'] == 0 or s2['n'] == 0:
            print(f"\n{'='*80}\n{label}\n  SKIPPED (insufficient data)\n")
            continue
        compare(label, df1, s1, df2, s2)

    # =====================================================================
    # Delta-delta: Is arsenite effect L1-specific?
    # =====================================================================
    print(f"\n{'='*80}")
    print("Delta-Delta: Is Arsenite Effect L1-Specific?")
    print("(Compare arsenite-induced change in L1 vs Control)")
    print(f"{'='*80}")

    for mod, col in [('m6A', 'm6a_per_kb'), ('psi', 'psi_per_kb')]:
        # L1: Ars - HeLa
        l1_hela_mean = hela_l1[col].mean()
        l1_ars_mean = ars_l1[col].mean()
        l1_delta = l1_ars_mean - l1_hela_mean

        # Ctrl: Ars - HeLa
        ctrl_hela_mean = hela_ctrl[col].mean()
        ctrl_ars_mean = ars_ctrl[col].mean()
        ctrl_delta = ctrl_ars_mean - ctrl_hela_mean

        dd = l1_delta - ctrl_delta

        print(f"\n  {mod} /kb:")
        print(f"    L1:   HeLa={l1_hela_mean:.3f} -> Ars={l1_ars_mean:.3f}  delta={l1_delta:+.3f}")
        print(f"    Ctrl: HeLa={ctrl_hela_mean:.3f} -> Ars={ctrl_ars_mean:.3f}  delta={ctrl_delta:+.3f}")
        print(f"    Delta-delta (L1 - Ctrl): {dd:+.3f}")
        if abs(ctrl_delta) > 0.001:
            print(f"    L1 delta / Ctrl delta: {l1_delta/ctrl_delta:.2f}")

    # =====================================================================
    # Per-group breakdown
    # =====================================================================
    print(f"\n{'='*80}")
    print("Per-Group Breakdown")
    print(f"{'='*80}")
    print(f"{'Group':<15} {'Type':<10} {'N':>6} {'m6A%':>7} {'m6A/kb':>8} {'psi%':>7} {'psi/kb':>8} {'AvgLen':>7}")
    print("-" * 75)

    for label, gdf in [('L1', hela_l1), ('L1', ars_l1), ('Ctrl', hela_ctrl), ('Ctrl', ars_ctrl)]:
        for g, sub in gdf.groupby('group'):
            s = summarize(sub, label)
            print(f"{g:<15} {label:<10} {s['n']:>6,} {s['m6a_rate']:>6.2f}% {s['m6a_per_kb']:>8.3f} "
                  f"{s['psi_rate']:>6.2f}% {s['psi_per_kb']:>8.3f} {s['avg_len']:>7.0f}")

    # =====================================================================
    # Save
    # =====================================================================
    all_reads = pd.concat([hela_l1, ars_l1, hela_ctrl, ars_ctrl], ignore_index=True)
    all_reads.to_csv(OUTPUT_DIR / "hela_ars_comparison_per_read.tsv", sep='\t', index=False)

    summary_df = pd.DataFrame(list(sums.values()))
    summary_df.to_csv(OUTPUT_DIR / "hela_ars_comparison_summary.tsv", sep='\t', index=False)

    print(f"\nSaved: hela_ars_comparison_per_read.tsv")
    print(f"Saved: hela_ars_comparison_summary.tsv")


if __name__ == "__main__":
    main()
