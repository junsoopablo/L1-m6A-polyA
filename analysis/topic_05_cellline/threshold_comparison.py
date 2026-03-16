#!/usr/bin/env python3
"""
Compare m6A threshold 0.5 vs 0.8 on L1 key metrics:
1. L1 vs Ctrl m6A/kb enrichment
2. Young vs Ancient m6A/kb
3. m6A-poly(A) dose-response (arsenite)

Parses MAFIA BAMs directly at multiple thresholds.
"""
import os
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from collections import defaultdict

PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
TOPIC = f'{PROJECT}/analysis/01_exploration/topic_05_cellline'

# Thresholds to compare (ML 0-255 scale)
THRESHOLDS = {
    '0.50': 128,
    '0.60': 153,
    '0.70': 178,
    '0.80': 204,
    '0.90': 229,
}

GROUPS = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

# L1 subfamily classification
YOUNG_SUBFAMS = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}


def parse_bam_multi_threshold(bam_path, thresholds):
    """Parse MAFIA BAM, count m6A at multiple thresholds simultaneously.

    Returns list of dicts: {read_id, read_length, m6a_N_128, m6a_N_204, ...}
    """
    if not os.path.exists(bam_path):
        print(f"    SKIP: {bam_path}")
        return []

    bam = pysam.AlignmentFile(bam_path, 'rb')
    results = []

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        mm_tag = None
        ml_tag = None
        for tag_name in ['MM', 'Mm']:
            if read.has_tag(tag_name):
                mm_tag = read.get_tag(tag_name)
                break
        for tag_name in ['ML', 'Ml']:
            if read.has_tag(tag_name):
                ml_tag = read.get_tag(tag_name)
                break

        read_len = read.query_alignment_length
        if read_len is None or read_len < 100:
            continue

        read_id = read.query_name
        row = {'read_id': read_id, 'read_length': read_len}

        if mm_tag is None or ml_tag is None:
            for thr_name, thr_val in thresholds.items():
                row[f'm6a_{thr_name}'] = 0
            results.append(row)
            continue

        # Extract all m6A ML values
        ml_values = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        ml_offset = 0
        m6a_mls = []

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(',')
            header = parts[0]
            skip_counts = [int(x) for x in parts[1:] if x]
            n_sites = len(skip_counts)
            if n_sites == 0:
                continue

            is_m6a = '21891' in header
            site_mls = ml_values[ml_offset:ml_offset + n_sites]
            ml_offset += n_sites

            if is_m6a:
                m6a_mls.extend(site_mls)

        # Count at each threshold
        for thr_name, thr_val in thresholds.items():
            row[f'm6a_{thr_name}'] = sum(1 for v in m6a_mls if v > thr_val)

        results.append(row)

    bam.close()
    return results


def load_l1_summary(group):
    """Load L1 summary to get subfamily and poly(A) info."""
    path = f'{PROJECT}/results_group/{group}/g_summary/{group}_L1_summary.tsv'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t')
    return df


# ══════════════════════════════════════════════════════════════
print("Threshold comparison: 0.5 vs 0.8 for L1 metrics\n")

all_l1 = []
all_ctrl = []

for condition, reps in GROUPS.items():
    for group in reps:
        print(f"=== {group} ===")

        # L1 BAM
        l1_bam = f'{PROJECT}/results_group/{group}/h_mafia/{group}.mAFiA.reads.bam'
        print(f"  L1: ", end='', flush=True)
        l1_reads = parse_bam_multi_threshold(l1_bam, THRESHOLDS)
        print(f"{len(l1_reads):,} reads")

        # Ctrl BAM
        ctrl_bam = f'{PROJECT}/results_group/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam'
        print(f"  Ctrl: ", end='', flush=True)
        ctrl_reads = parse_bam_multi_threshold(ctrl_bam, THRESHOLDS)
        print(f"{len(ctrl_reads):,} reads")

        # L1 summary for subfamily + poly(A)
        summary = load_l1_summary(group)

        # Merge L1 reads with summary (for subfamily, poly(A))
        l1_df = pd.DataFrame(l1_reads)
        l1_df['group'] = group
        l1_df['condition'] = condition
        l1_df['type'] = 'L1'

        if not summary.empty and 'read_id' in summary.columns:
            # Get subfamily and poly(A) from summary
            keep_cols = ['read_id']
            if 'subfamily' in summary.columns:
                keep_cols.append('subfamily')
            if 'polya_length' in summary.columns:
                keep_cols.append('polya_length')
            elif 'polya_tail_length' in summary.columns:
                keep_cols.append('polya_tail_length')
            merge_df = summary[keep_cols].drop_duplicates('read_id')
            l1_df = l1_df.merge(merge_df, on='read_id', how='left')

        ctrl_df = pd.DataFrame(ctrl_reads)
        ctrl_df['group'] = group
        ctrl_df['condition'] = condition
        ctrl_df['type'] = 'Ctrl'

        all_l1.append(l1_df)
        all_ctrl.append(ctrl_df)

df_l1 = pd.concat(all_l1, ignore_index=True)
df_ctrl = pd.concat(all_ctrl, ignore_index=True)

# ── Analysis 1: L1 vs Ctrl enrichment at each threshold ──
print(f"\n{'='*70}")
print("1. L1 vs Ctrl m6A/kb enrichment")
print(f"{'='*70}")

for cond in ['HeLa', 'HeLa-Ars']:
    print(f"\n  --- {cond} ---")
    l1_sub = df_l1[df_l1['condition'] == cond]
    ctrl_sub = df_ctrl[df_ctrl['condition'] == cond]

    print(f"  {'Threshold':>10s}  {'L1 /kb':>8s}  {'Ctrl /kb':>9s}  {'Ratio':>7s}  {'MWU P':>12s}")
    for thr_name in THRESHOLDS:
        col = f'm6a_{thr_name}'
        l1_dens = l1_sub[col] / (l1_sub['read_length'] / 1000.0)
        ctrl_dens = ctrl_sub[col] / (ctrl_sub['read_length'] / 1000.0)
        l1_mean = l1_dens.mean()
        ctrl_mean = ctrl_dens.mean()
        ratio = l1_mean / ctrl_mean if ctrl_mean > 0 else float('nan')
        _, p = stats.mannwhitneyu(l1_dens, ctrl_dens, alternative='two-sided')
        print(f"  {thr_name:>10s}  {l1_mean:>8.3f}  {ctrl_mean:>9.3f}  {ratio:>7.3f}  {p:>12.2e}")

# ── Analysis 2: Young vs Ancient at each threshold ──
print(f"\n{'='*70}")
print("2. Young vs Ancient m6A/kb")
print(f"{'='*70}")

if 'subfamily' in df_l1.columns:
    df_l1['age'] = df_l1['subfamily'].apply(
        lambda x: 'Young' if x in YOUNG_SUBFAMS else 'Ancient' if pd.notna(x) else None)

    for cond in ['HeLa', 'HeLa-Ars']:
        print(f"\n  --- {cond} ---")
        sub = df_l1[(df_l1['condition'] == cond) & df_l1['age'].notna()]

        print(f"  {'Threshold':>10s}  {'Young/kb':>9s}  {'Ancient/kb':>11s}  {'Y/A ratio':>10s}")
        for thr_name in THRESHOLDS:
            col = f'm6a_{thr_name}'
            young = sub[sub['age'] == 'Young']
            ancient = sub[sub['age'] == 'Ancient']
            y_dens = (young[col] / (young['read_length'] / 1000.0)).mean()
            a_dens = (ancient[col] / (ancient['read_length'] / 1000.0)).mean()
            ratio = y_dens / a_dens if a_dens > 0 else float('nan')
            print(f"  {thr_name:>10s}  {y_dens:>9.3f}  {a_dens:>11.3f}  {ratio:>10.3f}")

# ── Analysis 3: m6A-poly(A) dose-response (Arsenite) ──
print(f"\n{'='*70}")
print("3. m6A-poly(A) dose-response (HeLa vs HeLa-Ars)")
print(f"{'='*70}")

# Find poly(A) column
polya_col = None
for c in ['polya_length', 'polya_tail_length']:
    if c in df_l1.columns:
        polya_col = c
        break

if polya_col:
    # Compare HeLa vs HeLa-Ars: quartile analysis
    for thr_name in THRESHOLDS:
        col = f'm6a_{thr_name}'
        print(f"\n  --- Threshold {thr_name} ---")

        # HeLa-Ars only for dose-response
        ars = df_l1[(df_l1['condition'] == 'HeLa-Ars') & df_l1[polya_col].notna()].copy()
        if len(ars) < 50:
            print("    Too few reads with poly(A)")
            continue

        ars['m6a_kb'] = ars[col] / (ars['read_length'] / 1000.0)

        # Quartile analysis
        try:
            ars['m6a_q'] = pd.qcut(ars['m6a_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                   duplicates='drop')
        except ValueError:
            # Many ties (zeros) — use rank-based quartiles
            ars['_rank'] = ars['m6a_kb'].rank(method='first')
            ars['m6a_q'] = pd.qcut(ars['_rank'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        print(f"    {'Quartile':>10s}  {'m6A/kb':>8s}  {'polyA':>8s}  {'N':>6s}")
        for q in sorted(ars['m6a_q'].unique()):
            qsub = ars[ars['m6a_q'] == q]
            print(f"    {q:>10s}  {qsub['m6a_kb'].mean():>8.2f}  {qsub[polya_col].mean():>8.1f}  {len(qsub):>6d}")

        # Spearman correlation
        rho, p = stats.spearmanr(ars['m6a_kb'], ars[polya_col])
        print(f"    Spearman: rho={rho:.4f}, P={p:.2e}")

        # Also HeLa (baseline)
        hela = df_l1[(df_l1['condition'] == 'HeLa') & df_l1[polya_col].notna()].copy()
        if len(hela) > 50:
            hela['m6a_kb'] = hela[col] / (hela['read_length'] / 1000.0)
            rho_h, p_h = stats.spearmanr(hela['m6a_kb'], hela[polya_col])
            print(f"    HeLa (baseline): rho={rho_h:.4f}, P={p_h:.2e}")
else:
    print("  No poly(A) column found in L1 summary")

# ── Summary table ──
print(f"\n{'='*70}")
print("SUMMARY: Key metric comparison across thresholds")
print(f"{'='*70}")
print(f"\n  {'Metric':>30s}", end='')
for thr_name in THRESHOLDS:
    print(f"  {thr_name:>8s}", end='')
print()

# L1/Ctrl enrichment (HeLa)
l1_hela = df_l1[df_l1['condition'] == 'HeLa']
ctrl_hela = df_ctrl[df_ctrl['condition'] == 'HeLa']
print(f"  {'L1/Ctrl ratio (HeLa)':>30s}", end='')
for thr_name in THRESHOLDS:
    col = f'm6a_{thr_name}'
    l1_d = (l1_hela[col] / (l1_hela['read_length'] / 1000.0)).mean()
    c_d = (ctrl_hela[col] / (ctrl_hela['read_length'] / 1000.0)).mean()
    print(f"  {l1_d/c_d:>8.3f}", end='')
print()

# L1/Ctrl enrichment (HeLa-Ars)
l1_ars = df_l1[df_l1['condition'] == 'HeLa-Ars']
ctrl_ars = df_ctrl[df_ctrl['condition'] == 'HeLa-Ars']
print(f"  {'L1/Ctrl ratio (Ars)':>30s}", end='')
for thr_name in THRESHOLDS:
    col = f'm6a_{thr_name}'
    l1_d = (l1_ars[col] / (l1_ars['read_length'] / 1000.0)).mean()
    c_d = (ctrl_ars[col] / (ctrl_ars['read_length'] / 1000.0)).mean()
    print(f"  {l1_d/c_d:>8.3f}", end='')
print()

# Young/Ancient ratio (HeLa)
if 'age' in df_l1.columns:
    print(f"  {'Young/Ancient (HeLa)':>30s}", end='')
    for thr_name in THRESHOLDS:
        col = f'm6a_{thr_name}'
        sub = df_l1[(df_l1['condition'] == 'HeLa') & df_l1['age'].notna()]
        y = (sub[sub['age'] == 'Young'][col] / (sub[sub['age'] == 'Young']['read_length'] / 1000.0)).mean()
        a = (sub[sub['age'] == 'Ancient'][col] / (sub[sub['age'] == 'Ancient']['read_length'] / 1000.0)).mean()
        print(f"  {y/a:>8.3f}", end='')
    print()

# m6A-polyA Spearman (HeLa-Ars)
if polya_col:
    print(f"  {'m6A-polyA rho (Ars)':>30s}", end='')
    ars = df_l1[(df_l1['condition'] == 'HeLa-Ars') & df_l1[polya_col].notna()].copy()
    for thr_name in THRESHOLDS:
        col = f'm6a_{thr_name}'
        ars_d = ars[col] / (ars['read_length'] / 1000.0)
        rho, _ = stats.spearmanr(ars_d, ars[polya_col])
        print(f"  {rho:>8.4f}", end='')
    print()

    print(f"  {'m6A-polyA rho (HeLa)':>30s}", end='')
    hela = df_l1[(df_l1['condition'] == 'HeLa') & df_l1[polya_col].notna()].copy()
    for thr_name in THRESHOLDS:
        col = f'm6a_{thr_name}'
        hela_d = hela[col] / (hela['read_length'] / 1000.0)
        rho, _ = stats.spearmanr(hela_d, hela[polya_col])
        print(f"  {rho:>8.4f}", end='')
    print()

print("\n\nDone.")
