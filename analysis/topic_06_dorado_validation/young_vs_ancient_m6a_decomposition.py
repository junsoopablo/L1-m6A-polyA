#!/usr/bin/env python3
"""
Young vs Ancient L1 m6A Decomposition
======================================
Decomposes the Young/Ancient m6A/kb difference into two components:
  1. DRACH motif density (targets/kb) — sequence property
  2. Per-DRACH methylation rate — enzyme efficiency

Approach:
  A. From reference genome: count DRACH motifs/kb in L1 elements (Young vs Ancient)
  B. From dorado per-read data: methylated DRACH sites/kb
  C. Methylation rate = B / A

Also computes per-read level decomposition using the dorado BAM:
  For each read, count total DRACH adenines in aligned reference → density/kb
  Then: rate = methylated DRACH / total DRACH

Usage:
  conda run -n research python young_vs_ancient_m6a_decomposition.py
"""

import os
import re
import gzip
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
from scipy import stats

# ============================================================
# Configuration
# ============================================================
REF_PATH = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta"
L1_BED = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/L1_TE_L1_family.bed"
BAM_PATH = "/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/HeLa_1_1.dorado.m6A.sorted.bam"
TE_GTF = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/hg38_rmsk_TE.gtf"

PERREAD_TSV = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/"
                   "topic_06_dorado_validation/dorado_m6a_results/dorado_per_read_m6a.tsv.gz")

OUTDIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/"
              "topic_06_dorado_validation/dorado_m6a_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}
DRACH_RE = re.compile(r"[AGT][AG]AC[ACT]")  # non-anchored for scanning
DRACH_RE_5MER = re.compile(r"^[AGT][AG]AC[ACT]$")  # anchored for single check

M6A_PROB_THRESHOLD = 204


# ============================================================
# Part A: Reference-level DRACH density
# ============================================================
def count_drach_in_reference():
    """
    For each L1 element in the BED, extract the reference sequence
    and count DRACH motifs per kb.
    """
    print("=" * 70)
    print("PART A: Reference-level DRACH motif density")
    print("=" * 70)

    # Load L1 BED
    l1_elements = []
    with open(L1_BED) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, start, end, repname = parts[0], int(parts[1]), int(parts[2]), parts[3]
            age = "Young" if repname in YOUNG_SUBFAMILIES else "Ancient"
            l1_elements.append({
                'chrom': chrom, 'start': start, 'end': end,
                'repname': repname, 'age': age,
                'length_bp': end - start
            })

    print(f"  L1 elements: {len(l1_elements):,}")

    # Load L1 strand info from TE GTF
    l1_strand = {}
    with open(TE_GTF) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 9 or 'family_id "L1"' not in fields[8]:
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            key = (chrom, start, end)
            l1_strand[key] = strand

    # Open reference
    ref = pysam.FastaFile(REF_PATH)

    results = []
    for i, elem in enumerate(l1_elements):
        if i % 100000 == 0 and i > 0:
            print(f"    ... {i:,}/{len(l1_elements):,}")

        chrom = elem['chrom']
        start = elem['start']
        end = elem['end']
        length = elem['length_bp']

        try:
            seq = ref.fetch(chrom, start, end).upper()
        except (ValueError, KeyError):
            continue

        if len(seq) < 10:
            continue

        # Determine strand — use RNA strand for DRACH counting
        strand = l1_strand.get((chrom, start, end), '+')
        if strand == '-':
            # Reverse complement
            comp = str.maketrans("ACGT", "TGCA")
            seq = seq.translate(comp)[::-1]

        # Count DRACH motifs (overlapping allowed)
        n_drach = 0
        n_adenines = seq.count('A')
        for j in range(len(seq) - 4):
            kmer = seq[j:j+5]
            if DRACH_RE_5MER.match(kmer):
                n_drach += 1

        length_kb = length / 1000.0
        drach_per_kb = n_drach / length_kb if length_kb > 0 else 0
        adenines_per_kb = n_adenines / length_kb if length_kb > 0 else 0

        results.append({
            'chrom': chrom, 'start': start, 'end': end,
            'repname': elem['repname'], 'age': elem['age'],
            'length_bp': length,
            'n_adenines': n_adenines,
            'n_drach_motifs': n_drach,
            'drach_per_kb': drach_per_kb,
            'adenines_per_kb': adenines_per_kb,
        })

    ref.close()

    df = pd.DataFrame(results)
    print(f"\n  Processed {len(df):,} L1 elements")

    # Filter to elements >= 500 bp for meaningful density
    df_filt = df[df['length_bp'] >= 500].copy()
    print(f"  Elements >= 500bp: {len(df_filt):,}")

    # Summary by age
    print(f"\n  {'Category':<12s} {'N':>8s} {'Len(bp)':>10s} {'DRACH/kb':>10s} {'A/kb':>8s} {'DRACH/A':>10s}")
    print("  " + "-" * 62)

    for age in ['Young', 'Ancient']:
        sub = df_filt[df_filt['age'] == age]
        if len(sub) == 0:
            continue
        med_len = np.median(sub['length_bp'])
        med_drach = np.median(sub['drach_per_kb'])
        med_a = np.median(sub['adenines_per_kb'])
        # DRACH per A (what fraction of A's are in DRACH context)
        drach_per_a = sub['n_drach_motifs'].sum() / sub['n_adenines'].sum() if sub['n_adenines'].sum() > 0 else 0
        print(f"  {age:<12s} {len(sub):>8,} {med_len:>10.0f} {med_drach:>10.2f} {med_a:>8.0f} {drach_per_a:>10.4f}")

    # Statistical test
    young = df_filt[df_filt['age'] == 'Young']['drach_per_kb']
    ancient = df_filt[df_filt['age'] == 'Ancient']['drach_per_kb']
    if len(young) > 0 and len(ancient) > 0:
        ratio = np.median(young) / np.median(ancient) if np.median(ancient) > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(young, ancient, alternative='two-sided')
        print(f"\n  Young vs Ancient DRACH/kb: {np.median(young):.2f} vs {np.median(ancient):.2f} = {ratio:.2f}x  (MWU P={pval:.2e})")

    # Full-length L1 (>= 5kb) for cleaner comparison
    df_full = df_filt[df_filt['length_bp'] >= 5000].copy()
    print(f"\n  Full-length L1 (>= 5kb): {len(df_full):,}")
    for age in ['Young', 'Ancient']:
        sub = df_full[df_full['age'] == age]
        if len(sub) == 0:
            continue
        med_drach = np.median(sub['drach_per_kb'])
        med_len = np.median(sub['length_bp'])
        print(f"    {age}: N={len(sub):,}, median DRACH/kb={med_drach:.2f}, median length={med_len:.0f}bp")

    if len(df_full[df_full['age'] == 'Young']) > 0 and len(df_full[df_full['age'] == 'Ancient']) > 0:
        y = df_full[df_full['age'] == 'Young']['drach_per_kb']
        a = df_full[df_full['age'] == 'Ancient']['drach_per_kb']
        ratio = np.median(y) / np.median(a) if np.median(a) > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(y, a, alternative='two-sided')
        print(f"    Full-length Young vs Ancient: {np.median(y):.2f} vs {np.median(a):.2f} = {ratio:.2f}x  (P={pval:.2e})")

    # Top subfamilies
    print(f"\n  Top subfamilies by DRACH/kb (>=500bp, top 15):")
    subfam_stats = df_filt.groupby('repname').agg(
        n=('drach_per_kb', 'count'),
        drach_per_kb_median=('drach_per_kb', 'median'),
        length_median=('length_bp', 'median'),
    ).sort_values('drach_per_kb_median', ascending=False)
    subfam_stats = subfam_stats[subfam_stats['n'] >= 10].head(15)
    print(f"  {'Subfamily':<12s} {'N':>7s} {'DRACH/kb':>10s} {'Length':>8s}")
    for name, row in subfam_stats.iterrows():
        print(f"  {name:<12s} {int(row['n']):>7,} {row['drach_per_kb_median']:>10.2f} {row['length_median']:>8.0f}")

    return df_filt


# ============================================================
# Part B: Per-read DRACH decomposition from dorado BAM
# ============================================================
def per_read_drach_decomposition():
    """
    For each L1/Control read in the dorado data:
    1. Count total DRACH motifs in aligned reference region
    2. Count methylated DRACH (from MM/ML tag, prob >= 204)
    3. Compute: density = total_DRACH/kb, rate = methylated/total
    """
    print("\n" + "=" * 70)
    print("PART B: Per-read DRACH decomposition (BAM-based)")
    print("=" * 70)

    # Load existing per-read data for read IDs and categories
    print("  Loading per-read data...")
    df = pd.read_csv(PERREAD_TSV, sep='\t', compression='gzip')
    print(f"  Loaded {len(df):,} reads ({df['category'].value_counts().to_dict()})")

    target_ids = set(df['read_id'].values)
    read_info = df.set_index('read_id')[['category', 'age_class', 'subfamily', 'ref_len',
                                          'n_drach', 'n_nondrach', 'drach_per_kb']].to_dict('index')

    # Open reference for DRACH scanning
    ref = pysam.FastaFile(REF_PATH)

    results = []
    n_found = 0

    print("  Scanning BAM for aligned reference regions...")
    with pysam.AlignmentFile(BAM_PATH, 'rb') as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i % 500000 == 0 and i > 0:
                print(f"    ... {i:,} reads scanned, {n_found:,}/{len(target_ids):,} found")

            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            rid = read.query_name
            if rid not in target_ids:
                continue

            n_found += 1
            info = read_info[rid]

            chrom = read.reference_name
            ref_start = read.reference_start
            ref_end = read.reference_end
            is_reverse = read.is_reverse

            if ref_end is None or ref_start is None:
                continue

            ref_len = ref_end - ref_start
            if ref_len <= 0:
                continue

            # Extract aligned reference sequence
            try:
                ref_seq = ref.fetch(chrom, ref_start, ref_end).upper()
            except (ValueError, KeyError):
                continue

            # Orient to RNA strand
            if is_reverse:
                comp = str.maketrans("ACGT", "TGCA")
                ref_seq = ref_seq.translate(comp)[::-1]

            # Count total DRACH motifs in this reference region
            n_total_drach = 0
            n_total_a = ref_seq.count('A')
            for j in range(len(ref_seq) - 4):
                kmer = ref_seq[j:j+5]
                if DRACH_RE_5MER.match(kmer):
                    n_total_drach += 1

            ref_kb = ref_len / 1000.0
            total_drach_per_kb = n_total_drach / ref_kb if ref_kb > 0 else 0

            # Methylated DRACH (from existing data)
            methylated_drach = info['n_drach']
            methylated_drach_per_kb = info['drach_per_kb']

            # Methylation rate
            meth_rate = methylated_drach / n_total_drach if n_total_drach > 0 else np.nan

            results.append({
                'read_id': rid,
                'category': info['category'],
                'age_class': info['age_class'],
                'subfamily': info['subfamily'],
                'ref_len': ref_len,
                'ref_kb': ref_kb,
                'n_total_A': n_total_a,
                'n_total_drach': n_total_drach,
                'total_drach_per_kb': total_drach_per_kb,
                'n_methylated_drach': methylated_drach,
                'methylated_drach_per_kb': methylated_drach_per_kb,
                'drach_methylation_rate': meth_rate,
            })

            if n_found >= len(target_ids):
                break

    ref.close()
    print(f"  Processed {n_found:,} reads")

    rdf = pd.DataFrame(results)

    # Save
    out_file = OUTDIR / 'young_vs_ancient_drach_decomposition.tsv.gz'
    rdf.to_csv(out_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved: {out_file}")

    return rdf


# ============================================================
# Part C: Analysis & decomposition
# ============================================================
def analyze_decomposition(rdf):
    """Analyze the decomposition results."""
    print("\n" + "=" * 70)
    print("PART C: DECOMPOSITION ANALYSIS")
    print("=" * 70)

    l1 = rdf[rdf['category'] == 'L1']
    ctrl = rdf[rdf['category'] == 'Control']
    young = l1[l1['age_class'] == 'Young']
    ancient = l1[l1['age_class'] == 'Ancient']

    # ---- Summary table ----
    print(f"\n{'Category':<15s} {'N':>6s} {'DRACH_den':>10s} {'meth_DRACH':>11s} {'meth_rate':>10s} {'m6A/kb':>8s}")
    print("-" * 65)

    summary_rows = []
    for label, sub in [("Young L1", young), ("Ancient L1", ancient),
                        ("L1 (all)", l1), ("Control", ctrl)]:
        if len(sub) == 0:
            continue

        # Filter out reads with no DRACH for rate calculation
        has_drach = sub[sub['n_total_drach'] > 0]

        den_med = np.median(sub['total_drach_per_kb'])
        meth_med = np.median(sub['methylated_drach_per_kb'])
        rate_med = np.median(has_drach['drach_methylation_rate']) if len(has_drach) > 0 else 0
        # Total m6A/kb from original data
        orig = pd.read_csv(PERREAD_TSV, sep='\t', compression='gzip')
        orig_sub = orig[orig['read_id'].isin(sub['read_id'])]
        m6a_med = np.median(orig_sub['m6a_per_kb']) if len(orig_sub) > 0 else 0

        print(f"  {label:<13s} {len(sub):>6,} {den_med:>10.2f} {meth_med:>11.3f} {rate_med:>10.4f} {m6a_med:>8.2f}")

        summary_rows.append({
            'category': label,
            'n_reads': len(sub),
            'total_drach_per_kb_median': den_med,
            'total_drach_per_kb_mean': np.mean(sub['total_drach_per_kb']),
            'methylated_drach_per_kb_median': meth_med,
            'methylated_drach_per_kb_mean': np.mean(sub['methylated_drach_per_kb']),
            'drach_meth_rate_median': rate_med,
            'drach_meth_rate_mean': np.mean(has_drach['drach_methylation_rate']) if len(has_drach) > 0 else 0,
            'n_with_drach': len(has_drach),
            'frac_with_drach': len(has_drach) / len(sub) if len(sub) > 0 else 0,
        })

    # ---- Young vs Ancient decomposition ----
    print(f"\n{'='*70}")
    print("YOUNG vs ANCIENT DECOMPOSITION")
    print(f"{'='*70}")

    if len(young) > 0 and len(ancient) > 0:
        # Component 1: DRACH density
        y_den = np.median(young['total_drach_per_kb'])
        a_den = np.median(ancient['total_drach_per_kb'])
        den_ratio = y_den / a_den if a_den > 0 else float('inf')
        stat_den, p_den = stats.mannwhitneyu(
            young['total_drach_per_kb'], ancient['total_drach_per_kb'], alternative='two-sided')

        print(f"\n  1. DRACH motif density (targets/kb):")
        print(f"     Young: {y_den:.2f}/kb  vs  Ancient: {a_den:.2f}/kb  = {den_ratio:.2f}x  (P={p_den:.2e})")

        # Component 2: Per-DRACH methylation rate
        y_has = young[young['n_total_drach'] > 0]
        a_has = ancient[ancient['n_total_drach'] > 0]
        y_rate = np.median(y_has['drach_methylation_rate']) if len(y_has) > 0 else 0
        a_rate = np.median(a_has['drach_methylation_rate']) if len(a_has) > 0 else 0
        rate_ratio = y_rate / a_rate if a_rate > 0 else float('inf')

        if len(y_has) > 0 and len(a_has) > 0:
            stat_rate, p_rate = stats.mannwhitneyu(
                y_has['drach_methylation_rate'], a_has['drach_methylation_rate'], alternative='two-sided')
        else:
            p_rate = np.nan

        print(f"\n  2. Per-DRACH methylation rate:")
        print(f"     Young: {y_rate:.4f}  vs  Ancient: {a_rate:.4f}  = {rate_ratio:.2f}x  (P={p_rate:.2e})")

        # Component 3: Methylated DRACH/kb (product)
        y_meth = np.median(young['methylated_drach_per_kb'])
        a_meth = np.median(ancient['methylated_drach_per_kb'])
        meth_ratio = y_meth / a_meth if a_meth > 0 else float('inf')

        print(f"\n  3. Methylated DRACH/kb (observed):")
        print(f"     Young: {y_meth:.3f}/kb  vs  Ancient: {a_meth:.3f}/kb  = {meth_ratio:.2f}x")

        # Decomposition summary
        print(f"\n  DECOMPOSITION:")
        print(f"     density_ratio × rate_ratio = {den_ratio:.2f} × {rate_ratio:.2f} = {den_ratio * rate_ratio:.2f}")
        print(f"     observed methylated_DRACH/kb ratio = {meth_ratio:.2f}")
        print(f"     (slight discrepancy due to medians of products ≠ product of medians)")

        # Relative contribution
        # log decomposition: log(meth_ratio) = log(den_ratio) + log(rate_ratio)
        if den_ratio > 0 and rate_ratio > 0 and meth_ratio > 0:
            log_den = np.log(den_ratio)
            log_rate = np.log(rate_ratio)
            log_total = log_den + log_rate
            if log_total > 0:
                pct_den = log_den / log_total * 100
                pct_rate = log_rate / log_total * 100
                print(f"\n  RELATIVE CONTRIBUTION (log-scale):")
                print(f"     DRACH density (sequence):     {pct_den:.1f}%")
                print(f"     Per-DRACH rate (enzyme):       {pct_rate:.1f}%")

    # ---- Length-matched analysis ----
    print(f"\n{'='*70}")
    print("LENGTH-MATCHED DECOMPOSITION (2-5 kb reads)")
    print(f"{'='*70}")

    lm = rdf[(rdf['ref_len'] >= 2000) & (rdf['ref_len'] <= 5000)]
    lm_young = lm[(lm['category'] == 'L1') & (lm['age_class'] == 'Young')]
    lm_ancient = lm[(lm['category'] == 'L1') & (lm['age_class'] == 'Ancient')]
    lm_ctrl = lm[lm['category'] == 'Control']

    print(f"  Young: {len(lm_young):,}, Ancient: {len(lm_ancient):,}, Control: {len(lm_ctrl):,}")

    for label, sub in [("Young", lm_young), ("Ancient", lm_ancient), ("Control", lm_ctrl)]:
        if len(sub) == 0:
            continue
        has_d = sub[sub['n_total_drach'] > 0]
        den = np.median(sub['total_drach_per_kb'])
        rate = np.median(has_d['drach_methylation_rate']) if len(has_d) > 0 else 0
        meth = np.median(sub['methylated_drach_per_kb'])
        print(f"  {label:<10s}: DRACH_den={den:.2f}/kb, rate={rate:.4f}, meth_DRACH={meth:.3f}/kb")

    if len(lm_young) > 0 and len(lm_ancient) > 0:
        y_den = np.median(lm_young['total_drach_per_kb'])
        a_den = np.median(lm_ancient['total_drach_per_kb'])
        y_has = lm_young[lm_young['n_total_drach'] > 0]
        a_has = lm_ancient[lm_ancient['n_total_drach'] > 0]
        y_rate = np.median(y_has['drach_methylation_rate']) if len(y_has) > 0 else 0
        a_rate = np.median(a_has['drach_methylation_rate']) if len(a_has) > 0 else 0
        den_r = y_den / a_den if a_den > 0 else float('inf')
        rate_r = y_rate / a_rate if a_rate > 0 else float('inf')
        print(f"\n  Length-matched Young/Ancient:")
        print(f"    Density ratio: {den_r:.2f}x")
        print(f"    Rate ratio:    {rate_r:.2f}x")
        print(f"    Product:       {den_r * rate_r:.2f}x")

    # ---- L1 vs Control decomposition ----
    print(f"\n{'='*70}")
    print("L1 vs CONTROL DECOMPOSITION")
    print(f"{'='*70}")

    if len(l1) > 0 and len(ctrl) > 0:
        l1_den = np.median(l1['total_drach_per_kb'])
        c_den = np.median(ctrl['total_drach_per_kb'])
        den_ratio = l1_den / c_den if c_den > 0 else float('inf')

        l1_has = l1[l1['n_total_drach'] > 0]
        c_has = ctrl[ctrl['n_total_drach'] > 0]
        l1_rate = np.median(l1_has['drach_methylation_rate']) if len(l1_has) > 0 else 0
        c_rate = np.median(c_has['drach_methylation_rate']) if len(c_has) > 0 else 0
        rate_ratio = l1_rate / c_rate if c_rate > 0 else float('inf')

        print(f"  DRACH density: L1 {l1_den:.2f} vs Ctrl {c_den:.2f} = {den_ratio:.2f}x")
        print(f"  Meth rate:     L1 {l1_rate:.4f} vs Ctrl {c_rate:.4f} = {rate_ratio:.2f}x")
        print(f"  Product:       {den_ratio * rate_ratio:.2f}x")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUTDIR / 'young_vs_ancient_decomposition_summary.tsv'
    summary_df.to_csv(out_summary, sep='\t', index=False, float_format='%.4f')
    print(f"\n  Summary saved: {out_summary}")


# ============================================================
# Main
# ============================================================
def main():
    print("Young vs Ancient L1 m6A Decomposition Analysis")
    print("=" * 70)

    # Part A: Reference-level analysis
    ref_df = count_drach_in_reference()

    # Part B: Per-read decomposition from BAM
    rdf = per_read_drach_decomposition()

    # Part C: Analysis
    analyze_decomposition(rdf)

    print("\n\nDone.")


if __name__ == '__main__':
    main()
