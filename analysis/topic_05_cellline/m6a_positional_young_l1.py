#!/usr/bin/env python3
"""m6A positional distribution within young L1 elements.

Hwang et al. 2021 (Nat Comm) showed m6A clusters at L1 5'UTR (A332 site).
Question: does our ONT DRS data reproduce this 5'UTR enrichment in young L1?

Key challenge: DRS has 3' bias, so most reads cover only the 3' portion.
We need to:
1. Map m6A read positions → L1 element coordinates
2. Normalize by read coverage at each L1 position
3. Compare young vs ancient L1 positional distributions

L1HS consensus structure (~6,064 bp):
  5'UTR: 1-910 (0-15%)
  ORF1:  911-1924 (15-32%)
  inter-ORF: 1925-1990 (32-33%)
  ORF2:  1991-5817 (33-96%)
  3'UTR: 5818-6064 (96-100%)
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_DIR = TOPIC_05 / 'part3_l1_per_read_cache'
OUTDIR = TOPIC_05 / 'm6a_positional_young'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Full-length L1HS = ~6064bp
L1HS_LENGTH = 6064
L1HS_5UTR_END = 910      # fraction 0.15
L1HS_ORF1_END = 1924     # fraction 0.317
L1HS_ORF2_START = 1991   # fraction 0.328
L1HS_ORF2_END = 5817     # fraction 0.959

# Auto-detect groups from cache files, exclude special conditions
EXCLUDE_PREFIXES = ('HeLa-Ars', 'MCF7-EV')
ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# =====================================================================
# 1. Load and merge data
# =====================================================================
print("Loading data...")
all_reads = []

for grp in ALL_GROUPS:
    # L1 summary
    summary_f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    if not summary_f.exists():
        continue

    summary = pd.read_csv(summary_f, sep='\t', usecols=[
        'read_id', 'chr', 'start', 'end', 'read_length',
        'te_start', 'te_end', 'gene_id', 'te_strand', 'read_strand',
        'dist_to_3prime', 'transcript_id', 'qc_tag',
    ])
    summary = summary[summary['qc_tag'] == 'PASS'].copy()

    # Part3 cache
    cache_f = CACHE_DIR / f'{grp}_l1_per_read.tsv'
    if not cache_f.exists():
        continue

    cache = pd.read_csv(cache_f, sep='\t', usecols=[
        'read_id', 'read_length', 'm6a_sites_high', 'm6a_positions'
    ])

    # Merge
    merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_positions']],
                           on='read_id', how='inner')
    merged['group'] = grp
    merged['is_young'] = merged['gene_id'].isin(YOUNG)
    merged['age'] = np.where(merged['is_young'], 'young', 'ancient')
    merged['te_length'] = merged['te_end'] - merged['te_start']

    all_reads.append(merged)

df = pd.concat(all_reads, ignore_index=True)
print(f"  Total PASS reads with MAFIA: {len(df)}")
print(f"  Young: {df['is_young'].sum()}, Ancient: {(~df['is_young']).sum()}")

# =====================================================================
# 2. Map m6A positions to L1 element coordinates
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 1: m6A POSITION MAPPING TO L1 ELEMENT")
print("=" * 70)

def map_m6a_to_l1_fraction(row):
    """Map m6A read positions to fractional position within L1 element.

    MM tag positions are stored relative to BAM SEQ (left-to-right = reference orientation).
    - SEQ position p → genomic coordinate ≈ start + p
    - For + strand L1: fraction from 5' = (genomic - te_start) / te_length
    - For - strand L1: fraction from 5' = (te_end - genomic) / te_length
    """
    positions_str = row['m6a_positions']
    if pd.isna(positions_str) or positions_str == '[]':
        return []

    try:
        positions = ast.literal_eval(positions_str)
    except (ValueError, SyntaxError):
        return []

    if not positions:
        return []

    te_start = row['te_start']
    te_end = row['te_end']
    te_len = te_end - te_start
    if te_len <= 0:
        return []

    read_start = row['start']  # genomic start of read alignment
    te_strand = row['te_strand']

    fractions = []
    for p in positions:
        # Map read position to genomic coordinate
        genomic_pos = read_start + p

        # Check if within L1 element boundaries
        if genomic_pos < te_start or genomic_pos > te_end:
            continue

        # Calculate fraction from L1 5' end
        if te_strand == '+':
            frac = (genomic_pos - te_start) / te_len
        else:
            frac = (te_end - genomic_pos) / te_len

        if 0 <= frac <= 1:
            fractions.append(frac)

    return fractions

def map_coverage_to_l1_fraction(row):
    """Map read coverage to L1 element fractions (for normalization).

    Returns (frac_start, frac_end) where coverage exists.
    """
    te_start = row['te_start']
    te_end = row['te_end']
    te_len = te_end - te_start
    if te_len <= 0:
        return (0, 0)

    read_start = row['start']
    read_end = row['end']
    te_strand = row['te_strand']

    # Clip read to L1 boundaries
    cov_start = max(read_start, te_start)
    cov_end = min(read_end, te_end)

    if cov_start >= cov_end:
        return (0, 0)

    if te_strand == '+':
        frac_start = (cov_start - te_start) / te_len
        frac_end = (cov_end - te_start) / te_len
    else:
        frac_start = (te_end - cov_end) / te_len
        frac_end = (te_end - cov_start) / te_len

    return (frac_start, frac_end)


# Process young and ancient separately
print("\nMapping m6A positions to L1 element coordinates...")

results = {}
for age in ['young', 'ancient']:
    sub = df[df['age'] == age].copy()
    print(f"\n--- {age} L1 (n={len(sub)}) ---")

    # Map m6A positions
    m6a_fracs = []
    coverage_bins = np.zeros(20)  # 20 bins across L1

    for _, row in sub.iterrows():
        fracs = map_m6a_to_l1_fraction(row)
        for f in fracs:
            m6a_fracs.append({
                'frac': f,
                'te_length': row['te_length'],
                'gene_id': row['gene_id'],
                'group': row['group'],
            })

        # Coverage
        cov_s, cov_e = map_coverage_to_l1_fraction(row)
        if cov_e > cov_s:
            for b in range(20):
                bin_s = b / 20
                bin_e = (b + 1) / 20
                # Overlap between coverage and bin
                overlap = max(0, min(cov_e, bin_e) - max(cov_s, bin_s))
                coverage_bins[b] += overlap * 20  # normalize by bin width

    m6a_df = pd.DataFrame(m6a_fracs)
    print(f"  m6A sites within L1: {len(m6a_df)}")

    if len(m6a_df) > 0:
        # Bin m6A sites
        m6a_hist, _ = np.histogram(m6a_df['frac'], bins=20, range=(0, 1))

        # Normalize by coverage
        m6a_density = np.where(coverage_bins > 0, m6a_hist / coverage_bins, 0)

        results[age] = {
            'm6a_df': m6a_df,
            'm6a_hist': m6a_hist,
            'coverage': coverage_bins,
            'density': m6a_density,
            'n_reads': len(sub),
        }

        print(f"  m6A raw distribution (20 bins, 5'→3'):")
        for i in range(20):
            frac_s = i * 5
            frac_e = (i + 1) * 5
            bar = '#' * int(m6a_hist[i] / max(m6a_hist) * 30) if max(m6a_hist) > 0 else ''
            print(f"    {frac_s:3d}-{frac_e:3d}%: {m6a_hist[i]:5d} {bar}")

        print(f"\n  Coverage profile (relative reads):")
        for i in range(20):
            frac_s = i * 5
            frac_e = (i + 1) * 5
            bar = '#' * int(coverage_bins[i] / max(coverage_bins) * 30) if max(coverage_bins) > 0 else ''
            print(f"    {frac_s:3d}-{frac_e:3d}%: {coverage_bins[i]:8.1f} {bar}")

        print(f"\n  m6A density (normalized by coverage):")
        for i in range(20):
            frac_s = i * 5
            frac_e = (i + 1) * 5
            bar = '#' * int(m6a_density[i] / max(m6a_density) * 30) if max(m6a_density) > 0 else ''
            print(f"    {frac_s:3d}-{frac_e:3d}%: {m6a_density[i]:.4f} {bar}")

# =====================================================================
# 3. Young L1 by element size (full-length vs truncated)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: YOUNG L1 BY ELEMENT SIZE")
print("=" * 70)

young = df[df['age'] == 'young'].copy()
print(f"\nYoung L1 element length distribution:")
print(f"  Mean: {young['te_length'].mean():.0f}")
print(f"  Median: {young['te_length'].median():.0f}")
for pct in [10, 25, 50, 75, 90]:
    print(f"  P{pct}: {young['te_length'].quantile(pct/100):.0f}")

# Split by element size
size_bins = [
    ('short (<2kb)', 0, 2000),
    ('medium (2-5kb)', 2000, 5000),
    ('long (≥5kb)', 5000, 100000),
]

print(f"\nYoung L1 by element size:")
for label, lo, hi in size_bins:
    sub = young[(young['te_length'] >= lo) & (young['te_length'] < hi)]
    print(f"  {label}: {len(sub)} reads, median RL={sub['read_length'].median():.0f}")

# Focus on long young L1 (≥5kb, near full-length)
long_young = young[young['te_length'] >= 5000].copy()
print(f"\n--- Long Young L1 (≥5kb, near full-length): {len(long_young)} reads ---")

if len(long_young) > 0:
    print(f"  Element length: median={long_young['te_length'].median():.0f}")
    print(f"  Read length: median={long_young['read_length'].median():.0f}")
    print(f"  dist_to_3prime: median={long_young['dist_to_3prime'].median():.0f}")

    # How many reads cover the 5'UTR region (0-15% from 5' end)?
    reads_covering_5utr = 0
    for _, row in long_young.iterrows():
        cov_s, cov_e = map_coverage_to_l1_fraction(row)
        if cov_s <= 0.15:
            reads_covering_5utr += 1

    print(f"  Reads covering 5'UTR (0-15%): {reads_covering_5utr} ({reads_covering_5utr/len(long_young)*100:.1f}%)")

    # m6A distribution within long young L1
    m6a_fracs_long = []
    coverage_long = np.zeros(20)

    for _, row in long_young.iterrows():
        fracs = map_m6a_to_l1_fraction(row)
        for f in fracs:
            m6a_fracs_long.append(f)

        cov_s, cov_e = map_coverage_to_l1_fraction(row)
        if cov_e > cov_s:
            for b in range(20):
                bin_s = b / 20
                bin_e = (b + 1) / 20
                overlap = max(0, min(cov_e, bin_e) - max(cov_s, bin_s))
                coverage_long[b] += overlap * 20

    if m6a_fracs_long:
        hist_long, _ = np.histogram(m6a_fracs_long, bins=20, range=(0, 1))
        density_long = np.where(coverage_long > 0, hist_long / coverage_long, 0)

        print(f"\n  m6A density in long young L1 (coverage-normalized):")
        for i in range(20):
            frac_s = i * 5
            frac_e = (i + 1) * 5
            bar = '#' * int(density_long[i] / max(density_long) * 30) if max(density_long) > 0 else ''
            print(f"    {frac_s:3d}-{frac_e:3d}%: density={density_long[i]:.4f}, raw={hist_long[i]:4d}, cov={coverage_long[i]:.1f} {bar}")

        # 5'UTR region specifically
        utr5_density = density_long[:3].mean()  # 0-15%
        body_density = density_long[3:19].mean()  # 15-95%
        utr3_density = density_long[19]  # 95-100%

        print(f"\n  5'UTR (0-15%) density: {utr5_density:.4f}")
        print(f"  Body (15-95%) density: {body_density:.4f}")
        print(f"  3'UTR (95-100%) density: {utr3_density:.4f}")
        if body_density > 0:
            print(f"  5'UTR / Body ratio: {utr5_density/body_density:.2f}x")

# =====================================================================
# 4. Focus on reads that reach the 5' end
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: READS REACHING 5' END OF YOUNG L1")
print("=" * 70)

# Among young L1 with long elements, find reads that actually cover the 5' region
if len(long_young) > 0:
    # Reads where coverage starts at 5' end (fraction < 0.1)
    reaching_5 = []
    for _, row in long_young.iterrows():
        cov_s, cov_e = map_coverage_to_l1_fraction(row)
        if cov_s <= 0.10:  # reaches within 10% of 5' end
            reaching_5.append(row)

    reaching_df = pd.DataFrame(reaching_5)
    print(f"\nReads reaching within 10% of 5' end: {len(reaching_df)}")

    if len(reaching_df) > 10:
        print(f"  Read length: median={reaching_df['read_length'].median():.0f}")
        print(f"  Element length: median={reaching_df['te_length'].median():.0f}")

        # m6A distribution in these reads
        m6a_fracs_5 = []
        for _, row in reaching_df.iterrows():
            fracs = map_m6a_to_l1_fraction(row)
            m6a_fracs_5.extend(fracs)

        if m6a_fracs_5:
            print(f"  Total m6A sites: {len(m6a_fracs_5)}")
            fracs_arr = np.array(m6a_fracs_5)

            # Compare 5'UTR vs body density
            n_5utr = np.sum(fracs_arr <= 0.15)
            n_body = np.sum((fracs_arr > 0.15) & (fracs_arr <= 0.96))
            n_3utr = np.sum(fracs_arr > 0.96)

            len_5utr = 0.15
            len_body = 0.81
            len_3utr = 0.04

            rate_5utr = n_5utr / len_5utr if len_5utr > 0 else 0
            rate_body = n_body / len_body if len_body > 0 else 0
            rate_3utr = n_3utr / len_3utr if len_3utr > 0 else 0

            print(f"\n  5'UTR (0-15%): {n_5utr} sites, density={rate_5utr:.1f}")
            print(f"  Body (15-96%): {n_body} sites, density={rate_body:.1f}")
            print(f"  3'UTR (96-100%): {n_3utr} sites, density={rate_3utr:.1f}")
            if rate_body > 0:
                print(f"  5'UTR / Body ratio: {rate_5utr/rate_body:.2f}x")

# =====================================================================
# 5. Young vs Ancient comparison (read-relative fractional position)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: YOUNG vs ANCIENT (READ-RELATIVE POSITION)")
print("=" * 70)

for age in ['young', 'ancient']:
    sub = df[df['age'] == age].copy()
    fracs = []
    for _, row in sub.iterrows():
        pos_str = row['m6a_positions']
        if pd.isna(pos_str) or pos_str == '[]':
            continue
        try:
            positions = ast.literal_eval(pos_str)
        except:
            continue
        rl = row['read_length']
        if rl <= 0:
            continue
        for p in positions:
            frac = p / rl  # 0 = 3' end (seq start), 1 = 5' end
            if 0 <= frac <= 1:
                fracs.append(frac)

    fracs_arr = np.array(fracs)
    print(f"\n--- {age} (n_sites={len(fracs_arr)}) ---")
    print(f"  Mean frac pos (0=3',1=5'): {fracs_arr.mean():.3f}")
    print(f"  Median: {fracs_arr.median() if hasattr(fracs_arr, 'median') else np.median(fracs_arr):.3f}")

    # Quarter distribution
    q1 = np.mean(fracs_arr < 0.25)
    q2 = np.mean((fracs_arr >= 0.25) & (fracs_arr < 0.50))
    q3 = np.mean((fracs_arr >= 0.50) & (fracs_arr < 0.75))
    q4 = np.mean(fracs_arr >= 0.75)
    print(f"  0-25% (3'): {q1*100:.1f}%")
    print(f"  25-50%:     {q2*100:.1f}%")
    print(f"  50-75%:     {q3*100:.1f}%")
    print(f"  75-100%(5'): {q4*100:.1f}%")

    ks_stat, ks_p = stats.kstest(fracs_arr, 'uniform')
    print(f"  KS test vs uniform: D={ks_stat:.4f}, p={ks_p:.2e}")

# =====================================================================
# 6. Young L1 subfamily breakdown
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: YOUNG L1 SUBFAMILY BREAKDOWN")
print("=" * 70)

for sf in ['L1HS', 'L1PA1', 'L1PA2', 'L1PA3']:
    sub = df[df['gene_id'] == sf]
    print(f"\n--- {sf} (n={len(sub)}) ---")
    print(f"  Element length: median={sub['te_length'].median():.0f}")
    print(f"  Read length: median={sub['read_length'].median():.0f}")
    print(f"  m6A sites/read: {sub['m6a_sites_high'].mean():.2f}")

    # m6A position in element coordinates
    m6a_fracs = []
    for _, row in sub.iterrows():
        fracs = map_m6a_to_l1_fraction(row)
        m6a_fracs.extend(fracs)

    if m6a_fracs:
        arr = np.array(m6a_fracs)
        print(f"  m6A sites in L1: {len(arr)}")
        print(f"  Mean L1 position (0=5',1=3'): {arr.mean():.3f}")

        # 5'UTR fraction for near-full-length
        full = sub[sub['te_length'] >= 5000]
        if len(full) > 0:
            full_fracs = []
            for _, row in full.iterrows():
                full_fracs.extend(map_m6a_to_l1_fraction(row))
            if full_fracs:
                farr = np.array(full_fracs)
                n5 = np.sum(farr <= 0.15)
                print(f"  Full-length (≥5kb): {len(full)} reads, {len(full_fracs)} m6A sites")
                print(f"    5'UTR (≤15%): {n5} ({n5/len(full_fracs)*100:.1f}%)")

# =====================================================================
# 7. Summary table
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6: SUMMARY")
print("=" * 70)

summary_rows = []
for age in ['young', 'ancient']:
    sub = df[df['age'] == age]

    all_fracs = []
    for _, row in sub.iterrows():
        all_fracs.extend(map_m6a_to_l1_fraction(row))

    if all_fracs:
        arr = np.array(all_fracs)
        summary_rows.append({
            'category': age,
            'n_reads': len(sub),
            'n_m6a_sites': len(arr),
            'mean_te_length': sub['te_length'].mean(),
            'mean_read_length': sub['read_length'].mean(),
            'mean_m6a_frac_from_5prime': arr.mean(),
            'pct_in_5utr_015': (arr <= 0.15).mean() * 100,
            'pct_in_body_015_096': ((arr > 0.15) & (arr <= 0.96)).mean() * 100,
            'pct_in_3utr_096': (arr > 0.96).mean() * 100,
        })

# Long young L1
long_y = df[(df['age'] == 'young') & (df['te_length'] >= 5000)]
lf = []
for _, row in long_y.iterrows():
    lf.extend(map_m6a_to_l1_fraction(row))
if lf:
    arr = np.array(lf)
    summary_rows.append({
        'category': 'young_fullength',
        'n_reads': len(long_y),
        'n_m6a_sites': len(arr),
        'mean_te_length': long_y['te_length'].mean(),
        'mean_read_length': long_y['read_length'].mean(),
        'mean_m6a_frac_from_5prime': arr.mean(),
        'pct_in_5utr_015': (arr <= 0.15).mean() * 100,
        'pct_in_body_015_096': ((arr > 0.15) & (arr <= 0.96)).mean() * 100,
        'pct_in_3utr_096': (arr > 0.96).mean() * 100,
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
summary.to_csv(OUTDIR / 'm6a_positional_summary.tsv', sep='\t', index=False)

print(f"\nResults saved to: {OUTDIR}")
print("Done!")
