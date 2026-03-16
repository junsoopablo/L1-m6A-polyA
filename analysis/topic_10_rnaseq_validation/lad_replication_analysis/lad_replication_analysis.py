#!/usr/bin/env python3
"""
LAD and Replication Timing Analysis of L1 Poly(A) and m6A

Analyzes whether L1 elements in:
1. Lamina-associated domains (LADs) vs non-LAD regions
2. Early vs late replicating regions
show different poly(A) length, stress response, and m6A behavior.

Data sources:
- LADs: UCSC laminB1Lads (Guelen et al. 2008, hg19 -> liftover to hg38)
- Replication timing: ENCODE UW Repli-seq HeLa-S3 peaks (hg19 -> liftover)
- L1 data: l1_chromhmm_annotated.tsv (all cell lines)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pyliftover import LiftOver
import gzip
import os
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
BASE_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration'
OUT_DIR = os.path.join(BASE_DIR, 'topic_10_rnaseq_validation/lad_replication_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

L1_FILE = os.path.join(BASE_DIR, 'topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv')

# === Step 1: Load and liftover LAD coordinates ===
print("=" * 70)
print("Step 1: Loading and lifting over LAD coordinates (hg19 -> hg38)")
print("=" * 70)

lo = LiftOver('hg19', 'hg38')

# Load LADs from UCSC (format: bin, chrom, chromStart, chromEnd)
lad_regions = []
with gzip.open('/tmp/laminB1Lads_hg19.txt.gz', 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        chrom = parts[1]
        start_19 = int(parts[2])
        end_19 = int(parts[3])

        # Liftover start and end
        new_start = lo.convert_coordinate(chrom, start_19)
        new_end = lo.convert_coordinate(chrom, end_19)

        if new_start and new_end:
            new_chrom = new_start[0][0]
            ns = new_start[0][1]
            ne = new_end[0][1]
            if new_chrom == new_end[0][0] and ns < ne:
                lad_regions.append((new_chrom, ns, ne))

lad_df = pd.DataFrame(lad_regions, columns=['chrom', 'start', 'end'])
lad_df = lad_df.sort_values(['chrom', 'start']).reset_index(drop=True)
total_lad_bp = (lad_df['end'] - lad_df['start']).sum()
print(f"LAD regions after liftover: {len(lad_df)} (total {total_lad_bp/1e6:.0f} Mb)")

# Save LAD BED for bedtools
lad_bed = os.path.join(OUT_DIR, 'lads_hg38.bed')
lad_df.to_csv(lad_bed, sep='\t', header=False, index=False)

# === Step 2: Load and liftover Replication Timing ===
print("\n" + "=" * 70)
print("Step 2: Loading and lifting over Replication Timing (hg19 -> hg38)")
print("=" * 70)

repli_regions = []
with gzip.open('/tmp/repliSeq_helas3_hg19.txt.gz', 'rt') as f:
    for line in f:
        parts = line.strip().split('\t')
        chrom = parts[1]
        start_19 = int(parts[2])
        end_19 = int(parts[3])
        score = int(parts[5])

        new_start = lo.convert_coordinate(chrom, start_19)
        new_end = lo.convert_coordinate(chrom, end_19)

        if new_start and new_end:
            new_chrom = new_start[0][0]
            ns = new_start[0][1]
            ne = new_end[0][1]
            if new_chrom == new_end[0][0] and ns < ne:
                repli_regions.append((new_chrom, ns, ne, score))

repli_df = pd.DataFrame(repli_regions, columns=['chrom', 'start', 'end', 'score'])
repli_df = repli_df.sort_values(['chrom', 'start']).reset_index(drop=True)
print(f"Repli-seq peaks after liftover: {len(repli_df)}")
print(f"Score range: {repli_df['score'].min()}-{repli_df['score'].max()}")
print(f"Score median: {repli_df['score'].median()}")

# Classify: Early (score >= 600), Late (score <= 400)
early_df = repli_df[repli_df['score'] >= 600].copy()
late_df = repli_df[repli_df['score'] <= 400].copy()
print(f"Early-replicating peaks (score>=600): {len(early_df)}")
print(f"Late-replicating peaks (score<=400): {len(late_df)}")

# Save BED files
early_bed = os.path.join(OUT_DIR, 'early_replicating_hg38.bed')
late_bed = os.path.join(OUT_DIR, 'late_replicating_hg38.bed')
early_df[['chrom', 'start', 'end']].to_csv(early_bed, sep='\t', header=False, index=False)
late_df[['chrom', 'start', 'end']].to_csv(late_bed, sep='\t', header=False, index=False)

# === Step 3: Load L1 data ===
print("\n" + "=" * 70)
print("Step 3: Loading L1 data")
print("=" * 70)

l1 = pd.read_csv(L1_FILE, sep='\t')
print(f"Total L1 reads: {len(l1)}")
print(f"Cell lines: {sorted(l1['cellline'].unique())}")
print(f"Conditions: {sorted(l1['condition'].unique())}")

# Create L1 BED file for intersection
l1_bed_path = os.path.join(OUT_DIR, 'l1_positions.bed')
l1_bed = l1[['chr', 'start', 'end', 'read_id']].copy()
l1_bed = l1_bed.sort_values(['chr', 'start'])
l1_bed.to_csv(l1_bed_path, sep='\t', header=False, index=False)

# === Step 4: Intersect L1 with LADs and Replication Timing (Python interval) ===
print("\n" + "=" * 70)
print("Step 4: Intersecting L1 with LADs and replication timing")
print("=" * 70)

from collections import defaultdict

def build_interval_index(regions_df, chrom_col='chrom', start_col='start', end_col='end'):
    """Build a dict of sorted intervals per chromosome for fast overlap."""
    idx = defaultdict(list)
    for _, row in regions_df.iterrows():
        idx[row[chrom_col]].append((row[start_col], row[end_col]))
    # Sort by start
    for chrom in idx:
        idx[chrom].sort()
    return idx

def check_overlap(chrom, start, end, interval_idx):
    """Check if a query interval overlaps any interval in the index using binary search."""
    import bisect
    intervals = interval_idx.get(chrom, [])
    if not intervals:
        return False
    # Find intervals that could overlap: start < query_end AND end > query_start
    # Binary search for the first interval whose start < end
    pos = bisect.bisect_right([iv[0] for iv in intervals], end) - 1
    # Check backwards from pos
    for i in range(max(0, pos), -1, -1):
        iv_start, iv_end = intervals[i]
        if iv_start >= end:
            continue
        if iv_end > start:
            return True
        if iv_end <= start and iv_start < start:
            # Could still be more intervals earlier
            continue
        if iv_start + 10_000_000 < start:
            # Too far back, stop
            break
    # Also check forward
    pos2 = bisect.bisect_left([iv[0] for iv in intervals], start)
    for i in range(pos2, min(len(intervals), pos2 + 5)):
        iv_start, iv_end = intervals[i]
        if iv_start >= end:
            break
        if iv_end > start:
            return True
    return False

def get_repli_score(chrom, start, end, repli_idx):
    """Get the replication timing score for a query interval (mean of overlapping peaks)."""
    import bisect
    intervals = repli_idx.get(chrom, [])
    if not intervals:
        return None
    midpoint = (start + end) // 2
    starts = [iv[0] for iv in intervals]
    pos = bisect.bisect_right(starts, midpoint) - 1
    # Check nearby peaks (within 500kb)
    best_dist = float('inf')
    best_score = None
    for i in range(max(0, pos - 5), min(len(intervals), pos + 6)):
        iv_start, iv_end, iv_score = intervals[i]
        iv_mid = (iv_start + iv_end) // 2
        dist = abs(iv_mid - midpoint)
        if dist < best_dist:
            best_dist = dist
            best_score = iv_score
    if best_dist <= 500_000:
        return best_score
    return None

# Build LAD interval index
lad_idx = build_interval_index(lad_df)

# Build repli-seq index (include score)
repli_idx = defaultdict(list)
for _, row in repli_df.iterrows():
    repli_idx[row['chrom']].append((row['start'], row['end'], row['score']))
for chrom in repli_idx:
    repli_idx[chrom].sort()

# Annotate each L1 read
print("Annotating L1 reads...")
lad_flags = []
repli_scores = []
for _, row in l1.iterrows():
    lad_flags.append(check_overlap(row['chr'], row['start'], row['end'], lad_idx))
    repli_scores.append(get_repli_score(row['chr'], row['start'], row['end'], repli_idx))

l1['in_lad'] = lad_flags
l1['repli_score'] = repli_scores

lad_count = l1['in_lad'].sum()
print(f"L1 reads in LADs: {lad_count}")
print(f"L1 reads NOT in LADs: {len(l1) - lad_count}")

# Classify replication timing
l1['repli_timing'] = 'unknown'
l1.loc[l1['repli_score'].notna() & (l1['repli_score'] >= 600), 'repli_timing'] = 'early'
l1.loc[l1['repli_score'].notna() & (l1['repli_score'] <= 400), 'repli_timing'] = 'late'
l1.loc[l1['repli_score'].notna() & (l1['repli_score'] > 400) & (l1['repli_score'] < 600), 'repli_timing'] = 'mid'

early_count = (l1['repli_timing'] == 'early').sum()
late_count = (l1['repli_timing'] == 'late').sum()
mid_count = (l1['repli_timing'] == 'mid').sum()
unk_count = (l1['repli_timing'] == 'unknown').sum()
print(f"L1 reads in early-replicating: {early_count}")
print(f"L1 reads in late-replicating: {late_count}")
print(f"L1 reads in mid-replicating: {mid_count}")
print(f"L1 reads unknown: {unk_count}")

# === Step 5: Summary of annotations ===
print("\n" + "=" * 70)
print("Step 5: Annotation summary")
print("=" * 70)

print(f"\nLAD distribution:")
print(l1['in_lad'].value_counts())
print(f"\nReplication timing distribution:")
print(l1['repli_timing'].value_counts())

# === Step 6: Analysis functions ===
def compare_groups(df, group_col, group_a, group_b, label_a, label_b, metric_col, test='mannwhitneyu'):
    """Compare two groups on a metric."""
    a = df[df[group_col] == group_a][metric_col].dropna()
    b = df[df[group_col] == group_b][metric_col].dropna()

    if len(a) < 10 or len(b) < 10:
        return {'n_a': len(a), 'n_b': len(b), 'med_a': np.nan, 'med_b': np.nan, 'delta': np.nan, 'p': np.nan}

    if test == 'mannwhitneyu':
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    else:
        stat, p = stats.ttest_ind(a, b)

    return {
        'n_a': len(a), 'n_b': len(b),
        'med_a': a.median(), 'med_b': b.median(),
        'mean_a': a.mean(), 'mean_b': b.mean(),
        'delta': a.median() - b.median(),
        'ratio': a.median() / b.median() if b.median() != 0 else np.nan,
        'p': p
    }

# === Step 7: LAD Analysis ===
print("\n" + "=" * 70)
print("Step 7: LAD vs non-LAD Analysis")
print("=" * 70)

results_rows = []

# Focus on HeLa for stress comparison (HeLa = normal, HeLa-Ars = stress)
hela_normal = l1[l1['cellline'] == 'HeLa'].copy()
hela_stress = l1[l1['cellline'] == 'HeLa-Ars'].copy()
hela = pd.concat([hela_normal, hela_stress])

# All cell lines for baseline
for desc, subset in [('All CL', l1), ('HeLa only', hela)]:
    print(f"\n--- {desc} ---")

    # Baseline (normal only)
    normal = subset[subset['condition'] == 'normal']

    for metric, label in [('polya_length', 'Poly(A) length'), ('m6a_per_kb', 'm6A/kb')]:
        res = compare_groups(normal, 'in_lad', True, False, 'LAD', 'non-LAD', metric)
        print(f"  {label}: LAD={res['med_a']:.1f} vs non-LAD={res['med_b']:.1f} "
              f"(delta={res['delta']:+.1f}, ratio={res.get('ratio', np.nan):.3f}, P={res['p']:.2e}, "
              f"n={res['n_a']}/{res['n_b']})")
        results_rows.append({
            'comparison': 'LAD vs non-LAD', 'dataset': desc, 'condition': 'normal',
            'metric': label, **res
        })

# Stress-specific (HeLa)
print("\n--- HeLa Stress ---")
for metric, label in [('polya_length', 'Poly(A) length'), ('m6a_per_kb', 'm6A/kb')]:
    res = compare_groups(hela_stress, 'in_lad', True, False, 'LAD', 'non-LAD', metric)
    print(f"  {label}: LAD={res['med_a']:.1f} vs non-LAD={res['med_b']:.1f} "
          f"(delta={res['delta']:+.1f}, P={res['p']:.2e}, n={res['n_a']}/{res['n_b']})")
    results_rows.append({
        'comparison': 'LAD vs non-LAD', 'dataset': 'HeLa only', 'condition': 'stress',
        'metric': label, **res
    })

# Poly(A) delta (stress - normal) per LAD status
print("\n--- HeLa Poly(A) Delta (Stress vs Normal) ---")
for lad_status, lad_label in [(True, 'LAD'), (False, 'non-LAD')]:
    norm = hela_normal[hela_normal['in_lad'] == lad_status]['polya_length']
    strs = hela_stress[hela_stress['in_lad'] == lad_status]['polya_length']
    if len(norm) >= 10 and len(strs) >= 10:
        delta = strs.median() - norm.median()
        stat, p = stats.mannwhitneyu(strs, norm, alternative='two-sided')
        print(f"  {lad_label}: Normal={norm.median():.1f} -> Stress={strs.median():.1f} "
              f"(Delta={delta:+.1f}nt, P={p:.2e}, n={len(norm)}/{len(strs)})")
        results_rows.append({
            'comparison': f'Stress Delta ({lad_label})', 'dataset': 'HeLa', 'condition': 'stress-normal',
            'metric': 'Poly(A) delta', 'med_a': strs.median(), 'med_b': norm.median(),
            'n_a': len(strs), 'n_b': len(norm), 'delta': delta, 'p': p
        })

# === Step 8: ChromHMM validation - LAD L1 should be mostly Quiescent ===
print("\n" + "=" * 70)
print("Step 8: ChromHMM State Distribution in LAD vs non-LAD")
print("=" * 70)

# Use HeLa data only (ChromHMM = E117)
hela_normal_chrom = hela_normal.dropna(subset=['chromhmm_group'])

for lad_status, lad_label in [(True, 'LAD'), (False, 'non-LAD')]:
    subset = hela_normal_chrom[hela_normal_chrom['in_lad'] == lad_status]
    if len(subset) > 0:
        dist = subset['chromhmm_group'].value_counts(normalize=True) * 100
        print(f"\n{lad_label} (n={len(subset)}):")
        for state in ['Quiescent', 'Transcribed', 'Heterochromatin', 'Enhancer', 'Promoter', 'Regulatory']:
            pct = dist.get(state, 0)
            print(f"  {state}: {pct:.1f}%")

# Chi-square test for chromHMM distribution
ct = pd.crosstab(hela_normal_chrom['in_lad'], hela_normal_chrom['chromhmm_group'])
if ct.shape[0] == 2 and ct.shape[1] >= 2:
    chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-square test: chi2={chi2:.1f}, P={p_chi:.2e}, dof={dof}")

# === Step 9: Replication Timing Analysis ===
print("\n" + "=" * 70)
print("Step 9: Early vs Late Replication Timing Analysis")
print("=" * 70)

for desc, subset in [('All CL', l1), ('HeLa only', hela)]:
    print(f"\n--- {desc} ---")
    normal = subset[subset['condition'] == 'normal']

    for metric, label in [('polya_length', 'Poly(A) length'), ('m6a_per_kb', 'm6A/kb')]:
        res = compare_groups(normal, 'repli_timing', 'late', 'early', 'Late', 'Early', metric)
        print(f"  {label}: Late={res['med_a']:.1f} vs Early={res['med_b']:.1f} "
              f"(delta={res['delta']:+.1f}, ratio={res.get('ratio', np.nan):.3f}, P={res['p']:.2e}, "
              f"n={res['n_a']}/{res['n_b']})")
        results_rows.append({
            'comparison': 'Late vs Early replication', 'dataset': desc, 'condition': 'normal',
            'metric': label, **res
        })

# HeLa Stress
print("\n--- HeLa Stress ---")
for metric, label in [('polya_length', 'Poly(A) length'), ('m6a_per_kb', 'm6A/kb')]:
    res = compare_groups(hela_stress, 'repli_timing', 'late', 'early', 'Late', 'Early', metric)
    print(f"  {label}: Late={res['med_a']:.1f} vs Early={res['med_b']:.1f} "
          f"(delta={res['delta']:+.1f}, P={res['p']:.2e}, n={res['n_a']}/{res['n_b']})")
    results_rows.append({
        'comparison': 'Late vs Early replication', 'dataset': 'HeLa only', 'condition': 'stress',
        'metric': label, **res
    })

# Poly(A) delta by replication timing
print("\n--- HeLa Poly(A) Delta by Replication Timing ---")
for timing in ['early', 'late']:
    norm = hela_normal[hela_normal['repli_timing'] == timing]['polya_length']
    strs = hela_stress[hela_stress['repli_timing'] == timing]['polya_length']
    if len(norm) >= 10 and len(strs) >= 10:
        delta = strs.median() - norm.median()
        stat, p = stats.mannwhitneyu(strs, norm, alternative='two-sided')
        print(f"  {timing.capitalize()}: Normal={norm.median():.1f} -> Stress={strs.median():.1f} "
              f"(Delta={delta:+.1f}nt, P={p:.2e}, n={len(norm)}/{len(strs)})")
        results_rows.append({
            'comparison': f'Stress Delta ({timing})', 'dataset': 'HeLa', 'condition': 'stress-normal',
            'metric': 'Poly(A) delta', 'med_a': strs.median(), 'med_b': norm.median(),
            'n_a': len(strs), 'n_b': len(norm), 'delta': delta, 'p': p
        })

# === Step 10: Interaction: LAD × Replication Timing ===
print("\n" + "=" * 70)
print("Step 10: LAD × Replication Timing Interaction (HeLa normal)")
print("=" * 70)

hela_anno = hela_normal[(hela_normal['repli_timing'] != 'unknown')].copy()
for lad in [True, False]:
    for timing in ['early', 'late']:
        subset = hela_anno[(hela_anno['in_lad'] == lad) & (hela_anno['repli_timing'] == timing)]
        lad_label = 'LAD' if lad else 'non-LAD'
        if len(subset) >= 5:
            print(f"  {lad_label} + {timing}: n={len(subset)}, "
                  f"poly(A)={subset['polya_length'].median():.1f}, "
                  f"m6A/kb={subset['m6a_per_kb'].median():.2f}")

# === Step 11: Age stratification within LAD ===
print("\n" + "=" * 70)
print("Step 11: Young vs Ancient within LAD/non-LAD (HeLa normal)")
print("=" * 70)

for lad in [True, False]:
    lad_label = 'LAD' if lad else 'non-LAD'
    subset = hela_normal[hela_normal['in_lad'] == lad]
    for age in ['young', 'ancient']:
        age_sub = subset[subset['l1_age'] == age]
        if len(age_sub) >= 5:
            print(f"  {lad_label} + {age}: n={len(age_sub)}, "
                  f"poly(A)={age_sub['polya_length'].median():.1f}, "
                  f"m6A/kb={age_sub['m6a_per_kb'].median():.2f}")

# Stress delta by LAD × age
print("\n--- Stress Delta by LAD × Age (HeLa) ---")
for lad in [True, False]:
    lad_label = 'LAD' if lad else 'non-LAD'
    for age in ['young', 'ancient']:
        norm = hela_normal[(hela_normal['in_lad'] == lad) & (hela_normal['l1_age'] == age)]['polya_length']
        strs = hela_stress[(hela_stress['in_lad'] == lad) & (hela_stress['l1_age'] == age)]['polya_length']
        if len(norm) >= 10 and len(strs) >= 10:
            delta = strs.median() - norm.median()
            stat, p = stats.mannwhitneyu(strs, norm, alternative='two-sided')
            print(f"  {lad_label} + {age}: Delta={delta:+.1f}nt (P={p:.2e}, n={len(norm)}/{len(strs)})")

# === Step 12: Cross-cell-line consistency ===
print("\n" + "=" * 70)
print("Step 12: Cross-cell-line LAD effects on m6A/kb (normal only)")
print("=" * 70)

for cl in sorted(l1['cellline'].unique()):
    cl_normal = l1[(l1['cellline'] == cl) & (l1['condition'] == 'normal')]
    lad_m6a = cl_normal[cl_normal['in_lad'] == True]['m6a_per_kb']
    nonlad_m6a = cl_normal[cl_normal['in_lad'] == False]['m6a_per_kb']
    if len(lad_m6a) >= 10 and len(nonlad_m6a) >= 10:
        ratio = lad_m6a.median() / nonlad_m6a.median() if nonlad_m6a.median() > 0 else np.nan
        stat, p = stats.mannwhitneyu(lad_m6a, nonlad_m6a, alternative='two-sided')
        print(f"  {cl}: LAD={lad_m6a.median():.2f} vs non-LAD={nonlad_m6a.median():.2f} "
              f"(ratio={ratio:.3f}, P={p:.2e}, n={len(lad_m6a)}/{len(nonlad_m6a)})")

# === Step 13: Save results ===
print("\n" + "=" * 70)
print("Step 13: Saving results")
print("=" * 70)

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUT_DIR, 'lad_replication_results.tsv'), sep='\t', index=False)
print(f"Results saved to {OUT_DIR}/lad_replication_results.tsv")

# Save annotated L1 data
l1[['read_id', 'in_lad', 'repli_timing']].to_csv(
    os.path.join(OUT_DIR, 'l1_lad_repli_annotations.tsv'), sep='\t', index=False)
print(f"Annotations saved to {OUT_DIR}/l1_lad_repli_annotations.tsv")

# === Summary ===
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"LAD regions (hg38): {len(lad_df)} ({total_lad_bp/1e6:.0f} Mb)")
print(f"L1 in LAD: {l1['in_lad'].sum()} ({l1['in_lad'].mean()*100:.1f}%)")
print(f"L1 in early-replicating: {(l1['repli_timing']=='early').sum()}")
print(f"L1 in late-replicating: {(l1['repli_timing']=='late').sum()}")
print(f"L1 unknown replication timing: {(l1['repli_timing']=='unknown').sum()}")
