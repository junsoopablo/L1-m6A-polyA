#!/usr/bin/env python3
"""
Unified L1 autonomous vs read-through classifier — GENCODE-independent.

Goal: Find the optimal structural features to distinguish autonomous L1 transcripts
from host gene read-through, using only RepeatMasker + alignment features.

Features tested:
  1. L1 overlap fraction (overlap_length / read_span)
  2. dist_to_3prime (L1 3'end ↔ read 3'end)
  3. Flanking extension (read extends beyond L1 on 5' and 3' sides)
  4. Multi-TE span (read overlaps multiple distinct TE families)
  5. Combinations of above

Validation: arsenite poly(A) response as the biological ground truth
  - Autonomous L1 → arsenite-sensitive (Δ < 0)
  - Host read-through → arsenite-immune (Δ ≈ 0)
"""

import re
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'unified_classifier'
OUTDIR.mkdir(exist_ok=True)

TE_GTF = PROJECT / 'reference/hg38_rmsk_TE.gtf'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

# =========================================================================
# Step 1: Build L1 element coordinate lookup from RepeatMasker GTF
# =========================================================================
print("Step 1: Building L1 element lookup from RepeatMasker...")

l1_elements = {}  # transcript_id → {chr, start, end, strand}
with open(TE_GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 9:
            continue
        # Check if L1 family
        m_fam = re.search(r'family_id "([^"]*)"', fields[8])
        if not m_fam or m_fam.group(1) != 'L1':
            continue
        m_tid = re.search(r'transcript_id "([^"]*)"', fields[8])
        if not m_tid:
            continue
        tid = m_tid.group(1)
        chrom = fields[0]
        start = int(fields[3]) - 1  # 0-based
        end = int(fields[4])
        strand = fields[6]
        l1_elements[tid] = {'chr': chrom, 'start': start, 'end': end, 'strand': strand}

print(f"  L1 elements loaded: {len(l1_elements)}")

# Also build ALL TE BED for multi-TE detection
print("  Building all-TE BED...")
all_te_bed = OUTDIR / 'all_te_elements.bed'
if not all_te_bed.exists():
    te_lines = []
    with open(TE_GTF) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            m_fam = re.search(r'family_id "([^"]*)"', fields[8])
            m_gid = re.search(r'gene_id "([^"]*)"', fields[8])
            fam = m_fam.group(1) if m_fam else 'unknown'
            gid = m_gid.group(1) if m_gid else 'unknown'
            te_lines.append(f"{chrom}\t{start}\t{end}\t{fam}|{gid}\n")
    tmp = str(all_te_bed) + '.unsorted'
    with open(tmp, 'w') as f:
        f.writelines(te_lines)
    subprocess.run(f"sort -k1,1 -k2,2n {tmp} > {all_te_bed} && rm {tmp}",
                   shell=True, check=True)
    print(f"  All TE BED: {len(te_lines)} entries")
else:
    print("  All-TE BED exists, skipping.")

# =========================================================================
# Step 2: Load PASS L1 reads (HeLa + HeLa-Ars)
# =========================================================================
print("\nStep 2: Loading PASS L1 reads...")

pass_list = []
for cl, groups in CELL_LINES.items():
    for grp in groups:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        df = pd.read_csv(f, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['cell_line'] = cl
        df['group'] = grp
        df['source'] = 'PASS'
        df['age'] = df['class'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
        # Compute features
        df['overlap_frac'] = df['overlap_length'] / df['read_length']
        # dist_to_3prime already exists
        # Compute flanking extensions using L1 element lookup
        flanking_5 = []
        flanking_3 = []
        for _, row in df.iterrows():
            tid = row.get('transcript_id', '')
            el = l1_elements.get(tid, None)
            if el and el['chr'] == row['chr']:
                if el['strand'] == '+':
                    f5 = max(0, el['start'] - row['start'])
                    f3 = max(0, row['end'] - el['end'])
                else:
                    f5 = max(0, row['end'] - el['end'])
                    f3 = max(0, el['start'] - row['start'])
                flanking_5.append(f5)
                flanking_3.append(f3)
            else:
                flanking_5.append(np.nan)
                flanking_3.append(np.nan)
        df['flank_5prime'] = flanking_5
        df['flank_3prime'] = flanking_3
        df['total_flank'] = df['flank_5prime'].fillna(0) + df['flank_3prime'].fillna(0)
        df['flank_frac'] = df['total_flank'] / df['read_length']

        pass_list.append(df[['read_id', 'chr', 'start', 'end', 'read_length',
                             'overlap_length', 'overlap_frac', 'dist_to_3prime',
                             'flank_5prime', 'flank_3prime', 'total_flank', 'flank_frac',
                             'transcript_id', 'gene_id', 'cell_line', 'group',
                             'polya_length', 'age', 'source', 'TE_group']])

pass_df = pd.concat(pass_list, ignore_index=True)
print(f"  PASS reads (HeLa + HeLa-Ars): {len(pass_df)}")

# =========================================================================
# Step 3: Load Cat B reads (HeLa + HeLa-Ars) and compute features
# =========================================================================
print("\nStep 3: Loading Cat B reads and computing features...")

catb_list = []
for cl, groups in CELL_LINES.items():
    for grp in groups:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if not f.exists():
            continue
        df = pd.read_csv(f, sep='\t')
        df['cell_line'] = cl
        df['source'] = 'CatB'
        df['age'] = df['age'].str.strip()

        # Compute features using L1 element lookup
        overlap_lengths = []
        dist_to_3primes = []
        flanking_5 = []
        flanking_3 = []

        for _, row in df.iterrows():
            tid = row['locus_id']
            el = l1_elements.get(tid, None)
            if el and el['chr'] == row['chr']:
                # Overlap = intersection of read and L1 element
                ov_start = max(row['start'], el['start'])
                ov_end = min(row['end'], el['end'])
                ov = max(0, ov_end - ov_start)
                overlap_lengths.append(ov)

                # dist_to_3prime: distance from L1's 3' end to read's 3' end
                if el['strand'] == '+':
                    d3p = abs(row['end'] - el['end'])
                    f5 = max(0, el['start'] - row['start'])
                    f3 = max(0, row['end'] - el['end'])
                else:
                    d3p = abs(el['start'] - row['start'])
                    f5 = max(0, row['end'] - el['end'])
                    f3 = max(0, el['start'] - row['start'])
                dist_to_3primes.append(d3p)
                flanking_5.append(f5)
                flanking_3.append(f3)
            else:
                overlap_lengths.append(np.nan)
                dist_to_3primes.append(np.nan)
                flanking_5.append(np.nan)
                flanking_3.append(np.nan)

        df['overlap_length'] = overlap_lengths
        df['overlap_frac'] = df['overlap_length'] / df['read_span']
        df['dist_to_3prime'] = dist_to_3primes
        df['flank_5prime'] = flanking_5
        df['flank_3prime'] = flanking_3
        df['total_flank'] = pd.Series(flanking_5).fillna(0) + pd.Series(flanking_3).fillna(0)
        df['flank_frac'] = df['total_flank'] / df['read_span']

        catb_list.append(df.rename(columns={
            'read_span': 'read_length', 'locus_id': 'transcript_id',
            'subfamily': 'gene_id'
        })[['read_id', 'chr', 'start', 'end', 'read_length',
            'overlap_length', 'overlap_frac', 'dist_to_3prime',
            'flank_5prime', 'flank_3prime', 'total_flank', 'flank_frac',
            'transcript_id', 'gene_id', 'cell_line', 'group',
            'age', 'source']])

catb_df = pd.concat(catb_list, ignore_index=True)
print(f"  Cat B reads (HeLa + HeLa-Ars): {len(catb_df)}")

# Merge Cat B with poly(A)
catb_polya = []
for cl, groups in CELL_LINES.items():
    for grp in groups:
        f = RESULTS / grp / 'j_catB' / f'{grp}.catB.nanopolish.polya.tsv.gz'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df = df[df['qc_tag'] == 'PASS'].copy()
            catb_polya.append(df[['readname', 'polya_length']].rename(
                columns={'readname': 'read_id', 'polya_length': 'polya_length'}))
catb_polya = pd.concat(catb_polya, ignore_index=True)
catb_df = catb_df.merge(catb_polya, on='read_id', how='left')
print(f"  Cat B with poly(A): {catb_df['polya_length'].notna().sum()}")

# Add TE_group placeholder for Cat B
catb_df['TE_group'] = 'catB'

# =========================================================================
# Step 4: Multi-TE detection
# =========================================================================
print("\nStep 4: Multi-TE detection...")

# Write all reads (PASS + Cat B) as BED
all_reads = pd.concat([
    pass_df[['read_id', 'chr', 'start', 'end']],
    catb_df[['read_id', 'chr', 'start', 'end']]
], ignore_index=True).drop_duplicates(subset='read_id')

tmp_reads = tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False, prefix='all_reads_')
for _, row in all_reads.iterrows():
    tmp_reads.write(f"{row['chr']}\t{row['start']}\t{row['end']}\t{row['read_id']}\n")
tmp_reads.close()

tmp_sorted = tmp_reads.name + '.sorted'
subprocess.run(f"sort -k1,1 -k2,2n {tmp_reads.name} > {tmp_sorted}", shell=True, check=True)

# Intersect with all TEs
print("  Running bedtools intersect (all TEs)...")
result = subprocess.run(
    f"bedtools intersect -a {tmp_sorted} -b {all_te_bed} -wo",
    shell=True, capture_output=True, text=True)

# Count distinct TE families per read
te_per_read = defaultdict(set)
l1_overlap_per_read = defaultdict(int)
for line in result.stdout.strip().split('\n'):
    if not line:
        continue
    fields = line.split('\t')
    rid = fields[3]
    te_info = fields[7]  # family|gene_id
    te_family = te_info.split('|')[0] if '|' in te_info else te_info
    ov = int(fields[-1])
    if ov >= 50:  # minimum overlap to count
        te_per_read[rid].add(te_family)
    if te_family == 'L1' and ov > l1_overlap_per_read[rid]:
        l1_overlap_per_read[rid] = ov

import os
os.unlink(tmp_reads.name)
os.unlink(tmp_sorted)

# Map to dataframes
pass_df['n_te_families'] = pass_df['read_id'].map(lambda x: len(te_per_read.get(x, set())))
pass_df['has_non_l1_te'] = pass_df['read_id'].map(
    lambda x: len(te_per_read.get(x, set()) - {'L1'}) > 0)

catb_df['n_te_families'] = catb_df['read_id'].map(lambda x: len(te_per_read.get(x, set())))
catb_df['has_non_l1_te'] = catb_df['read_id'].map(
    lambda x: len(te_per_read.get(x, set()) - {'L1'}) > 0)

print(f"  PASS: {pass_df['has_non_l1_te'].mean()*100:.1f}% overlap non-L1 TE")
print(f"  Cat B: {catb_df['has_non_l1_te'].mean()*100:.1f}% overlap non-L1 TE")
print(f"  PASS n_te_families: median={pass_df['n_te_families'].median():.0f}, "
      f"mean={pass_df['n_te_families'].mean():.1f}")
print(f"  Cat B n_te_families: median={catb_df['n_te_families'].median():.0f}, "
      f"mean={catb_df['n_te_families'].mean():.1f}")

# =========================================================================
# Step 5: Merge into unified dataset
# =========================================================================
print("\nStep 5: Merging into unified dataset...")

# Ensure consistent columns
cols = ['read_id', 'chr', 'start', 'end', 'read_length', 'overlap_length',
        'overlap_frac', 'dist_to_3prime', 'flank_5prime', 'flank_3prime',
        'total_flank', 'flank_frac', 'transcript_id', 'gene_id',
        'cell_line', 'group', 'polya_length', 'age', 'source', 'TE_group',
        'n_te_families', 'has_non_l1_te']

unified = pd.concat([pass_df[cols], catb_df[cols]], ignore_index=True)
unified = unified[unified['age'] == 'ancient'].copy()  # ancient only for clean comparison
unified = unified.dropna(subset=['polya_length'])

print(f"  Unified ancient reads with poly(A): {len(unified)}")
print(f"    PASS: {(unified['source']=='PASS').sum()}")
print(f"    CatB: {(unified['source']=='CatB').sum()}")

# =========================================================================
# Step 6: Feature distributions — PASS vs Cat B
# =========================================================================
print("\n" + "=" * 80)
print("FEATURE DISTRIBUTIONS — PASS vs Cat B")
print("=" * 80)

for feat in ['overlap_frac', 'dist_to_3prime', 'flank_frac', 'n_te_families']:
    p_vals = unified[unified['source'] == 'PASS'][feat].dropna()
    c_vals = unified[unified['source'] == 'CatB'][feat].dropna()
    _, p = stats.mannwhitneyu(p_vals, c_vals)
    print(f"\n  {feat}:")
    print(f"    PASS: median={p_vals.median():.3f}, mean={p_vals.mean():.3f}")
    print(f"    CatB: median={c_vals.median():.3f}, mean={c_vals.mean():.3f}")
    print(f"    MWU p={p:.2e}")

# =========================================================================
# Step 7: Single-feature classifiers — Arsenite response
# =========================================================================
print("\n" + "=" * 80)
print("CLASSIFIER TEST: Single features")
print("=" * 80)

def ars_test(data, label):
    """Compute arsenite Δpoly(A) for a subset."""
    hela = data[data['cell_line'] == 'HeLa']['polya_length'].dropna()
    ars = data[data['cell_line'] == 'HeLa-Ars']['polya_length'].dropna()
    if len(hela) < 5 or len(ars) < 5:
        return {'label': label, 'n_hela': len(hela), 'n_ars': len(ars),
                'hela_med': np.nan, 'ars_med': np.nan, 'delta': np.nan, 'p': np.nan}
    _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
    return {
        'label': label, 'n_hela': len(hela), 'n_ars': len(ars),
        'hela_med': hela.median(), 'ars_med': ars.median(),
        'delta': ars.median() - hela.median(), 'p': p
    }

def print_ars_results(results):
    print(f"  {'Label':45s} {'HeLa':>8s} {'Ars':>8s} {'Δ':>8s} {'p':>12s} {'n':>10s}")
    print("  " + "-" * 95)
    for r in results:
        if np.isnan(r['delta']):
            print(f"  {r['label']:45s}  (too few: {r['n_hela']}+{r['n_ars']})")
        else:
            print(f"  {r['label']:45s} {r['hela_med']:8.1f} {r['ars_med']:8.1f} "
                  f"{r['delta']:+8.1f} {r['p']:12.2e} {r['n_hela']+r['n_ars']:10d}")

# --- Feature 1: overlap_frac ---
print("\n--- Feature 1: overlap_frac (L1 overlap / read length) ---")
results = []
for lo, hi, label in [(0, 0.2, 'ov<0.2'), (0.2, 0.4, '0.2≤ov<0.4'),
                        (0.4, 0.6, '0.4≤ov<0.6'), (0.6, 0.8, '0.6≤ov<0.8'),
                        (0.8, 1.5, 'ov≥0.8')]:
    sub = unified[(unified['overlap_frac'] >= lo) & (unified['overlap_frac'] < hi)]
    n_pass = (sub['source'] == 'PASS').sum()
    n_catb = (sub['source'] == 'CatB').sum()
    results.append(ars_test(sub, f'{label} (PASS={n_pass}, CatB={n_catb})'))
print_ars_results(results)

# --- Feature 2: dist_to_3prime ---
print("\n--- Feature 2: dist_to_3prime (L1 3'end ↔ read 3'end) ---")
results = []
for lo, hi, label in [(0, 30, 'd3p<30'), (30, 100, '30≤d3p<100'),
                        (100, 300, '100≤d3p<300'), (300, 1000, '300≤d3p<1000'),
                        (1000, 999999, 'd3p≥1000')]:
    sub = unified[(unified['dist_to_3prime'] >= lo) & (unified['dist_to_3prime'] < hi)]
    n_pass = (sub['source'] == 'PASS').sum()
    n_catb = (sub['source'] == 'CatB').sum()
    results.append(ars_test(sub, f'{label} (PASS={n_pass}, CatB={n_catb})'))
print_ars_results(results)

# --- Feature 3: flank_frac ---
print("\n--- Feature 3: flank_frac (non-L1 flanking / read length) ---")
results = []
for lo, hi, label in [(0, 0.1, 'flank<10%'), (0.1, 0.3, '10-30%'),
                        (0.3, 0.5, '30-50%'), (0.5, 0.7, '50-70%'),
                        (0.7, 2.0, '≥70%')]:
    sub = unified[(unified['flank_frac'] >= lo) & (unified['flank_frac'] < hi)]
    n_pass = (sub['source'] == 'PASS').sum()
    n_catb = (sub['source'] == 'CatB').sum()
    results.append(ars_test(sub, f'{label} (PASS={n_pass}, CatB={n_catb})'))
print_ars_results(results)

# --- Feature 4: n_te_families ---
print("\n--- Feature 4: n_te_families (distinct TE families overlapping read) ---")
results = []
for n, label in [(1, 'n_te=1 (L1 only)'), (2, 'n_te=2'), (3, 'n_te=3'),
                  (None, 'n_te≥4')]:
    if n is not None:
        sub = unified[unified['n_te_families'] == n]
    else:
        sub = unified[unified['n_te_families'] >= 4]
    n_pass = (sub['source'] == 'PASS').sum()
    n_catb = (sub['source'] == 'CatB').sum()
    results.append(ars_test(sub, f'{label} (PASS={n_pass}, CatB={n_catb})'))
print_ars_results(results)

# --- Feature 5: has_non_l1_te ---
print("\n--- Feature 5: has_non_l1_te (read overlaps Alu/SINE/etc) ---")
results = []
for val, label in [(False, 'L1 only (no other TE)'), (True, 'Has non-L1 TE')]:
    sub = unified[unified['has_non_l1_te'] == val]
    n_pass = (sub['source'] == 'PASS').sum()
    n_catb = (sub['source'] == 'CatB').sum()
    results.append(ars_test(sub, f'{label} (PASS={n_pass}, CatB={n_catb})'))
print_ars_results(results)

# =========================================================================
# Step 8: Combined classifiers
# =========================================================================
print("\n" + "=" * 80)
print("CLASSIFIER TEST: Combinations")
print("=" * 80)

# Create composite scores
unified['composite_1'] = unified['overlap_frac'] * (1 - unified['flank_frac'].clip(0, 1))
# Higher = more L1-like

# Approach A: overlap_frac > 0.5 AND n_te_families ≤ 2
print("\n--- Approach A: overlap_frac > 0.5 AND n_te ≤ 2 ---")
mask_auto = (unified['overlap_frac'] > 0.5) & (unified['n_te_families'] <= 2)
results = [
    ars_test(unified[mask_auto], 'Autonomous (ov>0.5, nte≤2)'),
    ars_test(unified[~mask_auto], 'Read-through (rest)'),
]
print_ars_results(results)

# Approach B: overlap_frac > 0.3 AND dist_to_3prime < 200
print("\n--- Approach B: overlap_frac > 0.3 AND d3p < 200 ---")
mask_auto = (unified['overlap_frac'] > 0.3) & (unified['dist_to_3prime'] < 200)
results = [
    ars_test(unified[mask_auto], 'Autonomous (ov>0.3, d3p<200)'),
    ars_test(unified[~mask_auto], 'Read-through (rest)'),
]
print_ars_results(results)

# Approach C: n_te_families == 1 (L1 only, no other TE)
print("\n--- Approach C: n_te_families == 1 (pure L1 reads) ---")
mask_auto = unified['n_te_families'] == 1
results = [
    ars_test(unified[mask_auto], 'Pure L1 (nte=1)'),
    ars_test(unified[~mask_auto], 'Multi-TE (nte≥2)'),
]
print_ars_results(results)

# Approach D: overlap_frac > 0.4
print("\n--- Approach D: overlap_frac > 0.4 ---")
mask_auto = unified['overlap_frac'] > 0.4
results = [
    ars_test(unified[mask_auto], 'High overlap (ov>0.4)'),
    ars_test(unified[~mask_auto], 'Low overlap (ov≤0.4)'),
]
print_ars_results(results)

# Approach E: flank_frac < 0.5 (read is mostly L1 + flanking sequence, not too much non-L1)
print("\n--- Approach E: flank_frac < 0.5 ---")
mask_auto = unified['flank_frac'] < 0.5
results = [
    ars_test(unified[mask_auto], 'Low flank (ff<0.5)'),
    ars_test(unified[~mask_auto], 'High flank (ff≥0.5)'),
]
print_ars_results(results)

# Approach F: n_te_families ≤ 2 AND overlap_frac > 0.3
print("\n--- Approach F: n_te ≤ 2 AND overlap_frac > 0.3 ---")
mask_auto = (unified['n_te_families'] <= 2) & (unified['overlap_frac'] > 0.3)
results = [
    ars_test(unified[mask_auto], 'Autonomous (nte≤2, ov>0.3)'),
    ars_test(unified[~mask_auto], 'Read-through (rest)'),
]
print_ars_results(results)

# Approach G: has_non_l1_te == False AND overlap_frac > 0.3
print("\n--- Approach G: no non-L1 TE AND overlap_frac > 0.3 ---")
mask_auto = (~unified['has_non_l1_te']) & (unified['overlap_frac'] > 0.3)
results = [
    ars_test(unified[mask_auto], 'Autonomous (no-other-TE, ov>0.3)'),
    ars_test(unified[~mask_auto], 'Read-through (rest)'),
]
print_ars_results(results)

# =========================================================================
# Step 9: Systematic threshold sweep
# =========================================================================
print("\n" + "=" * 80)
print("THRESHOLD SWEEP: overlap_frac cutoff")
print("=" * 80)

print(f"  {'Threshold':>10s} {'n_auto':>8s} {'n_RT':>8s} {'Δ_auto':>8s} {'Δ_RT':>8s} "
      f"{'p_auto':>12s} {'p_RT':>12s} {'|ΔΔ|':>8s}")
print("  " + "-" * 80)

best_sep = {'threshold': 0, 'delta_delta': 0}
for thresh in np.arange(0.15, 0.90, 0.05):
    auto = unified[unified['overlap_frac'] > thresh]
    rt = unified[unified['overlap_frac'] <= thresh]
    r_auto = ars_test(auto, f'ov>{thresh:.2f}')
    r_rt = ars_test(rt, f'ov≤{thresh:.2f}')
    if not np.isnan(r_auto['delta']) and not np.isnan(r_rt['delta']):
        dd = abs(r_auto['delta'] - r_rt['delta'])
        print(f"  {thresh:10.2f} {r_auto['n_hela']+r_auto['n_ars']:8d} "
              f"{r_rt['n_hela']+r_rt['n_ars']:8d} {r_auto['delta']:+8.1f} "
              f"{r_rt['delta']:+8.1f} {r_auto['p']:12.2e} {r_rt['p']:12.2e} {dd:8.1f}")
        if dd > best_sep['delta_delta']:
            best_sep = {'threshold': thresh, 'delta_delta': dd,
                        'auto_delta': r_auto['delta'], 'rt_delta': r_rt['delta']}

print(f"\n  Best separation at overlap_frac = {best_sep['threshold']:.2f}: "
      f"|ΔΔ| = {best_sep['delta_delta']:.1f} "
      f"(auto Δ={best_sep['auto_delta']:+.1f}, RT Δ={best_sep['rt_delta']:+.1f})")

# =========================================================================
# Step 10: Systematic threshold sweep — n_te_families
# =========================================================================
print("\n" + "=" * 80)
print("THRESHOLD SWEEP: n_te_families cutoff")
print("=" * 80)

print(f"  {'Threshold':>10s} {'n_auto':>8s} {'n_RT':>8s} {'Δ_auto':>8s} {'Δ_RT':>8s} "
      f"{'p_auto':>12s} {'p_RT':>12s} {'|ΔΔ|':>8s}")
print("  " + "-" * 80)

for thresh in [1, 2, 3, 4, 5]:
    auto = unified[unified['n_te_families'] <= thresh]
    rt = unified[unified['n_te_families'] > thresh]
    r_auto = ars_test(auto, f'nte≤{thresh}')
    r_rt = ars_test(rt, f'nte>{thresh}')
    if not np.isnan(r_auto['delta']) and not np.isnan(r_rt['delta']):
        dd = abs(r_auto['delta'] - r_rt['delta'])
        print(f"  nte≤{thresh:7d} {r_auto['n_hela']+r_auto['n_ars']:8d} "
              f"{r_rt['n_hela']+r_rt['n_ars']:8d} {r_auto['delta']:+8.1f} "
              f"{r_rt['delta']:+8.1f} {r_auto['p']:12.2e} {r_rt['p']:12.2e} {dd:8.1f}")

# =========================================================================
# Step 11: Compare with current PASS/CatB classification
# =========================================================================
print("\n" + "=" * 80)
print("REFERENCE: Current PASS vs Cat B classification")
print("=" * 80)

results = [
    ars_test(unified[unified['source'] == 'PASS'], 'Current PASS'),
    ars_test(unified[unified['source'] == 'CatB'], 'Current Cat B'),
]
print_ars_results(results)

# What if we just merge everything?
print("\n--- All reads merged (no filter) ---")
results = [ars_test(unified, 'All reads (PASS + CatB merged)')]
print_ars_results(results)

# =========================================================================
# Step 12: Save unified dataset
# =========================================================================
unified.to_csv(OUTDIR / 'unified_hela_ars_features.tsv', sep='\t', index=False)
print(f"\nSaved to: {OUTDIR}/unified_hela_ars_features.tsv")
print("Done!")
