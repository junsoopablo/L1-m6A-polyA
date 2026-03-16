#!/usr/bin/env python3
"""
Part 2: Check if exon overlap within PASS reads affects arsenite poly(A) response.
Also check poly(A) of near-threshold reads (50-99bp overlap) vs clean reads (0bp).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTDIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/exon_overlap_validation'

# Load exon overlap data
ov = pd.read_csv(OUTDIR / 'exon_overlap_all_stage1.tsv', sep='\t')
ov_pass = ov[ov['is_pass']].copy()

# Load L1 summary data for poly(A)
summary_files = {
    'HeLa': ['HeLa_1_1', 'HeLa_2_1', 'HeLa_3_1'],
    'HeLa-Ars': ['HeLa-Ars_1_1', 'HeLa-Ars_2_1', 'HeLa-Ars_3_1'],
}

polya_dfs = []
for cond, samples in summary_files.items():
    for sample in samples:
        path = PROJECT / f'results_group/{sample.rsplit("_", 1)[0]}/g_summary/{sample.rsplit("_", 1)[0]}_L1_summary.tsv'
        # Actually, summaries are grouped differently. Let me load per-sample
        pass

# Load from results_group summaries
group_map = {
    'HeLa_1_1': 'HeLa_1', 'HeLa_2_1': 'HeLa_2', 'HeLa_3_1': 'HeLa_3',
    'HeLa-Ars_1_1': 'HeLa-Ars_1', 'HeLa-Ars_2_1': 'HeLa-Ars_2', 'HeLa-Ars_3_1': 'HeLa-Ars_3',
}

polya_dfs = []
for sample, group in group_map.items():
    path = PROJECT / f'results_group/{group}/g_summary/{group}_L1_summary.tsv'
    if path.exists():
        df = pd.read_csv(path, sep='\t')
        df['sample'] = sample
        df['condition'] = 'HeLa' if 'Ars' not in sample else 'HeLa-Ars'
        polya_dfs.append(df)
    else:
        print(f'  Not found: {path}')

polya = pd.concat(polya_dfs, ignore_index=True)
print(f'Loaded {len(polya)} reads with poly(A) data')
print(f'Columns: {list(polya.columns[:10])}...')

# Merge overlap with poly(A)
polya_cols = ['read_id', 'polya_length']
if 'subfamily' in polya.columns:
    polya_cols.append('subfamily')
merged = ov_pass.merge(polya[polya_cols], on='read_id', how='inner')
print(f'Merged: {len(merged)} reads with both overlap and poly(A) data')

# === Analysis 1: Poly(A) by exon overlap bins within PASS ===
print('\n' + '='*60)
print('POLY(A) BY EXON OVERLAP BIN (PASS READS)')
print('='*60)

bins_pass = [(0, 0, '0 (exact)'),
             (1, 9, '1-9'),
             (10, 24, '10-24'),
             (25, 49, '25-49'),
             (50, 74, '50-74'),
             (75, 99, '75-99')]

for lo, hi, label in bins_pass:
    if lo == 0 and hi == 0:
        mask = merged['exon_overlap'] == 0
    else:
        mask = (merged['exon_overlap'] >= lo) & (merged['exon_overlap'] <= hi)

    sub = merged[mask]
    if len(sub) < 5:
        print(f'  {label:>12s}: n={len(sub)} (too few)')
        continue

    hela = sub[sub['condition'] == 'HeLa']['polya_length'].dropna()
    ars = sub[sub['condition'] == 'HeLa-Ars']['polya_length'].dropna()

    if len(hela) >= 5 and len(ars) >= 5:
        delta = ars.median() - hela.median()
        _, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
        print(f'  {label:>12s}: n={len(sub)} (HeLa={len(hela)}, Ars={len(ars)}) '
              f'| HeLa median={hela.median():.1f}, Ars={ars.median():.1f}, '
              f'Δ={delta:.1f}, p={p:.2e}')
    else:
        print(f'  {label:>12s}: n={len(sub)} (HeLa={len(hela)}, Ars={len(ars)}) '
              f'| insufficient per condition')

# === Analysis 2: Compare 0bp vs near-threshold (50-99bp) ===
print('\n' + '='*60)
print('CLEAN (0bp) vs NEAR-THRESHOLD (50-99bp)')
print('='*60)

clean = merged[merged['exon_overlap'] == 0]
near_thr = merged[(merged['exon_overlap'] >= 50) & (merged['exon_overlap'] < 100)]

for cond in ['HeLa', 'HeLa-Ars']:
    c_clean = clean[clean['condition'] == cond]['polya_length'].dropna()
    c_near = near_thr[near_thr['condition'] == cond]['polya_length'].dropna()
    if len(c_clean) >= 5 and len(c_near) >= 5:
        _, p = stats.mannwhitneyu(c_clean, c_near, alternative='two-sided')
        print(f'  {cond:>12s}: clean median={c_clean.median():.1f} (n={len(c_clean)}), '
              f'near-thr median={c_near.median():.1f} (n={len(c_near)}), '
              f'Δ={c_near.median() - c_clean.median():.1f}, p={p:.2e}')
    else:
        print(f'  {cond:>12s}: clean n={len(c_clean)}, near-thr n={len(c_near)}')

# === Analysis 3: What fraction of rejected reads are rejected ONLY by exon overlap? ===
print('\n' + '='*60)
print('REJECTION REASON ANALYSIS')
print('='*60)

rejected = ov[~ov['is_pass']].copy()
# Reads with overlap <100bp but still rejected = rejected by OTHER filters
rej_by_other = rejected[rejected['exon_overlap'] < 100]
rej_by_exon = rejected[rejected['exon_overlap'] >= 100]
print(f'Total rejected: {len(rejected)}')
print(f'  Rejected by exon overlap ≥100bp: {len(rej_by_exon)} ({100*len(rej_by_exon)/len(rejected):.1f}%)')
print(f'  Rejected by OTHER filters (overlap <100bp): {len(rej_by_other)} ({100*len(rej_by_other)/len(rejected):.1f}%)')

# Among exon-overlap-rejected, overlap distribution
if len(rej_by_exon) > 0:
    print(f'\n  Exon-overlap-rejected distribution:')
    for lo, hi in [(100, 124), (125, 149), (150, 199), (200, 299), (300, 500)]:
        n = ((rej_by_exon['exon_overlap'] >= lo) & (rej_by_exon['exon_overlap'] < hi)).sum()
        print(f'    {lo}-{hi}bp: {n}')

# === Analysis 4: How many reads would change status at different thresholds? ===
print('\n' + '='*60)
print('IMPACT OF THRESHOLD ON PASS SET')
print('='*60)
print('Note: reads must pass ALL filters. Changing exon threshold only affects')
print('reads currently rejected SOLELY by exon overlap.')
print()

# Reads that PASS all other filters but fail exon overlap
# = reads with overlap ≥100bp AND is_pass=False AND would pass if threshold raised
# We can approximate: reads with overlap 100-149bp that are currently rejected
# BUT we don't know if they pass other filters
# We DO know: PASS reads with overlap <100 all pass other filters
# And rejected reads with overlap <100 all FAIL other filters

# Count PASS reads with non-zero overlap (would be affected by lowering threshold)
print('If threshold LOWERED from 100bp:')
for thr in [75, 50, 25]:
    lost = ((ov_pass['exon_overlap'] >= thr) & (ov_pass['exon_overlap'] < 100)).sum()
    print(f'  Threshold={thr}: would lose {lost} currently PASS reads')

print(f'\nTotal PASS reads: {len(ov_pass)}')
print(f'  With overlap 0bp: {(ov_pass["exon_overlap"] == 0).sum()} '
      f'({100*(ov_pass["exon_overlap"]==0).sum()/len(ov_pass):.1f}%)')

# === Analysis 5: Is there a bimodal split? ===
print('\n' + '='*60)
print('BIMODAL ANALYSIS: PASS vs REJECTED OVERLAP DISTRIBUTIONS')
print('='*60)

# For ALL stage1 reads
for lo in [0, 25, 50, 75, 100, 125, 150]:
    n_pass = ((ov['exon_overlap'] >= lo) & (ov['exon_overlap'] < lo + 25) & ov['is_pass']).sum()
    n_total = ((ov['exon_overlap'] >= lo) & (ov['exon_overlap'] < lo + 25)).sum()
    if n_total > 0:
        pct_pass = 100 * n_pass / n_total
        print(f'  Overlap {lo:>3d}-{lo+24:>3d}bp: {n_total:>6d} total, {n_pass:>5d} PASS ({pct_pass:.1f}%)')

# Compute: what % of reads in each overlap bin are rejected by OTHER filters?
print('\n  Of reads with overlap <100bp:')
for lo in [0, 25, 50, 75]:
    mask = (ov['exon_overlap'] >= lo) & (ov['exon_overlap'] < lo + 25)
    n_total = mask.sum()
    n_pass = (mask & ov['is_pass']).sum()
    if n_total > 0:
        print(f'    {lo:>3d}-{lo+24:>3d}bp: {n_total:>6d} total, {n_pass:>5d} PASS '
              f'({100*n_pass/n_total:.1f}%), {n_total-n_pass:>5d} rejected by other filters')

print('\nDone!')
