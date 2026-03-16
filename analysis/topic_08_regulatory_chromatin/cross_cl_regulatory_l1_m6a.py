#!/usr/bin/env python3
"""
Cross-cell-line regulatory L1 m6A comparison.

For each cell line with available Roadmap/ENCODE ChromHMM data,
annotate L1 reads with that cell line's own chromatin states,
then compare m6A/kb on regulatory (enhancer+promoter) vs non-regulatory L1.

Available ChromHMM:
  E117 = HeLa-S3  → HeLa, HeLa-Ars
  E123 = K562      → K562
  E118 = HepG2     → HepG2
  E114 = A549      → A549
  E003 = H1-hESC   → H9 (proxy, both ESC)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import subprocess
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

STATE_GROUPS = {
    '1_TssA': 'Promoter', '2_TssAFlnk': 'Promoter',
    '3_TxFlnk': 'Transcribed', '4_Tx': 'Transcribed', '5_TxWk': 'Transcribed',
    '6_EnhG': 'Enhancer', '7_Enh': 'Enhancer',
    '8_ZNF/Rpts': 'ZNF/Repeats',
    '9_Het': 'Heterochromatin',
    '10_TssBiv': 'Bivalent', '11_BivFlnk': 'Bivalent', '12_EnhBiv': 'Bivalent',
    '13_ReprPC': 'Repressed', '14_ReprPCWk': 'Repressed',
    '15_Quies': 'Quiescent',
}

# ChromHMM mapping: cell line → EID
CL_TO_EID = {
    'HeLa': 'E117',
    'HeLa-Ars': 'E117',
    'K562': 'E123',
    'HepG2': 'E118',
    'A549': 'E114',
    'H9': 'E003',
}

###############################################################################
# 1. Load L1 summary files
###############################################################################
print("=== Loading L1 summary files ===")

summary_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    if 'sample' in df.columns:
        df['group'] = df['sample'].str.rsplit('_', n=1).str[0]
    summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
summary = summary[summary['qc_tag'] == 'PASS'].copy()
summary = summary[summary['polya_length'] > 0].copy()

def classify_age(gene_id):
    subfamily = gene_id.split('_dup')[0] if '_dup' in gene_id else gene_id
    return 'young' if subfamily in YOUNG_SUBFAMILIES else 'ancient'

summary['l1_age'] = summary['gene_id'].apply(classify_age)
summary['cellline'] = summary['group'].apply(lambda x: x.rsplit('_', 1)[0])
summary['condition'] = summary['cellline'].apply(
    lambda x: 'stress' if 'Ars' in x else 'normal'
)

print(f"  Total PASS L1 reads: {len(summary):,}")

###############################################################################
# 2. Load Part3 cache → m6A/kb
###############################################################################
print("\n=== Loading Part3 per-read cache ===")

cache_rows = []
for f in sorted(CACHE_DIR.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_rows.append(df)

cache = pd.concat(cache_rows, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
print(f"  Cached reads: {len(cache):,}")

###############################################################################
# 3. Merge
###############################################################################
print("\n=== Merging ===")
merged = summary.merge(cache[['read_id', 'm6a_per_kb', 'read_length']],
                       on='read_id', how='inner')
merged['midpoint'] = ((merged['start'] + merged['end']) // 2).astype(int)
print(f"  Merged: {len(merged):,}")

###############################################################################
# 4. Per-cell-line ChromHMM annotation
###############################################################################
print("\n=== Per-cell-line ChromHMM annotation ===")

def intersect_chromhmm(reads_df, chromhmm_bed):
    """Intersect read midpoints with ChromHMM BED, return dict read_id → state."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as tmp:
        tmp_bed = tmp.name
        for _, row in reads_df.iterrows():
            mid = int(row['midpoint'])
            tmp.write(f"{row['chr']}\t{mid}\t{mid+1}\t{row['read_id']}\n")

    cmd = f"bedtools intersect -a {tmp_bed} -b {chromhmm_bed} -wa -wb"
    result = subprocess.run(['bash', '-c', cmd],
                          capture_output=True, text=True, timeout=300)
    os.unlink(tmp_bed)

    state_map = {}
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) >= 8:
            state_map[parts[3]] = parts[7]
    return state_map

results = []
all_annotated = []

for cl, eid in CL_TO_EID.items():
    bed_file = OUT_DIR / f'{eid}_15_coreMarks_hg38lift_mnemonics.bed.gz'
    if not bed_file.exists():
        print(f"  {cl} ({eid}): BED file not found, skipping")
        continue

    cl_reads = merged[merged['cellline'] == cl].copy()
    if len(cl_reads) < 50:
        print(f"  {cl} ({eid}): only {len(cl_reads)} reads, skipping")
        continue

    print(f"\n  {cl} ({eid}): {len(cl_reads):,} reads")

    state_map = intersect_chromhmm(cl_reads, bed_file)
    cl_reads['chromhmm_state'] = cl_reads['read_id'].map(state_map)
    cl_reads['chromhmm_group'] = cl_reads['chromhmm_state'].map(STATE_GROUPS)
    cl_reads = cl_reads.dropna(subset=['chromhmm_state'])
    cl_reads['is_regulatory'] = cl_reads['chromhmm_group'].isin(['Enhancer', 'Promoter'])

    print(f"    Annotated: {len(cl_reads):,}")

    all_annotated.append(cl_reads)

    # Ancient only
    ancient = cl_reads[cl_reads['l1_age'] == 'ancient']
    reg = ancient[ancient['is_regulatory']]
    nonreg = ancient[~ancient['is_regulatory']]

    # State distribution
    gc = ancient['chromhmm_group'].value_counts()
    n_ancient = len(ancient)
    print(f"    Ancient reads: {n_ancient:,}")
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed', 'Heterochromatin', 'Quiescent']:
        n = gc.get(grp, 0)
        print(f"      {grp:20s}: {n:5,} ({100*n/n_ancient:5.1f}%)")

    # m6A comparison
    reg_m6a = reg['m6a_per_kb']
    nonreg_m6a = nonreg['m6a_per_kb']
    all_m6a = ancient['m6a_per_kb']

    if len(reg_m6a) >= 10 and len(nonreg_m6a) >= 10:
        _, p = stats.mannwhitneyu(reg_m6a, nonreg_m6a, alternative='two-sided')
        ratio = reg_m6a.median() / nonreg_m6a.median() if nonreg_m6a.median() > 0 else np.nan
        print(f"    m6A/kb: Reg={reg_m6a.median():.2f} vs Non-reg={nonreg_m6a.median():.2f} "
              f"(ratio={ratio:.2f}, p={p:.2e})")
    elif len(reg_m6a) >= 3:
        print(f"    m6A/kb: Reg={reg_m6a.median():.2f} (n={len(reg_m6a)}) — too few for test")

    # Poly(A) comparison
    reg_polya = reg['polya_length']
    nonreg_polya = nonreg['polya_length']
    if len(reg_polya) >= 10 and len(nonreg_polya) >= 10:
        _, p = stats.mannwhitneyu(reg_polya, nonreg_polya, alternative='two-sided')
        print(f"    poly(A): Reg={reg_polya.median():.0f} vs Non-reg={nonreg_polya.median():.0f} "
              f"(Δ={reg_polya.median()-nonreg_polya.median():+.1f}, p={p:.2e})")

    # By chromatin state
    m6a_by_state = {}
    polya_by_state = {}
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed', 'Quiescent']:
        sub = ancient[ancient['chromhmm_group'] == grp]
        if len(sub) >= 10:
            m6a_by_state[grp] = sub['m6a_per_kb'].median()
            polya_by_state[grp] = sub['polya_length'].median()

    results.append({
        'cellline': cl,
        'eid': eid,
        'n_ancient': n_ancient,
        'n_regulatory': len(reg),
        'pct_regulatory': 100 * len(reg) / n_ancient if n_ancient > 0 else 0,
        'reg_m6a_median': reg_m6a.median() if len(reg_m6a) > 0 else np.nan,
        'nonreg_m6a_median': nonreg_m6a.median() if len(nonreg_m6a) > 0 else np.nan,
        'reg_polya_median': reg_polya.median() if len(reg_polya) > 0 else np.nan,
        'nonreg_polya_median': nonreg_polya.median() if len(nonreg_polya) > 0 else np.nan,
        'all_m6a_median': all_m6a.median(),
        **{f'm6a_{k}': v for k, v in m6a_by_state.items()},
        **{f'polya_{k}': v for k, v in polya_by_state.items()},
    })

###############################################################################
# 5. Summary across cell lines
###############################################################################
print("\n" + "="*70)
print("CROSS-CELL-LINE REGULATORY L1 SUMMARY")
print("="*70)

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    # Exclude HeLa-Ars from cross-CL (it's a treatment, not a separate cell line)
    res_no_ars = res_df[res_df['cellline'] != 'HeLa-Ars']

    print(f"\n{'Cell Line':12s} {'EID':5s} {'N_anc':>6s} {'N_reg':>6s} {'%Reg':>6s} "
          f"{'Reg m6A':>8s} {'NonR m6A':>9s} {'Ratio':>6s} {'Reg pA':>7s} {'NonR pA':>8s}")
    print("-"*85)
    for _, row in res_no_ars.iterrows():
        ratio = row['reg_m6a_median'] / row['nonreg_m6a_median'] if row['nonreg_m6a_median'] > 0 else np.nan
        print(f"{row['cellline']:12s} {row['eid']:5s} {row['n_ancient']:6,} {row['n_regulatory']:6,} "
              f"{row['pct_regulatory']:5.1f}% "
              f"{row['reg_m6a_median']:7.2f} {row['nonreg_m6a_median']:8.2f} {ratio:6.2f} "
              f"{row['reg_polya_median']:6.0f} {row['nonreg_polya_median']:7.0f}")

    # Consistency: is regulatory m6A < non-regulatory in all cell lines?
    consistent = (res_no_ars['reg_m6a_median'] < res_no_ars['nonreg_m6a_median']).all()
    print(f"\n  Regulatory m6A < Non-reg in all CLs: {consistent}")

    # m6A by state across cell lines
    print(f"\n  m6A/kb by chromatin state across cell lines:")
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed', 'Quiescent']:
        col = f'm6a_{grp}'
        if col in res_no_ars.columns:
            vals = res_no_ars[col].dropna()
            if len(vals) > 0:
                print(f"    {grp:15s}: {vals.median():.2f} (range {vals.min():.2f}-{vals.max():.2f}, n={len(vals)} CLs)")

    # Poly(A) by state across cell lines
    print(f"\n  Poly(A) by chromatin state across cell lines:")
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed', 'Quiescent']:
        col = f'polya_{grp}'
        if col in res_no_ars.columns:
            vals = res_no_ars[col].dropna()
            if len(vals) > 0:
                print(f"    {grp:15s}: {vals.median():.0f}nt (range {vals.min():.0f}-{vals.max():.0f}, n={len(vals)} CLs)")

    # Regulatory poly(A) < non-regulatory in all CLs?
    polya_consistent = (res_no_ars['reg_polya_median'] < res_no_ars['nonreg_polya_median']).all()
    print(f"\n  Regulatory poly(A) < Non-reg in all CLs: {polya_consistent}")

###############################################################################
# 6. Save
###############################################################################
print("\n=== Saving ===")
res_df.to_csv(OUT_DIR / 'cross_cl_regulatory_l1_m6a.tsv', sep='\t', index=False)
print(f"  Saved: cross_cl_regulatory_l1_m6a.tsv ({len(res_df)} rows)")

if len(all_annotated) > 0:
    all_ann_df = pd.concat(all_annotated, ignore_index=True)
    out_cols = ['read_id', 'chr', 'start', 'end', 'gene_id', 'l1_age',
                'polya_length', 'm6a_per_kb', 'cellline', 'condition',
                'chromhmm_state', 'chromhmm_group', 'is_regulatory']
    out_cols = [c for c in out_cols if c in all_ann_df.columns]
    all_ann_df[out_cols].to_csv(OUT_DIR / 'cross_cl_chromhmm_annotated.tsv',
                                sep='\t', index=False)
    print(f"  Saved: cross_cl_chromhmm_annotated.tsv ({len(all_ann_df):,} reads)")

print("\n=== DONE ===")
