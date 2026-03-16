#!/usr/bin/env python3
"""Check whether Cat B results are driven by a few dominant loci."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_07 = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic'
OUTDIR = TOPIC_07 / 'catB_vs_pass_analysis'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

# =============================================================================
# Load Cat B metadata + poly(A)
# =============================================================================
print("Loading Cat B metadata...")
catB_meta_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = TOPIC_07 / f'catB_reads_{grp}.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['cell_line'] = group
            catB_meta_list.append(df)

catB_meta = pd.concat(catB_meta_list, ignore_index=True)
catB_meta['is_young'] = catB_meta['subfamily'].isin(YOUNG)

print("Loading Cat B poly(A)...")
catB_polya_list = []
for group, reps in CELL_LINES.items():
    for grp in reps:
        f = RESULTS / grp / 'j_catB' / f'{grp}.catB.nanopolish.polya.tsv.gz'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df = df[df['qc_tag'] == 'PASS'].copy()
            df['group'] = grp
            df['cell_line'] = group
            df = df.rename(columns={'readname': 'read_id', 'polya_length': 'polya'})
            catB_polya_list.append(df[['read_id', 'group', 'cell_line', 'polya']])

catB_polya = pd.concat(catB_polya_list, ignore_index=True)
catB_polya = catB_polya.merge(
    catB_meta[['read_id', 'group', 'locus_id', 'subfamily', 'is_young', 'age']].drop_duplicates(),
    on=['read_id', 'group'], how='inner'
)
print(f"  Cat B poly(A) with locus info: {len(catB_polya)}")

# =============================================================================
# 1. Loci concentration analysis (all CLs)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOCI CONCENTRATION")
print("=" * 70)

locus_counts = catB_meta.groupby('locus_id').size().sort_values(ascending=False)
print(f"\nTotal loci: {len(locus_counts)}")
print(f"Singleton loci (1 read): {(locus_counts == 1).sum()} ({(locus_counts == 1).mean()*100:.1f}%)")
print(f"Multi-read loci (≥2): {(locus_counts >= 2).sum()}")
print(f"High-read loci (≥10): {(locus_counts >= 10).sum()}")
print(f"Very high (≥50): {(locus_counts >= 50).sum()}")

print(f"\nTop 20 loci:")
top20 = locus_counts.head(20)
for locus, cnt in top20.items():
    sub = catB_meta[catB_meta['locus_id'] == locus].iloc[0]
    print(f"  {locus:30s}  {cnt:5d} reads  {sub['subfamily']:12s}  {sub['age']}")

total_reads = len(catB_meta)
top10_reads = locus_counts.head(10).sum()
top50_reads = locus_counts.head(50).sum()
print(f"\nTop 10 loci: {top10_reads} reads ({top10_reads/total_reads*100:.1f}% of total)")
print(f"Top 50 loci: {top50_reads} reads ({top50_reads/total_reads*100:.1f}% of total)")

# =============================================================================
# 2. HeLa/HeLa-Ars loci concentration
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: HELA / HELA-ARS LOCI")
print("=" * 70)

hela_catB = catB_polya[catB_polya['cell_line'] == 'HeLa']
ars_catB = catB_polya[catB_polya['cell_line'] == 'HeLa-Ars']

print(f"\nHeLa Cat B: {len(hela_catB)} reads, {hela_catB['locus_id'].nunique()} loci")
print(f"HeLa-Ars Cat B: {len(ars_catB)} reads, {ars_catB['locus_id'].nunique()} loci")

hela_locus_cnt = hela_catB.groupby('locus_id').size().sort_values(ascending=False)
ars_locus_cnt = ars_catB.groupby('locus_id').size().sort_values(ascending=False)

print(f"\nHeLa singleton: {(hela_locus_cnt == 1).sum()} / {len(hela_locus_cnt)} ({(hela_locus_cnt == 1).mean()*100:.1f}%)")
print(f"HeLa-Ars singleton: {(ars_locus_cnt == 1).sum()} / {len(ars_locus_cnt)} ({(ars_locus_cnt == 1).mean()*100:.1f}%)")

print(f"\nHeLa top 10 loci:")
for locus, cnt in hela_locus_cnt.head(10).items():
    med = hela_catB[hela_catB['locus_id'] == locus]['polya'].median()
    print(f"  {locus:30s}  {cnt:3d} reads  poly(A)={med:.1f}")

print(f"\nHeLa-Ars top 10 loci:")
for locus, cnt in ars_locus_cnt.head(10).items():
    med = ars_catB[ars_catB['locus_id'] == locus]['polya'].median()
    print(f"  {locus:30s}  {cnt:3d} reads  poly(A)={med:.1f}")

# =============================================================================
# 3. Per-locus HeLa vs HeLa-Ars poly(A) comparison (shared loci)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: SHARED LOCI — PER-LOCUS ARSENITE EFFECT")
print("=" * 70)

shared_loci = set(hela_catB['locus_id'].unique()) & set(ars_catB['locus_id'].unique())
print(f"\nShared loci between HeLa and HeLa-Ars: {len(shared_loci)}")
print(f"HeLa-only loci: {hela_catB['locus_id'].nunique() - len(shared_loci)}")
print(f"Ars-only loci: {ars_catB['locus_id'].nunique() - len(shared_loci)}")

# Per-locus comparison
locus_rows = []
for locus in shared_loci:
    h = hela_catB[hela_catB['locus_id'] == locus]['polya']
    a = ars_catB[ars_catB['locus_id'] == locus]['polya']
    locus_rows.append({
        'locus_id': locus,
        'hela_n': len(h), 'ars_n': len(a),
        'hela_median': h.median(), 'ars_median': a.median(),
        'delta': a.median() - h.median(),
    })

locus_df = pd.DataFrame(locus_rows)
locus_df = locus_df.sort_values('hela_n', ascending=False)

print(f"\nPer-locus delta (all {len(locus_df)} shared loci):")
print(f"  Mean delta: {locus_df['delta'].mean():+.1f} nt")
print(f"  Median delta: {locus_df['delta'].median():+.1f} nt")
print(f"  Loci with shortening (Δ<0): {(locus_df['delta'] < 0).sum()} ({(locus_df['delta'] < 0).mean()*100:.1f}%)")
print(f"  Loci with lengthening (Δ>0): {(locus_df['delta'] > 0).sum()} ({(locus_df['delta'] > 0).mean()*100:.1f}%)")

# Filter to loci with ≥3 reads in each condition
robust = locus_df[(locus_df['hela_n'] >= 3) & (locus_df['ars_n'] >= 3)]
print(f"\nRobust loci (≥3 reads each): {len(robust)}")
if len(robust) > 0:
    print(f"  Mean delta: {robust['delta'].mean():+.1f} nt")
    print(f"  Median delta: {robust['delta'].median():+.1f} nt")
    print(f"  Shortening: {(robust['delta'] < 0).sum()} / {len(robust)}")
    print(f"\n  Top loci by read count:")
    for _, row in robust.head(20).iterrows():
        print(f"    {row['locus_id']:30s}  HeLa={row['hela_n']:3.0f}r/{row['hela_median']:6.1f}nt  "
              f"Ars={row['ars_n']:3.0f}r/{row['ars_median']:6.1f}nt  Δ={row['delta']:+.1f}")

locus_df.to_csv(OUTDIR / 'hela_ars_shared_loci_polya.tsv', sep='\t', index=False)

# =============================================================================
# 4. Exclude top loci and re-check overall
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: SENSITIVITY — EXCLUDE TOP LOCI")
print("=" * 70)

# Overall top loci (across HeLa + HeLa-Ars)
combined = pd.concat([hela_catB, ars_catB])
combined_locus_cnt = combined.groupby('locus_id').size().sort_values(ascending=False)

for exclude_n in [5, 10, 20, 50]:
    top_loci = set(combined_locus_cnt.head(exclude_n).index)
    h_filt = hela_catB[~hela_catB['locus_id'].isin(top_loci)]['polya']
    a_filt = ars_catB[~ars_catB['locus_id'].isin(top_loci)]['polya']
    if len(h_filt) > 5 and len(a_filt) > 5:
        _, p = stats.mannwhitneyu(h_filt.dropna(), a_filt.dropna(), alternative='two-sided')
        delta = a_filt.median() - h_filt.median()
        print(f"  Exclude top {exclude_n:3d} loci: "
              f"HeLa={h_filt.median():.1f}(n={len(h_filt)}), "
              f"Ars={a_filt.median():.1f}(n={len(a_filt)}), "
              f"Δ={delta:+.1f} nt, p={p:.2e}")

# =============================================================================
# 5. Singleton-only analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: SINGLETON LOCI ONLY")
print("=" * 70)

hela_singleton_loci = set(hela_locus_cnt[hela_locus_cnt == 1].index)
ars_singleton_loci = set(ars_locus_cnt[ars_locus_cnt == 1].index)

h_sing = hela_catB[hela_catB['locus_id'].isin(hela_singleton_loci)]['polya']
a_sing = ars_catB[ars_catB['locus_id'].isin(ars_singleton_loci)]['polya']
if len(h_sing) > 5 and len(a_sing) > 5:
    _, p = stats.mannwhitneyu(h_sing.dropna(), a_sing.dropna(), alternative='two-sided')
    delta = a_sing.median() - h_sing.median()
    print(f"  Singleton only: HeLa={h_sing.median():.1f}(n={len(h_sing)}), "
          f"Ars={a_sing.median():.1f}(n={len(a_sing)}), Δ={delta:+.1f} nt, p={p:.2e}")

# =============================================================================
# 6. Read count distribution: reads per locus (Gini coefficient)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: CONCENTRATION METRICS")
print("=" * 70)

def gini(values):
    v = np.sort(np.array(values, dtype=float))
    n = len(v)
    if n == 0:
        return 0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))

for label, lc in [('Cat B all', locus_counts),
                   ('HeLa Cat B', hela_locus_cnt),
                   ('HeLa-Ars Cat B', ars_locus_cnt)]:
    g = gini(lc.values)
    print(f"  {label:20s}: Gini={g:.3f}, median reads/locus={np.median(lc.values):.0f}, "
          f"mean={np.mean(lc.values):.1f}")

# =============================================================================
# 7. Compare PASS L1 singleton pattern (for reference)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: COMPARISON WITH PASS L1 (HeLa/Ars)")
print("=" * 70)

# Load PASS poly(A) for HeLa/Ars with locus info
pass_polya_list = []
for cl in ['HeLa', 'HeLa-Ars']:
    for grp in CELL_LINES[cl]:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            if 'polya_length' in df.columns and 'transcript_id' in df.columns:
                df2 = df[df['polya_length'].notna()].copy()
                df2['group'] = grp
                df2['cell_line'] = cl
                df2 = df2.rename(columns={'polya_length': 'polya', 'transcript_id': 'locus_id'})
                pass_polya_list.append(df2[['read_id', 'group', 'cell_line', 'polya', 'locus_id']])

pass_polya_ha = pd.concat(pass_polya_list, ignore_index=True)
hela_pass = pass_polya_ha[pass_polya_ha['cell_line'] == 'HeLa']
ars_pass = pass_polya_ha[pass_polya_ha['cell_line'] == 'HeLa-Ars']

hela_pass_lc = hela_pass.groupby('locus_id').size().sort_values(ascending=False)
ars_pass_lc = ars_pass.groupby('locus_id').size().sort_values(ascending=False)

print(f"PASS HeLa: {len(hela_pass)} reads, {hela_pass['locus_id'].nunique()} loci, "
      f"singleton={( hela_pass_lc == 1).mean()*100:.1f}%")
print(f"PASS Ars:  {len(ars_pass)} reads, {ars_pass['locus_id'].nunique()} loci, "
      f"singleton={( ars_pass_lc == 1).mean()*100:.1f}%")
print(f"Cat B HeLa: {len(hela_catB)} reads, {hela_catB['locus_id'].nunique()} loci, "
      f"singleton={( hela_locus_cnt == 1).mean()*100:.1f}%")
print(f"Cat B Ars:  {len(ars_catB)} reads, {ars_catB['locus_id'].nunique()} loci, "
      f"singleton={( ars_locus_cnt == 1).mean()*100:.1f}%")

# PASS shared loci arsenite effect for comparison
pass_shared = set(hela_pass['locus_id'].unique()) & set(ars_pass['locus_id'].unique())
pass_locus_rows = []
for locus in pass_shared:
    h = hela_pass[hela_pass['locus_id'] == locus]['polya']
    a = ars_pass[ars_pass['locus_id'] == locus]['polya']
    pass_locus_rows.append({
        'locus_id': locus,
        'hela_n': len(h), 'ars_n': len(a),
        'hela_median': h.median(), 'ars_median': a.median(),
        'delta': a.median() - h.median(),
    })

pass_locus_df = pd.DataFrame(pass_locus_rows)
pass_robust = pass_locus_df[(pass_locus_df['hela_n'] >= 3) & (pass_locus_df['ars_n'] >= 3)]
print(f"\nPASS shared loci (≥3 each): {len(pass_robust)}")
if len(pass_robust) > 0:
    print(f"  Mean delta: {pass_robust['delta'].mean():+.1f} nt")
    print(f"  Median delta: {pass_robust['delta'].median():+.1f} nt")
    print(f"  Shortening: {(pass_robust['delta'] < 0).sum()} / {len(pass_robust)} "
          f"({(pass_robust['delta'] < 0).mean()*100:.1f}%)")

print(f"\nCat B shared loci (≥3 each): {len(robust)}")
if len(robust) > 0:
    print(f"  Mean delta: {robust['delta'].mean():+.1f} nt")
    print(f"  Median delta: {robust['delta'].median():+.1f} nt")
    print(f"  Shortening: {(robust['delta'] < 0).sum()} / {len(robust)} "
          f"({(robust['delta'] < 0).mean()*100:.1f}%)")

print("\nDone!")
