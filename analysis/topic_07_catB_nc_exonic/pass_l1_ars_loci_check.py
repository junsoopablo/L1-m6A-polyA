#!/usr/bin/env python3
"""Check whether PASS L1 arsenite poly(A) shortening is driven by specific loci."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
OUTDIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/catB_vs_pass_analysis'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =============================================================================
# Load PASS L1 data for HeLa / HeLa-Ars
# =============================================================================
print("Loading PASS L1 data for HeLa / HeLa-Ars...")

groups = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

all_data = []
for cl, grps in groups.items():
    for grp in grps:
        f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            df['group'] = grp
            df['cell_line'] = cl
            if 'transcript_id' in df.columns:
                df['locus_id'] = df['transcript_id']
                df['subfamily'] = df['gene_id']
            df['is_young'] = df['subfamily'].isin(YOUNG)
            df['age'] = np.where(df['is_young'], 'young', 'ancient')
            if 'polya_length' in df.columns:
                df = df.rename(columns={'polya_length': 'polya'})
            all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data = data[data['polya'].notna()].copy()

hela = data[data['cell_line'] == 'HeLa']
ars = data[data['cell_line'] == 'HeLa-Ars']

print(f"  HeLa: {len(hela)} reads, {hela['locus_id'].nunique()} loci")
print(f"  HeLa-Ars: {len(ars)} reads, {ars['locus_id'].nunique()} loci")

# Overall
_, p_all = stats.mannwhitneyu(hela['polya'], ars['polya'], alternative='two-sided')
print(f"\n  Overall: HeLa={hela['polya'].median():.1f}, Ars={ars['polya'].median():.1f}, "
      f"Δ={ars['polya'].median() - hela['polya'].median():+.1f}, p={p_all:.2e}")

# =============================================================================
# 1. Loci read count distribution
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: LOCI CONCENTRATION")
print("=" * 70)

hela_lc = hela.groupby('locus_id').size().sort_values(ascending=False)
ars_lc = ars.groupby('locus_id').size().sort_values(ascending=False)

for label, lc in [('HeLa', hela_lc), ('HeLa-Ars', ars_lc)]:
    print(f"\n  {label}:")
    print(f"    Total loci: {len(lc)}")
    print(f"    Singleton: {(lc == 1).sum()} ({(lc == 1).mean()*100:.1f}%)")
    print(f"    ≥3 reads: {(lc >= 3).sum()}")
    print(f"    ≥5 reads: {(lc >= 5).sum()}")
    print(f"    ≥10 reads: {(lc >= 10).sum()}")
    print(f"    Top 10:")
    for locus, cnt in lc.head(10).items():
        sub = data[(data['locus_id'] == locus) & (data['cell_line'] == label.replace('-', '-'))]
        age = sub['age'].iloc[0] if len(sub) > 0 else '?'
        med = sub['polya'].median()
        print(f"      {locus:30s}  {cnt:4d} reads  {age:8s}  poly(A)={med:.1f}")

# =============================================================================
# 2. Sensitivity: exclude top loci
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: SENSITIVITY — EXCLUDE TOP LOCI")
print("=" * 70)

combined = pd.concat([hela, ars])
combined_lc = combined.groupby('locus_id').size().sort_values(ascending=False)

print(f"\n  Top 10 loci account for {combined_lc.head(10).sum()} / {len(combined)} reads "
      f"({combined_lc.head(10).sum()/len(combined)*100:.1f}%)")
print(f"  Top 50 loci account for {combined_lc.head(50).sum()} / {len(combined)} reads "
      f"({combined_lc.head(50).sum()/len(combined)*100:.1f}%)")

for exclude_n in [0, 5, 10, 20, 50, 100]:
    top_loci = set(combined_lc.head(exclude_n).index) if exclude_n > 0 else set()
    h = hela[~hela['locus_id'].isin(top_loci)]['polya']
    a = ars[~ars['locus_id'].isin(top_loci)]['polya']
    if len(h) > 5 and len(a) > 5:
        _, p = stats.mannwhitneyu(h.dropna(), a.dropna(), alternative='two-sided')
        delta = a.median() - h.median()
        print(f"  Exclude top {exclude_n:4d}: "
              f"HeLa={h.median():.1f}(n={len(h)}), Ars={a.median():.1f}(n={len(a)}), "
              f"Δ={delta:+.1f} nt, p={p:.2e}")

# =============================================================================
# 3. Singleton-only analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: SINGLETON LOCI ONLY")
print("=" * 70)

hela_sing_loci = set(hela_lc[hela_lc == 1].index)
ars_sing_loci = set(ars_lc[ars_lc == 1].index)

h_sing = hela[hela['locus_id'].isin(hela_sing_loci)]['polya']
a_sing = ars[ars['locus_id'].isin(ars_sing_loci)]['polya']
_, p = stats.mannwhitneyu(h_sing.dropna(), a_sing.dropna(), alternative='two-sided')
delta = a_sing.median() - h_sing.median()
print(f"  Singleton: HeLa={h_sing.median():.1f}(n={len(h_sing)}), "
      f"Ars={a_sing.median():.1f}(n={len(a_sing)}), Δ={delta:+.1f} nt, p={p:.2e}")

# =============================================================================
# 4. Per-locus analysis (shared loci, all sizes)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: PER-LOCUS ANALYSIS (SHARED LOCI)")
print("=" * 70)

shared = set(hela['locus_id'].unique()) & set(ars['locus_id'].unique())
print(f"\n  Shared loci: {len(shared)}")
print(f"  HeLa-only: {hela['locus_id'].nunique() - len(shared)}")
print(f"  Ars-only: {ars['locus_id'].nunique() - len(shared)}")

locus_rows = []
for locus in shared:
    h = hela[hela['locus_id'] == locus]['polya']
    a = ars[ars['locus_id'] == locus]['polya']
    locus_rows.append({
        'locus_id': locus,
        'hela_n': len(h), 'ars_n': len(a),
        'hela_median': h.median(), 'ars_median': a.median(),
        'delta': a.median() - h.median(),
        'age': hela[hela['locus_id'] == locus]['age'].iloc[0],
    })

locus_df = pd.DataFrame(locus_rows)

# All shared
print(f"\n  All shared ({len(locus_df)} loci):")
print(f"    Mean delta: {locus_df['delta'].mean():+.1f}")
print(f"    Median delta: {locus_df['delta'].median():+.1f}")
print(f"    Shortening (Δ<0): {(locus_df['delta'] < 0).sum()} ({(locus_df['delta'] < 0).mean()*100:.1f}%)")

# By minimum reads
for min_n in [2, 3, 5, 10]:
    robust = locus_df[(locus_df['hela_n'] >= min_n) & (locus_df['ars_n'] >= min_n)]
    if len(robust) > 0:
        # Read-weighted delta
        robust_reads = []
        for _, row in robust.iterrows():
            h = hela[hela['locus_id'] == row['locus_id']]['polya']
            a = ars[ars['locus_id'] == row['locus_id']]['polya']
            robust_reads.append({'hela_polya': h.values, 'ars_polya': a.values})

        h_all = np.concatenate([r['hela_polya'] for r in robust_reads])
        a_all = np.concatenate([r['ars_polya'] for r in robust_reads])
        _, p = stats.mannwhitneyu(h_all, a_all, alternative='two-sided')

        print(f"\n  ≥{min_n} reads each ({len(robust)} loci, {len(h_all)}+{len(a_all)} reads):")
        print(f"    Locus-level: mean Δ={robust['delta'].mean():+.1f}, median Δ={robust['delta'].median():+.1f}")
        print(f"    Shortening: {(robust['delta'] < 0).sum()}/{len(robust)} ({(robust['delta'] < 0).mean()*100:.1f}%)")
        print(f"    Read-level: HeLa={np.median(h_all):.1f}, Ars={np.median(a_all):.1f}, "
              f"Δ={np.median(a_all)-np.median(h_all):+.1f}, p={p:.2e}")

        # Ancient only
        robust_anc = robust[robust['age'] == 'ancient']
        if len(robust_anc) > 0:
            h_anc = np.concatenate([hela[(hela['locus_id'] == row['locus_id'])]['polya'].values
                                    for _, row in robust_anc.iterrows()])
            a_anc = np.concatenate([ars[(ars['locus_id'] == row['locus_id'])]['polya'].values
                                    for _, row in robust_anc.iterrows()])
            _, p_anc = stats.mannwhitneyu(h_anc, a_anc, alternative='two-sided')
            print(f"    Ancient only ({len(robust_anc)} loci): "
                  f"HeLa={np.median(h_anc):.1f}, Ars={np.median(a_anc):.1f}, "
                  f"Δ={np.median(a_anc)-np.median(h_anc):+.1f}, p={p_anc:.2e}")

# =============================================================================
# 5. Locus-level shortening distribution
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: LOCUS-LEVEL DELTA DISTRIBUTION (≥3 reads each)")
print("=" * 70)

robust = locus_df[(locus_df['hela_n'] >= 3) & (locus_df['ars_n'] >= 3)].copy()
if len(robust) > 0:
    robust = robust.sort_values('delta')
    print(f"\n  Percentiles of per-locus delta:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct}: {robust['delta'].quantile(pct/100):+.1f} nt")

    print(f"\n  Top 15 most shortened loci:")
    for _, row in robust.head(15).iterrows():
        print(f"    {row['locus_id']:30s}  {row['age']:8s}  "
              f"HeLa={row['hela_n']:2.0f}r/{row['hela_median']:6.1f}  "
              f"Ars={row['ars_n']:2.0f}r/{row['ars_median']:6.1f}  Δ={row['delta']:+.1f}")

    print(f"\n  Top 15 most lengthened loci:")
    for _, row in robust.tail(15).iterrows():
        print(f"    {row['locus_id']:30s}  {row['age']:8s}  "
              f"HeLa={row['hela_n']:2.0f}r/{row['hela_median']:6.1f}  "
              f"Ars={row['ars_n']:2.0f}r/{row['ars_median']:6.1f}  Δ={row['delta']:+.1f}")

# =============================================================================
# 6. Ars-only vs HeLa-only loci poly(A)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: ARS-ONLY vs HELA-ONLY LOCI")
print("=" * 70)

hela_only_loci = set(hela['locus_id'].unique()) - set(ars['locus_id'].unique())
ars_only_loci = set(ars['locus_id'].unique()) - set(hela['locus_id'].unique())

h_only = hela[hela['locus_id'].isin(hela_only_loci)]['polya']
a_only = ars[ars['locus_id'].isin(ars_only_loci)]['polya']
h_shared_reads = hela[hela['locus_id'].isin(shared)]['polya']
a_shared_reads = ars[ars['locus_id'].isin(shared)]['polya']

print(f"\n  HeLa-only loci: {len(hela_only_loci)} loci, {len(h_only)} reads, "
      f"median poly(A)={h_only.median():.1f}")
print(f"  Ars-only loci:  {len(ars_only_loci)} loci, {len(a_only)} reads, "
      f"median poly(A)={a_only.median():.1f}")
print(f"  Shared (HeLa):  {len(shared)} loci, {len(h_shared_reads)} reads, "
      f"median poly(A)={h_shared_reads.median():.1f}")
print(f"  Shared (Ars):   {len(shared)} loci, {len(a_shared_reads)} reads, "
      f"median poly(A)={a_shared_reads.median():.1f}")

# Shared loci read-level
if len(h_shared_reads) > 5 and len(a_shared_reads) > 5:
    _, p_sh = stats.mannwhitneyu(h_shared_reads, a_shared_reads, alternative='two-sided')
    delta_sh = a_shared_reads.median() - h_shared_reads.median()
    print(f"\n  Shared loci read-level: Δ={delta_sh:+.1f} nt, p={p_sh:.2e}")

# Compare Ars-only loci poly(A) vs shared loci in Ars
if len(a_only) > 5 and len(a_shared_reads) > 5:
    _, p_ao = stats.mannwhitneyu(a_only, a_shared_reads, alternative='two-sided')
    print(f"  Ars-only vs Ars-shared: {a_only.median():.1f} vs {a_shared_reads.median():.1f}, p={p_ao:.2e}")

locus_df.to_csv(OUTDIR / 'pass_hela_ars_loci_delta.tsv', sep='\t', index=False)
print("\nDone!")
