#!/usr/bin/env python3
"""
Decompose the arsenite poly(A) shortening into:
  1. Post-transcriptional effect (same loci, poly(A) changes)
  2. Compositional effect (different loci with different baseline poly(A))

Key approach:
  - Compare poly(A) of reads from shared vs condition-specific loci
  - Lower threshold for per-locus analysis (≥2 reads)
  - Quantify how much of the Δ=-32.6nt is attributable to each component
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load data
# =========================================================================
print("Loading data...")

def load_l1(groups, label):
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['condition'] = label
    combined['l1_age'] = combined['gene_id'].apply(
        lambda x: 'young' if x in YOUNG else 'ancient')
    return combined

hela = load_l1(['HeLa_1', 'HeLa_2', 'HeLa_3'], 'HeLa')
ars = load_l1(['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'], 'HeLa-Ars')

hela_anc = hela[hela['l1_age'] == 'ancient'].copy()
ars_anc = ars[ars['l1_age'] == 'ancient'].copy()

hela_loci = set(hela_anc['transcript_id'].unique())
ars_loci = set(ars_anc['transcript_id'].unique())
shared = hela_loci & ars_loci
hela_only = hela_loci - ars_loci
ars_only = ars_loci - hela_loci

# Tag each read
hela_anc['locus_type'] = hela_anc['transcript_id'].apply(
    lambda x: 'shared' if x in shared else 'HeLa-only')
ars_anc['locus_type'] = ars_anc['transcript_id'].apply(
    lambda x: 'shared' if x in shared else 'Ars-only')

# =========================================================================
# 2. Poly(A) by locus category
# =========================================================================
print(f"\n{'='*90}")
print("Poly(A) by Locus Category")
print(f"{'='*90}")

categories = [
    ('HeLa ALL',          hela_anc),
    ('HeLa shared-loci',  hela_anc[hela_anc['locus_type'] == 'shared']),
    ('HeLa HeLa-only',    hela_anc[hela_anc['locus_type'] == 'HeLa-only']),
    ('Ars ALL',            ars_anc),
    ('Ars shared-loci',    ars_anc[ars_anc['locus_type'] == 'shared']),
    ('Ars Ars-only',       ars_anc[ars_anc['locus_type'] == 'Ars-only']),
]

print(f"\n{'Category':<22} {'N':>6} {'N loci':>8} {'Median':>8} {'Mean':>8} {'Q25':>6} {'Q75':>6}")
print("-" * 68)
for label, d in categories:
    n_loci = d['transcript_id'].nunique()
    print(f"{label:<22} {len(d):>6,} {n_loci:>8,} {d['polya_length'].median():>8.1f} "
          f"{d['polya_length'].mean():>8.1f} {d['polya_length'].quantile(0.25):>6.1f} "
          f"{d['polya_length'].quantile(0.75):>6.1f}")

# =========================================================================
# 3. Decomposition: shared vs condition-specific contribution
# =========================================================================
print(f"\n{'='*90}")
print("Decomposition of Overall Δ=-32.6nt")
print(f"{'='*90}")

hela_all_med = hela_anc['polya_length'].median()
ars_all_med = ars_anc['polya_length'].median()
overall_delta = ars_all_med - hela_all_med

hela_shared_med = hela_anc[hela_anc['locus_type'] == 'shared']['polya_length'].median()
ars_shared_med = ars_anc[ars_anc['locus_type'] == 'shared']['polya_length'].median()
shared_delta = ars_shared_med - hela_shared_med

hela_only_med = hela_anc[hela_anc['locus_type'] == 'HeLa-only']['polya_length'].median()
ars_only_med = ars_anc[ars_anc['locus_type'] == 'Ars-only']['polya_length'].median()

print(f"\n  Overall: HeLa={hela_all_med:.1f} → Ars={ars_all_med:.1f}  Δ={overall_delta:+.1f}")
print(f"  Shared loci: HeLa={hela_shared_med:.1f} → Ars={ars_shared_med:.1f}  Δ={shared_delta:+.1f}")
print(f"  HeLa-only loci: median={hela_only_med:.1f} (n={len(hela_anc[hela_anc['locus_type']=='HeLa-only']):,})")
print(f"  Ars-only loci:  median={ars_only_med:.1f} (n={len(ars_anc[ars_anc['locus_type']=='Ars-only']):,})")

# What fraction of reads are shared vs condition-specific?
hela_shared_frac = (hela_anc['locus_type'] == 'shared').mean()
ars_shared_frac = (ars_anc['locus_type'] == 'shared').mean()
print(f"\n  Read fraction from shared loci: HeLa={hela_shared_frac:.1%}, Ars={ars_shared_frac:.1%}")

# =========================================================================
# 4. Weighted decomposition
# =========================================================================
print(f"\n{'='*90}")
print("Weighted Decomposition (Mean-Based)")
print(f"{'='*90}")

# Use means for decomposition (additive)
hela_shared_mean = hela_anc[hela_anc['locus_type'] == 'shared']['polya_length'].mean()
hela_only_mean = hela_anc[hela_anc['locus_type'] == 'HeLa-only']['polya_length'].mean()
ars_shared_mean = ars_anc[ars_anc['locus_type'] == 'shared']['polya_length'].mean()
ars_only_mean = ars_anc[ars_anc['locus_type'] == 'Ars-only']['polya_length'].mean()

hela_all_mean = hela_anc['polya_length'].mean()
ars_all_mean = ars_anc['polya_length'].mean()
overall_delta_mean = ars_all_mean - hela_all_mean

n_hela_shared = (hela_anc['locus_type'] == 'shared').sum()
n_hela_only = (hela_anc['locus_type'] == 'HeLa-only').sum()
n_ars_shared = (ars_anc['locus_type'] == 'shared').sum()
n_ars_only = (ars_anc['locus_type'] == 'Ars-only').sum()

print(f"\n  Overall mean Δ: {overall_delta_mean:+.1f} nt")
print(f"  HeLa: shared mean={hela_shared_mean:.1f} ({n_hela_shared} reads), "
      f"HeLa-only mean={hela_only_mean:.1f} ({n_hela_only} reads)")
print(f"  Ars:  shared mean={ars_shared_mean:.1f} ({n_ars_shared} reads), "
      f"Ars-only mean={ars_only_mean:.1f} ({n_ars_only} reads)")

# Decompose: overall mean = (shared_frac * shared_mean) + (specific_frac * specific_mean)
# Effect 1 (post-transcriptional): shared loci mean changes
post_tx = ars_shared_mean - hela_shared_mean
# Effect 2 (compositional): condition-specific loci contribute differently
# HeLa overall = f_shared * shared_mean + f_only * only_mean
# Ars overall  = f_shared * shared_mean + f_only * only_mean (different f and mean)
print(f"\n  Post-transcriptional component (shared loci Δmean): {post_tx:+.1f} nt")
print(f"  Ars-only loci mean vs HeLa-only loci mean: {ars_only_mean:.1f} vs {hela_only_mean:.1f} = Δ{ars_only_mean-hela_only_mean:+.1f}")

# =========================================================================
# 5. Simulation: what would the overall Δ be if NO post-tx change?
# =========================================================================
print(f"\n{'='*90}")
print("Counterfactual: What if No Post-Transcriptional Change?")
print(f"{'='*90}")

# If shared loci had the SAME poly(A) in Ars as in HeLa,
# what would the overall Ars mean be?
counterfactual_ars_mean = (n_ars_shared * hela_shared_mean + n_ars_only * ars_only_mean) / len(ars_anc)
counterfactual_delta = counterfactual_ars_mean - hela_all_mean

print(f"\n  Actual Ars mean:         {ars_all_mean:.1f}")
print(f"  Counterfactual Ars mean: {counterfactual_ars_mean:.1f} (shared loci keep HeLa poly(A))")
print(f"  Actual Δ (mean):         {overall_delta_mean:+.1f}")
print(f"  Counterfactual Δ:        {counterfactual_delta:+.1f} (composition-only)")
print(f"  Post-tx contribution:    {overall_delta_mean - counterfactual_delta:+.1f}")
print(f"\n  Attribution:")
print(f"    Compositional effect:       {counterfactual_delta:+.1f} nt ({abs(counterfactual_delta)/abs(overall_delta_mean)*100:.0f}%)")
print(f"    Post-transcriptional effect:{overall_delta_mean - counterfactual_delta:+.1f} nt ({abs(overall_delta_mean - counterfactual_delta)/abs(overall_delta_mean)*100:.0f}%)")

# =========================================================================
# 6. Per-locus paired test with lower threshold
# =========================================================================
print(f"\n{'='*90}")
print("Per-Locus Paired Test (Lower Thresholds)")
print(f"{'='*90}")

hela_locus_polya = hela_anc.groupby('transcript_id')['polya_length'].agg(['median', 'count'])
ars_locus_polya = ars_anc.groupby('transcript_id')['polya_length'].agg(['median', 'count'])

for min_n in [2, 3, 5, 10]:
    valid_loci = []
    hela_meds = []
    ars_meds = []
    for locus in shared:
        h = hela_locus_polya.loc[locus] if locus in hela_locus_polya.index else None
        a = ars_locus_polya.loc[locus] if locus in ars_locus_polya.index else None
        if h is not None and a is not None and h['count'] >= min_n and a['count'] >= min_n:
            valid_loci.append(locus)
            hela_meds.append(h['median'])
            ars_meds.append(a['median'])

    if len(valid_loci) >= 5:
        hela_meds = np.array(hela_meds)
        ars_meds = np.array(ars_meds)
        deltas = ars_meds - hela_meds
        n_short = (deltas < 0).sum()
        n_long = (deltas > 0).sum()
        _, p_wilcox = stats.wilcoxon(hela_meds, ars_meds, alternative='two-sided')
        p_sign = stats.binomtest(n_short, n_short + n_long, 0.5).pvalue
        print(f"\n  min_n={min_n}: {len(valid_loci)} loci")
        print(f"    {n_short} shortened, {n_long} lengthened (sign test p={p_sign:.3f})")
        print(f"    Mean Δ={deltas.mean():+.1f}, Median Δ={np.median(deltas):+.1f}")
        print(f"    Wilcoxon p={p_wilcox:.3f}")

# =========================================================================
# 7. Are Ars-only loci inherently short-polyA loci?
# =========================================================================
print(f"\n{'='*90}")
print("Are Condition-Specific Loci Different?")
print(f"{'='*90}")

# Among HeLa reads, compare: reads from shared loci vs HeLa-only loci
hela_shared_pa = hela_anc[hela_anc['locus_type'] == 'shared']['polya_length']
hela_only_pa = hela_anc[hela_anc['locus_type'] == 'HeLa-only']['polya_length']
_, p1 = stats.mannwhitneyu(hela_shared_pa, hela_only_pa, alternative='two-sided')

ars_shared_pa = ars_anc[ars_anc['locus_type'] == 'shared']['polya_length']
ars_only_pa = ars_anc[ars_anc['locus_type'] == 'Ars-only']['polya_length']
_, p2 = stats.mannwhitneyu(ars_shared_pa, ars_only_pa, alternative='two-sided')

print(f"\n  Within HeLa:")
print(f"    Shared loci reads: median={hela_shared_pa.median():.1f} (n={len(hela_shared_pa):,})")
print(f"    HeLa-only reads:  median={hela_only_pa.median():.1f} (n={len(hela_only_pa):,})")
print(f"    MW p={p1:.2e}")

print(f"\n  Within HeLa-Ars:")
print(f"    Shared loci reads: median={ars_shared_pa.median():.1f} (n={len(ars_shared_pa):,})")
print(f"    Ars-only reads:   median={ars_only_pa.median():.1f} (n={len(ars_only_pa):,})")
print(f"    MW p={p2:.2e}")

# =========================================================================
# 8. Read count per locus distribution
# =========================================================================
print(f"\n{'='*90}")
print("Read Count Distribution per Locus")
print(f"{'='*90}")

hela_counts = hela_anc['transcript_id'].value_counts()
ars_counts = ars_anc['transcript_id'].value_counts()

for label, counts in [('HeLa', hela_counts), ('HeLa-Ars', ars_counts)]:
    n_1 = (counts == 1).sum()
    n_2_5 = ((counts >= 2) & (counts <= 5)).sum()
    n_6_plus = (counts > 5).sum()
    print(f"  {label}: total loci={len(counts)}, "
          f"1 read={n_1} ({n_1/len(counts)*100:.0f}%), "
          f"2-5 reads={n_2_5} ({n_2_5/len(counts)*100:.0f}%), "
          f">5 reads={n_6_plus} ({n_6_plus/len(counts)*100:.0f}%)")
    # Reads from singleton loci
    singleton_loci = set(counts[counts == 1].index)
    n_singleton_reads = hela_anc[hela_anc['transcript_id'].isin(singleton_loci)].shape[0] if label == 'HeLa' else \
                        ars_anc[ars_anc['transcript_id'].isin(singleton_loci)].shape[0]
    print(f"         Singleton reads: {n_singleton_reads} ({n_singleton_reads/len(hela_anc if label=='HeLa' else ars_anc)*100:.1f}%)")

# =========================================================================
# 9. Final Summary
# =========================================================================
print(f"\n{'='*90}")
print("FINAL SUMMARY")
print(f"{'='*90}")
print(f"""
  Overall ancient L1 poly(A) Δ = {overall_delta_mean:+.1f} nt (mean)

  Decomposition:
    Compositional effect:        {counterfactual_delta:+.1f} nt ({abs(counterfactual_delta)/abs(overall_delta_mean)*100:.0f}%)
      (Ars-only loci have {'shorter' if ars_only_mean < hela_only_mean else 'longer'} poly(A) than HeLa-only)
    Post-transcriptional effect: {overall_delta_mean - counterfactual_delta:+.1f} nt ({abs(overall_delta_mean - counterfactual_delta)/abs(overall_delta_mean)*100:.0f}%)
      (Same loci: shared Δmean = {post_tx:+.1f} nt)

  Caveats:
    - Most loci have 1-2 reads (sparse sampling)
    - "Shared" vs "condition-specific" partly reflects sampling noise
    - Singleton loci are {(hela_counts==1).sum()}/{len(hela_counts)} in HeLa,
      {(ars_counts==1).sum()}/{len(ars_counts)} in Ars
""")
