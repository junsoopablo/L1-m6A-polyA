#!/usr/bin/env python3
"""
Cross-cell-line validation of Ars-only loci poly(A).

If Ars-only loci have short poly(A) INHERENTLY (in other cell lines too),
→ compositional: arsenite activates inherently-short-polyA loci.
If Ars-only loci have normal poly(A) in other cell lines,
→ post-transcriptional: arsenite shortens poly(A) at these loci.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load HeLa / HeLa-Ars data and classify loci
# =========================================================================
print("Loading HeLa/HeLa-Ars data...")

def load_l1(groups):
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined['l1_age'] = combined['gene_id'].apply(
        lambda x: 'young' if x in YOUNG else 'ancient')
    return combined

hela = load_l1(['HeLa_1', 'HeLa_2', 'HeLa_3'])
ars = load_l1(['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'])

hela_anc = hela[hela['l1_age'] == 'ancient']
ars_anc = ars[ars['l1_age'] == 'ancient']

hela_loci = set(hela_anc['transcript_id'].unique())
ars_loci = set(ars_anc['transcript_id'].unique())
shared = hela_loci & ars_loci
hela_only = hela_loci - ars_loci
ars_only = ars_loci - hela_loci

print(f"  HeLa ancient: {len(hela_anc):,} reads, {len(hela_loci):,} loci")
print(f"  HeLa-Ars ancient: {len(ars_anc):,} reads, {len(ars_loci):,} loci")
print(f"  Shared: {len(shared)}, HeLa-only: {len(hela_only)}, Ars-only: {len(ars_only)}")

# =========================================================================
# 2. Load ALL other cell lines
# =========================================================================
print("\nLoading other cell lines...")

other_groups = {
    'A549': ['A549_1', 'A549_2', 'A549_3'],
    'H9': ['H9_1', 'H9_2', 'H9_3'],
    'Hct116': ['Hct116_1', 'Hct116_2', 'Hct116_3'],
    'HEK293': ['HEK293_1', 'HEK293_2', 'HEK293_3'],
    'HepG2': ['HepG2_1', 'HepG2_2', 'HepG2_3'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_1', 'K562_2', 'K562_3'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

other_data = {}
for cell_line, groups in other_groups.items():
    df = load_l1(groups)
    if len(df) > 0:
        df_anc = df[df['l1_age'] == 'ancient']
        other_data[cell_line] = df_anc
        print(f"  {cell_line}: {len(df_anc):,} ancient reads, {df_anc['transcript_id'].nunique():,} loci")

# Pool all other cell lines
all_other = pd.concat(other_data.values(), ignore_index=True)
print(f"\n  Total other: {len(all_other):,} reads, {all_other['transcript_id'].nunique():,} loci")

# =========================================================================
# 3. Cross-reference Ars-only / HeLa-only / shared loci with other cell lines
# =========================================================================
print(f"\n{'='*90}")
print("Ars-Only / HeLa-Only / Shared Loci: Presence in Other Cell Lines")
print(f"{'='*90}")

other_loci = set(all_other['transcript_id'].unique())

ars_only_in_other = ars_only & other_loci
hela_only_in_other = hela_only & other_loci
shared_in_other = shared & other_loci

print(f"\n  {'Category':<20} {'Total':>7} {'In other CL':>13} {'%':>6}")
print(f"  {'-'*50}")
print(f"  {'Ars-only':<20} {len(ars_only):>7} {len(ars_only_in_other):>13} {len(ars_only_in_other)/len(ars_only)*100:>5.1f}%")
print(f"  {'HeLa-only':<20} {len(hela_only):>7} {len(hela_only_in_other):>13} {len(hela_only_in_other)/len(hela_only)*100:>5.1f}%")
print(f"  {'Shared':<20} {len(shared):>7} {len(shared_in_other):>13} {len(shared_in_other)/len(shared)*100:>5.1f}%")

# =========================================================================
# 4. KEY TEST: Poly(A) of Ars-only loci in other cell lines
# =========================================================================
print(f"\n{'='*90}")
print("KEY: Poly(A) of Ars-Only Loci in Other Cell Lines vs in HeLa-Ars")
print(f"{'='*90}")

# Get poly(A) for Ars-only loci in HeLa-Ars
ars_only_reads_in_ars = ars_anc[ars_anc['transcript_id'].isin(ars_only)]
# Get poly(A) for same loci in other cell lines
ars_only_reads_in_other = all_other[all_other['transcript_id'].isin(ars_only_in_other)]

print(f"\n  Ars-only loci in HeLa-Ars:     n={len(ars_only_reads_in_ars):,}  "
      f"median polyA={ars_only_reads_in_ars['polya_length'].median():.1f}  "
      f"mean={ars_only_reads_in_ars['polya_length'].mean():.1f}")
print(f"  Same loci in other cell lines: n={len(ars_only_reads_in_other):,}  "
      f"median polyA={ars_only_reads_in_other['polya_length'].median():.1f}  "
      f"mean={ars_only_reads_in_other['polya_length'].mean():.1f}")

if len(ars_only_reads_in_other) >= 10:
    _, p = stats.mannwhitneyu(ars_only_reads_in_ars['polya_length'],
                               ars_only_reads_in_other['polya_length'],
                               alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  MW p={p:.2e} ({sig})")

# Same for HeLa-only loci
hela_only_reads_in_hela = hela_anc[hela_anc['transcript_id'].isin(hela_only)]
hela_only_reads_in_other = all_other[all_other['transcript_id'].isin(hela_only_in_other)]

print(f"\n  HeLa-only loci in HeLa:        n={len(hela_only_reads_in_hela):,}  "
      f"median polyA={hela_only_reads_in_hela['polya_length'].median():.1f}  "
      f"mean={hela_only_reads_in_hela['polya_length'].mean():.1f}")
print(f"  Same loci in other cell lines: n={len(hela_only_reads_in_other):,}  "
      f"median polyA={hela_only_reads_in_other['polya_length'].median():.1f}  "
      f"mean={hela_only_reads_in_other['polya_length'].mean():.1f}")

if len(hela_only_reads_in_other) >= 10:
    _, p2 = stats.mannwhitneyu(hela_only_reads_in_hela['polya_length'],
                                hela_only_reads_in_other['polya_length'],
                                alternative='two-sided')
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
    print(f"  MW p={p2:.2e} ({sig2})")

# =========================================================================
# 5. Per-cell-line breakdown
# =========================================================================
print(f"\n{'='*90}")
print("Per-Cell-Line: Ars-Only Loci Poly(A)")
print(f"{'='*90}")

print(f"\n  {'Cell Line':<12} {'N reads':>8} {'N loci':>8} {'Median':>8} {'Mean':>8}")
print(f"  {'-'*48}")
print(f"  {'HeLa-Ars':<12} {len(ars_only_reads_in_ars):>8} "
      f"{ars_only_reads_in_ars['transcript_id'].nunique():>8} "
      f"{ars_only_reads_in_ars['polya_length'].median():>8.1f} "
      f"{ars_only_reads_in_ars['polya_length'].mean():>8.1f}")

for cl, cl_df in sorted(other_data.items()):
    cl_ars_loci = cl_df[cl_df['transcript_id'].isin(ars_only_in_other)]
    if len(cl_ars_loci) >= 5:
        print(f"  {cl:<12} {len(cl_ars_loci):>8} "
              f"{cl_ars_loci['transcript_id'].nunique():>8} "
              f"{cl_ars_loci['polya_length'].median():>8.1f} "
              f"{cl_ars_loci['polya_length'].mean():>8.1f}")

# =========================================================================
# 6. Per-locus paired: same locus in HeLa-Ars vs other cell lines
# =========================================================================
print(f"\n{'='*90}")
print("Per-Locus Paired: Ars-Only Loci in HeLa-Ars vs Other Cell Lines")
print(f"{'='*90}")

ars_locus_polya = ars_anc[ars_anc['transcript_id'].isin(ars_only_in_other)].groupby('transcript_id')['polya_length'].median()
other_locus_polya = all_other[all_other['transcript_id'].isin(ars_only_in_other)].groupby('transcript_id')['polya_length'].median()

common = set(ars_locus_polya.index) & set(other_locus_polya.index)
if len(common) >= 5:
    ars_vals = [ars_locus_polya[l] for l in common]
    other_vals = [other_locus_polya[l] for l in common]
    deltas = [a - o for a, o in zip(ars_vals, other_vals)]

    n_shorter = sum(1 for d in deltas if d < 0)
    n_longer = sum(1 for d in deltas if d > 0)

    print(f"\n  Loci compared: {len(common)}")
    print(f"  Ars-only loci median polyA in HeLa-Ars: {np.median(ars_vals):.1f}")
    print(f"  Same loci median polyA in other CL:     {np.median(other_vals):.1f}")
    print(f"  Per-locus Δ (Ars - other): mean={np.mean(deltas):+.1f}, median={np.median(deltas):+.1f}")
    print(f"  {n_shorter} shorter in Ars, {n_longer} longer in Ars")

    if len(common) >= 10:
        _, p_wilcox = stats.wilcoxon(ars_vals, other_vals, alternative='two-sided')
        p_sign = stats.binomtest(n_shorter, n_shorter + n_longer, 0.5).pvalue
        print(f"  Wilcoxon p={p_wilcox:.2e}, Sign test p={p_sign:.3f}")

# =========================================================================
# 7. The critical comparison table
# =========================================================================
print(f"\n{'='*90}")
print("Critical Comparison: Are Ars-Only Loci Inherently Short-PolyA?")
print(f"{'='*90}")

# For each locus category, show poly(A) in HeLa, HeLa-Ars, and Other
print(f"\n  {'Locus category':<20} {'In HeLa':>10} {'In HeLa-Ars':>12} {'In Other CL':>12} {'Interpretation'}")
print(f"  {'-'*75}")

# Shared loci
shared_hela = hela_anc[hela_anc['transcript_id'].isin(shared)]['polya_length'].median()
shared_ars = ars_anc[ars_anc['transcript_id'].isin(shared)]['polya_length'].median()
shared_other_reads = all_other[all_other['transcript_id'].isin(shared_in_other)]
shared_other = shared_other_reads['polya_length'].median() if len(shared_other_reads) > 0 else float('nan')
print(f"  {'Shared':<20} {shared_hela:>10.1f} {shared_ars:>12.1f} {shared_other:>12.1f}")

# HeLa-only
hela_only_hela = hela_anc[hela_anc['transcript_id'].isin(hela_only)]['polya_length'].median()
hela_only_other_val = hela_only_reads_in_other['polya_length'].median() if len(hela_only_reads_in_other) > 0 else float('nan')
print(f"  {'HeLa-only':<20} {hela_only_hela:>10.1f} {'n/a':>12} {hela_only_other_val:>12.1f}")

# Ars-only
ars_only_ars = ars_anc[ars_anc['transcript_id'].isin(ars_only)]['polya_length'].median()
ars_only_other_val = ars_only_reads_in_other['polya_length'].median() if len(ars_only_reads_in_other) > 0 else float('nan')
print(f"  {'Ars-only':<20} {'n/a':>10} {ars_only_ars:>12.1f} {ars_only_other_val:>12.1f}")

print(f"\n  Interpretation:")
if ars_only_other_val < hela_only_other_val - 10:
    print(f"    Ars-only loci have INHERENTLY shorter poly(A) in other cell lines too")
    print(f"    (Other CL: {ars_only_other_val:.1f} vs {hela_only_other_val:.1f})")
    print(f"    → COMPOSITIONAL: arsenite activates loci that inherently have short poly(A)")
elif abs(ars_only_other_val - hela_only_other_val) <= 10:
    print(f"    Ars-only loci have SIMILAR poly(A) in other cell lines")
    print(f"    (Other CL: {ars_only_other_val:.1f} vs {hela_only_other_val:.1f})")
    print(f"    But shorter in HeLa-Ars ({ars_only_ars:.1f})")
    print(f"    → POST-TRANSCRIPTIONAL: arsenite shortens poly(A) at these loci")
else:
    print(f"    Ars-only loci have LONGER poly(A) in other cell lines")
    print(f"    → POST-TRANSCRIPTIONAL: arsenite shortens poly(A) at these loci")

# Additional: compare Ars-only loci poly(A) in HeLa-Ars vs same loci in other CL
if len(ars_only_reads_in_other) >= 10:
    delta_ars_vs_other = ars_only_ars - ars_only_other_val
    print(f"\n    Ars-only loci: HeLa-Ars median={ars_only_ars:.1f} vs Other CL median={ars_only_other_val:.1f}")
    print(f"    Δ = {delta_ars_vs_other:+.1f} nt")
    if delta_ars_vs_other < -10:
        print(f"    → These loci are SHORTER in HeLa-Ars than in other cell lines")
        print(f"    → Suggests POST-TRANSCRIPTIONAL shortening by arsenite")
    elif abs(delta_ars_vs_other) <= 10:
        print(f"    → These loci have SIMILAR poly(A) in HeLa-Ars and other cell lines")
        print(f"    → Suggests INHERENT property of these loci (compositional)")

# =========================================================================
# 8. Summary
# =========================================================================
print(f"\n{'='*90}")
print("FINAL ANSWER")
print(f"{'='*90}")
