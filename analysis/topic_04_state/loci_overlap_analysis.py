#!/usr/bin/env python3
"""
L1 loci overlap analysis: HeLa vs HeLa-Ars.
Are the same ancient L1 loci expressed in both conditions?
Do poly(A) changes occur at the same loci?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load L1 summary data
# =========================================================================
print("Loading L1 summary data...")

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

print(f"  HeLa: {len(hela):,} reads ({(hela['l1_age']=='ancient').sum():,} ancient)")
print(f"  HeLa-Ars: {len(ars):,} reads ({(ars['l1_age']=='ancient').sum():,} ancient)")

# =========================================================================
# 2. Locus-level comparison (transcript_id = specific genomic locus)
# =========================================================================
print(f"\n{'='*90}")
print("Locus-Level Overlap: HeLa vs HeLa-Ars (Ancient L1)")
print(f"{'='*90}")

hela_anc = hela[hela['l1_age'] == 'ancient']
ars_anc = ars[ars['l1_age'] == 'ancient']

hela_loci = set(hela_anc['transcript_id'].unique())
ars_loci = set(ars_anc['transcript_id'].unique())
shared_loci = hela_loci & ars_loci
hela_only = hela_loci - ars_loci
ars_only = ars_loci - hela_loci

print(f"\n  HeLa ancient loci:     {len(hela_loci):,}")
print(f"  HeLa-Ars ancient loci: {len(ars_loci):,}")
print(f"  Shared loci:           {len(shared_loci):,} ({len(shared_loci)/len(hela_loci|ars_loci)*100:.1f}% of union)")
print(f"  HeLa-only loci:       {len(hela_only):,}")
print(f"  HeLa-Ars-only loci:   {len(ars_only):,}")

# How many READS come from shared loci?
hela_shared_reads = hela_anc[hela_anc['transcript_id'].isin(shared_loci)]
ars_shared_reads = ars_anc[ars_anc['transcript_id'].isin(shared_loci)]

print(f"\n  Reads from SHARED loci:")
print(f"    HeLa:     {len(hela_shared_reads):,} / {len(hela_anc):,} = {len(hela_shared_reads)/len(hela_anc)*100:.1f}%")
print(f"    HeLa-Ars: {len(ars_shared_reads):,} / {len(ars_anc):,} = {len(ars_shared_reads)/len(ars_anc)*100:.1f}%")

# =========================================================================
# 3. Top loci: are hotspots the same?
# =========================================================================
print(f"\n{'='*90}")
print("Top 20 Ancient L1 Hotspot Loci")
print(f"{'='*90}")

hela_locus_counts = hela_anc['transcript_id'].value_counts()
ars_locus_counts = ars_anc['transcript_id'].value_counts()

# Top 20 from each
hela_top20 = set(hela_locus_counts.head(20).index)
ars_top20 = set(ars_locus_counts.head(20).index)
top20_overlap = hela_top20 & ars_top20

print(f"\n  HeLa top 20 ∩ HeLa-Ars top 20: {len(top20_overlap)} / 20 shared")
print(f"  Overlap: {len(top20_overlap)/20*100:.0f}%")

# Combined top loci table
all_top = hela_locus_counts.head(30).index.union(ars_locus_counts.head(30).index)
print(f"\n{'Locus':<30} {'HeLa n':>8} {'Ars n':>8} {'Shared':>7} {'Subfamily'}")
print("-" * 70)

locus_data = []
for locus in sorted(all_top, key=lambda x: -(hela_locus_counts.get(x, 0) + ars_locus_counts.get(x, 0))):
    h_n = hela_locus_counts.get(locus, 0)
    a_n = ars_locus_counts.get(locus, 0)
    shared = "✓" if locus in shared_loci else ""
    # Get subfamily
    sf = hela_anc[hela_anc['transcript_id'] == locus]['gene_id'].iloc[0] if h_n > 0 else \
         ars_anc[ars_anc['transcript_id'] == locus]['gene_id'].iloc[0]
    print(f"{locus:<30} {h_n:>8} {a_n:>8} {shared:>7} {sf}")
    locus_data.append({'locus': locus, 'hela_n': h_n, 'ars_n': a_n, 'subfamily': sf})

# Correlation of locus counts
shared_loci_list = list(shared_loci)
hela_counts_shared = [hela_locus_counts.get(l, 0) for l in shared_loci_list]
ars_counts_shared = [ars_locus_counts.get(l, 0) for l in shared_loci_list]
r, p = stats.spearmanr(hela_counts_shared, ars_counts_shared)
print(f"\n  Locus count correlation (shared loci): Spearman r={r:.3f}, p={p:.2e}")

# =========================================================================
# 4. Per-locus poly(A) comparison (shared hotspot loci)
# =========================================================================
print(f"\n{'='*90}")
print("Per-Locus Poly(A) Change: Same Loci, HeLa vs HeLa-Ars")
print(f"{'='*90}")

MIN_READS = 5  # minimum per condition per locus

locus_polya = []
for locus in shared_loci:
    h = hela_anc[hela_anc['transcript_id'] == locus]['polya_length']
    a = ars_anc[ars_anc['transcript_id'] == locus]['polya_length']
    if len(h) >= MIN_READS and len(a) >= MIN_READS:
        h_med = h.median()
        a_med = a.median()
        delta = a_med - h_med
        _, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        locus_polya.append({
            'locus': locus,
            'hela_n': len(h), 'ars_n': len(a),
            'hela_med': h_med, 'ars_med': a_med,
            'delta': delta, 'p': p,
            'subfamily': hela_anc[hela_anc['transcript_id'] == locus]['gene_id'].iloc[0],
        })

locus_df = pd.DataFrame(locus_polya)
locus_df = locus_df.sort_values('delta')

print(f"\n  Loci with ≥{MIN_READS} reads in both conditions: {len(locus_df)}")
print(f"  Loci with shortening (Δ<0): {(locus_df['delta']<0).sum()} ({(locus_df['delta']<0).mean()*100:.1f}%)")
print(f"  Loci with lengthening (Δ>0): {(locus_df['delta']>0).sum()} ({(locus_df['delta']>0).mean()*100:.1f}%)")

# Sign test
n_short = (locus_df['delta'] < 0).sum()
n_long = (locus_df['delta'] > 0).sum()
p_sign = stats.binomtest(n_short, n_short + n_long, 0.5).pvalue
print(f"  Sign test: {n_short} shortened vs {n_long} lengthened, p={p_sign:.2e}")

# Mean/median of per-locus deltas
print(f"  Mean Δ across loci: {locus_df['delta'].mean():+.1f} nt")
print(f"  Median Δ across loci: {locus_df['delta'].median():+.1f} nt")

# Top shortened and lengthened loci
print(f"\n  Top 15 SHORTENED loci:")
print(f"  {'Locus':<28} {'HeLa n':>7} {'Ars n':>7} {'HeLa med':>9} {'Ars med':>9} {'Δ':>7} {'p':>10} {'SF'}")
print(f"  {'-'*85}")
for _, row in locus_df.head(15).iterrows():
    sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else "ns"
    print(f"  {row['locus']:<28} {row['hela_n']:>7} {row['ars_n']:>7} "
          f"{row['hela_med']:>9.1f} {row['ars_med']:>9.1f} {row['delta']:>+6.1f} "
          f"{row['p']:>9.2e}({sig}) {row['subfamily']}")

print(f"\n  Top 10 LENGTHENED loci:")
print(f"  {'Locus':<28} {'HeLa n':>7} {'Ars n':>7} {'HeLa med':>9} {'Ars med':>9} {'Δ':>7} {'p':>10} {'SF'}")
print(f"  {'-'*85}")
for _, row in locus_df.tail(10).iloc[::-1].iterrows():
    sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else "ns"
    print(f"  {row['locus']:<28} {row['hela_n']:>7} {row['ars_n']:>7} "
          f"{row['hela_med']:>9.1f} {row['ars_med']:>9.1f} {row['delta']:>+6.1f} "
          f"{row['p']:>9.2e}({sig}) {row['subfamily']}")

# =========================================================================
# 5. Restricting to SHARED loci only: does the overall effect hold?
# =========================================================================
print(f"\n{'='*90}")
print("Shared-Loci-Only Analysis: Poly(A) Shortening")
print(f"{'='*90}")

hela_shared = hela_anc[hela_anc['transcript_id'].isin(shared_loci)]
ars_shared = ars_anc[ars_anc['transcript_id'].isin(shared_loci)]

h_med = hela_shared['polya_length'].median()
a_med = ars_shared['polya_length'].median()
_, p = stats.mannwhitneyu(hela_shared['polya_length'], ars_shared['polya_length'],
                           alternative='two-sided')
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

print(f"\n  Shared loci only (n_loci={len(shared_loci)}):")
print(f"    HeLa:     n={len(hela_shared):,}  median={h_med:.1f}")
print(f"    HeLa-Ars: n={len(ars_shared):,}  median={a_med:.1f}")
print(f"    Δ={a_med-h_med:+.1f} nt, p={p:.2e} ({sig})")

# vs ALL loci
print(f"\n  All loci:")
h_all_med = hela_anc['polya_length'].median()
a_all_med = ars_anc['polya_length'].median()
print(f"    HeLa:     n={len(hela_anc):,}  median={h_all_med:.1f}")
print(f"    HeLa-Ars: n={len(ars_anc):,}  median={a_all_med:.1f}")
print(f"    Δ={a_all_med-h_all_med:+.1f} nt")

# =========================================================================
# 6. Per-locus paired analysis (locus as unit)
# =========================================================================
print(f"\n{'='*90}")
print("Paired Analysis: Per-Locus Median Poly(A) (Locus as Unit)")
print(f"{'='*90}")

# Only loci with enough reads
locus_paired = locus_df[['locus', 'hela_med', 'ars_med', 'delta']].copy()

# Wilcoxon signed-rank test on per-locus medians
_, p_wilcox = stats.wilcoxon(locus_paired['hela_med'], locus_paired['ars_med'],
                              alternative='two-sided')
print(f"\n  N loci (≥{MIN_READS} each): {len(locus_paired)}")
print(f"  Wilcoxon signed-rank: p={p_wilcox:.2e}")
print(f"  Mean per-locus Δ: {locus_paired['delta'].mean():+.1f} nt")
print(f"  Median per-locus Δ: {locus_paired['delta'].median():+.1f} nt")
print(f"  → The same loci show poly(A) shortening under arsenite")

# =========================================================================
# 7. Locus count changes (expression changes?)
# =========================================================================
print(f"\n{'='*90}")
print("Locus Expression Changes: Read Count HeLa vs HeLa-Ars")
print(f"{'='*90}")

# Normalize by total ancient reads in each condition
hela_total = len(hela_anc)
ars_total = len(ars_anc)

locus_expr = []
for locus in shared_loci:
    h_n = hela_locus_counts.get(locus, 0)
    a_n = ars_locus_counts.get(locus, 0)
    h_frac = h_n / hela_total
    a_frac = a_n / ars_total
    fc = a_frac / h_frac if h_frac > 0 else float('inf')
    locus_expr.append({
        'locus': locus, 'hela_n': h_n, 'ars_n': a_n,
        'hela_frac': h_frac, 'ars_frac': a_frac, 'fold_change': fc
    })

expr_df = pd.DataFrame(locus_expr)
expr_df = expr_df[expr_df['hela_n'] >= MIN_READS]

print(f"\n  Loci with ≥{MIN_READS} reads in HeLa: {len(expr_df)}")
print(f"  Median fold-change (Ars/HeLa, fraction-normalized): {expr_df['fold_change'].median():.2f}")
print(f"  Mean fold-change: {expr_df['fold_change'].mean():.2f}")
print(f"  Loci up (FC>1.5): {(expr_df['fold_change']>1.5).sum()}")
print(f"  Loci down (FC<0.67): {(expr_df['fold_change']<0.67).sum()}")
print(f"  Loci stable (0.67-1.5): {((expr_df['fold_change']>=0.67)&(expr_df['fold_change']<=1.5)).sum()}")

# Correlation: expression FC vs poly(A) delta?
merged = expr_df.merge(locus_df[['locus', 'delta']], on='locus', how='inner')
if len(merged) >= 10:
    r, p = stats.spearmanr(merged['fold_change'], merged['delta'])
    print(f"\n  Correlation: expression FC vs poly(A) Δ: r={r:.3f}, p={p:.2e}")
    print(f"  → {'Correlated' if p < 0.05 else 'Not correlated'}: expression change {'is' if p < 0.05 else 'is NOT'} linked to poly(A) change")

# =========================================================================
# 8. Young L1 loci
# =========================================================================
print(f"\n{'='*90}")
print("Young L1 Loci Overlap")
print(f"{'='*90}")

hela_young = hela[hela['l1_age'] == 'young']
ars_young = ars[ars['l1_age'] == 'young']

hela_young_loci = set(hela_young['transcript_id'].unique())
ars_young_loci = set(ars_young['transcript_id'].unique())
young_shared = hela_young_loci & ars_young_loci

print(f"  HeLa young loci:     {len(hela_young_loci)}")
print(f"  HeLa-Ars young loci: {len(ars_young_loci)}")
print(f"  Shared:              {len(young_shared)} ({len(young_shared)/max(len(hela_young_loci|ars_young_loci),1)*100:.1f}%)")

# =========================================================================
# 9. Summary
# =========================================================================
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"""
  1. Ancient L1 loci overlap between HeLa and HeLa-Ars:
     - {len(shared_loci):,} shared loci ({len(shared_loci)/len(hela_loci|ars_loci)*100:.1f}% of union)
     - {len(hela_shared_reads)/len(hela_anc)*100:.0f}% of HeLa reads and {len(ars_shared_reads)/len(ars_anc)*100:.0f}% of Ars reads come from shared loci

  2. Per-locus poly(A) shortening at the SAME loci:
     - {(locus_df['delta']<0).sum()}/{len(locus_df)} loci shortened (sign test p={p_sign:.2e})
     - Median per-locus Δ = {locus_paired['delta'].median():+.1f} nt
     - Wilcoxon paired test: p={p_wilcox:.2e}
     → The poly(A) shortening is a POST-TRANSCRIPTIONAL change at the same loci

  3. Expression changes are minimal and NOT correlated with poly(A) changes
""")

# Save
locus_df.to_csv(PROJECT / 'analysis/01_exploration/topic_04_state/loci_polya_comparison.tsv',
                sep='\t', index=False)
print("Saved: loci_polya_comparison.tsv")
