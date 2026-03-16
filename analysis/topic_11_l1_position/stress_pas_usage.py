#!/usr/bin/env python3
"""
Does arsenite stress change L1 PAS usage (proportion of reads at 3' end)?
Hypothesis: if stress preferentially degrades L1 transcripts using downstream PAS,
the surviving pool should be enriched for own-PAS reads.
"""
import pandas as pd
import numpy as np
import glob, os
from scipy import stats

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'

frames = []
for f in sorted(glob.glob(f'{BASE}/results_group/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'read_length', 'overlap_length',
        'te_start', 'te_end', 'gene_id', 'dist_to_3prime',
        'polya_length', 'qc_tag', 'TE_group'
    ])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['group'] = group
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    frames.append(tmp)

df = pd.concat(frames, ignore_index=True)
df = df[df['qc_tag'] == 'PASS'].copy()
df['is_young'] = df['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')
df['overlap_frac'] = df['overlap_length'] / df['read_length']
df['l1_at_3end'] = df['dist_to_3prime'] <= 50

hela = df[df['cell_line'] == 'HeLa']
ars = df[df['cell_line'] == 'HeLa-Ars']

print("=" * 70)
print("STRESS vs L1 PAS USAGE")
print("=" * 70)

# ── 1. Overall proportions ──
print("\n[1] Overall: L1 at 3' end proportion")
print(f"  HeLa:     {hela['l1_at_3end'].sum()}/{len(hela)} = {hela['l1_at_3end'].mean()*100:.1f}%")
print(f"  HeLa-Ars: {ars['l1_at_3end'].sum()}/{len(ars)} = {ars['l1_at_3end'].mean()*100:.1f}%")
ct = pd.crosstab(
    pd.Categorical(['HeLa']*len(hela) + ['HeLa-Ars']*len(ars)),
    list(hela['l1_at_3end']) + list(ars['l1_at_3end'])
)
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"  χ² = {chi2:.2f}, P = {p:.3f}")

# ── 2. Stratified by age ──
print("\n[2] By L1 age")
for is_y, label in [(True, "Young"), (False, "Ancient")]:
    h = hela[hela['is_young'] == is_y]
    a = ars[ars['is_young'] == is_y]
    h_frac = h['l1_at_3end'].mean() * 100
    a_frac = a['l1_at_3end'].mean() * 100
    if len(h) > 20 and len(a) > 20:
        ct2 = pd.crosstab(
            pd.Categorical(['H']*len(h) + ['A']*len(a)),
            list(h['l1_at_3end']) + list(a['l1_at_3end'])
        )
        chi2, p2, _, _ = stats.chi2_contingency(ct2)
    else:
        p2 = np.nan
    print(f"  {label}: HeLa {h_frac:.1f}% (n={len(h)}) → Ars {a_frac:.1f}% (n={len(a)}), P={p2:.3f}")

# ── 3. Stratified by genomic context ──
print("\n[3] By genomic context")
for ctx in ['intergenic', 'intronic']:
    h = hela[hela['TE_group'] == ctx]
    a = ars[ars['TE_group'] == ctx]
    h_frac = h['l1_at_3end'].mean() * 100
    a_frac = a['l1_at_3end'].mean() * 100
    ct3 = pd.crosstab(
        pd.Categorical(['H']*len(h) + ['A']*len(a)),
        list(h['l1_at_3end']) + list(a['l1_at_3end'])
    )
    chi2, p3, _, _ = stats.chi2_contingency(ct3)
    print(f"  {ctx}: HeLa {h_frac:.1f}% (n={len(h)}) → Ars {a_frac:.1f}% (n={len(a)}), P={p3:.3f}")

# ── 4. Absolute read counts — does stress differentially remove non-PAS reads? ──
print("\n[4] Absolute read counts (per-replicate normalized)")
# Calculate reads per replicate
for cl in ['HeLa', 'HeLa-Ars']:
    sub = df[df['cell_line'] == cl]
    n_reps = sub['group'].nunique()
    at3 = sub['l1_at_3end'].sum() / n_reps
    away = (~sub['l1_at_3end']).sum() / n_reps
    print(f"  {cl} ({n_reps} reps): at-3'={at3:.0f}/rep, away={away:.0f}/rep, "
          f"total={at3+away:.0f}/rep, ratio at-3'={at3/(at3+away)*100:.1f}%")

# ── 5. Read length matched comparison ──
print("\n[5] Read-length matched: does apparent PAS usage change?")
print("  (Longer reads have L1 further from 3' → length confound?)")
for rl_bin, lo, hi in [("short (<700bp)", 0, 700), ("medium (700-1200bp)", 700, 1200), ("long (>1200bp)", 1200, 10000)]:
    h = hela[(hela['read_length'] >= lo) & (hela['read_length'] < hi)]
    a = ars[(ars['read_length'] >= lo) & (ars['read_length'] < hi)]
    if len(h) > 30 and len(a) > 30:
        h_frac = h['l1_at_3end'].mean() * 100
        a_frac = a['l1_at_3end'].mean() * 100
        print(f"  {rl_bin}: HeLa {h_frac:.1f}% (n={len(h)}) → Ars {a_frac:.1f}% (n={len(a)})")

# ── 6. Does dist_to_3prime distribution shift under stress? ──
print("\n[6] dist_to_3prime distribution shift (Ancient only)")
h_dist = hela[~hela['is_young']]['dist_to_3prime'].values
a_dist = ars[~ars['is_young']]['dist_to_3prime'].values
u, p_mw = stats.mannwhitneyu(h_dist, a_dist, alternative='two-sided')
print(f"  HeLa Ancient: median={np.median(h_dist):.0f}, mean={np.mean(h_dist):.0f} (n={len(h_dist)})")
print(f"  HeLa-Ars Ancient: median={np.median(a_dist):.0f}, mean={np.mean(a_dist):.0f} (n={len(a_dist)})")
print(f"  MW U P = {p_mw:.4f}")
print(f"  Percentiles (HeLa/Ars): 25%={np.percentile(h_dist,25):.0f}/{np.percentile(a_dist,25):.0f}, "
      f"50%={np.percentile(h_dist,50):.0f}/{np.percentile(a_dist,50):.0f}, "
      f"75%={np.percentile(h_dist,75):.0f}/{np.percentile(a_dist,75):.0f}")

# ── 7. Survival analysis perspective ──
print("\n[7] 'Survival' perspective: among reads with poly(A) < 30nt (decay zone)")
print("    Are decay-zone reads more likely to be non-PAS (L1 away from 3')?")
for cl, sub in [("HeLa", hela), ("HeLa-Ars", ars)]:
    decay = sub[sub['polya_length'] < 30]
    alive = sub[sub['polya_length'] >= 30]
    if len(decay) > 10:
        d_frac = decay['l1_at_3end'].mean() * 100
        a_frac = alive['l1_at_3end'].mean() * 100
        print(f"  {cl}: decay zone at-3'={d_frac:.1f}% (n={len(decay)}), "
              f"alive at-3'={a_frac:.1f}% (n={len(alive)})")

# ── 8. Per-replicate variability ──
print("\n[8] Per-replicate at-3'-end fraction")
for cl in ['HeLa', 'HeLa-Ars']:
    sub = df[df['cell_line'] == cl]
    for grp in sorted(sub['group'].unique()):
        g = sub[sub['group'] == grp]
        frac = g['l1_at_3end'].mean() * 100
        print(f"  {grp}: {frac:.1f}% (n={len(g)})")

print("\n" + "=" * 70)
