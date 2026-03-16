#!/usr/bin/env python3
"""
Read length distribution analysis: MCF7 vs MCF7-EV L1.
Is EV enriched for longer / more full-length L1 transcripts?
"""

import pandas as pd
import numpy as np
from scipy import stats

PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'

# Load data
mcf7_dfs = []
for g in ['MCF7_2', 'MCF7_3', 'MCF7_4']:
    df = pd.read_csv(f'{PROJECT}/results_group/{g}/g_summary/{g}_L1_summary.tsv', sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    df['group'] = g
    mcf7_dfs.append(df)
mcf7 = pd.concat(mcf7_dfs)

ev = pd.read_csv(f'{PROJECT}/results_group/MCF7-EV_1/g_summary/MCF7-EV_1_L1_summary.tsv', sep='\t')
ev = ev[ev['qc_tag'] == 'PASS']
ev['group'] = 'MCF7-EV_1'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
mcf7['l1_age'] = mcf7['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
ev['l1_age'] = ev['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')

# =========================================================================
# 1. Overall read length by source x age
# =========================================================================
print("=" * 90)
print("Read Length Distribution: MCF7 vs MCF7-EV")
print("=" * 90)

categories = [
    ('MCF7 L1 young', mcf7[mcf7['l1_age']=='young']),
    ('MCF7-EV L1 young', ev[ev['l1_age']=='young']),
    ('MCF7 L1 ancient', mcf7[mcf7['l1_age']=='ancient']),
    ('MCF7-EV L1 ancient', ev[ev['l1_age']=='ancient']),
    ('MCF7 L1 all', mcf7),
    ('MCF7-EV L1 all', ev),
]

print(f"\n{'Label':<22} {'N':>6} {'Mean':>7} {'Median':>7} {'Q25':>6} {'Q75':>6} "
      f"{'>=1kb':>6} {'>=2kb':>6} {'>=3kb':>6} {'>=5kb':>6} {'Max':>6}")
print("-" * 95)
for label, d in categories:
    n = len(d)
    print(f"{label:<22} {n:>6,} {d['read_length'].mean():>7.0f} {d['read_length'].median():>7.0f} "
          f"{d['read_length'].quantile(0.25):>6.0f} {d['read_length'].quantile(0.75):>6.0f} "
          f"{(d['read_length']>=1000).mean()*100:>5.1f}% "
          f"{(d['read_length']>=2000).mean()*100:>5.1f}% "
          f"{(d['read_length']>=3000).mean()*100:>5.1f}% "
          f"{(d['read_length']>=5000).mean()*100:>5.1f}% "
          f"{d['read_length'].max():>6}")

# =========================================================================
# 2. Young L1 by subfamily
# =========================================================================
print(f"\n{'='*90}")
print("Young L1 by Subfamily (full-length ~ 6kb)")
print(f"{'='*90}")

for sf in sorted(YOUNG):
    m = mcf7[mcf7['gene_id'] == sf]
    e = ev[ev['gene_id'] == sf]
    if len(m) == 0 and len(e) == 0:
        continue
    print(f"\n  {sf}:")
    if len(m) > 0:
        print(f"    MCF7:    n={len(m):>4}  mean={m['read_length'].mean():.0f}  "
              f"median={m['read_length'].median():.0f}  "
              f">=1kb: {(m['read_length']>=1000).mean()*100:.1f}%  "
              f">=3kb: {(m['read_length']>=3000).mean()*100:.1f}%  "
              f">=5kb: {(m['read_length']>=5000).mean()*100:.1f}%")
    else:
        print(f"    MCF7:    n=0")
    if len(e) > 0:
        print(f"    MCF7-EV: n={len(e):>4}  mean={e['read_length'].mean():.0f}  "
              f"median={e['read_length'].median():.0f}  "
              f">=1kb: {(e['read_length']>=1000).mean()*100:.1f}%  "
              f">=3kb: {(e['read_length']>=3000).mean()*100:.1f}%  "
              f">=5kb: {(e['read_length']>=5000).mean()*100:.1f}%")
    else:
        print(f"    MCF7-EV: n=0")
    if len(m) > 0 and len(e) > 0:
        _, p = stats.mannwhitneyu(m['read_length'], e['read_length'], alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    MW p={p:.2e} ({sig})")

# =========================================================================
# 3. Read length statistical tests
# =========================================================================
print(f"\n{'='*90}")
print("Read Length Statistical Tests (Mann-Whitney U)")
print(f"{'='*90}")

tests = [
    ("MCF7 vs EV (young)",      mcf7[mcf7['l1_age']=='young'],   ev[ev['l1_age']=='young']),
    ("MCF7 vs EV (ancient)",    mcf7[mcf7['l1_age']=='ancient'], ev[ev['l1_age']=='ancient']),
    ("MCF7 vs EV (all)",        mcf7, ev),
    ("Young vs Ancient (MCF7)", mcf7[mcf7['l1_age']=='young'],   mcf7[mcf7['l1_age']=='ancient']),
    ("Young vs Ancient (EV)",   ev[ev['l1_age']=='young'],       ev[ev['l1_age']=='ancient']),
]

for name, d1, d2 in tests:
    _, p = stats.mannwhitneyu(d1['read_length'], d2['read_length'], alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {name:<30} median {d1['read_length'].median():.0f} vs {d2['read_length'].median():.0f}  "
          f"p={p:.2e} ({sig})")

# =========================================================================
# 4. Young L1 read length bins
# =========================================================================
print(f"\n{'='*90}")
print("Young L1 Read Length Bins: MCF7 vs MCF7-EV")
print(f"{'='*90}")

bins = [0, 500, 1000, 2000, 3000, 5000, 10000]
bin_labels = ['<0.5kb', '0.5-1kb', '1-2kb', '2-3kb', '3-5kb', '>=5kb']

m_young = mcf7[mcf7['l1_age']=='young']['read_length']
e_young = ev[ev['l1_age']=='young']['read_length']

m_hist = pd.cut(m_young, bins=bins, labels=bin_labels).value_counts().sort_index()
e_hist = pd.cut(e_young, bins=bins, labels=bin_labels).value_counts().sort_index()

m_pct = m_hist / len(m_young) * 100
e_pct = e_hist / len(e_young) * 100

print(f"\n{'Bin':<12} {'MCF7 n':>8} {'MCF7 %':>8} {'EV n':>8} {'EV %':>8} {'EV/MCF7':>8}")
print("-" * 55)
for b in bin_labels:
    mn = m_hist.get(b, 0)
    en = e_hist.get(b, 0)
    mp = m_pct.get(b, 0)
    ep = e_pct.get(b, 0)
    ratio = ep / mp if mp > 0 else float('inf')
    print(f"{b:<12} {mn:>8} {mp:>7.1f}% {en:>8} {ep:>7.1f}% {ratio:>7.2f}x")

# =========================================================================
# 5. overlap_length: how much of L1 element is covered
# =========================================================================
print(f"\n{'='*90}")
print("overlap_length: How Much of L1 Element is Covered by Read")
print(f"{'='*90}")

for label, d in categories:
    print(f"  {label:<22} mean={d['overlap_length'].mean():.0f}  median={d['overlap_length'].median():.0f}  "
          f">=1kb: {(d['overlap_length']>=1000).mean()*100:.1f}%  "
          f">=3kb: {(d['overlap_length']>=3000).mean()*100:.1f}%  "
          f">=5kb: {(d['overlap_length']>=5000).mean()*100:.1f}%")

# Overlap length tests
print(f"\n  Mann-Whitney tests (overlap_length):")
for name, d1, d2 in tests:
    _, p = stats.mannwhitneyu(d1['overlap_length'], d2['overlap_length'], alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {name:<30} median {d1['overlap_length'].median():.0f} vs {d2['overlap_length'].median():.0f}  "
          f"p={p:.2e} ({sig})")

# =========================================================================
# 6. dist_to_3prime: how close to L1 3' end
# =========================================================================
print(f"\n{'='*90}")
print("dist_to_3prime: Distance from Read to L1 3' End")
print("(Lower = closer to 3' end; DRS reads from 3' so this reflects capture start)")
print(f"{'='*90}")

for label, d in categories:
    print(f"  {label:<22} mean={d['dist_to_3prime'].mean():.0f}  median={d['dist_to_3prime'].median():.0f}")

print(f"\n  Mann-Whitney tests (dist_to_3prime):")
for name, d1, d2 in tests[:3]:  # MCF7 vs EV only
    _, p = stats.mannwhitneyu(d1['dist_to_3prime'], d2['dist_to_3prime'], alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {name:<30} median {d1['dist_to_3prime'].median():.0f} vs {d2['dist_to_3prime'].median():.0f}  "
          f"p={p:.2e} ({sig})")
