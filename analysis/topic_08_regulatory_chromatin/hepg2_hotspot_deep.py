#!/usr/bin/env python3
"""
Analyze m6A modification pattern on L1PA7_dup11216 hotspot reads vs rest of HepG2 L1.
Also check: are these reads actually from LTR12C read-through?
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'

# 1. Load HepG2 L1 summary
print("=== Loading HepG2 L1 summary ===")
summary_rows = []
for f in sorted(RESULTS.glob('HepG2_*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    print(f"  {f.name}: {len(df)} reads")
    summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
summary = summary[summary['qc_tag'] == 'PASS'].copy()

# 2. Identify reads from the hotspot locus
hotspot = summary[
    (summary['chr'] == 'chr14') & 
    (summary['start'] >= 23578000) & 
    (summary['end'] <= 23595000)
].copy()
rest = summary[~summary.index.isin(hotspot.index)].copy()

print(f"\nHotspot reads: {len(hotspot)}")
print(f"Rest of HepG2: {len(rest)}")

# 3. Load Part3 cache for m6A data
print("\n=== Loading Part3 cache ===")
cache_rows = []
for f in sorted(CACHE_DIR.glob('HepG2_*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t')
    cache_rows.append(df)

cache = pd.concat(cache_rows, ignore_index=True)
# Rename to avoid conflict with summary's read_length
cache.rename(columns={'read_length': 'cache_read_length'}, inplace=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['cache_read_length'] / 1000)
cache['psi_per_kb'] = cache['psi_sites_high'] / (cache['cache_read_length'] / 1000)

# 4. Merge
merge_cols = ['read_id', 'm6a_per_kb', 'psi_per_kb', 'm6a_sites_high', 'psi_sites_high', 'cache_read_length']
hotspot_m = hotspot.merge(cache[merge_cols], on='read_id', how='inner')
rest_m = rest.merge(cache[merge_cols], on='read_id', how='inner')

print(f"\nHotspot merged: {len(hotspot_m)}")
print(f"Rest merged: {len(rest_m)}")

# 5. Compare properties
print("\n=== COMPARISON: Hotspot vs Rest ===")
for col, label in [('polya_length', 'Poly(A)'), ('read_length', 'Read length'),
                     ('m6a_per_kb', 'm6A/kb'), ('psi_per_kb', 'Psi/kb'),
                     ('m6a_sites_high', 'm6A sites'), ('psi_sites_high', 'Psi sites')]:
    if col in hotspot_m.columns and col in rest_m.columns:
        h_med = hotspot_m[col].median()
        r_med = rest_m[col].median()
        h_mean = hotspot_m[col].mean()
        r_mean = rest_m[col].mean()
        ratio = h_med / r_med if r_med > 0 else float('inf')
        _, p = stats.mannwhitneyu(hotspot_m[col], rest_m[col], alternative='two-sided')
        print(f"  {label:15s}: Hotspot={h_med:.1f} (mean={h_mean:.1f}), Rest={r_med:.1f} (mean={r_mean:.1f}), ratio={ratio:.2f}x, p={p:.2e}")

# 6. Read position distribution
print("\n=== Hotspot Read Positions ===")
print(f"  Start range: {hotspot['start'].min()} - {hotspot['start'].max()}")
print(f"  End range:   {hotspot['end'].min()} - {hotspot['end'].max()}")
print(f"  Median start: {hotspot['start'].median():.0f}")
print(f"  Median end:   {hotspot['end'].median():.0f}")

l1pa7_end = 23583899
extends_into_ltr = hotspot[hotspot['end'] > l1pa7_end]
print(f"\n  Reads extending beyond L1PA7 annotation (>{l1pa7_end}):")
print(f"    {len(extends_into_ltr)} / {len(hotspot)} ({100*len(extends_into_ltr)/len(hotspot):.1f}%)")
if len(extends_into_ltr) > 0:
    print(f"    Extension range: {extends_into_ltr['end'].min()} - {extends_into_ltr['end'].max()}")
    print(f"    Median extension beyond L1PA7: {extends_into_ltr['end'].median() - l1pa7_end:.0f} bp")

# 7. Strand distribution
if 'strand' in hotspot.columns:
    print(f"\n  Strand distribution:")
    print(hotspot['strand'].value_counts().to_string())
# Try read_strand or te_strand
for sc in ['read_strand', 'te_strand']:
    if sc in hotspot.columns:
        print(f"\n  {sc} distribution:")
        print(hotspot[sc].value_counts().to_string())

# 8. Gene ID distribution (subfamily)
print(f"\n  Gene ID (subfamily) distribution:")
print(hotspot['gene_id'].value_counts().head(10).to_string())

# 9. Read length distribution in more detail
print("\n=== Read Length Distribution ===")
labels_b = ['<300', '300-500', '500-750', '750-1K', '1-1.5K', '1.5-2K', '2-5K', '>5K']
bins = [0, 300, 500, 750, 1000, 1500, 2000, 5000, 100000]
for lbl, mdf in [('Hotspot', hotspot_m), ('Rest', rest_m)]:
    print(f"\n  {lbl}:")
    rl = mdf['read_length']
    print(f"    Q25={rl.quantile(0.25):.0f}, Median={rl.median():.0f}, "
          f"Q75={rl.quantile(0.75):.0f}, Max={rl.max():.0f}")
    mdf = mdf.copy()
    mdf['rl_bin'] = pd.cut(mdf['read_length'], bins=bins, labels=labels_b)
    rl_dist = mdf['rl_bin'].value_counts().sort_index()
    for b, n in rl_dist.items():
        pct = 100*n/len(mdf)
        print(f"    {b:10s}: {n:5d} ({pct:5.1f}%)")

# Store rl_bin back for later use
hotspot_m = hotspot_m.copy()
hotspot_m['rl_bin'] = pd.cut(hotspot_m['read_length'], bins=bins, labels=labels_b)
rest_m = rest_m.copy()
rest_m['rl_bin'] = pd.cut(rest_m['read_length'], bins=bins, labels=labels_b)

# 10. m6A/kb by read length bin for hotspot
print("\n=== m6A/kb by Read Length (Hotspot) ===")
for b in labels_b:
    sub = hotspot_m[hotspot_m['rl_bin'] == b]
    if len(sub) >= 5:
        print(f"  {b:10s}: n={len(sub):4d}, m6A/kb={sub['m6a_per_kb'].median():.2f}")

# 11. Compare m6A/kb at MATCHED read lengths
print("\n=== Read-Length Matched m6A Comparison ===")
for rl_min, rl_max in [(300, 500), (500, 750), (750, 1000), (1000, 2000)]:
    h_sub = hotspot_m[(hotspot_m['read_length'] >= rl_min) & (hotspot_m['read_length'] < rl_max)]
    r_sub = rest_m[(rest_m['read_length'] >= rl_min) & (rest_m['read_length'] < rl_max)]
    if len(h_sub) >= 10 and len(r_sub) >= 10:
        h_med = h_sub['m6a_per_kb'].median()
        r_med = r_sub['m6a_per_kb'].median()
        _, p = stats.mannwhitneyu(h_sub['m6a_per_kb'], r_sub['m6a_per_kb'], alternative='two-sided')
        ratio = h_med / r_med if r_med > 0 else float('inf')
        print(f"  RL {rl_min}-{rl_max}: Hotspot m6A/kb={h_med:.2f} (n={len(h_sub)}), "
              f"Rest={r_med:.2f} (n={len(r_sub)}), ratio={ratio:.2f}x, p={p:.2e}")

# 12. Cross-cell-line check: this locus in other CLs
print("\n=== Cross-Cell-Line: This Locus ===")
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    locus_reads = df[(df['chr'] == 'chr14') & (df['start'] >= 23578000) & (df['end'] <= 23595000)]
    if len(locus_reads) > 0:
        group = f.parent.parent.name
        total = len(df[df['qc_tag'] == 'PASS'])
        pct = 100 * len(locus_reads) / total if total > 0 else 0
        print(f"  {group:20s}: {len(locus_reads):4d} reads ({pct:.1f}% of PASS)")

print("\n=== DONE ===")
