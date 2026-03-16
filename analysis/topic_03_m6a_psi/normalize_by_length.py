#!/usr/bin/env python3
"""
Read-length normalized m6A/psi analysis: sites/kb instead of sites/read.
Reads pre-computed per-read data from compare_mcf7_vs_ev_by_age.py.
"""

import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_03_m6a_psi'
df = pd.read_csv(f'{OUTPUT_DIR}/mcf7_ev_by_age_per_read.tsv', sep='\t')

# Sites per kilobase
df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)
df['psi_per_kb'] = df['psi_sites_high'] / (df['read_length'] / 1000)

# Define groups
mcf7_young = df[(df['source'] == 'MCF7_L1') & (df['l1_age'] == 'young')]
mcf7_ancient = df[(df['source'] == 'MCF7_L1') & (df['l1_age'] == 'ancient')]
ev_young = df[(df['source'] == 'MCF7-EV_L1') & (df['l1_age'] == 'young')]
ev_ancient = df[(df['source'] == 'MCF7-EV_L1') & (df['l1_age'] == 'ancient')]
ctrl = df[df['source'] == 'MCF7_Control']

groups = {
    'MCF7_L1_young': mcf7_young,
    'MCF7_L1_ancient': mcf7_ancient,
    'MCF7-EV_L1_young': ev_young,
    'MCF7-EV_L1_ancient': ev_ancient,
    'MCF7_Control': ctrl,
}

# =========================================================================
# Summary table
# =========================================================================
print("=" * 90)
print("MCF7 vs MCF7-EV: m6A/psi Read-Length Normalized (sites/kb)")
print("=" * 90)

print(f"\n{'Label':<22} {'N':>6} {'AvgLen':>7} "
      f"{'m6A/read':>9} {'m6A/kb':>8} {'psi/read':>9} {'psi/kb':>8} "
      f"{'m6A%':>7} {'psi%':>7}")
print("-" * 90)

for label, gdf in groups.items():
    n = len(gdf)
    m6a_rate = (gdf['m6a_sites_high'] > 0).mean() * 100
    psi_rate = (gdf['psi_sites_high'] > 0).mean() * 100
    print(f"{label:<22} {n:>6,} {gdf['read_length'].mean():>7.0f} "
          f"{gdf['m6a_sites_high'].mean():>9.3f} {gdf['m6a_per_kb'].mean():>8.3f} "
          f"{gdf['psi_sites_high'].mean():>9.3f} {gdf['psi_per_kb'].mean():>8.3f} "
          f"{m6a_rate:>6.1f}% {psi_rate:>6.1f}%")

# =========================================================================
# Pairwise Mann-Whitney tests
# =========================================================================
def mw_test(v1, v2):
    if len(v1) == 0 or len(v2) == 0:
        return np.nan, np.nan, 'n/a'
    stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return stat, p, sig

comparisons = [
    ("YOUNG: MCF7 L1 vs Control",        mcf7_young,  ctrl),
    ("YOUNG: EV L1 vs Control",           ev_young,    ctrl),
    ("YOUNG: MCF7 L1 vs EV L1",           mcf7_young,  ev_young),
    ("ANCIENT: MCF7 L1 vs Control",       mcf7_ancient, ctrl),
    ("ANCIENT: EV L1 vs Control",         ev_ancient,  ctrl),
    ("ANCIENT: MCF7 L1 vs EV L1",         mcf7_ancient, ev_ancient),
    ("MCF7: Young vs Ancient",            mcf7_young,  mcf7_ancient),
    ("EV: Young vs Ancient",              ev_young,    ev_ancient),
]

print(f"\n{'='*90}")
print("Pairwise Comparisons (Mann-Whitney U)")
print(f"{'='*90}")

for title, df1, df2 in comparisons:
    print(f"\n  {title}")
    for mod, col_raw, col_norm in [('m6A', 'm6a_sites_high', 'm6a_per_kb'),
                                    ('psi', 'psi_sites_high', 'psi_per_kb')]:
        _, p_raw, sig_raw = mw_test(df1[col_raw].values, df2[col_raw].values)
        _, p_norm, sig_norm = mw_test(df1[col_norm].values, df2[col_norm].values)
        mean1_raw = df1[col_raw].mean()
        mean2_raw = df2[col_raw].mean()
        mean1_norm = df1[col_norm].mean()
        mean2_norm = df2[col_norm].mean()
        p_raw_str = f"{p_raw:.2e}" if not np.isnan(p_raw) else "n/a"
        p_norm_str = f"{p_norm:.2e}" if not np.isnan(p_norm) else "n/a"
        print(f"    {mod}: /read {mean1_raw:.2f} vs {mean2_raw:.2f} p={p_raw_str}({sig_raw}) "
              f"| /kb {mean1_norm:.3f} vs {mean2_norm:.3f} p={p_norm_str}({sig_norm})")

# =========================================================================
# Key question: EV psi enrichment artifact check
# =========================================================================
print(f"\n{'='*90}")
print("Key: Is EV psi/m6A enrichment a read-length artifact?")
print(f"{'='*90}")

for age in ['young', 'ancient']:
    mcf7_g = df[(df['source'] == 'MCF7_L1') & (df['l1_age'] == age)]
    ev_g = df[(df['source'] == 'MCF7-EV_L1') & (df['l1_age'] == age)]

    print(f"\n  [{age.upper()}]")
    print(f"    read length:  MCF7={mcf7_g['read_length'].mean():.0f}  EV={ev_g['read_length'].mean():.0f}  "
          f"(EV/MCF7={ev_g['read_length'].mean()/mcf7_g['read_length'].mean():.2f}x)")

    for mod in ['m6a', 'psi']:
        col_raw = f'{mod}_sites_high'
        col_norm = f'{mod}_per_kb'
        label = 'm6A' if mod == 'm6a' else 'psi'

        raw_ratio = ev_g[col_raw].mean() / mcf7_g[col_raw].mean() if mcf7_g[col_raw].mean() > 0 else float('inf')
        norm_ratio = ev_g[col_norm].mean() / mcf7_g[col_norm].mean() if mcf7_g[col_norm].mean() > 0 else float('inf')

        _, p_raw, sig_raw = mw_test(mcf7_g[col_raw].values, ev_g[col_raw].values)
        _, p_norm, sig_norm = mw_test(mcf7_g[col_norm].values, ev_g[col_norm].values)

        print(f"    {label} sites/read: MCF7={mcf7_g[col_raw].mean():.2f} EV={ev_g[col_raw].mean():.2f} "
              f"({raw_ratio:.2f}x) p={p_raw:.2e}({sig_raw})")
        print(f"    {label} sites/kb:   MCF7={mcf7_g[col_norm].mean():.3f} EV={ev_g[col_norm].mean():.3f} "
              f"({norm_ratio:.2f}x) p={p_norm:.2e}({sig_norm})")

# Save
df.to_csv(f'{OUTPUT_DIR}/mcf7_ev_by_age_per_read_normalized.tsv', sep='\t', index=False)
print(f"\nSaved: mcf7_ev_by_age_per_read_normalized.tsv")
