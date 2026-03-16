#!/usr/bin/env python3
"""
Cross-cell-line comparison of L1 m6A / psi / poly(A) landscape.

For each cell line:
  - Poly(A) length distribution (median, mean, IQR)
  - m6A rate (fraction of reads with m6A=1)
  - Psi rate (fraction of reads with psi=1)
  - Split by young vs ancient L1
  - Also m6A/psi sites/kb from MAFIA per-read extraction
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
MIN_READS = 200  # Minimum reads per replicate to include

# Cell line → replicate groups mapping
# Excluded: HEK293 (1 rep, 194 reads), HEK293T (1 valid rep after filtering), THP1 (1 valid rep)
CELL_LINES = {
    'A549':     ['A549_4', 'A549_5', 'A549_6'],
    'H9':       ['H9_2', 'H9_3', 'H9_4'],
    'Hct116':   ['Hct116_3', 'Hct116_4'],
    'HeLa':     ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':    ['HepG2_5', 'HepG2_6'],
    'HEYA8':    ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':     ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':     ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV':  ['MCF7-EV_1'],
    'SHSY5Y':   ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

# =========================================================================
# 1. Load all L1 summaries
# =========================================================================
print("Loading all cell line L1 summaries...")

all_data = {}
for cl, groups in CELL_LINES.items():
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        if len(df) < MIN_READS:
            print(f"  Skipping {g}: {len(df)} reads < MIN_READS={MIN_READS}")
            continue
        df['group'] = g
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined['cell_line'] = cl
        combined['l1_age'] = combined['gene_id'].apply(
            lambda x: 'young' if x in YOUNG else 'ancient')
        all_data[cl] = combined
        print(f"  {cl}: {len(combined):,} reads ({len(dfs)} replicates)")

# =========================================================================
# 2. Overall landscape table
# =========================================================================
print(f"\n{'='*110}")
print("L1 Landscape by Cell Line (ALL reads)")
print(f"{'='*110}")

header = (f"{'Cell Line':<12} {'N':>6} {'polyA med':>10} {'polyA mean':>10} "
          f"{'polyA Q25':>9} {'polyA Q75':>9} "
          f"{'m6A rate':>9} {'psi rate':>9} "
          f"{'rdLen med':>10}")
print(f"\n{header}")
print("-" * 110)

rows = []
for cl in CELL_LINES:
    if cl not in all_data:
        continue
    d = all_data[cl]
    pa_med = d['polya_length'].median()
    pa_mean = d['polya_length'].mean()
    pa_q25 = d['polya_length'].quantile(0.25)
    pa_q75 = d['polya_length'].quantile(0.75)
    m6a_rate = d['m6A'].mean() * 100
    psi_rate = d['psi'].mean() * 100
    rdlen_med = d['read_length'].median()

    print(f"{cl:<12} {len(d):>6,} {pa_med:>10.1f} {pa_mean:>10.1f} "
          f"{pa_q25:>9.1f} {pa_q75:>9.1f} "
          f"{m6a_rate:>8.1f}% {psi_rate:>8.1f}% "
          f"{rdlen_med:>10.0f}")

    rows.append({
        'cell_line': cl, 'n_reads': len(d), 'l1_age': 'all',
        'polya_median': pa_med, 'polya_mean': pa_mean,
        'polya_q25': pa_q25, 'polya_q75': pa_q75,
        'm6a_rate': m6a_rate, 'psi_rate': psi_rate,
        'rdlen_median': rdlen_med,
    })

# =========================================================================
# 3. Split by young vs ancient
# =========================================================================
for age in ['ancient', 'young']:
    print(f"\n{'='*110}")
    print(f"L1 Landscape by Cell Line ({age.upper()} L1)")
    print(f"{'='*110}")
    print(f"\n{header}")
    print("-" * 110)

    for cl in CELL_LINES:
        if cl not in all_data:
            continue
        d = all_data[cl][all_data[cl]['l1_age'] == age]
        if len(d) < 5:
            continue
        pa_med = d['polya_length'].median()
        pa_mean = d['polya_length'].mean()
        pa_q25 = d['polya_length'].quantile(0.25)
        pa_q75 = d['polya_length'].quantile(0.75)
        m6a_rate = d['m6A'].mean() * 100
        psi_rate = d['psi'].mean() * 100
        rdlen_med = d['read_length'].median()

        print(f"{cl:<12} {len(d):>6,} {pa_med:>10.1f} {pa_mean:>10.1f} "
              f"{pa_q25:>9.1f} {pa_q75:>9.1f} "
              f"{m6a_rate:>8.1f}% {psi_rate:>8.1f}% "
              f"{rdlen_med:>10.0f}")

        rows.append({
            'cell_line': cl, 'n_reads': len(d), 'l1_age': age,
            'polya_median': pa_med, 'polya_mean': pa_mean,
            'polya_q25': pa_q25, 'polya_q75': pa_q75,
            'm6a_rate': m6a_rate, 'psi_rate': psi_rate,
            'rdlen_median': rdlen_med,
        })

# =========================================================================
# 4. Statistical tests: Kruskal-Wallis across cell lines
# =========================================================================
print(f"\n{'='*110}")
print("Kruskal-Wallis Tests: Are Cell Lines Different?")
print(f"{'='*110}")

# Exclude treatment variants for clean comparison
base_cls = [cl for cl in CELL_LINES if cl not in ('HeLa-Ars', 'MCF7-EV')]
base_cls_with_data = [cl for cl in base_cls if cl in all_data]

for age_label, age_filter in [('ALL', None), ('Ancient', 'ancient'), ('Young', 'young')]:
    groups_pa = []
    groups_m6a = []
    groups_psi = []
    cl_labels = []

    for cl in base_cls_with_data:
        d = all_data[cl]
        if age_filter:
            d = d[d['l1_age'] == age_filter]
        if len(d) < 10:
            continue
        groups_pa.append(d['polya_length'].values)
        groups_m6a.append(d['m6A'].values)
        groups_psi.append(d['psi'].values)
        cl_labels.append(cl)

    if len(groups_pa) >= 3:
        kw_pa = stats.kruskal(*groups_pa)
        kw_psi = stats.kruskal(*groups_psi)
        kw_m6a = stats.kruskal(*groups_m6a)

        sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"\n  {age_label} L1 ({len(cl_labels)} cell lines):")
        print(f"    Poly(A):  H={kw_pa.statistic:.1f}, p={kw_pa.pvalue:.2e} ({sig(kw_pa.pvalue)})")
        print(f"    psi rate: H={kw_psi.statistic:.1f}, p={kw_psi.pvalue:.2e} ({sig(kw_psi.pvalue)})")
        print(f"    m6A rate: H={kw_m6a.statistic:.1f}, p={kw_m6a.pvalue:.2e} ({sig(kw_m6a.pvalue)})")

# =========================================================================
# 5. Young L1 fraction and count by cell line
# =========================================================================
print(f"\n{'='*110}")
print("Young L1 Fraction by Cell Line")
print(f"{'='*110}")

print(f"\n  {'Cell Line':<12} {'Total':>7} {'Young':>7} {'Ancient':>8} {'Young%':>7} {'Young/kb med':>12}")
print(f"  {'-'*60}")

for cl in CELL_LINES:
    if cl not in all_data:
        continue
    d = all_data[cl]
    n_young = (d['l1_age'] == 'young').sum()
    n_ancient = (d['l1_age'] == 'ancient').sum()
    frac = n_young / len(d) * 100

    print(f"  {cl:<12} {len(d):>7,} {n_young:>7,} {n_ancient:>8,} {frac:>6.1f}%")

# =========================================================================
# 6. Pairwise comparisons: poly(A) and psi
# =========================================================================
print(f"\n{'='*110}")
print("Pairwise: Poly(A) Median and Psi Rate (Ancient L1, base cell lines)")
print(f"{'='*110}")

base_ancient = {}
for cl in base_cls_with_data:
    d = all_data[cl][all_data[cl]['l1_age'] == 'ancient']
    if len(d) >= 10:
        base_ancient[cl] = d

# Sort by poly(A) median
sorted_cls = sorted(base_ancient.keys(),
                    key=lambda cl: base_ancient[cl]['polya_length'].median())

print(f"\n  Sorted by poly(A) median (ascending):")
print(f"  {'Cell Line':<12} {'N':>6} {'polyA med':>10} {'psi rate':>9} {'m6A rate':>9}")
print(f"  {'-'*50}")
for cl in sorted_cls:
    d = base_ancient[cl]
    print(f"  {cl:<12} {len(d):>6,} {d['polya_length'].median():>10.1f} "
          f"{d['psi'].mean()*100:>8.1f}% {d['m6A'].mean()*100:>8.1f}%")

# =========================================================================
# 7. Correlation: poly(A) vs psi rate across cell lines
# =========================================================================
print(f"\n{'='*110}")
print("Cross-Cell-Line Correlation: Poly(A) vs Psi Rate (Ancient L1)")
print(f"{'='*110}")

cl_polya_medians = []
cl_psi_rates = []
cl_m6a_rates = []
cl_names = []

for cl in base_cls_with_data:
    d = all_data[cl][all_data[cl]['l1_age'] == 'ancient']
    if len(d) >= 10:
        cl_polya_medians.append(d['polya_length'].median())
        cl_psi_rates.append(d['psi'].mean() * 100)
        cl_m6a_rates.append(d['m6A'].mean() * 100)
        cl_names.append(cl)

if len(cl_names) >= 5:
    r_psi, p_psi = stats.spearmanr(cl_polya_medians, cl_psi_rates)
    r_m6a, p_m6a = stats.spearmanr(cl_polya_medians, cl_m6a_rates)
    r_psi_m6a, p_psi_m6a = stats.spearmanr(cl_psi_rates, cl_m6a_rates)

    print(f"\n  N cell lines: {len(cl_names)}")
    print(f"  polyA median vs psi rate:  Spearman r={r_psi:.3f}, p={p_psi:.3f}")
    print(f"  polyA median vs m6A rate:  Spearman r={r_m6a:.3f}, p={p_m6a:.3f}")
    print(f"  psi rate vs m6A rate:      Spearman r={r_psi_m6a:.3f}, p={p_psi_m6a:.3f}")

    print(f"\n  Cell line data:")
    for i, cl in enumerate(cl_names):
        print(f"    {cl:<12} polyA={cl_polya_medians[i]:.1f}  psi={cl_psi_rates[i]:.1f}%  m6A={cl_m6a_rates[i]:.1f}%")

# =========================================================================
# 8. Replicate consistency check
# =========================================================================
print(f"\n{'='*110}")
print("Replicate Consistency (Ancient L1)")
print(f"{'='*110}")

print(f"\n  {'Cell Line':<12} {'Group':<15} {'N':>6} {'polyA med':>10} {'psi rate':>9} {'m6A rate':>9}")
print(f"  {'-'*68}")

for cl in CELL_LINES:
    if cl not in all_data:
        continue
    d = all_data[cl][all_data[cl]['l1_age'] == 'ancient']
    for g in sorted(d['group'].unique()):
        gd = d[d['group'] == g]
        if len(gd) < 5:
            continue
        print(f"  {cl:<12} {g:<15} {len(gd):>6,} {gd['polya_length'].median():>10.1f} "
              f"{gd['psi'].mean()*100:>8.1f}% {gd['m6A'].mean()*100:>8.1f}%")

# =========================================================================
# 9. Control comparison (poly(A) only - control summaries don't have m6A/psi)
# =========================================================================
print(f"\n{'='*110}")
print("Control (non-L1) Poly(A) by Cell Line")
print(f"{'='*110}")

ctrl_rows = []
print(f"\n  {'Cell Line':<12} {'N reads':>8} {'polyA med':>10} {'polyA mean':>10}")
print(f"  {'-'*45}")

for cl, groups in CELL_LINES.items():
    ctrl_dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
        if not path.exists():
            continue
        cdf = pd.read_csv(path, sep='\t')
        cdf = cdf[cdf['qc_tag'] == 'PASS'].copy()
        ctrl_dfs.append(cdf)
    if ctrl_dfs:
        ctrl = pd.concat(ctrl_dfs, ignore_index=True)
        print(f"  {cl:<12} {len(ctrl):>8,} {ctrl['polya_length'].median():>10.1f} "
              f"{ctrl['polya_length'].mean():>10.1f}")
        ctrl_rows.append({
            'cell_line': cl, 'n_reads': len(ctrl),
            'polya_median': ctrl['polya_length'].median(),
            'polya_mean': ctrl['polya_length'].mean(),
        })

# =========================================================================
# 10. L1 vs Control poly(A) difference by cell line
# =========================================================================
print(f"\n{'='*110}")
print("L1 vs Control Poly(A) Difference (Median)")
print(f"{'='*110}")

ctrl_dict = {r['cell_line']: r for r in ctrl_rows}

print(f"\n  {'Cell Line':<12} {'L1 polyA':>9} {'Ctrl polyA':>11} {'Δ(L1-Ctrl)':>11} {'p-value':>12}")
print(f"  {'-'*60}")

for cl in CELL_LINES:
    if cl not in all_data or cl not in ctrl_dict:
        continue
    l1_pa = all_data[cl]['polya_length']
    # Reload control for test
    ctrl_dfs = []
    for g in CELL_LINES[cl]:
        path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
        if path.exists():
            cdf = pd.read_csv(path, sep='\t')
            cdf = cdf[cdf['qc_tag'] == 'PASS']
            ctrl_dfs.append(cdf)
    if not ctrl_dfs:
        continue
    ctrl_pa = pd.concat(ctrl_dfs)['polya_length']

    delta = l1_pa.median() - ctrl_pa.median()
    _, p = stats.mannwhitneyu(l1_pa, ctrl_pa, alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"  {cl:<12} {l1_pa.median():>9.1f} {ctrl_pa.median():>11.1f} "
          f"{delta:>+11.1f} {p:.2e} ({sig})")

# =========================================================================
# Save results
# =========================================================================
out_dir = PROJECT / 'analysis/01_exploration/topic_05_cellline'
out_dir.mkdir(parents=True, exist_ok=True)

df_rows = pd.DataFrame(rows)
df_rows.to_csv(out_dir / 'cross_cellline_landscape.tsv', sep='\t', index=False)
print(f"\nSaved: {out_dir / 'cross_cellline_landscape.tsv'}")

# =========================================================================
# 11. Summary
# =========================================================================
print(f"\n{'='*110}")
print("SUMMARY")
print(f"{'='*110}")
