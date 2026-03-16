#!/usr/bin/env python3
"""
HepG2 Hotspot / Locus Concentration Analysis.

Investigates whether HepG2 is an outlier in L1 analyses due to
extreme read concentration at specific loci.

Tasks:
  1. HepG2 top 20 loci with read counts, chromhmm_group, m6A/kb, poly(A)
  2. Top-10 loci share of total HepG2 reads
  3. Cross-CL Gini coefficients and top-5 loci per CL
  4. HepG2 regulatory L1 locus distribution & sensitivity test
  5. Check if dominant locus L1PA7_dup11216 is in regulatory chromatin
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Settings
# =============================================================================
INFILE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/'
              'analysis/01_exploration/topic_08_regulatory_chromatin/'
              'cross_cl_chromhmm_annotated.tsv')

# Also load the broader cross-CL landscape for all 11 cell lines
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
SUMMARY_DIR = PROJECT / 'results_group'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Cell line -> group mapping (all 11 CLs)
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

# =============================================================================
# Helper: Gini coefficient
# =============================================================================
def gini(values):
    """Compute Gini coefficient from an array of non-negative values."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def print_sep(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# =============================================================================
# 1. Load ChromHMM-annotated data (6 CLs with Roadmap epigenomes)
# =============================================================================
print_sep("Loading ChromHMM-annotated data")
df = pd.read_csv(INFILE, sep='\t')
print(f"Total reads: {len(df):,}")
print(f"Cell lines: {sorted(df['cellline'].unique())}")
for cl in sorted(df['cellline'].unique()):
    n = len(df[df['cellline'] == cl])
    print(f"  {cl}: {n:,} reads")

# =============================================================================
# 2. HepG2 top 20 loci
# =============================================================================
print_sep("Task 1: HepG2 Top 20 Loci")
hepg2 = df[df['cellline'] == 'HepG2'].copy()
print(f"HepG2 total reads: {len(hepg2):,}")

# Per-locus stats
locus_stats = hepg2.groupby('gene_id').agg(
    n_reads=('read_id', 'count'),
    l1_age=('l1_age', 'first'),
    chromhmm_groups=('chromhmm_group', lambda x: ', '.join(sorted(x.unique()))),
    n_regulatory=('is_regulatory', 'sum'),
    m6a_median=('m6a_per_kb', 'median'),
    m6a_mean=('m6a_per_kb', 'mean'),
    polya_median=('polya_length', 'median'),
    polya_mean=('polya_length', 'mean'),
    chr_vals=('chr', 'first'),
).sort_values('n_reads', ascending=False)

print(f"\nTotal unique loci: {len(locus_stats):,}")
print(f"Singleton loci: {(locus_stats['n_reads'] == 1).sum():,} "
      f"({(locus_stats['n_reads'] == 1).mean()*100:.1f}%)")

print(f"\nTop 20 HepG2 loci:")
print(f"{'Rank':>4}  {'Locus':<25} {'Age':<8} {'N':>5} {'%Tot':>6}  "
      f"{'ChromHMM':<25} {'Reg':>3}  {'m6A/kb':>7} {'polyA':>7}  {'Chr':<6}")
print("-" * 120)

total_hepg2 = len(hepg2)
cumul = 0
for i, (locus, row) in enumerate(locus_stats.head(20).iterrows()):
    cumul += row['n_reads']
    pct = row['n_reads'] / total_hepg2 * 100
    cum_pct = cumul / total_hepg2 * 100
    age = 'young' if locus.split('_')[0] in YOUNG else 'ancient'
    print(f"{i+1:>4}  {locus:<25} {age:<8} {int(row['n_reads']):>5} {pct:>5.1f}%  "
          f"{row['chromhmm_groups']:<25} {int(row['n_regulatory']):>3}  "
          f"{row['m6a_median']:>7.2f} {row['polya_median']:>7.1f}  {row['chr_vals']:<6}")

# =============================================================================
# 3. Top-10 loci share
# =============================================================================
print_sep("Task 2: Top-10 Loci Share of HepG2 Total Reads")
top10_reads = locus_stats.head(10)['n_reads'].sum()
print(f"Top 10 loci: {top10_reads:,} reads = {top10_reads/total_hepg2*100:.1f}% of HepG2 total ({total_hepg2:,})")
top5_reads = locus_stats.head(5)['n_reads'].sum()
print(f"Top  5 loci: {top5_reads:,} reads = {top5_reads/total_hepg2*100:.1f}%")
top1_reads = locus_stats.head(1)['n_reads'].sum()
print(f"Top  1 locus: {top1_reads:,} reads = {top1_reads/total_hepg2*100:.1f}%")

# Cumulative curve
cumul_pcts = np.cumsum(locus_stats['n_reads'].values) / total_hepg2 * 100
for n in [1, 5, 10, 20, 50, 100]:
    if n <= len(cumul_pcts):
        print(f"  Top {n:>3} loci: {cumul_pcts[n-1]:.1f}%")

# =============================================================================
# 4. Cross-CL Gini & top loci (ChromHMM-annotated CLs only: 6 CLs)
# =============================================================================
print_sep("Task 3: Cross-CL Gini Coefficients (ChromHMM-annotated, 6 CLs)")

cl_gini_data = []
for cl in sorted(df['cellline'].unique()):
    cl_df = df[df['cellline'] == cl]
    lc = cl_df.groupby('gene_id').size()
    g = gini(lc.values)
    total = len(cl_df)
    n_loci = len(lc)
    top10_sum = lc.nlargest(10).sum()
    top10_pct = top10_sum / total * 100
    cl_gini_data.append({
        'cellline': cl, 'n_reads': total, 'n_loci': n_loci,
        'gini': g, 'top10_pct': top10_pct,
        'top1_locus': lc.idxmax(), 'top1_reads': lc.max(),
        'top1_pct': lc.max() / total * 100,
    })

cl_gini = pd.DataFrame(cl_gini_data).sort_values('gini', ascending=False)
print(f"\n{'CL':<12} {'N reads':>7} {'N loci':>7} {'Gini':>6} {'Top10%':>7} "
      f"{'Top1 locus':<25} {'Top1 N':>6} {'Top1%':>6}")
print("-" * 90)
for _, row in cl_gini.iterrows():
    print(f"{row['cellline']:<12} {int(row['n_reads']):>7} {int(row['n_loci']):>7} "
          f"{row['gini']:>6.3f} {row['top10_pct']:>6.1f}% "
          f"{row['top1_locus']:<25} {int(row['top1_reads']):>6} {row['top1_pct']:>5.1f}%")

# Top 5 loci per CL
for cl in sorted(df['cellline'].unique()):
    cl_df = df[df['cellline'] == cl]
    lc = cl_df.groupby('gene_id').size().sort_values(ascending=False)
    total = len(cl_df)
    print(f"\n  {cl} top 5 loci:")
    for j, (locus, cnt) in enumerate(lc.head(5).items()):
        pct = cnt / total * 100
        # Get chromHMM for this locus
        locus_rows = cl_df[cl_df['gene_id'] == locus]
        chrom = locus_rows['chromhmm_group'].mode().iloc[0] if len(locus_rows) > 0 else '?'
        age = 'young' if locus.split('_')[0] in YOUNG else 'ancient'
        print(f"    {j+1}. {locus:<25} {cnt:>5} ({pct:>5.1f}%) [{age}, {chrom}]")

# =============================================================================
# 5. Load ALL 11 CLs from L1 summaries for broader Gini comparison
# =============================================================================
print_sep("Task 3b: Cross-CL Gini (All 11 Cell Lines from L1 Summaries)")

all_cl_dfs = []
for cl, groups in CELL_LINES.items():
    for grp in groups:
        fpath = SUMMARY_DIR / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if fpath.exists():
            tmp = pd.read_csv(fpath, sep='\t')
            tmp['cellline'] = cl
            all_cl_dfs.append(tmp)
        else:
            print(f"  WARNING: {fpath} not found")

all_cl = pd.concat(all_cl_dfs, ignore_index=True)
print(f"Total reads across 11 CLs: {len(all_cl):,}")

# gene_id column check
gene_col = 'gene_id' if 'gene_id' in all_cl.columns else 'l1_id'
if gene_col not in all_cl.columns:
    # Check what columns are available
    print(f"Available columns: {list(all_cl.columns)}")
    # Try to find an appropriate locus ID column
    for candidate in ['gene_id', 'l1_id', 'repeat_name', 'name']:
        if candidate in all_cl.columns:
            gene_col = candidate
            break
print(f"Using locus column: {gene_col}")

all_cl_gini_data = []
for cl in sorted(CELL_LINES.keys()):
    cl_df = all_cl[all_cl['cellline'] == cl]
    if len(cl_df) == 0:
        continue
    lc = cl_df.groupby(gene_col).size()
    g = gini(lc.values)
    total = len(cl_df)
    n_loci = len(lc)
    top10_sum = lc.nlargest(10).sum()
    top10_pct = top10_sum / total * 100
    singleton_pct = (lc == 1).mean() * 100
    all_cl_gini_data.append({
        'cellline': cl, 'n_reads': total, 'n_loci': n_loci,
        'gini': g, 'top10_pct': top10_pct,
        'singleton_pct': singleton_pct,
        'top1_locus': lc.idxmax(), 'top1_reads': lc.max(),
        'top1_pct': lc.max() / total * 100,
    })

all_cl_gini = pd.DataFrame(all_cl_gini_data).sort_values('gini', ascending=False)
print(f"\n{'CL':<12} {'N reads':>7} {'N loci':>7} {'Gini':>6} {'Top10%':>7} "
      f"{'Singl%':>7} {'Top1 locus':<25} {'Top1 N':>6} {'Top1%':>6}")
print("-" * 100)
for _, row in all_cl_gini.iterrows():
    print(f"{row['cellline']:<12} {int(row['n_reads']):>7} {int(row['n_loci']):>7} "
          f"{row['gini']:>6.3f} {row['top10_pct']:>6.1f}% {row['singleton_pct']:>6.1f}% "
          f"{row['top1_locus']:<25} {int(row['top1_reads']):>6} {row['top1_pct']:>5.1f}%")

# =============================================================================
# 6. HepG2 Regulatory L1 locus distribution
# =============================================================================
print_sep("Task 4: HepG2 Regulatory L1 Locus Distribution")
hepg2_reg = hepg2[hepg2['is_regulatory'] == True].copy()
print(f"HepG2 regulatory reads: {len(hepg2_reg):,} ({len(hepg2_reg)/len(hepg2)*100:.1f}% of total)")
print(f"HepG2 non-regulatory reads: {len(hepg2) - len(hepg2_reg):,}")

reg_loci = hepg2_reg.groupby('gene_id').agg(
    n_reads=('read_id', 'count'),
    m6a_median=('m6a_per_kb', 'median'),
    m6a_mean=('m6a_per_kb', 'mean'),
    polya_median=('polya_length', 'median'),
    chromhmm_state=('chromhmm_state', lambda x: ', '.join(sorted(x.unique()))),
    chr_vals=('chr', 'first'),
    l1_age=('l1_age', 'first'),
).sort_values('n_reads', ascending=False)

print(f"Unique regulatory loci: {len(reg_loci):,}")
print(f"Singleton regulatory loci: {(reg_loci['n_reads'] == 1).sum():,} "
      f"({(reg_loci['n_reads'] == 1).mean()*100:.1f}%)")

print(f"\nTop 20 HepG2 regulatory loci:")
print(f"{'Rank':>4}  {'Locus':<25} {'Age':<8} {'N':>5} {'%Reg':>6}  "
      f"{'State':<20} {'m6A/kb':>7} {'polyA':>7} {'Chr':<6}")
print("-" * 100)

total_reg = len(hepg2_reg)
for i, (locus, row) in enumerate(reg_loci.head(20).iterrows()):
    pct = row['n_reads'] / total_reg * 100
    age = 'young' if locus.split('_')[0] in YOUNG else 'ancient'
    print(f"{i+1:>4}  {locus:<25} {age:<8} {int(row['n_reads']):>5} {pct:>5.1f}%  "
          f"{row['chromhmm_state']:<20} {row['m6a_median']:>7.2f} {row['polya_median']:>7.1f} "
          f"{row['chr_vals']:<6}")

# Gini of regulatory loci
reg_gini = gini(reg_loci['n_reads'].values)
print(f"\nGini of regulatory reads across loci: {reg_gini:.3f}")
top5_reg = reg_loci.head(5)['n_reads'].sum()
print(f"Top 5 regulatory loci: {top5_reg:,} reads = {top5_reg/total_reg*100:.1f}% of regulatory")

# =============================================================================
# 7. Sensitivity test: remove top regulatory loci
# =============================================================================
print_sep("Task 4b: Sensitivity — Remove Top Regulatory Loci")

# Baseline: full regulatory vs non-regulatory
reg_m6a = hepg2_reg['m6a_per_kb'].median()
nonreg_m6a = hepg2[hepg2['is_regulatory'] == False]['m6a_per_kb'].median()
ratio_full = reg_m6a / nonreg_m6a if nonreg_m6a > 0 else float('inf')
print(f"\nBaseline: Regulatory m6A/kb = {reg_m6a:.2f}, Non-reg = {nonreg_m6a:.2f}, "
      f"Ratio = {ratio_full:.2f}x")

# Remove top N regulatory loci and recompute
for n_remove in [1, 2, 3, 5, 10, 20]:
    if n_remove > len(reg_loci):
        break
    top_n_loci = set(reg_loci.head(n_remove).index)
    remaining = hepg2_reg[~hepg2_reg['gene_id'].isin(top_n_loci)]
    if len(remaining) == 0:
        print(f"  Remove top {n_remove:>2}: no reads remaining")
        continue
    rem_m6a = remaining['m6a_per_kb'].median()
    ratio = rem_m6a / nonreg_m6a if nonreg_m6a > 0 else float('inf')
    print(f"  Remove top {n_remove:>2} loci ({sum(reg_loci.head(n_remove)['n_reads']):>4} reads): "
          f"remaining {len(remaining):>4} reads, m6A/kb = {rem_m6a:.2f}, "
          f"ratio = {ratio:.2f}x")

# Also: per-locus median comparison
print(f"\nPer-locus median m6A/kb (each locus = 1 observation):")
reg_locus_m6a = reg_loci['m6a_median'].median()
# Non-reg locus medians
nonreg_loci = hepg2[hepg2['is_regulatory'] == False].groupby('gene_id')['m6a_per_kb'].median()
nonreg_locus_m6a = nonreg_loci.median()
print(f"  Regulatory loci median of medians: {reg_locus_m6a:.2f}")
print(f"  Non-reg loci median of medians: {nonreg_locus_m6a:.2f}")
print(f"  Ratio: {reg_locus_m6a/nonreg_locus_m6a:.2f}x" if nonreg_locus_m6a > 0 else "  Ratio: N/A")

# =============================================================================
# 8. Is L1PA7_dup11216 regulatory?
# =============================================================================
print_sep("Task 5: Check L1PA7_dup11216 Chromatin State")

target = 'L1PA7_dup11216'
target_rows = hepg2[hepg2['gene_id'] == target]
if len(target_rows) == 0:
    # Try partial match
    matches = hepg2[hepg2['gene_id'].str.contains('L1PA7', na=False)]
    print(f"Exact match for '{target}' not found. Partial matches with 'L1PA7':")
    if len(matches) > 0:
        for gid, cnt in matches.groupby('gene_id').size().sort_values(ascending=False).head(10).items():
            chrom = matches[matches['gene_id'] == gid]['chromhmm_group'].mode().iloc[0]
            print(f"  {gid}: {cnt} reads, {chrom}")
    else:
        print("  No L1PA7 matches in ChromHMM-annotated data.")

    # The gene_id format might differ. Check all top loci
    print("\nLet me check gene_id format for top HepG2 loci:")
    for locus in locus_stats.head(5).index:
        print(f"  {locus}: {int(locus_stats.loc[locus, 'n_reads'])} reads")

    # Also check in the broader all_cl data
    if gene_col in all_cl.columns:
        hepg2_all = all_cl[all_cl['cellline'] == 'HepG2']
        target_all = hepg2_all[hepg2_all[gene_col].str.contains('L1PA7', na=False)]
        if len(target_all) > 0:
            print(f"\nIn L1 summary (all CLs), L1PA7 matches for HepG2:")
            for gid, cnt in target_all.groupby(gene_col).size().sort_values(ascending=False).head(5).items():
                print(f"  {gid}: {cnt} reads")
else:
    print(f"\n{target} in HepG2: {len(target_rows)} reads")
    print(f"  ChromHMM groups: {target_rows['chromhmm_group'].value_counts().to_dict()}")
    print(f"  ChromHMM states: {target_rows['chromhmm_state'].value_counts().to_dict()}")
    print(f"  is_regulatory: {target_rows['is_regulatory'].sum()} / {len(target_rows)}")
    print(f"  m6A/kb: median={target_rows['m6a_per_kb'].median():.2f}, "
          f"mean={target_rows['m6a_per_kb'].mean():.2f}")
    print(f"  poly(A): median={target_rows['polya_length'].median():.1f}, "
          f"mean={target_rows['polya_length'].mean():.1f}")
    print(f"  l1_age: {target_rows['l1_age'].iloc[0]}")
    print(f"  chr: {target_rows['chr'].iloc[0]}")

# =============================================================================
# 9. Cross-CL comparison: HepG2 outlier metrics
# =============================================================================
print_sep("Summary: Is HepG2 an Outlier?")

# Compute per-CL metrics for ChromHMM-annotated CLs
print("\nPer-CL summary (ChromHMM-annotated, 6 CLs):")
print(f"{'CL':<12} {'N':>6} {'Gini':>6} {'Top10%':>7} {'%Reg':>6} "
      f"{'m6A/kb':>7} {'Reg m6A':>8} {'NonR m6A':>9} {'Ratio':>6}")
print("-" * 85)

for cl in sorted(df['cellline'].unique()):
    cl_df = df[df['cellline'] == cl]
    n = len(cl_df)
    lc = cl_df.groupby('gene_id').size()
    g = gini(lc.values)
    top10 = lc.nlargest(10).sum() / n * 100
    reg_n = cl_df['is_regulatory'].sum()
    reg_pct = reg_n / n * 100
    overall_m6a = cl_df['m6a_per_kb'].median()
    if reg_n > 0:
        r_m6a = cl_df[cl_df['is_regulatory'] == True]['m6a_per_kb'].median()
    else:
        r_m6a = np.nan
    nr_m6a = cl_df[cl_df['is_regulatory'] == False]['m6a_per_kb'].median()
    ratio = r_m6a / nr_m6a if (nr_m6a > 0 and not np.isnan(r_m6a)) else np.nan
    print(f"{cl:<12} {n:>6} {g:>6.3f} {top10:>6.1f}% {reg_pct:>5.1f}% "
          f"{overall_m6a:>7.2f} {r_m6a:>8.2f} {nr_m6a:>9.2f} {ratio:>5.2f}x"
          if not np.isnan(ratio) else
          f"{cl:<12} {n:>6} {g:>6.3f} {top10:>6.1f}% {reg_pct:>5.1f}% "
          f"{overall_m6a:>7.2f} {'N/A':>8} {nr_m6a:>9.2f} {'N/A':>6}")

# HepG2 vs others: is the regulatory % extreme?
hepg2_reg_pct = hepg2['is_regulatory'].mean() * 100
other_reg_pcts = []
for cl in sorted(df['cellline'].unique()):
    if cl != 'HepG2':
        cl_df = df[df['cellline'] == cl]
        other_reg_pcts.append(cl_df['is_regulatory'].mean() * 100)
print(f"\nHepG2 regulatory %: {hepg2_reg_pct:.1f}%")
print(f"Other CLs regulatory %: {np.mean(other_reg_pcts):.1f}% +/- {np.std(other_reg_pcts):.1f}% "
      f"(range: {np.min(other_reg_pcts):.1f}% - {np.max(other_reg_pcts):.1f}%)")
z_score = (hepg2_reg_pct - np.mean(other_reg_pcts)) / np.std(other_reg_pcts) if np.std(other_reg_pcts) > 0 else 0
print(f"HepG2 z-score for regulatory %: {z_score:.1f}")

# =============================================================================
# 10. HepG2 outlier in m6A/kb: driven by top loci?
# =============================================================================
print_sep("Sensitivity: HepG2 m6A/kb Without Top Loci")

# Remove top N loci and recompute overall m6A/kb
hepg2_m6a_full = hepg2['m6a_per_kb'].median()
print(f"HepG2 full m6A/kb median: {hepg2_m6a_full:.2f} ({len(hepg2):,} reads)")

for n_remove in [1, 5, 10, 20, 50]:
    if n_remove > len(locus_stats):
        break
    top_n = set(locus_stats.head(n_remove).index)
    remaining = hepg2[~hepg2['gene_id'].isin(top_n)]
    if len(remaining) == 0:
        break
    rem_m6a = remaining['m6a_per_kb'].median()
    print(f"  Remove top {n_remove:>2} loci ({sum(locus_stats.head(n_remove)['n_reads']):>5} reads removed): "
          f"{len(remaining):>5} remaining, m6A/kb = {rem_m6a:.2f}")

# Compare to other CLs
print(f"\nOther CL m6A/kb medians:")
for cl in sorted(df['cellline'].unique()):
    cl_df = df[df['cellline'] == cl]
    print(f"  {cl:<12}: {cl_df['m6a_per_kb'].median():.2f} ({len(cl_df):,} reads)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
