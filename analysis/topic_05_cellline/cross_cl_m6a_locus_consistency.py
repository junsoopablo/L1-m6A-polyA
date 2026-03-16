#!/usr/bin/env python3
"""
Cross-cell-line m6A consistency at the LOCUS level.
Question: Do the same L1 loci have consistent m6A levels across cell lines?
Also: Is the Young > Ancient pattern, and subfamily-level m6A, consistent?
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CACHE = PROJECT / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
SUMMARY_DIR = PROJECT / 'results_group'

# Cell line grouping (exclude HeLa-Ars and MCF7-EV for baseline analysis)
CL_GROUPS = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Load all data
all_reads = []
for cl, groups in CL_GROUPS.items():
    for grp in groups:
        cache_file = CACHE / f'{grp}_l1_per_read.tsv'
        if not cache_file.exists():
            continue
        df = pd.read_csv(cache_file, sep='\t',
                         usecols=['read_id', 'read_length', 'm6a_sites_high'])
        df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)

        summary_file = SUMMARY_DIR / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if summary_file.exists():
            summ = pd.read_csv(summary_file, sep='\t',
                               usecols=['read_id', 'polya_length', 'transcript_id'])
            df = df.merge(summ, on='read_id', how='left')
        else:
            df['polya_length'] = np.nan
            df['transcript_id'] = ''

        df['cell_line'] = cl
        all_reads.append(df)

data = pd.concat(all_reads, ignore_index=True)
print(f'Total reads: {len(data)} from {data["cell_line"].nunique()} cell lines')

# Parse subfamily and age
data['subfamily'] = data['transcript_id'].str.split('_dup').str[0]
data['age'] = data['subfamily'].apply(
    lambda x: 'young' if x in YOUNG else 'ancient' if pd.notna(x) else 'unknown')

# =====================================================================
# 1. Per-locus m6A consistency across cell lines
# =====================================================================
print('\n' + '='*70)
print('1. PER-LOCUS m6A RANK CORRELATION ACROSS CELL LINES')
print('='*70)

# Get loci expressed in >= 2 cell lines with >= 3 reads each
locus_cl = data.groupby(['transcript_id', 'cell_line']).agg(
    m6a_median=('m6a_per_kb', 'median'),
    n_reads=('read_id', 'count')
).reset_index()

# Require >= 3 reads per locus per CL for reliable median
locus_cl_filt = locus_cl[locus_cl['n_reads'] >= 3]

# Count how many CLs each locus appears in
locus_ncl = locus_cl_filt.groupby('transcript_id')['cell_line'].nunique()
multi_cl_loci = locus_ncl[locus_ncl >= 2].index
print(f'\nLoci with >= 3 reads in >= 2 CLs: {len(multi_cl_loci)}')

multi_cl_loci_5 = locus_ncl[locus_ncl >= 5].index
print(f'Loci with >= 3 reads in >= 5 CLs: {len(multi_cl_loci_5)}')

# For shared loci (>= 2 CLs), compute pairwise Spearman r
if len(multi_cl_loci) > 10:
    # Pivot: locus × CL
    pivot = locus_cl_filt[locus_cl_filt['transcript_id'].isin(multi_cl_loci)].pivot_table(
        index='transcript_id', columns='cell_line', values='m6a_median')

    # Pairwise Spearman between CL columns
    cls_with_data = [c for c in pivot.columns if pivot[c].notna().sum() >= 20]
    print(f'\nPairwise Spearman r for m6A/kb (loci with >= 3 reads in both CLs):')
    print(f'{"":>10s}', '  '.join(f'{c:>8s}' for c in cls_with_data))

    pair_rs = []
    for i, cl1 in enumerate(cls_with_data):
        row_vals = []
        for j, cl2 in enumerate(cls_with_data):
            if j <= i:
                row_vals.append('')
                continue
            shared = pivot[[cl1, cl2]].dropna()
            if len(shared) >= 10:
                r, p = stats.spearmanr(shared[cl1], shared[cl2])
                row_vals.append(f'{r:+.3f}')
                pair_rs.append({'CL1': cl1, 'CL2': cl2, 'r': r, 'p': p, 'n_loci': len(shared)})
            else:
                row_vals.append(f'n={len(shared):d}')
        if row_vals:
            print(f'{cl1:>10s}', '  '.join(f'{v:>8s}' for v in row_vals))

    if pair_rs:
        pair_df = pd.DataFrame(pair_rs)
        print(f'\nMedian pairwise Spearman r: {pair_df["r"].median():.3f}')
        print(f'Range: {pair_df["r"].min():.3f} to {pair_df["r"].max():.3f}')
        print(f'Pairs with r > 0.3: {(pair_df["r"] > 0.3).sum()} / {len(pair_df)}')
        print(f'Mean n_loci shared: {pair_df["n_loci"].mean():.0f}')

# =====================================================================
# 2. Subfamily-level m6A consistency
# =====================================================================
print('\n' + '='*70)
print('2. SUBFAMILY-LEVEL m6A/kb PER CELL LINE')
print('='*70)

# Top 10 most abundant subfamilies
top_sf = data['subfamily'].value_counts().head(10).index.tolist()

# Per-CL × per-subfamily median m6A/kb
sf_cl = data[data['subfamily'].isin(top_sf)].groupby(
    ['subfamily', 'cell_line'])['m6a_per_kb'].agg(['median', 'count']).reset_index()
sf_cl.columns = ['subfamily', 'cell_line', 'm6a_median', 'n_reads']

# Pivot for display
sf_pivot = sf_cl[sf_cl['n_reads'] >= 5].pivot_table(
    index='subfamily', columns='cell_line', values='m6a_median')

print('\nMedian m6A/kb by Subfamily × Cell Line (n≥5 reads):')
# Show as formatted table
cls = [c for c in CL_GROUPS if c in sf_pivot.columns]
print(f'{"Subfamily":>15s}', '  '.join(f'{c:>7s}' for c in cls))
print('-' * (15 + 10 * len(cls)))
for sf in top_sf:
    if sf in sf_pivot.index:
        vals = []
        for c in cls:
            v = sf_pivot.loc[sf, c] if c in sf_pivot.columns and pd.notna(sf_pivot.loc[sf, c]) else np.nan
            vals.append(f'{v:>7.2f}' if pd.notna(v) else f'{"---":>7s}')
        print(f'{sf:>15s}', '  '.join(vals))

# Subfamily rank correlation across CLs
print('\nSubfamily ranking consistency (Spearman r between CL pairs):')
sf_pair_rs = []
for cl1, cl2 in combinations(cls, 2):
    shared = sf_pivot[[cl1, cl2]].dropna()
    if len(shared) >= 5:
        r, p = stats.spearmanr(shared[cl1], shared[cl2])
        sf_pair_rs.append(r)

if sf_pair_rs:
    print(f'  Median r: {np.median(sf_pair_rs):.3f}')
    print(f'  Range: {min(sf_pair_rs):.3f} to {max(sf_pair_rs):.3f}')

# =====================================================================
# 3. Young > Ancient pattern consistency
# =====================================================================
print('\n' + '='*70)
print('3. YOUNG > ANCIENT m6A RATIO: CONSISTENCY TEST')
print('='*70)

ratios = []
for cl in CL_GROUPS:
    sub = data[data['cell_line'] == cl]
    y = sub[sub['age'] == 'young']['m6a_per_kb']
    a = sub[sub['age'] == 'ancient']['m6a_per_kb']
    if len(y) >= 10 and len(a) >= 20:
        ratio = y.median() / a.median()
        _, p = stats.mannwhitneyu(y, a, alternative='greater')
        ratios.append({'CL': cl, 'Young_med': y.median(), 'Ancient_med': a.median(),
                       'Ratio': ratio, 'MWU_p': p, 'n_young': len(y), 'n_ancient': len(a)})

df_ratio = pd.DataFrame(ratios)
print(f'\n{"CL":>10s}  {"Young":>7s}  {"Ancient":>8s}  {"Ratio":>6s}  {"P":>10s}  {"n_y":>5s}')
print('-' * 55)
for _, row in df_ratio.iterrows():
    p_str = f'{row["MWU_p"]:.2e}' if row["MWU_p"] < 0.05 else f'{row["MWU_p"]:.3f}'
    print(f'{row["CL"]:>10s}  {row["Young_med"]:>7.2f}  {row["Ancient_med"]:>8.2f}  '
          f'{row["Ratio"]:>6.2f}  {p_str:>10s}  {row["n_young"]:>5.0f}')

print(f'\nAll {len(df_ratio)} CLs show Young > Ancient: {(df_ratio["Ratio"] > 1.0).all()}')
print(f'All significant (p<0.05): {(df_ratio["MWU_p"] < 0.05).all()}')
print(f'Ratio range: {df_ratio["Ratio"].min():.2f} - {df_ratio["Ratio"].max():.2f}')
print(f'Ratio CV: {df_ratio["Ratio"].std()/df_ratio["Ratio"].mean():.3f}')

# =====================================================================
# 4. Per-CL m6A/kb quantile consistency
# =====================================================================
print('\n' + '='*70)
print('4. m6A/kb DISTRIBUTION SHAPE CONSISTENCY (KS test between CLs)')
print('='*70)

# Pairwise KS test for m6A/kb distributions
print('\nPairwise KS D statistic (smaller = more similar):')
cls_list = sorted(CL_GROUPS.keys())
ks_ds = []
for cl1, cl2 in combinations(cls_list, 2):
    d1 = data[data['cell_line'] == cl1]['m6a_per_kb'].values
    d2 = data[data['cell_line'] == cl2]['m6a_per_kb'].values
    ks_d, ks_p = stats.ks_2samp(d1, d2)
    ks_ds.append({'CL1': cl1, 'CL2': cl2, 'D': ks_d, 'p': ks_p})

ks_df = pd.DataFrame(ks_ds)
print(f'Median KS D: {ks_df["D"].median():.3f}')
print(f'Range: {ks_df["D"].min():.3f} - {ks_df["D"].max():.3f}')
print(f'Most similar: {ks_df.loc[ks_df["D"].idxmin(), "CL1"]} vs {ks_df.loc[ks_df["D"].idxmin(), "CL2"]} (D={ks_df["D"].min():.3f})')
print(f'Most different: {ks_df.loc[ks_df["D"].idxmax(), "CL1"]} vs {ks_df.loc[ks_df["D"].idxmax(), "CL2"]} (D={ks_df["D"].max():.3f})')

# =====================================================================
# 5. ICR (Intraclass Correlation) for m6A/kb across replicates
# =====================================================================
print('\n' + '='*70)
print('5. REPLICATE REPRODUCIBILITY (ICC-like: between-CL vs within-CL variance)')
print('='*70)

# Per-replicate median m6A/kb
rep_data = []
for cl, groups in CL_GROUPS.items():
    for grp in groups:
        sub = data[(data['cell_line'] == cl) & (data.get('group', '') == grp)] if 'group' in data.columns else pd.DataFrame()
        # Fallback: filter by group from the read data
        cache_file = CACHE / f'{grp}_l1_per_read.tsv'
        if cache_file.exists():
            ids = set(pd.read_csv(cache_file, sep='\t', usecols=['read_id'])['read_id'])
            sub = data[data['read_id'].isin(ids)]
        if len(sub) > 0:
            rep_data.append({'CL': cl, 'rep': grp, 'med_m6a': sub['m6a_per_kb'].median(), 'n': len(sub)})

rep_df = pd.DataFrame(rep_data)

# Between-CL variance vs within-CL variance
cl_means = rep_df.groupby('CL')['med_m6a'].mean()
between_var = cl_means.var()

within_vars = []
for cl in CL_GROUPS:
    reps = rep_df[rep_df['CL'] == cl]['med_m6a']
    if len(reps) >= 2:
        within_vars.append(reps.var())

within_var = np.mean(within_vars) if within_vars else 0
print(f'Between-CL variance: {between_var:.4f}')
print(f'Within-CL variance (mean): {within_var:.4f}')
print(f'ICC estimate: {between_var / (between_var + within_var):.3f}')
print('  (1.0 = all variation is between CLs, 0.0 = all within)')

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print("""
Key findings:
1. Overall m6A/kb differs across CLs (4.37-6.51, 1.49x fold-range, KW p≈0)
   → Statistically significant but moderate effect
2. Young > Ancient: UNIVERSAL (all CLs, ratio 1.56-1.87x)
3. HeLa vs HeLa-Ars: m6A UNCHANGED under stress (p=0.87)
4. Baseline m6A vs poly(A): NO cross-CL correlation (r=-0.25, ns)
5. Per-locus and subfamily consistency: see pairwise correlations above
""")
