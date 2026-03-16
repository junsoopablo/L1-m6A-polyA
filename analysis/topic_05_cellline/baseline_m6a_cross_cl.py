#!/usr/bin/env python3
"""
Cross-cell-line baseline m6A/kb on L1.
Question: do cell lines differ in L1 m6A density?
If m6A protects poly(A) under stress, baseline m6A level could predict arsenite sensitivity.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CACHE = PROJECT / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
SUMMARY_DIR = PROJECT / 'results_group'

# Cell line grouping
CL_GROUPS = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

# Load all Part3 cache + L1 summary (for poly(A), subfamily)
all_reads = []

for cl, groups in CL_GROUPS.items():
    for grp in groups:
        # Part3 cache (m6A/kb)
        cache_file = CACHE / f'{grp}_l1_per_read.tsv'
        if not cache_file.exists():
            print(f'  Missing: {cache_file}')
            continue
        df = pd.read_csv(cache_file, sep='\t')
        df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)
        df['psi_per_kb'] = df['psi_sites_high'] / (df['read_length'] / 1000)

        # L1 summary (for poly(A))
        summary_file = SUMMARY_DIR / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
        if summary_file.exists():
            summ = pd.read_csv(summary_file, sep='\t')
            df = df.merge(summ[['read_id', 'polya_length', 'transcript_id']],
                          on='read_id', how='left')
        else:
            df['polya_length'] = np.nan
            df['transcript_id'] = ''

        df['cell_line'] = cl
        df['group'] = grp
        all_reads.append(df)

data = pd.concat(all_reads, ignore_index=True)
print(f'Total reads: {len(data)}')
print(f'Cell lines: {data["cell_line"].nunique()}')

# Classify young vs ancient
young_prefixes = ('L1HS', 'L1PA1', 'L1PA2', 'L1PA3')
data['is_young'] = data['transcript_id'].str.split('_dup').str[0].isin(
    [p for p in young_prefixes]) if 'transcript_id' in data.columns else False

# Better: parse from transcript_id
def classify_age(tid):
    if pd.isna(tid):
        return 'unknown'
    base = tid.split('_dup')[0] if '_dup' in str(tid) else str(tid)
    if base in ('L1HS', 'L1PA1', 'L1PA2', 'L1PA3'):
        return 'young'
    return 'ancient'

data['age'] = data['transcript_id'].apply(classify_age)

# =====================================================================
# 1. Per-cell-line baseline m6A/kb
# =====================================================================
print('\n' + '='*70)
print('BASELINE m6A/kb PER CELL LINE (excluding HeLa-Ars)')
print('='*70)

normal_cls = [cl for cl in CL_GROUPS if cl != 'HeLa-Ars']

cl_stats = []
for cl in normal_cls:
    sub = data[(data['cell_line'] == cl)]
    m6a = sub['m6a_per_kb']
    polya = sub['polya_length'].dropna()

    cl_stats.append({
        'cell_line': cl,
        'n': len(sub),
        'm6a_kb_median': m6a.median(),
        'm6a_kb_mean': m6a.mean(),
        'm6a_kb_std': m6a.std(),
        'polya_median': polya.median() if len(polya) > 0 else np.nan,
    })

df_stats = pd.DataFrame(cl_stats).sort_values('m6a_kb_median', ascending=False)

print(f'\n{"Cell Line":>12s}  {"n":>6s}  {"m6A/kb med":>10s}  {"m6A/kb mean":>11s}  {"SD":>6s}  {"polyA med":>9s}')
print('-' * 65)
for _, row in df_stats.iterrows():
    print(f'{row["cell_line"]:>12s}  {row["n"]:>6.0f}  {row["m6a_kb_median"]:>10.2f}  '
          f'{row["m6a_kb_mean"]:>11.2f}  {row["m6a_kb_std"]:>6.2f}  {row["polya_median"]:>9.1f}')

# Range and variability
m6a_meds = df_stats['m6a_kb_median'].values
print(f'\nm6A/kb median range: {m6a_meds.min():.2f} – {m6a_meds.max():.2f}')
print(f'm6A/kb fold-range: {m6a_meds.max()/m6a_meds.min():.2f}x')
print(f'm6A/kb CV across CLs: {np.std(m6a_meds)/np.mean(m6a_meds):.3f}')

# Kruskal-Wallis
groups_m6a = [data[data['cell_line'] == cl]['m6a_per_kb'].values for cl in normal_cls]
kw_stat, kw_p = stats.kruskal(*groups_m6a)
print(f'Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.2e}')

# =====================================================================
# 2. Per-replicate m6A/kb (for reproducibility)
# =====================================================================
print('\n' + '='*70)
print('PER-REPLICATE m6A/kb MEDIAN')
print('='*70)

for cl in normal_cls:
    grps = CL_GROUPS[cl]
    meds = []
    for grp in grps:
        sub = data[(data['cell_line'] == cl) & (data['group'] == grp)]
        meds.append(sub['m6a_per_kb'].median())
    if len(meds) >= 2:
        cv = np.std(meds) / np.mean(meds) if np.mean(meds) > 0 else 0
        print(f'  {cl:>10s}: {" ".join(f"{m:.2f}" for m in meds)}  (CV={cv:.3f})')
    else:
        print(f'  {cl:>10s}: {meds[0]:.2f}  (1 replicate)')

# =====================================================================
# 3. Young vs Ancient m6A/kb per cell line
# =====================================================================
print('\n' + '='*70)
print('YOUNG vs ANCIENT m6A/kb PER CELL LINE')
print('='*70)

print(f'\n{"Cell Line":>12s}  {"Young m6A/kb":>12s}  {"Ancient m6A/kb":>14s}  {"Ratio":>6s}  {"Young n":>8s}')
print('-' * 60)
for cl in normal_cls:
    sub = data[data['cell_line'] == cl]
    young = sub[sub['age'] == 'young']['m6a_per_kb']
    ancient = sub[sub['age'] == 'ancient']['m6a_per_kb']
    if len(young) >= 5 and len(ancient) >= 5:
        ratio = young.median() / ancient.median() if ancient.median() > 0 else np.nan
        print(f'{cl:>12s}  {young.median():>12.2f}  {ancient.median():>14.2f}  '
              f'{ratio:>6.2f}  {len(young):>8d}')
    else:
        print(f'{cl:>12s}  n_young={len(young)}, n_ancient={len(ancient)} (too few)')

# =====================================================================
# 4. HeLa vs HeLa-Ars: m6A unchanged under stress?
# =====================================================================
print('\n' + '='*70)
print('HeLa vs HeLa-Ars: m6A/kb COMPARISON')
print('='*70)

hela = data[data['cell_line'] == 'HeLa']['m6a_per_kb']
ars = data[data['cell_line'] == 'HeLa-Ars']['m6a_per_kb']
_, p = stats.mannwhitneyu(hela, ars, alternative='two-sided')
print(f'HeLa median m6A/kb: {hela.median():.2f} (n={len(hela)})')
print(f'HeLa-Ars median m6A/kb: {ars.median():.2f} (n={len(ars)})')
print(f'Δ = {ars.median() - hela.median():.2f}, MWU p = {p:.4f}')
print(f'KS test: {stats.ks_2samp(hela, ars)}')

# Per-replicate
print('\nPer-replicate:')
for cl in ['HeLa', 'HeLa-Ars']:
    for grp in CL_GROUPS[cl]:
        sub = data[(data['cell_line'] == cl) & (data['group'] == grp)]
        print(f'  {grp}: median={sub["m6a_per_kb"].median():.2f}, n={len(sub)}')

# =====================================================================
# 5. m6A/kb vs poly(A) baseline correlation across cell lines
# =====================================================================
print('\n' + '='*70)
print('CROSS-CL: BASELINE m6A/kb vs BASELINE poly(A) CORRELATION')
print('='*70)

# Per-CL medians
cl_m6a = []
cl_polya = []
cl_names = []
for cl in normal_cls:
    sub = data[data['cell_line'] == cl]
    polya = sub['polya_length'].dropna()
    if len(polya) > 50:
        cl_m6a.append(sub['m6a_per_kb'].median())
        cl_polya.append(polya.median())
        cl_names.append(cl)

r, p = stats.spearmanr(cl_m6a, cl_polya)
print(f'Spearman r = {r:.3f}, p = {p:.4f} (n={len(cl_names)} cell lines)')
for name, m, pa in zip(cl_names, cl_m6a, cl_polya):
    print(f'  {name:>10s}: m6A/kb={m:.2f}, poly(A)={pa:.1f}')

# =====================================================================
# 6. Read-length controlled check
# =====================================================================
print('\n' + '='*70)
print('READ-LENGTH CONTROLLED m6A/kb (500-2000bp reads only)')
print('='*70)

rl_mask = (data['read_length'] >= 500) & (data['read_length'] <= 2000)
data_rl = data[rl_mask & (~data['cell_line'].isin(['HeLa-Ars']))]

print(f'\n{"Cell Line":>12s}  {"n":>6s}  {"m6A/kb med":>10s}  {"RL median":>10s}')
print('-' * 45)
for cl in normal_cls:
    sub = data_rl[data_rl['cell_line'] == cl]
    if len(sub) > 20:
        print(f'{cl:>12s}  {len(sub):>6d}  {sub["m6a_per_kb"].median():>10.2f}  '
              f'{sub["read_length"].median():>10.0f}')

# KW on RL-controlled
groups_rl = [data_rl[data_rl['cell_line'] == cl]['m6a_per_kb'].values
             for cl in normal_cls if len(data_rl[data_rl['cell_line'] == cl]) > 20]
if len(groups_rl) >= 3:
    kw_stat, kw_p = stats.kruskal(*groups_rl)
    print(f'\nKruskal-Wallis (RL-controlled): H={kw_stat:.1f}, p={kw_p:.2e}')

print('\nDone!')
