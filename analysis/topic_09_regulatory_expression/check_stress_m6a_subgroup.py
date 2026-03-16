#!/usr/bin/env python3
"""
Check: does L1 m6A level change under arsenite for specific subgroups?
- Young vs Ancient
- Regulatory vs non-regulatory (ChromHMM)
- By chromatin state
- High vs low body overlap
"""
import pandas as pd
import numpy as np
from scipy import stats
import glob, os

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/'
TOPIC = 'analysis/01_exploration/topic_05_cellline/'

# --- 1. Load L1 Part3 cache ---
l1_files = glob.glob(BASE + TOPIC + 'part3_l1_per_read_cache/*.tsv')
l1_dfs = []
for f in l1_files:
    df = pd.read_csv(f, sep='\t')
    group = os.path.basename(f).replace('_l1_per_read.tsv', '')
    df['group'] = group
    l1_dfs.append(df)
l1 = pd.concat(l1_dfs, ignore_index=True)
l1['m6a_kb'] = l1['m6a_sites_high'] / l1['read_length'] * 1000

# --- 2. Load L1 summary for subfamily info ---
YOUNG = ['L1HS', 'L1PA1', 'L1PA2', 'L1PA3']
sum_files = glob.glob(BASE + 'results_group/*/g_summary/*_L1_summary.tsv')
sum_dfs = []
for f in sum_files:
    df = pd.read_csv(f, sep='\t')
    sum_dfs.append(df)
l1_sum = pd.concat(sum_dfs, ignore_index=True)
# transcript_id = e.g. "L1MC4_dup15" -> subfamily = "L1MC4"
l1_sum['subfamily'] = l1_sum['transcript_id'].str.replace(r'_dup\d+$', '', regex=True)
l1_sum['is_young'] = l1_sum['subfamily'].isin(YOUNG)
l1_sum['gene_context'] = l1_sum['TE_group']
# body overlap fraction
l1_sum['body_overlap_frac'] = l1_sum['overlap_length'] / l1_sum['read_length']

# Merge: get subfamily & age
l1 = l1.merge(l1_sum[['read_id', 'subfamily', 'is_young', 'polya_length',
                        'gene_context', 'body_overlap_frac']].drop_duplicates('read_id'),
               on='read_id', how='left')

# --- 3. Load ChromHMM annotation ---
chrom_path = BASE + 'analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
if os.path.exists(chrom_path):
    chrom = pd.read_csv(chrom_path, sep='\t')
    l1 = l1.merge(chrom[['read_id', 'chromhmm_group']].drop_duplicates('read_id'),
                   on='read_id', how='left')
    has_chrom = True
else:
    has_chrom = False

# --- 4. Define conditions ---
hela_normal = ['HeLa_1', 'HeLa_2', 'HeLa_3']
hela_ars = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

l1['condition'] = 'other'
l1.loc[l1['group'].isin(hela_normal), 'condition'] = 'normal'
l1.loc[l1['group'].isin(hela_ars), 'condition'] = 'stress'

hela = l1[l1['condition'].isin(['normal', 'stress'])].copy()

def compare(sub, label):
    """Compare m6A/kb between normal and stress for a subset."""
    n = sub[sub['condition'] == 'normal']
    s = sub[sub['condition'] == 'stress']
    if len(n) < 10 or len(s) < 10:
        print(f"  {label:30s}: n_norm={len(n):5d}, n_stress={len(s):5d}  *** TOO FEW ***")
        return
    ratio = s['m6a_kb'].mean() / n['m6a_kb'].mean() if n['m6a_kb'].mean() > 0 else np.nan
    u, p = stats.mannwhitneyu(n['m6a_kb'], s['m6a_kb'], alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {label:30s}: norm={n['m6a_kb'].mean():.2f} (n={len(n):4d})  "
          f"stress={s['m6a_kb'].mean():.2f} (n={len(s):4d})  "
          f"ratio={ratio:.3f}  P={p:.2e} {sig}")

# --- 5. Young vs Ancient ---
print("=" * 90)
print("1. YOUNG vs ANCIENT L1: m6A/kb change under stress")
print("=" * 90)
compare(hela, "All L1")
compare(hela[hela['is_young'] == True], "Young L1")
compare(hela[hela['is_young'] == False], "Ancient L1")

# --- 6. By gene context ---
print(f"\n{'=' * 90}")
print("2. BY GENE CONTEXT: m6A/kb change under stress")
print("=" * 90)
for ctx in ['intronic', 'intergenic']:
    sub = hela[hela['gene_context'] == ctx]
    compare(sub, f"{ctx}")
    compare(sub[sub['is_young'] == False], f"  ancient × {ctx}")

# --- 7. By chromatin state ---
if has_chrom:
    print(f"\n{'=' * 90}")
    print("3. BY CHROMATIN STATE: m6A/kb change under stress")
    print("=" * 90)
    for state in ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent', 'Heterochromatin']:
        sub = hela[hela['chromhmm_group'] == state]
        compare(sub, state)

    # Regulatory vs non-regulatory
    hela['is_regulatory'] = hela['chromhmm_group'].isin(['Enhancer', 'Promoter'])
    print()
    compare(hela[hela['is_regulatory'] == True], "Regulatory (Enh+Prom)")
    compare(hela[hela['is_regulatory'] == False], "Non-regulatory")
    compare(hela[hela['chromhmm_group'] == 'Heterochromatin'], "Heterochromatin")

# --- 8. By body overlap fraction ---
print(f"\n{'=' * 90}")
print("4. BY BODY OVERLAP: m6A/kb change under stress")
print("=" * 90)
if 'body_overlap_frac' in hela.columns:
    for lo, hi, label in [(0, 0.3, '<0.30'), (0.3, 0.6, '0.30-0.60'),
                           (0.6, 0.8, '0.60-0.80'), (0.8, 1.01, '>=0.80')]:
        sub = hela[(hela['body_overlap_frac'] >= lo) & (hela['body_overlap_frac'] < hi)]
        compare(sub, f"body_frac {label}")

# --- 9. Top subfamilies ---
print(f"\n{'=' * 90}")
print("5. TOP SUBFAMILIES: m6A/kb change under stress")
print("=" * 90)
top_sf = hela['subfamily'].value_counts().head(15).index
for sf in top_sf:
    sub = hela[hela['subfamily'] == sf]
    compare(sub, sf)

# --- 10. m6A/kb distribution comparison (KS test per subgroup) ---
print(f"\n{'=' * 90}")
print("6. DISTRIBUTION SHIFTS (KS test)")
print("=" * 90)
for label, sub in [("All", hela),
                    ("Young", hela[hela['is_young']==True]),
                    ("Ancient", hela[hela['is_young']==False])]:
    n = sub[sub['condition']=='normal']['m6a_kb']
    s = sub[sub['condition']=='stress']['m6a_kb']
    if len(n) > 10 and len(s) > 10:
        ks, p = stats.ks_2samp(n, s)
        print(f"  {label:20s}: KS D={ks:.4f}, P={p:.2e}")
