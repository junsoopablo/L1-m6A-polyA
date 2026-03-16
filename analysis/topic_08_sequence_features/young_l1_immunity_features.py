#!/usr/bin/env python3
"""
Young L1 immunity features analysis.

What sequence-encoded features make young L1 immune to stress-induced
poly(A) shortening? Compare Young vs Ancient across multiple axes:
1. m6A/kb density
2. DRACH motif density on consensus
3. CpG density (ZAP evasion)
4. PAS retention
5. Structural completeness (full-length fraction)
6. Translational capacity (intact ORFs)

Then show: ancient L1 that retain young-like features are also immune.
"""
import pandas as pd
import numpy as np
from scipy import stats
import re

CACHE_DIR = 'topic_05_cellline/part3_l1_per_read_cache'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ── Load L1 reads with m6A ──
dfs = []
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    summ = pd.read_csv(f'../../results_group/{grp}/g_summary/{grp}_L1_summary.tsv', sep='\t')
    summ = summ[summ['qc_tag'] == 'PASS']
    cache = pd.read_csv(f'{CACHE_DIR}/{grp}_l1_per_read.tsv', sep='\t')
    cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
    merged = summ.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']], on='read_id', how='inner')
    merged['group'] = grp
    merged['is_stress'] = 1 if 'Ars' in grp else 0
    dfs.append(merged)
df = pd.concat(dfs, ignore_index=True)
df['subfamily'] = df['gene_id']
df['age'] = df['subfamily'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')

# ── Load rmsk ──
rmsk = pd.read_csv('topic_08_sequence_features/hg38_L1_rmsk_consensus.tsv', sep='\t')
def normalize(row):
    if row['strand'] == '+':
        return pd.Series({'cons_start': row['repStart'] + 1, 'cons_end': row['repEnd']})
    else:
        return pd.Series({'cons_start': row['repLeft'] + 1, 'cons_end': row['repEnd']})
rmsk[['cons_start', 'cons_end']] = rmsk.apply(normalize, axis=1)
rmsk['te_key'] = rmsk['genoName'] + ':' + rmsk['genoStart'].astype(str) + '-' + rmsk['genoEnd'].astype(str)
rmsk_slim = rmsk[['te_key', 'cons_start', 'cons_end']].drop_duplicates('te_key')

df['te_key'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)
df = df.merge(rmsk_slim, on='te_key', how='left')
df = df[df['cons_start'].notna()].copy()
df['cons_start'] = df['cons_start'].astype(int)
df['cons_end'] = df['cons_end'].astype(int)

# ── Domain assignment ──
FULL_DOMAINS = {
    '5UTR': (1, 908), 'ORF1': (909, 1990),
    'ORF2': (1991, 5817), '3UTR': (5818, 6064)
}

def get_n_domains(cs, ce):
    count = 0
    for name, (ds, de) in FULL_DOMAINS.items():
        overlap = max(0, min(ce, de) - max(cs, ds))
        if overlap > 50:
            count += 1
    return count

def en_coverage(cs, ce):
    return max(0, min(ce, 2708) - max(cs, 1991)) / (2708 - 1991 + 1)

df['n_domains'] = df.apply(lambda r: get_n_domains(r['cons_start'], r['cons_end']), axis=1)
df['is_full_length'] = df['n_domains'] == 4
df['en_cov'] = df.apply(lambda r: en_coverage(r['cons_start'], r['cons_end']), axis=1)
df['has_en'] = df['en_cov'] > 0.1
df['te_length'] = df['te_end'] - df['te_start']
df['consensus_span'] = df['cons_end'] - df['cons_start']

# ── PAS detection ──
# Check if read's transcript has PAS annotation (from L1_summary polya columns or separate)
# Use a simple proxy: does the L1 element's 3' end in consensus reach 3'UTR?
df['reaches_3utr'] = df['cons_end'] >= 5818

# ── Feature comparison: Young vs Ancient ──
print("="*70)
print("Feature 1: Young vs Ancient L1 — Immunity Feature Comparison")
print("="*70)

young = df[df['age'] == 'Young']
ancient = df[df['age'] == 'Ancient']

features = {
    'm6A/kb (mean)': (young['m6a_per_kb'].mean(), ancient['m6a_per_kb'].mean()),
    'm6A/kb (median)': (young['m6a_per_kb'].median(), ancient['m6a_per_kb'].median()),
    'Full-length (%)': (young['is_full_length'].mean()*100, ancient['is_full_length'].mean()*100),
    'Has EN domain (%)': (young['has_en'].mean()*100, ancient['has_en'].mean()*100),
    'Consensus span (bp)': (young['consensus_span'].median(), ancient['consensus_span'].median()),
    'Reaches 3\'UTR (%)': (young['reaches_3utr'].mean()*100, ancient['reaches_3utr'].mean()*100),
    'N domains (mean)': (young['n_domains'].mean(), ancient['n_domains'].mean()),
}

print(f"\n{'Feature':<25} {'Young':>10} {'Ancient':>10} {'Ratio':>8}")
print("-"*60)
for feat, (yval, aval) in features.items():
    ratio = yval / aval if aval > 0 else np.inf
    print(f"  {feat:<25} {yval:>10.1f} {aval:>10.1f} {ratio:>8.2f}x")

# ── Stress response ──
print(f"\n{'='*70}")
print("Feature 2: Stress poly(A) delta")
print("="*70)

for age in ['Young', 'Ancient']:
    sub = df[df['age'] == age]
    un = sub[sub['is_stress'] == 0]['polya_length']
    st = sub[sub['is_stress'] == 1]['polya_length']
    delta = st.median() - un.median()
    _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
    sig = '***' if p < 0.001 else 'ns'
    print(f"  {age:<10} N={len(un):>5}/{len(st):>5}  un={un.median():>7.1f}  st={st.median():>7.1f}  Δ={delta:>+7.1f}  P={p:.2e} {sig}")

# ── Ancient L1 with young-like features ──
print(f"\n{'='*70}")
print("Feature 3: Ancient L1 with young-like features → stress immunity?")
print("="*70)

anc = df[df['age'] == 'Ancient'].copy()

# Feature groups
feature_groups = [
    ('Full-length (4 domains)', anc['is_full_length']),
    ('Has EN domain', anc['has_en']),
    ('High m6A (≥ Young median)', anc['m6a_per_kb'] >= young['m6a_per_kb'].median()),
    ('Long consensus span (≥ Young median)', anc['consensus_span'] >= young['consensus_span'].median()),
    ('Reaches 3\'UTR', anc['reaches_3utr']),
]

print(f"\n{'Feature':<35} {'N_un':>6} {'N_st':>6} {'Δ_un→st':>8} {'P':>12} {'Sig':>4}")
print("-"*75)

immunity_results = []
for label, mask in feature_groups:
    # With feature
    sub_with = anc[mask]
    un_w = sub_with[sub_with['is_stress'] == 0]['polya_length']
    st_w = sub_with[sub_with['is_stress'] == 1]['polya_length']
    if len(un_w) >= 20 and len(st_w) >= 20:
        delta_w = st_w.median() - un_w.median()
        _, p_w = stats.mannwhitneyu(st_w, un_w, alternative='two-sided')
        sig_w = '***' if p_w < 0.001 else '**' if p_w < 0.01 else '*' if p_w < 0.05 else 'ns'
        print(f"  + {label:<33} {len(un_w):>6} {len(st_w):>6} {delta_w:>+8.1f} {p_w:>12.2e} {sig_w}")
        immunity_results.append({
            'feature': label, 'has_feature': True,
            'n_un': len(un_w), 'n_st': len(st_w),
            'delta': delta_w, 'p': p_w
        })

    # Without feature
    sub_without = anc[~mask]
    un_wo = sub_without[sub_without['is_stress'] == 0]['polya_length']
    st_wo = sub_without[sub_without['is_stress'] == 1]['polya_length']
    if len(un_wo) >= 20 and len(st_wo) >= 20:
        delta_wo = st_wo.median() - un_wo.median()
        _, p_wo = stats.mannwhitneyu(st_wo, un_wo, alternative='two-sided')
        sig_wo = '***' if p_wo < 0.001 else '**' if p_wo < 0.01 else '*' if p_wo < 0.05 else 'ns'
        print(f"  - {label:<33} {len(un_wo):>6} {len(st_wo):>6} {delta_wo:>+8.1f} {p_wo:>12.2e} {sig_wo}")
        immunity_results.append({
            'feature': label, 'has_feature': False,
            'n_un': len(un_wo), 'n_st': len(st_wo),
            'delta': delta_wo, 'p': p_wo
        })

# ── Composite immunity score ──
print(f"\n{'='*70}")
print("Feature 4: Composite immunity score")
print("="*70)

# Score: +1 for each young-like feature
anc_st = anc[anc['is_stress'] == 1].copy()
anc_un = anc[anc['is_stress'] == 0].copy()

for subset, label in [(anc_st, 'Stressed'), (anc_un, 'Unstressed')]:
    subset = subset.copy()
    subset['immunity_score'] = (
        subset['is_full_length'].astype(int) +
        subset['has_en'].astype(int) +
        (subset['m6a_per_kb'] >= young['m6a_per_kb'].median()).astype(int) +
        subset['reaches_3utr'].astype(int)
    )
    print(f"\n  {label}:")
    for score in sorted(subset['immunity_score'].unique()):
        sub = subset[subset['immunity_score'] == score]
        print(f"    Score {score}: N={len(sub):>5}  median poly(A)={sub['polya_length'].median():>7.1f}")

    rho, p = stats.spearmanr(subset['immunity_score'], subset['polya_length'])
    print(f"    Spearman rho={rho:.3f}  P={p:.2e}")

# ── Young L1 subfamily gradient ──
print(f"\n{'='*70}")
print("Feature 5: Within young L1 — is there a gradient?")
print("="*70)

for sf in ['L1HS', 'L1PA1', 'L1PA2', 'L1PA3']:
    sub = df[df['subfamily'] == sf]
    un = sub[sub['is_stress'] == 0]['polya_length']
    st = sub[sub['is_stress'] == 1]['polya_length']
    if len(un) >= 5 and len(st) >= 5:
        delta = st.median() - un.median()
        m6a_mean = sub['m6a_per_kb'].mean()
        print(f"  {sf:<8} N={len(un):>4}/{len(st):>4}  Δ={delta:>+7.1f}  m6A/kb={m6a_mean:.2f}")

# Save
pd.DataFrame(immunity_results).to_csv(
    'topic_08_sequence_features/young_like_immunity_results.tsv', sep='\t', index=False
)
print("\nSaved: young_like_immunity_results.tsv")
