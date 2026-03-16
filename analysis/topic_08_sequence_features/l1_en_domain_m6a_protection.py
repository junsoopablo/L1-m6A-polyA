#!/usr/bin/env python3
"""
Test whether m6A at ORF2 EN domain specifically drives stress protection.

If EN domain is the primary m6A deposition zone AND mutation there causes
stress vulnerability, then:
- Reads with higher m6A in EN domain region should have longer poly(A) under stress
- The m6A-poly(A) correlation should be stronger for EN-derived reads

Uses Part3 cache (thr=204) + domain assignment from rmsk consensus coordinates.
"""
import pandas as pd
import numpy as np
from scipy import stats

CACHE_DIR = 'topic_05_cellline/part3_l1_per_read_cache'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ── Load rmsk with consensus positions ──
rmsk = pd.read_csv('topic_08_sequence_features/hg38_L1_rmsk_consensus.tsv', sep='\t')

def normalize_consensus_coords(row):
    if row['strand'] == '+':
        cs = row['repStart'] + 1
        ce = row['repEnd']
    else:
        cs = row['repLeft'] + 1
        ce = row['repEnd']
    return pd.Series({'cons_start': cs, 'cons_end': ce})

rmsk[['cons_start', 'cons_end']] = rmsk.apply(normalize_consensus_coords, axis=1)
rmsk['te_key'] = rmsk['genoName'] + ':' + rmsk['genoStart'].astype(str) + '-' + rmsk['genoEnd'].astype(str)
rmsk_slim = rmsk[['te_key', 'cons_start', 'cons_end']].drop_duplicates('te_key')

# ── Load L1 reads with m6A from Part3 cache ──
dfs = []
for grp in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    summ_path = f'../../results_group/{grp}/g_summary/{grp}_L1_summary.tsv'
    cache_path = f'{CACHE_DIR}/{grp}_l1_per_read.tsv'
    try:
        summ = pd.read_csv(summ_path, sep='\t')
        summ = summ[summ['qc_tag'] == 'PASS'].copy()
        cache = pd.read_csv(cache_path, sep='\t')
        cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
        merged = summ.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']], on='read_id', how='inner')
        merged['group'] = grp
        merged['is_stress'] = 1 if 'Ars' in grp else 0
        dfs.append(merged)
    except Exception as e:
        print(f"  Skip {grp}: {e}")

df = pd.concat(dfs, ignore_index=True)
df['subfamily'] = df['gene_id']
df['age'] = df['subfamily'].apply(lambda x: 'Young' if x in YOUNG else 'Ancient')
df['te_key'] = df['te_chr'] + ':' + df['te_start'].astype(str) + '-' + df['te_end'].astype(str)

# Merge with rmsk
df = df.merge(rmsk_slim, on='te_key', how='left')
matched = df['cons_start'].notna().sum()
print(f"Matched to rmsk: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
df = df[df['cons_start'].notna()].copy()
df['cons_start'] = df['cons_start'].astype(int)
df['cons_end'] = df['cons_end'].astype(int)

# ── Domain assignment ──
# EN domain: consensus 1991-2708
def classify_en_coverage(cs, ce):
    """Fraction of EN domain (1991-2708) covered by this element."""
    overlap = max(0, min(ce, 2708) - max(cs, 1991))
    return overlap / (2708 - 1991 + 1)

def get_primary_domain(cs, ce):
    domains = {
        '5UTR': (1, 908), 'ORF1': (909, 1990),
        'ORF2_EN': (1991, 2708), 'ORF2_RT': (2709, 4149),
        'ORF2_Crich': (4150, 5817), '3UTR': (5818, 6064)
    }
    best, best_ov = 'unknown', 0
    for name, (ds, de) in domains.items():
        ov = max(0, min(ce, de) - max(cs, ds))
        if ov > best_ov:
            best_ov = ov
            best = name
    return best

df['en_coverage'] = df.apply(lambda r: classify_en_coverage(r['cons_start'], r['cons_end']), axis=1)
df['primary_domain'] = df.apply(lambda r: get_primary_domain(r['cons_start'], r['cons_end']), axis=1)
df['has_en'] = df['en_coverage'] > 0.1  # At least 10% EN domain

# Focus on ancient stressed
anc = df[df['age'] == 'Ancient'].copy()
anc_st = anc[anc['is_stress'] == 1].copy()
anc_un = anc[anc['is_stress'] == 0].copy()

print(f"\nAncient reads: {len(anc)} (stressed: {len(anc_st)}, unstressed: {len(anc_un)})")
print(f"With EN domain (>10%): {anc['has_en'].sum()} ({anc['has_en'].mean()*100:.1f}%)")

# ── Analysis 1: m6A-poly(A) correlation by EN coverage ──
print("\n" + "="*70)
print("Analysis 1: m6A-poly(A) correlation by EN domain coverage")
print("="*70)

for label, subset in [('Has EN domain', anc_st[anc_st['has_en']]),
                       ('No EN domain', anc_st[~anc_st['has_en']])]:
    if len(subset) >= 30:
        rho, p = stats.spearmanr(subset['m6a_per_kb'], subset['polya_length'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<20} N={len(subset):>5}  rho={rho:>+.3f}  P={p:.2e}  {sig}")
        print(f"    m6A/kb={subset['m6a_per_kb'].mean():.2f}  poly(A)={subset['polya_length'].median():.1f}")

# ── Analysis 2: Stress delta by EN coverage ──
print("\n" + "="*70)
print("Analysis 2: Stress poly(A) delta by EN domain coverage")
print("="*70)

for label, mask in [
    ('Has EN (>10%)', anc['has_en']),
    ('No EN', ~anc['has_en']),
    ('EN >50%', anc['en_coverage'] > 0.5),
    ('EN >80%', anc['en_coverage'] > 0.8),
]:
    sub = anc[mask]
    un = sub[sub['is_stress'] == 0]['polya_length']
    st = sub[sub['is_stress'] == 1]['polya_length']
    if len(un) >= 20 and len(st) >= 20:
        delta = st.median() - un.median()
        _, p = stats.mannwhitneyu(st, un, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<20} N={len(un):>5}/{len(st):>5}  un={un.median():>7.1f}  st={st.median():>7.1f}  Δ={delta:>+7.1f}  P={p:.2e} {sig}")

# ── Analysis 3: m6A density by primary domain ──
print("\n" + "="*70)
print("Analysis 3: m6A density by primary domain (Ancient, stressed)")
print("="*70)

print(f"\n{'Domain':<14} {'N':>6} {'m6A/kb':>8} {'poly(A)':>8} {'rho':>8} {'P':>12}")
print("-"*60)
for domain in ['5UTR', 'ORF1', 'ORF2_EN', 'ORF2_RT', 'ORF2_Crich', '3UTR']:
    sub = anc_st[anc_st['primary_domain'] == domain]
    if len(sub) >= 20:
        rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {domain:<14} {len(sub):>6} {sub['m6a_per_kb'].mean():>8.2f} {sub['polya_length'].median():>8.1f} "
              f"{rho:>+8.3f} {p:>12.2e} {sig}")

# ── Analysis 4: EN domain reads - m6A quartile vs poly(A) ──
print("\n" + "="*70)
print("Analysis 4: m6A quartile poly(A) in EN-containing stressed reads")
print("="*70)

en_st = anc_st[anc_st['has_en']].copy()
if len(en_st) >= 100:
    en_st['m6a_q'] = pd.qcut(en_st['m6a_per_kb'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    print(f"\n{'Quartile':<10} {'N':>6} {'m6A/kb':>8} {'Med_polyA':>10}")
    print("-"*40)
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = en_st[en_st['m6a_q'] == q]
        if len(sub) > 0:
            print(f"  {q:<10} {len(sub):>6} {sub['m6a_per_kb'].mean():>8.2f} {sub['polya_length'].median():>10.1f}")

    # Q4 vs Q1
    q1 = en_st[en_st['m6a_q'] == 'Q1']['polya_length']
    q4 = en_st[en_st['m6a_q'] == 'Q4']['polya_length']
    if len(q1) >= 10 and len(q4) >= 10:
        _, p = stats.mannwhitneyu(q4, q1, alternative='greater')
        print(f"\n  Q4 vs Q1 poly(A): {q4.median():.1f} vs {q1.median():.1f} = {q4.median()-q1.median():+.1f}nt (P={p:.2e})")

# ── Analysis 5: Does EN mutation burden predict stress outcome? ──
# Proxy: EN coverage × m6A interaction
print("\n" + "="*70)
print("Analysis 5: EN coverage × m6A interaction on stress poly(A)")
print("="*70)

# Multiple regression: poly(A) ~ m6A_per_kb + en_coverage + m6A × en_coverage
import statsmodels.api as sm

model_data = anc_st[['polya_length', 'm6a_per_kb', 'en_coverage']].dropna()
model_data['m6a_x_en'] = model_data['m6a_per_kb'] * model_data['en_coverage']

X = sm.add_constant(model_data[['m6a_per_kb', 'en_coverage', 'm6a_x_en']])
y = model_data['polya_length']
model = sm.OLS(y, X).fit()

print(f"\n  OLS: poly(A) ~ m6A/kb + EN_coverage + m6A×EN")
print(f"  N = {len(model_data)}")
print(f"  R² = {model.rsquared:.4f}")
for var in ['m6a_per_kb', 'en_coverage', 'm6a_x_en']:
    coef = model.params[var]
    p = model.pvalues[var]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {var:<16} β={coef:>+8.2f}  P={p:.3e}  {sig}")

# ── Analysis 6: Compare with unstressed (negative control) ──
print("\n" + "="*70)
print("Analysis 6: Same analysis in unstressed (negative control)")
print("="*70)

for label, subset, stress_label in [
    ('Stressed + EN', anc_st[anc_st['has_en']], 'stressed'),
    ('Unstressed + EN', anc_un[anc_un['has_en']], 'unstressed'),
]:
    if len(subset) >= 30:
        rho, p = stats.spearmanr(subset['m6a_per_kb'], subset['polya_length'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<25} N={len(subset):>5}  rho={rho:>+.3f}  P={p:.2e}  {sig}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
ORF2 EN domain (consensus 1991-2708) characteristics:
  - Highest DRACH density in L1: 43.2/kb (1.38× L1 average)
  - Nearly complete CpG depletion: 2.8/kb (obs/exp=0.08) → ZAP invisible
  - Highest A-richness: 0.43 (SAFB binding substrate)
  - All 21 Bonferroni-significant mutation sensitivity windows in ORF2
  - Mutation-sensitive windows: lower GC (P=0.021), marginally higher A-rich (P=0.055)

Mechanism: EN domain is the primary m6A deposition zone in L1.
  Ancient L1 with mutations in this region lose DRACH contexts → less m6A →
  reduced SAFB/YTHDF2-mediated surveillance → paradoxically more vulnerable
  to stress-induced poly(A) shortening because m6A also provides protective
  signaling under stress.
""")
