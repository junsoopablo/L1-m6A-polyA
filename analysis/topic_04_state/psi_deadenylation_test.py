#!/usr/bin/env python3
"""
Direct test: Do psi+ L1 reads show more poly(A) shortening under arsenite
than psi- L1 reads?

Approach:
  1. Compare poly(A) length of psi+ vs psi- reads within each condition
  2. Compare arsenite Δpoly(A) for psi+ vs psi- subgroups
  3. Test interaction: is the arsenite effect larger in psi+ reads?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
ANALYSIS = PROJECT / 'analysis/01_exploration'

# =========================================================================
# 1. Load coupled data
# =========================================================================
print("Loading coupled data...")
df = pd.read_csv(ANALYSIS / 'topic_02_polya/polya_modification_coupled.tsv', sep='\t')

l1 = df[df['l1_age'].isin(['young', 'ancient'])].copy()
l1['has_psi'] = l1['psi_sites_high'] > 0
l1['has_m6a'] = l1['m6a_sites_high'] > 0
l1['condition'] = l1['source'].map({'HeLa_L1': 'HeLa', 'HeLa-Ars_L1': 'HeLa-Ars'})

ctrl = df[df['source'].isin(['HeLa_Ctrl', 'HeLa-Ars_Ctrl'])].copy()
ctrl['has_psi'] = ctrl['psi_sites_high'] > 0
ctrl['has_m6a'] = ctrl['m6a_sites_high'] > 0
ctrl['condition'] = ctrl['source'].map({'HeLa_Ctrl': 'HeLa', 'HeLa-Ars_Ctrl': 'HeLa-Ars'})

# =========================================================================
# 2. Poly(A) by psi status x condition
# =========================================================================
print(f"\n{'='*90}")
print("Poly(A) Length by Psi Status x Arsenite Condition")
print(f"{'='*90}")

print(f"\n{'Group':<20} {'Psi':<6} {'N':>6} {'Median':>8} {'Mean':>8} {'Q25':>6} {'Q75':>6}")
print("-" * 62)

subgroups = {}
for cond in ['HeLa', 'HeLa-Ars']:
    for psi_label, psi_val in [('psi+', True), ('psi-', False)]:
        for age in ['all', 'young', 'ancient']:
            if age == 'all':
                subset = l1[(l1['condition'] == cond) & (l1['has_psi'] == psi_val)]
            else:
                subset = l1[(l1['condition'] == cond) & (l1['has_psi'] == psi_val) & (l1['l1_age'] == age)]
            key = (cond, psi_label, age)
            subgroups[key] = subset
            if age == 'all':
                print(f"{cond+' L1':<20} {psi_label:<6} {len(subset):>6} "
                      f"{subset['polya_length'].median():>8.1f} {subset['polya_length'].mean():>8.1f} "
                      f"{subset['polya_length'].quantile(0.25):>6.1f} {subset['polya_length'].quantile(0.75):>6.1f}")

# Control
print()
for cond in ['HeLa', 'HeLa-Ars']:
    for psi_label, psi_val in [('psi+', True), ('psi-', False)]:
        subset = ctrl[(ctrl['condition'] == cond) & (ctrl['has_psi'] == psi_val)]
        subgroups[('ctrl_' + cond, psi_label, 'all')] = subset
        print(f"{cond+' Ctrl':<20} {psi_label:<6} {len(subset):>6} "
              f"{subset['polya_length'].median():>8.1f} {subset['polya_length'].mean():>8.1f} "
              f"{subset['polya_length'].quantile(0.25):>6.1f} {subset['polya_length'].quantile(0.75):>6.1f}")

# =========================================================================
# 3. Arsenite effect BY psi status
# =========================================================================
print(f"\n{'='*90}")
print("Arsenite Poly(A) Shortening: psi+ vs psi- Subgroups")
print(f"{'='*90}")

def compare_ars_effect(label, hela_grp, ars_grp):
    """Compare poly(A) between HeLa and HeLa-Ars within a subgroup."""
    h_med = hela_grp['polya_length'].median()
    a_med = ars_grp['polya_length'].median()
    h_mean = hela_grp['polya_length'].mean()
    a_mean = ars_grp['polya_length'].mean()
    delta_med = a_med - h_med
    delta_mean = a_mean - h_mean
    _, p = stats.mannwhitneyu(hela_grp['polya_length'], ars_grp['polya_length'],
                               alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label:<30} HeLa={h_med:>7.1f}  Ars={a_med:>7.1f}  "
          f"Δmed={delta_med:>+7.1f}  Δmean={delta_mean:>+7.1f}  p={p:.2e}({sig})")
    return {'label': label, 'hela_med': h_med, 'ars_med': a_med,
            'delta_med': delta_med, 'delta_mean': delta_mean, 'p': p}

results = {}
print("\n  L1 ALL:")
results['psi+'] = compare_ars_effect('L1 psi+',
    subgroups[('HeLa', 'psi+', 'all')], subgroups[('HeLa-Ars', 'psi+', 'all')])
results['psi-'] = compare_ars_effect('L1 psi-',
    subgroups[('HeLa', 'psi-', 'all')], subgroups[('HeLa-Ars', 'psi-', 'all')])

print("\n  L1 ANCIENT:")
results['anc_psi+'] = compare_ars_effect('Ancient L1 psi+',
    subgroups[('HeLa', 'psi+', 'ancient')], subgroups[('HeLa-Ars', 'psi+', 'ancient')])
results['anc_psi-'] = compare_ars_effect('Ancient L1 psi-',
    subgroups[('HeLa', 'psi-', 'ancient')], subgroups[('HeLa-Ars', 'psi-', 'ancient')])

print("\n  L1 YOUNG:")
results['young_psi+'] = compare_ars_effect('Young L1 psi+',
    subgroups[('HeLa', 'psi+', 'young')], subgroups[('HeLa-Ars', 'psi+', 'young')])
results['young_psi-'] = compare_ars_effect('Young L1 psi-',
    subgroups[('HeLa', 'psi-', 'young')], subgroups[('HeLa-Ars', 'psi-', 'young')])

print("\n  CONTROL:")
compare_ars_effect('Control psi+',
    subgroups[('ctrl_HeLa', 'psi+', 'all')], subgroups[('ctrl_HeLa-Ars', 'psi+', 'all')])
compare_ars_effect('Control psi-',
    subgroups[('ctrl_HeLa', 'psi-', 'all')], subgroups[('ctrl_HeLa-Ars', 'psi-', 'all')])

# =========================================================================
# 4. Differential shortening: psi+ vs psi-
# =========================================================================
print(f"\n{'='*90}")
print("Differential Shortening: Is psi+ More Deadenylated than psi-?")
print(f"{'='*90}")

print(f"\n  L1 ALL:")
print(f"    psi+ Δmedian: {results['psi+']['delta_med']:+.1f} nt")
print(f"    psi- Δmedian: {results['psi-']['delta_med']:+.1f} nt")
print(f"    Difference:   {results['psi+']['delta_med'] - results['psi-']['delta_med']:+.1f} nt")

print(f"\n  ANCIENT L1:")
print(f"    psi+ Δmedian: {results['anc_psi+']['delta_med']:+.1f} nt")
print(f"    psi- Δmedian: {results['anc_psi-']['delta_med']:+.1f} nt")
print(f"    Difference:   {results['anc_psi+']['delta_med'] - results['anc_psi-']['delta_med']:+.1f} nt")

print(f"\n  YOUNG L1:")
print(f"    psi+ Δmedian: {results['young_psi+']['delta_med']:+.1f} nt")
print(f"    psi- Δmedian: {results['young_psi-']['delta_med']:+.1f} nt")
print(f"    Difference:   {results['young_psi+']['delta_med'] - results['young_psi-']['delta_med']:+.1f} nt")

# =========================================================================
# 5. Interaction test: 2-way test
# =========================================================================
print(f"\n{'='*90}")
print("Interaction Test: psi x arsenite on poly(A)")
print(f"{'='*90}")

# Method: Compare the arsenite effect size between psi+ and psi- using permutation
print("\n  Bootstrap test for differential arsenite effect (psi+ vs psi-):")

np.random.seed(42)
n_bootstrap = 10000

for age_label, age_filter in [('L1 all', ['young', 'ancient']),
                                ('Ancient L1', ['ancient']),
                                ('Young L1', ['young'])]:
    hela_psi_pos = l1[(l1['condition'] == 'HeLa') & (l1['has_psi']) &
                       (l1['l1_age'].isin(age_filter))]['polya_length'].values
    hela_psi_neg = l1[(l1['condition'] == 'HeLa') & (~l1['has_psi']) &
                       (l1['l1_age'].isin(age_filter))]['polya_length'].values
    ars_psi_pos = l1[(l1['condition'] == 'HeLa-Ars') & (l1['has_psi']) &
                      (l1['l1_age'].isin(age_filter))]['polya_length'].values
    ars_psi_neg = l1[(l1['condition'] == 'HeLa-Ars') & (~l1['has_psi']) &
                      (l1['l1_age'].isin(age_filter))]['polya_length'].values

    # Observed interaction
    obs_delta_pos = np.median(ars_psi_pos) - np.median(hela_psi_pos)
    obs_delta_neg = np.median(ars_psi_neg) - np.median(hela_psi_neg)
    obs_interaction = obs_delta_pos - obs_delta_neg

    # Bootstrap CI for interaction
    boot_interactions = []
    for _ in range(n_bootstrap):
        hp = np.random.choice(hela_psi_pos, size=len(hela_psi_pos), replace=True)
        hn = np.random.choice(hela_psi_neg, size=len(hela_psi_neg), replace=True)
        ap = np.random.choice(ars_psi_pos, size=len(ars_psi_pos), replace=True)
        an = np.random.choice(ars_psi_neg, size=len(ars_psi_neg), replace=True)
        dp = np.median(ap) - np.median(hp)
        dn = np.median(an) - np.median(hn)
        boot_interactions.append(dp - dn)

    boot_interactions = np.array(boot_interactions)
    ci_lo = np.percentile(boot_interactions, 2.5)
    ci_hi = np.percentile(boot_interactions, 97.5)
    # p-value: proportion of bootstrap samples with opposite sign
    if obs_interaction < 0:
        p_boot = (boot_interactions >= 0).mean()
    else:
        p_boot = (boot_interactions <= 0).mean()
    p_boot = max(p_boot * 2, 1/n_bootstrap)  # two-sided

    sig = "***" if p_boot < 0.001 else "**" if p_boot < 0.01 else "*" if p_boot < 0.05 else "ns"
    print(f"\n  {age_label}:")
    print(f"    psi+ arsenite Δ: {obs_delta_pos:+.1f} nt")
    print(f"    psi- arsenite Δ: {obs_delta_neg:+.1f} nt")
    print(f"    Interaction (psi+ Δ - psi- Δ): {obs_interaction:+.1f} nt")
    print(f"    95% CI: [{ci_lo:+.1f}, {ci_hi:+.1f}]")
    print(f"    Bootstrap p: {p_boot:.4f} ({sig})")

# =========================================================================
# 6. Psi density quantile analysis
# =========================================================================
print(f"\n{'='*90}")
print("Dose-Response: Arsenite Shortening by Psi Density Quantile")
print(f"{'='*90}")

# Among psi+ reads, does higher psi density = more shortening?
l1_psi_pos = l1[l1['has_psi']].copy()

# Quartiles of psi_per_kb
l1_psi_pos['psi_quartile'] = pd.qcut(l1_psi_pos['psi_per_kb'], q=4,
                                       labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])

print(f"\n  Among psi+ L1 reads, arsenite Δpoly(A) by psi density quartile:")
print(f"  {'Quartile':<12} {'psi/kb range':>16} {'HeLa med':>10} {'Ars med':>10} {'Δ':>8} {'p':>12}")
print(f"  {'-'*65}")

for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
    hela_q = l1_psi_pos[(l1_psi_pos['condition'] == 'HeLa') & (l1_psi_pos['psi_quartile'] == q)]
    ars_q = l1_psi_pos[(l1_psi_pos['condition'] == 'HeLa-Ars') & (l1_psi_pos['psi_quartile'] == q)]
    if len(hela_q) >= 10 and len(ars_q) >= 10:
        psi_range = f"{hela_q['psi_per_kb'].min():.1f}-{hela_q['psi_per_kb'].max():.1f}"
        h_med = hela_q['polya_length'].median()
        a_med = ars_q['polya_length'].median()
        delta = a_med - h_med
        _, p = stats.mannwhitneyu(hela_q['polya_length'], ars_q['polya_length'],
                                   alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {q:<12} {psi_range:>16} {h_med:>10.1f} {a_med:>10.1f} {delta:>+7.1f} {p:>11.2e}({sig})")

# Spearman: psi density vs arsenite delta (across reads)
# Can't measure per-read delta, but can test: among ars reads, is higher psi associated with shorter polyA?
print(f"\n  Within HeLa-Ars psi+ reads: psi_per_kb vs polya_length")
ars_psi = l1_psi_pos[l1_psi_pos['condition'] == 'HeLa-Ars']
r, p = stats.spearmanr(ars_psi['psi_per_kb'], ars_psi['polya_length'])
print(f"    Spearman r={r:.3f}, p={p:.2e}")
print(f"    (negative r = higher psi → shorter polyA)")

hela_psi = l1_psi_pos[l1_psi_pos['condition'] == 'HeLa']
r_h, p_h = stats.spearmanr(hela_psi['psi_per_kb'], hela_psi['polya_length'])
print(f"\n  Within HeLa psi+ reads: psi_per_kb vs polya_length")
print(f"    Spearman r={r_h:.3f}, p={p_h:.2e}")

# =========================================================================
# 7. Same analysis for m6A
# =========================================================================
print(f"\n{'='*90}")
print("Same Test for m6A: Does m6A Status Affect Deadenylation?")
print(f"{'='*90}")

for age_label, age_filter in [('L1 all', ['young', 'ancient']),
                                ('Ancient L1', ['ancient'])]:
    hela_m6a_pos = l1[(l1['condition'] == 'HeLa') & (l1['has_m6a']) &
                       (l1['l1_age'].isin(age_filter))]['polya_length']
    hela_m6a_neg = l1[(l1['condition'] == 'HeLa') & (~l1['has_m6a']) &
                       (l1['l1_age'].isin(age_filter))]['polya_length']
    ars_m6a_pos = l1[(l1['condition'] == 'HeLa-Ars') & (l1['has_m6a']) &
                      (l1['l1_age'].isin(age_filter))]['polya_length']
    ars_m6a_neg = l1[(l1['condition'] == 'HeLa-Ars') & (~l1['has_m6a']) &
                      (l1['l1_age'].isin(age_filter))]['polya_length']

    d_pos = ars_m6a_pos.median() - hela_m6a_pos.median()
    d_neg = ars_m6a_neg.median() - hela_m6a_neg.median()
    interaction = d_pos - d_neg

    print(f"\n  {age_label}:")
    print(f"    m6A+ arsenite Δ: {d_pos:+.1f} nt  (HeLa={hela_m6a_pos.median():.1f} → Ars={ars_m6a_pos.median():.1f})")
    print(f"    m6A- arsenite Δ: {d_neg:+.1f} nt  (HeLa={hela_m6a_neg.median():.1f} → Ars={ars_m6a_neg.median():.1f})")
    print(f"    Interaction: {interaction:+.1f} nt")

# =========================================================================
# 8. Summary
# =========================================================================
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print("""
  Question: Do psi-modified L1 reads undergo more deadenylation under arsenite?

  Key results:""")
