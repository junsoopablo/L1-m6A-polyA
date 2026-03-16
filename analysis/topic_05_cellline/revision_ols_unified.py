#!/usr/bin/env python3
"""
Unified OLS analysis for manuscript revision.
Produces authoritative numbers for:
  A2: OLS coefficient unification (single definitive run)
  B2: R², adjusted R², partial R² for m6A/kb
  B4: Decay zone threshold sensitivity (20, 30, 40, 50, 60 nt)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
RESULTS = BASE / 'results_group'
OUTDIR = BASE / 'analysis/01_exploration/topic_05_cellline/m6a_polya_stratified'
OUTDIR.mkdir(exist_ok=True)

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

###############################################################################
# 1. Load data (same as m6a_polya_stratified_correlation.py)
###############################################################################
print("=== Loading L1 summary files ===")
summary_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'read_length', 'gene_id', 'polya_length',
        'qc_tag', 'TE_group', 'sample'
    ])
    df['group'] = df['sample'].str.rsplit('_', n=1).str[0]
    summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
summary = summary[(summary['qc_tag'] == 'PASS') & (summary['polya_length'] > 0)].copy()
print(f"  After QC PASS + poly(A) > 0: {len(summary):,}")

summary['l1_age'] = summary['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG_SUBFAMILIES else 'ancient'
)

def classify_context(tg):
    if pd.isna(tg): return 'other'
    tg = str(tg).lower()
    if 'intronic' in tg: return 'intronic'
    elif 'intergenic' in tg: return 'intergenic'
    return 'other'
summary['genomic_context'] = summary['TE_group'].apply(classify_context)

print("=== Loading Part3 per-read cache ===")
cache_rows = []
for f in sorted(CACHE_DIR.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_rows.append(df)
cache = pd.concat(cache_rows, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)

merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']],
                       on='read_id', how='inner')
merged['cellline'] = merged['group'].apply(lambda g: g.rsplit('_', 1)[0])
merged['condition'] = merged['cellline'].apply(lambda x: 'stress' if 'Ars' in x else 'normal')
print(f"  Merged reads: {len(merged):,}")

# HeLa/HeLa-Ars subset for OLS
hela = merged[merged['cellline'].isin(['HeLa', 'HeLa-Ars'])].copy()
hela = hela[hela['genomic_context'].isin(['intronic', 'intergenic'])].copy()
hela['read_length_z'] = (hela['read_length'] - hela['read_length'].mean()) / hela['read_length'].std()
hela['is_stress'] = (hela['condition'] == 'stress').astype(int)
hela['is_young'] = (hela['l1_age'] == 'young').astype(int)

###############################################################################
# 2. A2: DEFINITIVE OLS — matching Table S3 specification (no intergenic terms)
###############################################################################
print("\n" + "="*80)
print("=== A2: DEFINITIVE OLS MODEL (Table S3 spec) ===")
print("="*80)

X = hela[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young']].copy()
X['stress_x_m6a'] = X['is_stress'] * X['m6a_per_kb']
X['young_x_m6a'] = X['is_young'] * X['m6a_per_kb']
X['stress_x_young'] = X['is_stress'] * X['is_young']
X = sm.add_constant(X)
y = hela['polya_length']

model = sm.OLS(y, X).fit()

# Print full summary
print(f"\n  N = {int(model.nobs)}")
print(f"  R² = {model.rsquared:.6f}")
print(f"  Adj R² = {model.rsquared_adj:.6f}")
print(f"\n  {'Variable':<25s} {'Coef':>8s} {'SE':>8s} {'t':>8s} {'P':>12s} {'95% CI':>20s}")
print("  " + "-"*85)

var_names = ['Intercept', 'm6A/kb', 'Read length (z)', 'Stress', 'Young',
             'Stress × m6A/kb', 'Young × m6A/kb', 'Stress × Young']

for i, (coef, se, t, p, ci_l, ci_h) in enumerate(zip(
        model.params, model.bse, model.tvalues, model.pvalues,
        model.conf_int()[0], model.conf_int()[1])):
    name = var_names[i] if i < len(var_names) else f'var_{i}'
    print(f"  {name:<25s} {coef:8.2f} {se:8.2f} {t:8.2f} {p:12.2e} [{ci_l:7.1f}, {ci_h:7.1f}]")

# Key number for manuscript
stress_x_m6a_idx = list(X.columns).index('stress_x_m6a')
key_coef = model.params[stress_x_m6a_idx]
key_p = model.pvalues[stress_x_m6a_idx]
key_ci = model.conf_int().iloc[stress_x_m6a_idx]
print(f"\n  *** KEY: stress×m6A/kb coefficient = {key_coef:.2f}, P = {key_p:.2e} ***")
print(f"  *** 95% CI: [{key_ci[0]:.1f}, {key_ci[1]:.1f}] ***")

###############################################################################
# 3. B2: R², Partial R² for m6A/kb
###############################################################################
print("\n" + "="*80)
print("=== B2: R² AND PARTIAL R² ===")
print("="*80)

print(f"\n  Full model R²:     {model.rsquared:.6f} ({model.rsquared*100:.2f}%)")
print(f"  Full model Adj R²: {model.rsquared_adj:.6f} ({model.rsquared_adj*100:.2f}%)")

# Partial R² for stress×m6A/kb via Type III comparison
# Compare full model vs model without stress_x_m6a
X_reduced = X.drop(columns=['stress_x_m6a'])
model_reduced = sm.OLS(y, X_reduced).fit()
partial_r2 = (model_reduced.ssr - model.ssr) / model_reduced.ssr
print(f"\n  Partial R² for stress×m6A/kb: {partial_r2:.6f} ({partial_r2*100:.4f}%)")

# Partial R² for m6A/kb (main effect + interaction combined)
X_no_m6a = X.drop(columns=['m6a_per_kb', 'stress_x_m6a', 'young_x_m6a'])
model_no_m6a = sm.OLS(y, X_no_m6a).fit()
combined_partial_r2 = (model_no_m6a.ssr - model.ssr) / model_no_m6a.ssr
print(f"  Combined partial R² for all m6A terms: {combined_partial_r2:.6f} ({combined_partial_r2*100:.4f}%)")

# Partial R² for read length
X_no_rl = X.drop(columns=['read_length_z'])
model_no_rl = sm.OLS(y, X_no_rl).fit()
rl_partial_r2 = (model_no_rl.ssr - model.ssr) / model_no_rl.ssr
print(f"  Partial R² for read length: {rl_partial_r2:.6f} ({rl_partial_r2*100:.4f}%)")

# Partial R² for stress
X_no_stress = X.drop(columns=['is_stress', 'stress_x_m6a', 'stress_x_young'])
model_no_stress = sm.OLS(y, X_no_stress).fit()
stress_partial_r2 = (model_no_stress.ssr - model.ssr) / model_no_stress.ssr
print(f"  Combined partial R² for all stress terms: {stress_partial_r2:.6f} ({stress_partial_r2*100:.4f}%)")

###############################################################################
# 4. B4: DECAY ZONE THRESHOLD SENSITIVITY
###############################################################################
print("\n" + "="*80)
print("=== B4: DECAY ZONE THRESHOLD SENSITIVITY ===")
print("="*80)

# Use HeLa-Ars stressed ancient L1 reads with m6A quartiles
stressed = hela[(hela['is_stress'] == 1) & (hela['l1_age'] == 'ancient')].copy()
stressed['m6a_q'] = pd.qcut(stressed['m6a_per_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                             duplicates='drop')

print(f"\n  Stressed ancient L1 reads: {len(stressed)}")
print(f"  Q1 (low m6A): n={len(stressed[stressed['m6a_q']=='Q1'])}, "
      f"median m6A/kb={stressed[stressed['m6a_q']=='Q1']['m6a_per_kb'].median():.2f}")
print(f"  Q4 (high m6A): n={len(stressed[stressed['m6a_q']=='Q4'])}, "
      f"median m6A/kb={stressed[stressed['m6a_q']=='Q4']['m6a_per_kb'].median():.2f}")

print(f"\n  {'Threshold':>10s} {'Q1_frac':>10s} {'Q4_frac':>10s} {'Ratio':>8s} {'OR':>8s} {'P':>12s}")
print("  " + "-"*65)

decay_results = []
for thr in [20, 25, 30, 35, 40, 50, 60]:
    q1 = stressed[stressed['m6a_q'] == 'Q1']
    q4 = stressed[stressed['m6a_q'] == 'Q4']

    q1_below = (q1['polya_length'] < thr).sum()
    q1_above = (q1['polya_length'] >= thr).sum()
    q4_below = (q4['polya_length'] < thr).sum()
    q4_above = (q4['polya_length'] >= thr).sum()

    q1_frac = q1_below / len(q1) if len(q1) > 0 else 0
    q4_frac = q4_below / len(q4) if len(q4) > 0 else 0
    ratio = q1_frac / q4_frac if q4_frac > 0 else np.inf

    # Fisher's exact
    table = [[q1_below, q1_above], [q4_below, q4_above]]
    odds_ratio, fisher_p = stats.fisher_exact(table)

    decay_results.append({
        'threshold_nt': thr,
        'Q1_below_frac': round(q1_frac, 4),
        'Q4_below_frac': round(q4_frac, 4),
        'Q1_Q4_ratio': round(ratio, 2),
        'odds_ratio': round(odds_ratio, 2),
        'fisher_p': fisher_p,
        'Q1_n': len(q1), 'Q4_n': len(q4),
        'Q1_below': q1_below, 'Q4_below': q4_below
    })

    print(f"  {thr:>6d} nt {q1_frac:>10.1%} {q4_frac:>10.1%} {ratio:>8.2f} {odds_ratio:>8.2f} {fisher_p:>12.2e}")

# Also for baseline (normal HeLa)
print("\n  --- Baseline (unstressed) for comparison ---")
baseline = hela[(hela['is_stress'] == 0) & (hela['l1_age'] == 'ancient')].copy()
baseline['m6a_q'] = pd.qcut(baseline['m6a_per_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                             duplicates='drop')

for thr in [20, 30, 40, 50]:
    q1 = baseline[baseline['m6a_q'] == 'Q1']
    q4 = baseline[baseline['m6a_q'] == 'Q4']
    q1_frac = (q1['polya_length'] < thr).sum() / len(q1) if len(q1) > 0 else 0
    q4_frac = (q4['polya_length'] < thr).sum() / len(q4) if len(q4) > 0 else 0
    ratio = q1_frac / q4_frac if q4_frac > 0 else np.inf
    table = [[(q1['polya_length'] < thr).sum(), (q1['polya_length'] >= thr).sum()],
             [(q4['polya_length'] < thr).sum(), (q4['polya_length'] >= thr).sum()]]
    odds_ratio, fisher_p = stats.fisher_exact(table)
    print(f"  {thr:>6d} nt {q1_frac:>10.1%} {q4_frac:>10.1%} {ratio:>8.2f} {odds_ratio:>8.2f} {fisher_p:>12.2e}")

###############################################################################
# 5. Save definitive outputs
###############################################################################
# Save OLS table for Table S3 update
ols_rows = []
for i, (var, coef, se, t, p, ci_l, ci_h) in enumerate(zip(
        var_names, model.params, model.bse, model.tvalues, model.pvalues,
        model.conf_int()[0], model.conf_int()[1])):
    ols_rows.append({'variable': var, 'coef': round(coef, 2), 'se': round(se, 2),
                     't': round(t, 2), 'p': p, 'ci_low': round(ci_l, 1), 'ci_high': round(ci_h, 1)})
ols_df = pd.DataFrame(ols_rows)
ols_df.to_csv(OUTDIR / 'definitive_ols_table_s3.tsv', sep='\t', index=False)

# Save R² summary
r2_summary = pd.DataFrame([
    {'metric': 'R_squared', 'value': model.rsquared},
    {'metric': 'Adj_R_squared', 'value': model.rsquared_adj},
    {'metric': 'Partial_R2_stress_x_m6a', 'value': partial_r2},
    {'metric': 'Partial_R2_all_m6a_terms', 'value': combined_partial_r2},
    {'metric': 'Partial_R2_read_length', 'value': rl_partial_r2},
    {'metric': 'Partial_R2_all_stress_terms', 'value': stress_partial_r2},
    {'metric': 'N_observations', 'value': model.nobs},
])
r2_summary.to_csv(OUTDIR / 'definitive_r2_summary.tsv', sep='\t', index=False)

# Save decay zone results
pd.DataFrame(decay_results).to_csv(OUTDIR / 'decay_zone_sensitivity.tsv', sep='\t', index=False)

print(f"\n=== Saved to {OUTDIR}/ ===")
print(f"  definitive_ols_table_s3.tsv")
print(f"  definitive_r2_summary.tsv")
print(f"  decay_zone_sensitivity.tsv")
print(f"\nDONE.")
