#!/usr/bin/env python3
"""
Unified OLS analysis for revision: A2 + B2 + B3.

Produces authoritative numbers for:
  - Table S3: Full OLS coefficients at thr=204
  - Table S5: Threshold robustness (5 thresholds)
  - B2: R², adjusted R², partial R² for stress×m6A

Model spec (matches Table S3):
  poly(A) = β₀ + β₁·m6A/kb + β₂·read_length_z + β₃·stress
            + β₄·young + β₅·stress×m6A + β₆·young×m6A + β₇·stress×young + ε

Data: HeLa (normal, n=3) + HeLa-Ars (stress, n=3), PASS L1 reads only.
m6A parsed from MAFIA BAMs at 5 ML thresholds.
"""
import os, sys, glob
import pandas as pd
import numpy as np
import pysam
from scipy import stats
import statsmodels.api as sm

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
BASE = f"{PROJECT}/analysis/01_exploration"
RESULTS = f"{PROJECT}/results_group"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']
THRESHOLDS = [128, 153, 178, 204, 229]

###############################################################################
# Step 1: Parse MAFIA BAMs — extract per-read ML probabilities for m6A sites
###############################################################################
print("=" * 70)
print("Step 1: Parsing MAFIA BAMs for per-read m6A ML values...")
print("=" * 70)

all_rows = []
for group in GROUPS:
    bam_path = f'{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam'
    if not os.path.exists(bam_path):
        print(f"  WARNING: {bam_path} not found, skipping")
        continue

    bam = pysam.AlignmentFile(bam_path, 'rb')
    n_reads = 0
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rlen = read.query_alignment_length
        if rlen is None or rlen < 100:
            continue

        # Parse MM/ML tags
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t)
                break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t)
                break

        m6a_ml_values = []
        if mm_tag and ml_tag:
            ml_list = list(ml_tag)
            entries = mm_tag.rstrip(';').split(';')
            idx = 0
            for entry in entries:
                parts = entry.strip().split(',')
                base_mod = parts[0]
                skips = [int(x) for x in parts[1:]] if len(parts) > 1 else []
                n_sites = len(skips)
                if '21891' in base_mod:
                    for i in range(n_sites):
                        if idx + i < len(ml_list):
                            m6a_ml_values.append(ml_list[idx + i])
                idx += n_sites

        all_rows.append({
            'read_id': read.query_name,
            'group': group,
            'read_length': rlen,
            'm6a_ml_values': m6a_ml_values,
        })
        n_reads += 1
    bam.close()
    print(f"  {group}: {n_reads} reads")

df_bam = pd.DataFrame(all_rows)
print(f"  Total: {len(df_bam)} reads from BAMs")

###############################################################################
# Step 2: Compute m6A counts at each threshold
###############################################################################
print("\n" + "=" * 70)
print("Step 2: Computing m6A counts at each threshold...")
print("=" * 70)

for thr in THRESHOLDS:
    col = f'm6a_sites_{thr}'
    df_bam[col] = df_bam['m6a_ml_values'].apply(
        lambda vals: sum(1 for v in vals if v >= thr))
    col_kb = f'm6a_per_kb_{thr}'
    df_bam[col_kb] = df_bam[col] / (df_bam['read_length'] / 1000)
    med = df_bam[col_kb].median()
    print(f"  thr={thr}: median m6A/kb = {med:.3f}")

###############################################################################
# Step 3: Merge with L1 summary (poly(A), age, genomic context)
###############################################################################
print("\n" + "=" * 70)
print("Step 3: Merging with L1 summary...")
print("=" * 70)

raw_frames = []
for group in GROUPS:
    f = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
    if not os.path.exists(f):
        continue
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'polya_length', 'qc_tag',
                                              'gene_id', 'read_length'])
    cl = group.rsplit('_', 1)[0]
    tmp['cell_line'] = cl
    raw_frames.append(tmp)
df_l1 = pd.concat(raw_frames, ignore_index=True)
df_l1 = df_l1[df_l1['qc_tag'] == 'PASS'].copy()
print(f"  PASS L1 reads: {len(df_l1)}")

# Merge
m6a_cols = [f'm6a_per_kb_{t}' for t in THRESHOLDS]
df = df_l1.merge(df_bam[['read_id'] + m6a_cols], on='read_id', how='inner')
df['is_young'] = df['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$').astype(int)
df['is_stress'] = (df['cell_line'] == 'HeLa-Ars').astype(int)
df['read_length_z'] = (df['read_length'] - df['read_length'].mean()) / df['read_length'].std()

print(f"  Merged: {len(df)} reads")
print(f"  HeLa: {(df['is_stress']==0).sum()}, HeLa-Ars: {(df['is_stress']==1).sum()}")
print(f"  Young: {df['is_young'].sum()}, Ancient: {(df['is_young']==0).sum()}")

###############################################################################
# Step 4: Run OLS at each threshold — Table S3 model spec
###############################################################################
print("\n" + "=" * 70)
print("Step 4: Running OLS at each threshold...")
print("=" * 70)

ols_results = []
full_model_at_204 = None

for thr in THRESHOLDS:
    m6a_col = f'm6a_per_kb_{thr}'

    df_ols = df[['polya_length', m6a_col, 'read_length_z',
                  'is_stress', 'is_young']].dropna().copy()
    df_ols.rename(columns={m6a_col: 'm6a_per_kb'}, inplace=True)

    # Interaction terms
    df_ols['stress_x_m6a'] = df_ols['is_stress'] * df_ols['m6a_per_kb']
    df_ols['young_x_m6a'] = df_ols['is_young'] * df_ols['m6a_per_kb']
    df_ols['stress_x_young'] = df_ols['is_stress'] * df_ols['is_young']

    y = df_ols['polya_length']
    X = df_ols[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young',
                'stress_x_m6a', 'young_x_m6a', 'stress_x_young']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    beta = model.params['stress_x_m6a']
    se = model.bse['stress_x_m6a']
    p = model.pvalues['stress_x_m6a']
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    n = int(model.nobs)

    ols_results.append({
        'threshold': thr,
        'prob_pct': int(round(thr / 255 * 100)),
        'beta_stress_x_m6a': beta,
        'se': se,
        'p': p,
        'r2': r2,
        'r2_adj': r2_adj,
        'n': n,
    })

    print(f"\n  thr={thr} ({int(round(thr/255*100))}%): "
          f"beta3={beta:.2f}, SE={se:.2f}, P={p:.2e}, R2={r2:.4f}, n={n}")

    if thr == 204:
        full_model_at_204 = model

###############################################################################
# Step 5: Full Table S3 output at thr=204
###############################################################################
print("\n" + "=" * 70)
print("Step 5: Full OLS coefficients at thr=204 (Table S3)")
print("=" * 70)

model = full_model_at_204
ci = model.conf_int()

var_names = {
    'const': 'Intercept',
    'm6a_per_kb': 'm6A/kb',
    'read_length_z': 'Read length (z)',
    'is_stress': 'Stress',
    'is_young': 'Young',
    'stress_x_m6a': 'Stress x m6A/kb',
    'young_x_m6a': 'Young x m6A/kb',
    'stress_x_young': 'Stress x Young',
}

table_s3_rows = []
for var in model.params.index:
    row = {
        'variable': var_names.get(var, var),
        'coef': model.params[var],
        'se': model.bse[var],
        't': model.tvalues[var],
        'p': model.pvalues[var],
        'ci_low': ci.loc[var, 0],
        'ci_high': ci.loc[var, 1],
    }
    table_s3_rows.append(row)
    print(f"  {row['variable']:25s}  coef={row['coef']:8.2f}  SE={row['se']:6.2f}  "
          f"t={row['t']:6.2f}  P={row['p']:.2e}  [{row['ci_low']:.1f}, {row['ci_high']:.1f}]")

df_s3 = pd.DataFrame(table_s3_rows)
df_s3.to_csv(f'{OUTDIR}/table_s3_ols_thr204.tsv', sep='\t', index=False)

print(f"\n  R2 = {model.rsquared:.4f}")
print(f"  Adj R2 = {model.rsquared_adj:.4f}")
print(f"  n = {model.nobs:.0f}")

###############################################################################
# Step 6: Partial R2 for stress x m6A (Plan B2)
###############################################################################
print("\n" + "=" * 70)
print("Step 6: Partial R2 for stress x m6A interaction")
print("=" * 70)

m6a_col_204 = 'm6a_per_kb_204'
df_ols_204 = df[['polya_length', m6a_col_204, 'read_length_z',
                  'is_stress', 'is_young']].dropna().copy()
df_ols_204.rename(columns={m6a_col_204: 'm6a_per_kb'}, inplace=True)
df_ols_204['stress_x_m6a'] = df_ols_204['is_stress'] * df_ols_204['m6a_per_kb']
df_ols_204['young_x_m6a'] = df_ols_204['is_young'] * df_ols_204['m6a_per_kb']
df_ols_204['stress_x_young'] = df_ols_204['is_stress'] * df_ols_204['is_young']

y = df_ols_204['polya_length']

# Full model
X_full = sm.add_constant(df_ols_204[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young',
                                      'stress_x_m6a', 'young_x_m6a', 'stress_x_young']])
model_full = sm.OLS(y, X_full).fit()

# Reduced: drop stress_x_m6a
X_reduced = sm.add_constant(df_ols_204[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young',
                                         'young_x_m6a', 'stress_x_young']])
model_reduced = sm.OLS(y, X_reduced).fit()
partial_r2_interaction = (model_reduced.ssr - model_full.ssr) / model_reduced.ssr

# Partial R2 for m6A/kb main effect
X_no_m6a = sm.add_constant(df_ols_204[['read_length_z', 'is_stress', 'is_young',
                                         'stress_x_m6a', 'young_x_m6a', 'stress_x_young']])
model_no_m6a = sm.OLS(y, X_no_m6a).fit()
partial_r2_m6a = (model_no_m6a.ssr - model_full.ssr) / model_no_m6a.ssr

# Minimal model: just intercept + read_length_z + stress
X_minimal = sm.add_constant(df_ols_204[['read_length_z', 'is_stress']])
model_minimal = sm.OLS(y, X_minimal).fit()

print(f"  Full model R2 = {model_full.rsquared:.4f}")
print(f"  Adj R2 = {model_full.rsquared_adj:.4f}")
print(f"  Partial R2 (stress x m6A) = {partial_r2_interaction:.6f}")
print(f"  Partial R2 (m6A/kb main) = {partial_r2_m6a:.6f}")
print(f"  Minimal model R2 (intercept + RL + stress) = {model_minimal.rsquared:.4f}")
print(f"  m6A terms add dR2 = {model_full.rsquared - model_minimal.rsquared:.4f}")

###############################################################################
# Step 7: Table S5 output
###############################################################################
print("\n" + "=" * 70)
print("Step 7: Table S5 — Threshold robustness")
print("=" * 70)

df_s5 = pd.DataFrame(ols_results)
df_s5.to_csv(f'{OUTDIR}/table_s5_threshold_robustness.tsv', sep='\t', index=False)

for _, row in df_s5.iterrows():
    print(f"  ML>={row['threshold']:3.0f} ({row['prob_pct']}%): "
          f"beta3={row['beta_stress_x_m6a']:.2f}, SE={row['se']:.2f}, "
          f"P={row['p']:.2e}, R2={row['r2']:.3f}")

###############################################################################
# Summary
###############################################################################
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nTable S3 caption: n = {model_full.nobs:.0f}, "
      f"R2 = {model_full.rsquared:.3f}, adj R2 = {model_full.rsquared_adj:.3f}")
print(f"Partial R2 stress x m6A = {partial_r2_interaction:.4f}")

print("\nTable S5 LaTeX rows:")
for _, row in df_s5.iterrows():
    thr = int(row['threshold'])
    pct = int(row['prob_pct'])
    beta = row['beta_stress_x_m6a']
    se = row['se']
    p = row['p']
    r2 = row['r2']
    exp = int(np.floor(np.log10(p)))
    mant = p / 10**exp
    print(f"ML $\\geq$ {thr} ({pct}\\%) & {beta:.2f} & {se:.2f} "
          f"& ${mant:.1f}\\times10^{{{exp}}}$ & {r2:.3f} \\\\")

print(f"\nOutputs saved to: {OUTDIR}/")
