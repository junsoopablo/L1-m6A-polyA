#!/usr/bin/env python3
"""
B3: Threshold robustness for Part 3 (m6A-poly(A) OLS interaction).
Parses HeLa + HeLa-Ars MAFIA BAMs at 5 thresholds,
joins with poly(A), and runs OLS at each threshold.
"""
import os, sys
import numpy as np
import pandas as pd
import pysam
from scipy import stats
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
OUTDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline/m6a_polya_stratified'

THRESHOLDS = {128: '0.50', 153: '0.60', 178: '0.70', 204: '0.80', 229: '0.90'}
YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Only HeLa groups (used in OLS)
GROUPS = {
    'HeLa_1': 'normal', 'HeLa_2': 'normal', 'HeLa_3': 'normal',
    'HeLa-Ars_1': 'stress', 'HeLa-Ars_2': 'stress', 'HeLa-Ars_3': 'stress',
}

###############################################################################
# 1. Parse MAFIA BAMs at multiple thresholds
###############################################################################
def parse_bam_multi(bam_path, thresholds):
    """Parse MAFIA BAM, count m6A sites at multiple ML thresholds."""
    if not os.path.exists(bam_path):
        print(f"    SKIP: {bam_path}")
        return []
    bam = pysam.AlignmentFile(bam_path, 'rb')
    rows = []
    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t); break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t); break
        rlen = read.query_alignment_length
        if rlen is None or rlen < 100:
            continue
        row = {'read_id': read.query_name, 'read_length': rlen}
        if mm_tag is None or ml_tag is None:
            for thr in thresholds:
                row[f'sites_{thr}'] = 0
            rows.append(row)
            continue
        # Extract m6A ML values (chemodCode 21891)
        ml_values = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        idx = 0
        m6a_ml_vals = []
        for entry in entries:
            parts = entry.strip().split(',')
            base_mod = parts[0]
            skip_counts = [int(x) for x in parts[1:]] if len(parts) > 1 else []
            n_sites = len(skip_counts)
            if '21891' in base_mod:
                for i in range(n_sites):
                    if idx + i < len(ml_values):
                        m6a_ml_vals.append(ml_values[idx + i])
            idx += n_sites
        for thr in thresholds:
            row[f'sites_{thr}'] = sum(1 for v in m6a_ml_vals if v >= thr)
        rows.append(row)
    bam.close()
    return rows

print("=== Parsing MAFIA BAMs at 5 thresholds ===")
all_rows = []
for group, condition in GROUPS.items():
    bam_path = RESULTS / group / 'h_mafia' / f'{group}.mAFiA.reads.bam'
    print(f"  {group} ({condition})...", end=' ', flush=True)
    rows = parse_bam_multi(str(bam_path), list(THRESHOLDS.keys()))
    for r in rows:
        r['group'] = group
        r['condition'] = condition
    all_rows.extend(rows)
    print(f"{len(rows)} reads")

mafia_df = pd.DataFrame(all_rows)
print(f"\n  Total MAFIA reads: {len(mafia_df):,}")

###############################################################################
# 2. Load L1 summary for poly(A) + age
###############################################################################
print("\n=== Loading L1 summary ===")
summary_rows = []
for group in GROUPS:
    f = RESULTS / group / 'g_summary' / f'{group}_L1_summary.tsv'
    if f.exists():
        df = pd.read_csv(f, sep='\t', usecols=[
            'read_id', 'read_length', 'gene_id', 'polya_length', 'qc_tag', 'sample'
        ])
        df['group'] = group
        summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
summary = summary[(summary['qc_tag'] == 'PASS') & (summary['polya_length'] > 0)].copy()
summary['l1_age'] = summary['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG_SUBFAMILIES else 'ancient'
)
print(f"  L1 reads with poly(A): {len(summary):,}")

###############################################################################
# 3. Merge and run OLS at each threshold
###############################################################################
print("\n=== Running OLS at each threshold ===")

merged = mafia_df.merge(summary[['read_id', 'polya_length', 'l1_age']],
                        on='read_id', how='inner')
merged['is_stress'] = (merged['condition'] == 'stress').astype(int)
merged['is_young'] = (merged['l1_age'] == 'young').astype(int)
merged['read_length_z'] = (merged['read_length'] - merged['read_length'].mean()) / merged['read_length'].std()
print(f"  Merged reads: {len(merged):,}")

results = []
print(f"\n  {'Threshold':>10s} {'β(stress×m6A)':>14s} {'SE':>8s} {'P':>12s} {'R²':>8s} {'N':>6s}")
print("  " + "-"*65)

for thr_val, thr_label in sorted(THRESHOLDS.items()):
    col = f'sites_{thr_val}'
    merged[f'm6a_kb_{thr_val}'] = merged[col] / (merged['read_length'] / 1000)

    X = merged[[f'm6a_kb_{thr_val}', 'read_length_z', 'is_stress', 'is_young']].copy()
    X.columns = ['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young']
    X['stress_x_m6a'] = X['is_stress'] * X['m6a_per_kb']
    X['young_x_m6a'] = X['is_young'] * X['m6a_per_kb']
    X['stress_x_young'] = X['is_stress'] * X['is_young']
    X = sm.add_constant(X)
    y = merged['polya_length']

    model = sm.OLS(y, X).fit()

    idx = list(X.columns).index('stress_x_m6a')
    coef = model.params[idx]
    se = model.bse[idx]
    pval = model.pvalues[idx]
    ci_l, ci_h = model.conf_int().iloc[idx]

    row = {
        'threshold_ml': thr_val,
        'threshold_prob': thr_label,
        'stress_x_m6a_coef': round(coef, 3),
        'stress_x_m6a_se': round(se, 3),
        'stress_x_m6a_p': pval,
        'stress_x_m6a_ci_low': round(ci_l, 1),
        'stress_x_m6a_ci_high': round(ci_h, 1),
        'r_squared': round(model.rsquared, 6),
        'n': int(model.nobs),
        'median_m6a_kb': round(merged[f'm6a_kb_{thr_val}'].median(), 2),
    }
    results.append(row)

    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  ML>={thr_val:3d} ({thr_label}) {coef:14.3f} {se:8.3f} {pval:12.2e} {model.rsquared:8.4f} {int(model.nobs):6d} {sig}")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv(OUTDIR / 'threshold_robustness_ols.tsv', sep='\t', index=False)
print(f"\nSaved to {OUTDIR}/threshold_robustness_ols.tsv")
print("\nDONE.")
