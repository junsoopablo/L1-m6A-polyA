#!/usr/bin/env python3
"""
Regenerate all intermediate TSVs needed by figure scripts after m6A threshold change.
Part3 per-read caches are already at PROB_THRESHOLD=204. This script re-derives
all downstream TSVs from those caches.

Outputs:
  1. m6a_validation/m6a_validation_per_group.tsv  (Fig 1b, 1c)
  2. m6a_polya_stratified/m6a_polya_ols_full.tsv  (Fig 3b)
  3. m6a_polya_stratified/m6a_polya_hela_stratified.tsv  (Fig 3d)
  4. METTL3 KO summary files  (Fig 1e)
  5. topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv  (Fig S6)
  6. topic_05_cellline/m6a_validation/m6a_body_vs_flanking_persite_summary.tsv  (Fig S5)
"""
import os, glob, sys
import pandas as pd
import numpy as np
from scipy import stats

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
BASE = f"{PROJECT}/analysis/01_exploration"
RESULTS = f"{PROJECT}/results_group"
METTL3 = "/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore/analysis/matched_guppy"

BASE_GROUPS = [
    'A549_4', 'A549_5', 'A549_6',
    'H9_2', 'H9_3', 'H9_4',
    'HeLa_1', 'HeLa_2', 'HeLa_3',
    'HepG2_5', 'HepG2_6',
    'HEYA8_1', 'HEYA8_2', 'HEYA8_3',
    'K562_4', 'K562_5', 'K562_6',
    'MCF7_2', 'MCF7_3', 'MCF7_4',
    'SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3',
]

HELA_ARS_GROUPS = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

# =====================================================================
# SECTION 1: Regenerate m6a_validation_per_group.tsv
# =====================================================================
print("=" * 70)
print("SECTION 1: Regenerate m6a_validation_per_group.tsv")
print("=" * 70)

# Load old file for motif data (independent of threshold)
old_val = pd.read_csv(f'{BASE}/topic_05_cellline/m6a_validation/m6a_validation_per_group.tsv', sep='\t')

# Build new m6A counts from Part3 caches
all_groups = BASE_GROUPS + HELA_ARS_GROUPS

new_rows = []
for group in all_groups:
    # L1 cache
    l1_f = f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/{group}_l1_per_read.tsv'
    ctrl_f = f'{BASE}/topic_05_cellline/part3_ctrl_per_read_cache/{group}_ctrl_per_read.tsv'

    if not os.path.exists(l1_f) or not os.path.exists(ctrl_f):
        print(f"  WARNING: Missing cache for {group}, skipping")
        continue

    l1_df = pd.read_csv(l1_f, sep='\t')
    ctrl_df = pd.read_csv(ctrl_f, sep='\t')

    l1_reads = len(l1_df)
    ctrl_reads = len(ctrl_df)
    l1_m6a_total = l1_df['m6a_sites_high'].sum()
    ctrl_m6a_total = ctrl_df['m6a_sites_high'].sum()
    l1_aligned_bp = l1_df['read_length'].sum()
    ctrl_aligned_bp = ctrl_df['read_length'].sum()

    # Get motif data from old file
    old_row = old_val[old_val['group'] == group]
    if len(old_row) == 0:
        print(f"  WARNING: No old data for {group}, skipping motif columns")
        l1_motif = np.nan
        ctrl_motif = np.nan
    else:
        old_row = old_row.iloc[0]
        l1_motif = old_row['l1_motif_sites']
        ctrl_motif = old_row['ctrl_motif_sites']

    l1_m6a_per_kb = l1_m6a_total / (l1_aligned_bp / 1000) if l1_aligned_bp > 0 else 0
    ctrl_m6a_per_kb = ctrl_m6a_total / (ctrl_aligned_bp / 1000) if ctrl_aligned_bp > 0 else 0
    l1_motif_per_kb = l1_motif / (l1_aligned_bp / 1000) if l1_aligned_bp > 0 and not np.isnan(l1_motif) else np.nan
    ctrl_motif_per_kb = ctrl_motif / (ctrl_aligned_bp / 1000) if ctrl_aligned_bp > 0 and not np.isnan(ctrl_motif) else np.nan
    l1_per_site = l1_m6a_total / l1_motif if l1_motif > 0 and not np.isnan(l1_motif) else np.nan
    ctrl_per_site = ctrl_m6a_total / ctrl_motif if ctrl_motif > 0 and not np.isnan(ctrl_motif) else np.nan

    new_rows.append({
        'group': group,
        'l1_reads': l1_reads,
        'ctrl_reads': ctrl_reads,
        'l1_m6a_high': l1_m6a_total,
        'ctrl_m6a_high': ctrl_m6a_total,
        'l1_motif_sites': l1_motif,
        'ctrl_motif_sites': ctrl_motif,
        'l1_aligned_bp': l1_aligned_bp,
        'ctrl_aligned_bp': ctrl_aligned_bp,
        'l1_per_site_rate': l1_per_site,
        'ctrl_per_site_rate': ctrl_per_site,
        'l1_m6a_per_kb': l1_m6a_per_kb,
        'ctrl_m6a_per_kb': ctrl_m6a_per_kb,
        'l1_motif_per_kb': l1_motif_per_kb,
        'ctrl_motif_per_kb': ctrl_motif_per_kb,
    })

new_val = pd.DataFrame(new_rows)
out_f1 = f'{BASE}/topic_05_cellline/m6a_validation/m6a_validation_per_group.tsv'
new_val.to_csv(out_f1, sep='\t', index=False)
print(f"  Saved: {out_f1}")
print(f"  {len(new_val)} groups. L1 m6A/kb median={new_val['l1_m6a_per_kb'].median():.3f}, Ctrl={new_val['ctrl_m6a_per_kb'].median():.3f}")
print(f"  L1/Ctrl ratio={new_val['l1_m6a_per_kb'].median()/new_val['ctrl_m6a_per_kb'].median():.3f}")

# =====================================================================
# SECTION 2 & 3: OLS and stratified correlations
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2 & 3: OLS + stratified correlations")
print("=" * 70)

# Load HeLa + HeLa-Ars per-read data
raw_frames = []
for group in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    f = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
    if not os.path.exists(f):
        continue
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'polya_length', 'qc_tag', 'gene_id',
                                              'read_length', 'overlapping_genes'])
    cl = group.rsplit('_', 1)[0]
    tmp['cell_line'] = cl
    raw_frames.append(tmp)
df_l1 = pd.concat(raw_frames, ignore_index=True)
df_l1 = df_l1[df_l1['qc_tag'] == 'PASS'].copy()

# Load Part3 caches for m6A
cache_frames = []
for group in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    f = f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/{group}_l1_per_read.tsv'
    if not os.path.exists(f):
        continue
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_frames.append(tmp)
df_cache = pd.concat(cache_frames, ignore_index=True)
df_cache['m6a_per_kb'] = df_cache['m6a_sites_high'] / (df_cache['read_length'] / 1000)

df_merged = df_l1.merge(df_cache[['read_id', 'm6a_per_kb']], on='read_id', how='inner')
df_merged['is_young'] = df_merged['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')
df_merged['is_stress'] = (df_merged['cell_line'] == 'HeLa-Ars').astype(int)
df_merged['is_intergenic'] = (df_merged['overlapping_genes'] == '.').astype(int)
df_merged['read_length_z'] = (df_merged['read_length'] - df_merged['read_length'].mean()) / df_merged['read_length'].std()

print(f"  Merged data: {len(df_merged)} reads (HeLa={sum(df_merged['cell_line']=='HeLa')}, Ars={sum(df_merged['cell_line']=='HeLa-Ars')})")

# --- OLS ---
import statsmodels.api as sm

df_ols = df_merged[['polya_length', 'm6a_per_kb', 'read_length_z',
                     'is_stress', 'is_young', 'is_intergenic']].dropna().copy()
df_ols['is_young'] = df_ols['is_young'].astype(int)

# Interaction terms
df_ols['stress_x_m6a'] = df_ols['is_stress'] * df_ols['m6a_per_kb']
df_ols['young_x_m6a'] = df_ols['is_young'] * df_ols['m6a_per_kb']
df_ols['intergenic_x_m6a'] = df_ols['is_intergenic'] * df_ols['m6a_per_kb']
df_ols['stress_x_young'] = df_ols['is_stress'] * df_ols['is_young']

y = df_ols['polya_length']
X = df_ols[['m6a_per_kb', 'read_length_z', 'is_stress', 'is_young',
            'is_intergenic', 'stress_x_m6a', 'young_x_m6a',
            'intergenic_x_m6a', 'stress_x_young']]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
ci = model.conf_int()

ols_rows = []
for var in model.params.index:
    ols_rows.append({
        'variable': var,
        'coef': model.params[var],
        'se': model.bse[var],
        't': model.tvalues[var],
        'p': model.pvalues[var],
        'ci_low': ci.loc[var, 0],
        'ci_high': ci.loc[var, 1],
    })

df_ols_out = pd.DataFrame(ols_rows)
out_f2 = f'{BASE}/topic_05_cellline/m6a_polya_stratified/m6a_polya_ols_full.tsv'
df_ols_out.to_csv(out_f2, sep='\t', index=False)
print(f"\n  OLS saved: {out_f2}")
print(f"  stress_x_m6a: coef={model.params['stress_x_m6a']:.3f}, P={model.pvalues['stress_x_m6a']:.2e}")
print(f"  R²={model.rsquared:.4f}, n={len(df_ols)}")

# --- Stratified correlations ---
strat_rows = []

def pearson_row(label, sub, stratification, condition, age, context):
    if len(sub) < 10:
        return None
    r, p = stats.pearsonr(sub['m6a_per_kb'], sub['polya_length'])
    return {
        'label': label,
        'n': len(sub),
        'r': r,
        'p': p,
        'polya_median': sub['polya_length'].median(),
        'm6a_kb_median': sub['m6a_per_kb'].median(),
        'm6a_kb_mean': sub['m6a_per_kb'].mean(),
        'polya_mean': sub['polya_length'].mean(),
        'stratification': stratification,
        'condition': condition,
        'age': age,
        'context': context,
    }

# Condition-level
for cond_lbl, is_s in [('normal', 0), ('stress', 1)]:
    sub = df_merged[df_merged['is_stress'] == is_s]
    row = pearson_row(f'condition={cond_lbl}', sub, 'condition', cond_lbl, 'all', 'all')
    if row: strat_rows.append(row)

# Condition x age
for cond_lbl, is_s in [('normal', 0), ('stress', 1)]:
    for age_lbl, is_y in [('young', True), ('ancient', False)]:
        sub = df_merged[(df_merged['is_stress'] == is_s) & (df_merged['is_young'] == is_y)]
        row = pearson_row(f'{cond_lbl}_{age_lbl}', sub, 'condition×age', cond_lbl, age_lbl, 'all')
        if row: strat_rows.append(row)

# Condition x context
for cond_lbl, is_s in [('normal', 0), ('stress', 1)]:
    for ctx_lbl, is_ig in [('intronic', 0), ('intergenic', 1)]:
        sub = df_merged[(df_merged['is_stress'] == is_s) & (df_merged['is_intergenic'] == is_ig)]
        row = pearson_row(f'{cond_lbl}_{ctx_lbl}', sub, 'condition×context', cond_lbl, 'all', ctx_lbl)
        if row: strat_rows.append(row)

# Condition x age x context
for cond_lbl, is_s in [('normal', 0), ('stress', 1)]:
    for age_lbl, is_y in [('young', True), ('ancient', False)]:
        for ctx_lbl, is_ig in [('intronic', 0), ('intergenic', 1)]:
            sub = df_merged[(df_merged['is_stress'] == is_s) &
                           (df_merged['is_young'] == is_y) &
                           (df_merged['is_intergenic'] == is_ig)]
            row = pearson_row(f'{cond_lbl}_{age_lbl}_{ctx_lbl}', sub,
                             'condition×age×context', cond_lbl, age_lbl, ctx_lbl)
            if row: strat_rows.append(row)

df_strat = pd.DataFrame(strat_rows)
out_f3 = f'{BASE}/topic_05_cellline/m6a_polya_stratified/m6a_polya_hela_stratified.tsv'
df_strat.to_csv(out_f3, sep='\t', index=False)
print(f"\n  Stratified saved: {out_f3}")
print(f"  Normal all: r={df_strat[df_strat['label']=='condition=normal']['r'].values[0]:.4f}")
print(f"  Stress all: r={df_strat[df_strat['label']=='condition=stress']['r'].values[0]:.4f}")

# =====================================================================
# SECTION 4: METTL3 KO summary at new threshold
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: METTL3 KO summary at new threshold")
print("=" * 70)

l1_ko = pd.read_csv(f'{METTL3}/ml204_l1_per_read.tsv', sep='\t')
ctrl_ko = pd.read_csv(f'{METTL3}/ml204_ctrl_per_read.tsv', sep='\t')

# Create summary in format matching old mettl3ko_matched_summary.tsv
summary_rows = []
for sample in l1_ko['sample'].unique():
    sub = l1_ko[l1_ko['sample'] == sample]
    cond = sub['condition'].iloc[0]
    rep = sample.split('_')[-1] if '_' in sample else sample
    summary_rows.append({
        'sample': sample,
        'condition': cond,
        'rep': rep,
        'n_reads': len(sub),
        'median_rl': sub['read_len'].median(),
        'm6a_per_kb_median': sub['m6a_per_kb'].median(),
        'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        'm6a_per_site_rate': sub['n_sites_above_thr'].sum() / sub['n_total_sites'].sum() if sub['n_total_sites'].sum() > 0 else 0,
    })
df_ko_summary = pd.DataFrame(summary_rows)

out_f4a = f'{METTL3}/ml204_l1_summary.tsv'
df_ko_summary.to_csv(out_f4a, sep='\t', index=False)
print(f"  L1 summary saved: {out_f4a}")

# Control summary
ctrl_summary_rows = []
for sample in ctrl_ko['sample'].unique():
    sub = ctrl_ko[ctrl_ko['sample'] == sample]
    cond = sub['condition'].iloc[0]
    rep = sample.split('_')[-1] if '_' in sample else sample
    ctrl_summary_rows.append({
        'sample': sample,
        'condition': cond,
        'rep': rep,
        'data_type': 'control',
        'n_reads': len(sub),
        'median_rl': sub['read_len'].median(),
        'm6a_per_kb_median': sub['m6a_per_kb'].median(),
        'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        'm6a_per_site_rate': sub['n_sites_above_thr'].sum() / sub['n_total_sites'].sum() if sub['n_total_sites'].sum() > 0 else 0,
    })
df_ctrl_ko_summary = pd.DataFrame(ctrl_summary_rows)

out_f4b = f'{METTL3}/ml204_ctrl_summary.tsv'
df_ctrl_ko_summary.to_csv(out_f4b, sep='\t', index=False)
print(f"  Ctrl summary saved: {out_f4b}")

wt_l1_m = df_ko_summary[df_ko_summary['condition'] == 'WT']['m6a_per_kb_median'].values
ko_l1_m = df_ko_summary[df_ko_summary['condition'] == 'KO']['m6a_per_kb_median'].values
wt_ctrl_m = df_ctrl_ko_summary[df_ctrl_ko_summary['condition'] == 'WT']['m6a_per_kb_median'].values
ko_ctrl_m = df_ctrl_ko_summary[df_ctrl_ko_summary['condition'] == 'KO']['m6a_per_kb_median'].values
print(f"  WT L1 m6A/kb median: {wt_l1_m}")
print(f"  KO L1 m6A/kb median: {ko_l1_m}")
print(f"  WT Ctrl m6A/kb median: {wt_ctrl_m}")
print(f"  KO Ctrl m6A/kb median: {ko_ctrl_m}")
print(f"  L1 FC={np.mean(ko_l1_m)/np.mean(wt_l1_m):.3f}, Ctrl FC={np.mean(ko_ctrl_m)/np.mean(wt_ctrl_m):.3f}")

# =====================================================================
# SECTION 5: Update l1_chromhmm_annotated.tsv m6A columns
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: Update l1_chromhmm_annotated.tsv m6A columns")
print("=" * 70)

chromhmm_f = f'{BASE}/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
df_chrom = pd.read_csv(chromhmm_f, sep='\t')
print(f"  Loaded: {len(df_chrom)} rows")

# Build read_id -> m6A lookup from ALL Part3 caches
m6a_lookup_frames = []
for f in sorted(glob.glob(f'{BASE}/topic_05_cellline/part3_l1_per_read_cache/*_l1_per_read.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    m6a_lookup_frames.append(tmp)
df_m6a_lookup = pd.concat(m6a_lookup_frames, ignore_index=True)
df_m6a_lookup['m6a_per_kb_new'] = df_m6a_lookup['m6a_sites_high'] / (df_m6a_lookup['read_length'] / 1000)
df_m6a_lookup = df_m6a_lookup[['read_id', 'm6a_sites_high', 'm6a_per_kb_new']].rename(
    columns={'m6a_sites_high': 'm6a_sites_high_new'})

n_before = len(df_chrom)
df_chrom = df_chrom.merge(df_m6a_lookup, on='read_id', how='left')

# Update columns
matched = df_chrom['m6a_per_kb_new'].notna()
print(f"  Matched: {matched.sum()}/{n_before} reads")
df_chrom.loc[matched, 'm6a_per_kb'] = df_chrom.loc[matched, 'm6a_per_kb_new']
df_chrom.loc[matched, 'm6a_sites_high'] = df_chrom.loc[matched, 'm6a_sites_high_new']
df_chrom.drop(columns=['m6a_per_kb_new', 'm6a_sites_high_new'], inplace=True)

df_chrom.to_csv(chromhmm_f, sep='\t', index=False)
print(f"  Updated: {chromhmm_f}")
print(f"  m6A/kb median={df_chrom['m6a_per_kb'].median():.3f}")

# =====================================================================
# SECTION 6: Regenerate body vs flanking summary (Fig S5)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6: Update body vs flanking summary from Part3 cache")
print("=" * 70)

# Load body vs flanking per-read data
bvf_f = f'{BASE}/topic_05_cellline/m6a_validation/m6a_body_vs_flanking_persite.tsv'
if os.path.exists(bvf_f):
    df_bvf = pd.read_csv(bvf_f, sep='\t')
    print(f"  Loaded per-read body/flanking: {len(df_bvf)} rows")

    # This file has per-SITE data, not per-read. It has columns:
    # read_id, site_pos, ml_prob, is_body, motif, ...
    # The summary is derived from this. Since the per-site data was parsed
    # from BAMs at threshold 128, we can't easily update it without re-parsing.
    # Instead, we'll regenerate the summary from Part3 cache approach.

    # For the summary, we need: L1 body rate, L1 flanking rate, Ctrl body rate, Ctrl flanking rate
    # The Part3 cache gives us total m6A per read, but not body vs flanking split.
    # So we need to use a different approach for this one.
    print("  NOTE: body_vs_flanking_persite requires BAM re-parsing for threshold update.")
    print("  Skipping - will need separate script or manual update.")
else:
    print("  File not found, skipping")

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
