#!/usr/bin/env python3
"""
Cross-cell-line ChromHMM switching analysis:
Do the SAME L1 loci show different m6A/poly(A) when in different chromatin states?

Logic:
1. Load per-read data, filter ancient L1 + normal condition
2. Cluster reads by locus (chr + gene_id + position window)
3. Find loci present in multiple cell lines with different chromhmm_group
4. Paired analysis: same locus, different chromatin states
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

OUT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/cross_cl_chromhmm_switching'

# ── 1. Load data ──────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv',
    sep='\t'
)

# Filter: ancient L1, normal condition only
# Cell lines with ChromHMM: A549(E114), HeLa(E117), HepG2(E118), K562(E123), H9(~E003)
# Note: MCF7, MCF7-EV, Hct116, HEYA8, SHSY5Y may not have matched EIDs
chromhmm_cls = ['A549', 'HeLa', 'HepG2', 'K562', 'H9']
mask = (df['l1_age'] == 'ancient') & (df['condition'] == 'normal')
df_anc = df[mask].copy()
print(f"Ancient normal reads: {len(df_anc)}")
print(f"Cell lines: {df_anc['cellline'].value_counts().to_dict()}")
print(f"ChromHMM groups: {df_anc['chromhmm_group'].value_counts().to_dict()}")

# ── 2. Define locus ID ───────────────────────────────────────
# Use chr + gene_id as primary locus identifier (L1 subfamily name at that location)
# But gene_id is subfamily name (L1MC4, L1PA3...), not unique per locus
# Better: cluster by chr + midpoint position (±500bp)

df_anc['midpoint'] = (df_anc['start'] + df_anc['end']) // 2
# Round midpoint to nearest 1000bp to cluster nearby reads
df_anc['locus_bin'] = df_anc['chr'] + ':' + (df_anc['midpoint'] // 1000 * 1000).astype(str)

# More precise: group by chr + gene_id + position bin
df_anc['locus_id'] = df_anc['chr'] + ':' + df_anc['gene_id'] + ':' + (df_anc['midpoint'] // 1000 * 1000).astype(str)

print(f"\nUnique loci (chr:gene_id:pos_bin): {df_anc['locus_id'].nunique()}")

# ── 3. Find state-switching loci ─────────────────────────────
# For each locus, find which chromhmm_groups it appears in across cell lines
locus_states = df_anc.groupby('locus_id').agg(
    n_reads=('read_id', 'count'),
    n_celllines=('cellline', 'nunique'),
    celllines=('cellline', lambda x: ','.join(sorted(x.unique()))),
    chromhmm_groups=('chromhmm_group', lambda x: ','.join(sorted(x.unique()))),
    n_states=('chromhmm_group', 'nunique'),
).reset_index()

# Loci in multiple cell lines
multi_cl = locus_states[locus_states['n_celllines'] >= 2]
print(f"\nLoci in ≥2 cell lines: {len(multi_cl)}")

# Loci with different chromhmm states
switching = locus_states[(locus_states['n_celllines'] >= 2) & (locus_states['n_states'] >= 2)]
print(f"State-switching loci (≥2 CL, ≥2 states): {len(switching)}")
print(f"Total reads at switching loci: {df_anc[df_anc['locus_id'].isin(switching['locus_id'])].shape[0]}")

# Show distribution
print("\nSwitching loci state combinations (top 20):")
print(switching['chromhmm_groups'].value_counts().head(20))

# ── 4. Paired analysis: Regulatory vs Non-Regulatory ─────────
# For switching loci, compute per-locus per-state medians
switching_ids = set(switching['locus_id'])
df_switch = df_anc[df_anc['locus_id'].isin(switching_ids)].copy()

# Define regulatory vs non-regulatory
df_switch['is_reg'] = df_switch['chromhmm_group'].isin(['Enhancer', 'Promoter'])

# Per-locus per-state aggregation
locus_state_agg = df_switch.groupby(['locus_id', 'is_reg']).agg(
    median_polya=('polya_length', 'median'),
    mean_polya=('polya_length', 'mean'),
    median_m6a=('m6a_per_kb', 'median'),
    mean_m6a=('m6a_per_kb', 'mean'),
    n_reads=('read_id', 'count'),
).reset_index()

# Pivot: need loci that have BOTH regulatory and non-regulatory reads
pivot_polya = locus_state_agg.pivot(index='locus_id', columns='is_reg', values='median_polya').dropna()
pivot_m6a = locus_state_agg.pivot(index='locus_id', columns='is_reg', values='median_m6a').dropna()
pivot_n = locus_state_agg.pivot(index='locus_id', columns='is_reg', values='n_reads').dropna()

# Require ≥3 reads in each state for reliability
min_reads = 3
n_info = locus_state_agg.pivot(index='locus_id', columns='is_reg', values='n_reads').dropna()
valid_loci = n_info[(n_info[True] >= min_reads) & (n_info[False] >= min_reads)].index

pivot_polya_v = pivot_polya.loc[pivot_polya.index.isin(valid_loci)]
pivot_m6a_v = pivot_m6a.loc[pivot_m6a.index.isin(valid_loci)]

print(f"\n═══ PAIRED ANALYSIS: Same Locus, Regulatory vs Non-Regulatory ═══")
print(f"Loci with both states (any reads): {len(pivot_polya)}")
print(f"Loci with ≥{min_reads} reads each state: {len(pivot_polya_v)}")

if len(pivot_polya_v) >= 5:
    reg_polya = pivot_polya_v[True].values
    nonreg_polya = pivot_polya_v[False].values

    # Wilcoxon signed-rank (paired)
    stat_w, pval_w = stats.wilcoxon(reg_polya, nonreg_polya)
    diff_polya = reg_polya - nonreg_polya

    print(f"\n--- Poly(A) length ---")
    print(f"Regulatory median: {np.median(reg_polya):.1f} nt")
    print(f"Non-regulatory median: {np.median(nonreg_polya):.1f} nt")
    print(f"Paired difference (Reg - NonReg): median={np.median(diff_polya):.1f}, mean={np.mean(diff_polya):.1f} nt")
    print(f"Wilcoxon signed-rank: W={stat_w:.0f}, P={pval_w:.2e}")

    reg_m6a = pivot_m6a_v[True].values
    nonreg_m6a = pivot_m6a_v[False].values
    stat_w2, pval_w2 = stats.wilcoxon(reg_m6a, nonreg_m6a)
    diff_m6a = reg_m6a - nonreg_m6a

    print(f"\n--- m6A/kb ---")
    print(f"Regulatory median: {np.median(reg_m6a):.2f}")
    print(f"Non-regulatory median: {np.median(nonreg_m6a):.2f}")
    print(f"Paired difference (Reg - NonReg): median={np.median(diff_m6a):.2f}, mean={np.mean(diff_m6a):.2f}")
    print(f"Wilcoxon signed-rank: W={stat_w2:.0f}, P={pval_w2:.2e}")
else:
    print("Too few loci with ≥3 reads in both states for paired analysis")

# ── 4b. More granular: all 4 states ─────────────────────────
print(f"\n═══ GRANULAR STATE COMPARISON (same-locus switching) ═══")

# For loci switching between specific state pairs
state_pairs = [
    ('Enhancer', 'Quiescent'),
    ('Enhancer', 'Transcribed'),
    ('Promoter', 'Quiescent'),
    ('Promoter', 'Transcribed'),
    ('Transcribed', 'Quiescent'),
]

pair_results = []
for s1, s2 in state_pairs:
    sub = df_switch[df_switch['chromhmm_group'].isin([s1, s2])]
    locus_agg = sub.groupby(['locus_id', 'chromhmm_group']).agg(
        median_polya=('polya_length', 'median'),
        median_m6a=('m6a_per_kb', 'median'),
        n=('read_id', 'count'),
    ).reset_index()

    piv = locus_agg.pivot(index='locus_id', columns='chromhmm_group', values='median_polya').dropna()
    piv_n = locus_agg.pivot(index='locus_id', columns='chromhmm_group', values='n').dropna()
    piv_m = locus_agg.pivot(index='locus_id', columns='chromhmm_group', values='median_m6a').dropna()

    # At least 2 reads each
    if s1 in piv_n.columns and s2 in piv_n.columns:
        valid = piv_n[(piv_n[s1] >= 2) & (piv_n[s2] >= 2)].index
        piv_v = piv.loc[piv.index.isin(valid)]
        piv_mv = piv_m.loc[piv_m.index.isin(valid)]

        if len(piv_v) >= 3:
            d_polya = piv_v[s1].values - piv_v[s2].values
            d_m6a = piv_mv[s1].values - piv_mv[s2].values
            _, p_polya = stats.wilcoxon(piv_v[s1].values, piv_v[s2].values) if len(piv_v) >= 5 else (0, np.nan)
            _, p_m6a = stats.wilcoxon(piv_mv[s1].values, piv_mv[s2].values) if len(piv_mv) >= 5 else (0, np.nan)

            res = {
                'pair': f"{s1} vs {s2}",
                'n_loci': len(piv_v),
                'polya_s1': np.median(piv_v[s1].values),
                'polya_s2': np.median(piv_v[s2].values),
                'polya_diff': np.median(d_polya),
                'polya_p': p_polya,
                'm6a_s1': np.median(piv_mv[s1].values),
                'm6a_s2': np.median(piv_mv[s2].values),
                'm6a_diff': np.median(d_m6a),
                'm6a_p': p_m6a,
            }
            pair_results.append(res)
            print(f"\n{s1} vs {s2}: {len(piv_v)} loci")
            print(f"  Poly(A): {res['polya_s1']:.1f} vs {res['polya_s2']:.1f} (Δ={res['polya_diff']:.1f}, P={p_polya:.2e})")
            print(f"  m6A/kb:  {res['m6a_s1']:.2f} vs {res['m6a_s2']:.2f} (Δ={res['m6a_diff']:.2f}, P={p_m6a:.2e})")
        else:
            print(f"\n{s1} vs {s2}: <3 valid loci")

# ── 5. ALL loci: chromatin activity score correlation ─────────
print(f"\n═══ ALL LOCI: CHROMATIN ACTIVITY SCORE CORRELATION ═══")

activity_map = {
    'Quiescent': 1,
    'Transcribed': 2,
    'Enhancer': 3,
    'Promoter': 3,
}

df_anc['activity_score'] = df_anc['chromhmm_group'].map(activity_map)
df_valid = df_anc.dropna(subset=['activity_score', 'polya_length', 'm6a_per_kb'])

# Per-read correlation
r_polya, p_polya = stats.spearmanr(df_valid['activity_score'], df_valid['polya_length'])
r_m6a, p_m6a = stats.spearmanr(df_valid['activity_score'], df_valid['m6a_per_kb'])
print(f"Per-read (N={len(df_valid)}):")
print(f"  Activity vs poly(A): rho={r_polya:.4f}, P={p_polya:.2e}")
print(f"  Activity vs m6A/kb:  rho={r_m6a:.4f}, P={p_m6a:.2e}")

# Per-locus aggregation
locus_agg_all = df_valid.groupby('locus_id').agg(
    median_polya=('polya_length', 'median'),
    median_m6a=('m6a_per_kb', 'median'),
    mean_activity=('activity_score', 'mean'),
    n_reads=('read_id', 'count'),
).reset_index()

locus_5 = locus_agg_all[locus_agg_all['n_reads'] >= 5]
r_polya_l, p_polya_l = stats.spearmanr(locus_5['mean_activity'], locus_5['median_polya'])
r_m6a_l, p_m6a_l = stats.spearmanr(locus_5['mean_activity'], locus_5['median_m6a'])
print(f"\nPer-locus ≥5 reads (N={len(locus_5)}):")
print(f"  Activity vs poly(A): rho={r_polya_l:.4f}, P={p_polya_l:.2e}")
print(f"  Activity vs m6A/kb:  rho={r_m6a_l:.4f}, P={p_m6a_l:.2e}")

# ── 5b. Per-state group means (all reads) ────────────────────
print(f"\n--- Per-state group statistics (all ancient normal reads) ---")
state_stats = df_anc.groupby('chromhmm_group').agg(
    n=('read_id', 'count'),
    polya_median=('polya_length', 'median'),
    polya_mean=('polya_length', 'mean'),
    m6a_median=('m6a_per_kb', 'median'),
    m6a_mean=('m6a_per_kb', 'mean'),
).reset_index()
print(state_stats.to_string(index=False))

# ── 6. Within-cell-line analysis ─────────────────────────────
# To control for cell-line batch effects, look at switching WITHIN each cell line
# This is less powerful but cleaner
print(f"\n═══ WITHIN-CELL-LINE CHROMATIN EFFECT ═══")

for cl in sorted(df_anc['cellline'].unique()):
    sub = df_anc[df_anc['cellline'] == cl]
    if sub['chromhmm_group'].nunique() < 2:
        continue
    n_reads = len(sub)

    # Compare Regulatory vs non-Regulatory within this cell line
    reg = sub[sub['chromhmm_group'].isin(['Enhancer', 'Promoter'])]
    nonreg = sub[~sub['chromhmm_group'].isin(['Enhancer', 'Promoter'])]

    if len(reg) >= 10 and len(nonreg) >= 10:
        u_polya, p_polya = stats.mannwhitneyu(reg['polya_length'], nonreg['polya_length'], alternative='two-sided')
        u_m6a, p_m6a = stats.mannwhitneyu(reg['m6a_per_kb'], nonreg['m6a_per_kb'], alternative='two-sided')

        print(f"\n{cl} (N={n_reads}, Reg={len(reg)}, NonReg={len(nonreg)}):")
        print(f"  Poly(A): Reg {reg['polya_length'].median():.1f} vs NonReg {nonreg['polya_length'].median():.1f} (P={p_polya:.2e})")
        print(f"  m6A/kb:  Reg {reg['m6a_per_kb'].median():.2f} vs NonReg {nonreg['m6a_per_kb'].median():.2f} (P={p_m6a:.2e})")

# ── 7. Key test: Same locus, different CL, different state ──
# The strongest test: find a locus in CL-A with state X and CL-B with state Y
print(f"\n═══ STRONGEST TEST: Same locus across cell lines with different states ═══")

# Per cell-line per-locus dominant state
cl_locus = df_anc.groupby(['locus_id', 'cellline']).agg(
    dominant_state=('chromhmm_group', lambda x: x.mode().iloc[0]),
    median_polya=('polya_length', 'median'),
    median_m6a=('m6a_per_kb', 'median'),
    n_reads=('read_id', 'count'),
).reset_index()

# Find loci in ≥2 cell lines
loci_multi = cl_locus.groupby('locus_id').filter(lambda x: len(x) >= 2 and x['dominant_state'].nunique() >= 2)
n_switching_loci = loci_multi['locus_id'].nunique()
print(f"Loci in ≥2 CLs with different dominant states: {n_switching_loci}")

if n_switching_loci >= 5:
    # For each switching locus, compute within-locus state effect
    locus_effects = []
    for lid, grp in loci_multi.groupby('locus_id'):
        # Classify each CL observation as regulatory or not
        grp = grp.copy()
        grp['is_reg'] = grp['dominant_state'].isin(['Enhancer', 'Promoter'])

        if grp['is_reg'].nunique() == 2:  # Has both reg and non-reg across CLs
            reg_vals = grp[grp['is_reg']]
            nonreg_vals = grp[~grp['is_reg']]

            locus_effects.append({
                'locus_id': lid,
                'reg_polya': reg_vals['median_polya'].mean(),
                'nonreg_polya': nonreg_vals['median_polya'].mean(),
                'reg_m6a': reg_vals['median_m6a'].mean(),
                'nonreg_m6a': nonreg_vals['median_m6a'].mean(),
                'reg_cls': ','.join(reg_vals['cellline'].values),
                'nonreg_cls': ','.join(nonreg_vals['cellline'].values),
                'reg_states': ','.join(reg_vals['dominant_state'].values),
                'nonreg_states': ','.join(nonreg_vals['dominant_state'].values),
            })

    if locus_effects:
        eff_df = pd.DataFrame(locus_effects)

        # Paired test
        print(f"\nPaired loci with Reg vs NonReg across CLs: {len(eff_df)}")

        if len(eff_df) >= 5:
            stat_p, pval_p = stats.wilcoxon(eff_df['reg_polya'], eff_df['nonreg_polya'])
            stat_m, pval_m = stats.wilcoxon(eff_df['reg_m6a'], eff_df['nonreg_m6a'])
        else:
            _, pval_p = stats.ttest_rel(eff_df['reg_polya'], eff_df['nonreg_polya'])
            _, pval_m = stats.ttest_rel(eff_df['reg_m6a'], eff_df['nonreg_m6a'])

        diff_p = eff_df['reg_polya'] - eff_df['nonreg_polya']
        diff_m = eff_df['reg_m6a'] - eff_df['nonreg_m6a']

        print(f"\nPoly(A): Reg={eff_df['reg_polya'].median():.1f} vs NonReg={eff_df['nonreg_polya'].median():.1f}")
        print(f"  Paired Δ: median={diff_p.median():.1f}, mean={diff_p.mean():.1f}")
        print(f"  P={pval_p:.2e}")

        print(f"\nm6A/kb: Reg={eff_df['reg_m6a'].median():.2f} vs NonReg={eff_df['nonreg_m6a'].median():.2f}")
        print(f"  Paired Δ: median={diff_m.median():.2f}, mean={diff_m.mean():.2f}")
        print(f"  P={pval_m:.2e}")

        # Save switching loci details
        eff_df.to_csv(os.path.join(OUT, 'switching_loci_details.tsv'), sep='\t', index=False)
        print(f"\nSaved: switching_loci_details.tsv")

# ── 8. Relaxed locus matching (±2kb window) ──────────────────
print(f"\n═══ RELAXED MATCHING (±2kb window) ═══")

# Re-bin with 2kb windows
df_anc['locus_2kb'] = df_anc['chr'] + ':' + df_anc['gene_id'] + ':' + (df_anc['midpoint'] // 2000 * 2000).astype(str)

cl_locus2 = df_anc.groupby(['locus_2kb', 'cellline']).agg(
    dominant_state=('chromhmm_group', lambda x: x.mode().iloc[0]),
    median_polya=('polya_length', 'median'),
    median_m6a=('m6a_per_kb', 'median'),
    n_reads=('read_id', 'count'),
).reset_index()

# Filter ≥2 reads per cell line per locus
cl_locus2 = cl_locus2[cl_locus2['n_reads'] >= 2]

loci_multi2 = cl_locus2.groupby('locus_2kb').filter(lambda x: len(x) >= 2 and x['dominant_state'].nunique() >= 2)
n_switching2 = loci_multi2['locus_2kb'].nunique()
print(f"Switching loci (2kb bins, ≥2 reads/CL): {n_switching2}")

if n_switching2 >= 5:
    locus_effects2 = []
    for lid, grp in loci_multi2.groupby('locus_2kb'):
        grp = grp.copy()
        grp['is_reg'] = grp['dominant_state'].isin(['Enhancer', 'Promoter'])
        if grp['is_reg'].nunique() == 2:
            reg_vals = grp[grp['is_reg']]
            nonreg_vals = grp[~grp['is_reg']]
            locus_effects2.append({
                'locus_id': lid,
                'reg_polya': reg_vals['median_polya'].mean(),
                'nonreg_polya': nonreg_vals['median_polya'].mean(),
                'reg_m6a': reg_vals['median_m6a'].mean(),
                'nonreg_m6a': nonreg_vals['median_m6a'].mean(),
                'n_cls': len(grp),
            })

    if len(locus_effects2) >= 5:
        eff2 = pd.DataFrame(locus_effects2)
        stat_p2, pval_p2 = stats.wilcoxon(eff2['reg_polya'], eff2['nonreg_polya'])
        stat_m2, pval_m2 = stats.wilcoxon(eff2['reg_m6a'], eff2['nonreg_m6a'])

        diff_p2 = eff2['reg_polya'] - eff2['nonreg_polya']
        diff_m2 = eff2['reg_m6a'] - eff2['nonreg_m6a']

        print(f"Paired loci (Reg vs NonReg): {len(eff2)}")
        print(f"Poly(A): Reg={eff2['reg_polya'].median():.1f} vs NonReg={eff2['nonreg_polya'].median():.1f}")
        print(f"  Δ median={diff_p2.median():.1f}, P={pval_p2:.2e}")
        print(f"m6A/kb: Reg={eff2['reg_m6a'].median():.2f} vs NonReg={eff2['nonreg_m6a'].median():.2f}")
        print(f"  Δ median={diff_m2.median():.2f}, P={pval_m2:.2e}")

        eff2.to_csv(os.path.join(OUT, 'switching_loci_2kb_details.tsv'), sep='\t', index=False)

# ── 9. Alternative: use cross-CL annotated file ─────────────
print(f"\n═══ CROSS-CL FILE ANALYSIS ═══")
df_cross = pd.read_csv(
    '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/cross_cl_chromhmm_annotated.tsv',
    sep='\t'
)
df_cross_anc = df_cross[(df_cross['l1_age'] == 'ancient') & (df_cross['condition'] == 'normal')].copy()
print(f"Cross-CL ancient normal: {len(df_cross_anc)} reads")
print(f"Cell lines: {df_cross_anc['cellline'].value_counts().to_dict()}")

# Same locus identification
df_cross_anc['midpoint'] = (df_cross_anc['start'] + df_cross_anc['end']) // 2
df_cross_anc['locus_id'] = df_cross_anc['chr'] + ':' + df_cross_anc['gene_id'] + ':' + (df_cross_anc['midpoint'] // 2000 * 2000).astype(str)

cl_locus_c = df_cross_anc.groupby(['locus_id', 'cellline']).agg(
    dominant_state=('chromhmm_group', lambda x: x.mode().iloc[0]),
    median_polya=('polya_length', 'median'),
    median_m6a=('m6a_per_kb', 'median'),
    n_reads=('read_id', 'count'),
    is_reg_frac=('is_regulatory', 'mean'),
).reset_index()

cl_locus_c = cl_locus_c[cl_locus_c['n_reads'] >= 2]
switch_c = cl_locus_c.groupby('locus_id').filter(lambda x: len(x) >= 2 and x['dominant_state'].nunique() >= 2)
n_switch_c = switch_c['locus_id'].nunique()
print(f"Switching loci (cross-CL file): {n_switch_c}")

if n_switch_c >= 5:
    effects_c = []
    for lid, grp in switch_c.groupby('locus_id'):
        grp = grp.copy()
        grp['is_reg'] = grp['dominant_state'].isin(['Enhancer', 'Promoter'])
        if grp['is_reg'].nunique() == 2:
            reg = grp[grp['is_reg']]
            nonreg = grp[~grp['is_reg']]
            effects_c.append({
                'locus_id': lid,
                'reg_polya': reg['median_polya'].mean(),
                'nonreg_polya': nonreg['median_polya'].mean(),
                'reg_m6a': reg['median_m6a'].mean(),
                'nonreg_m6a': nonreg['median_m6a'].mean(),
            })

    if len(effects_c) >= 5:
        efc = pd.DataFrame(effects_c)
        _, p_p = stats.wilcoxon(efc['reg_polya'], efc['nonreg_polya'])
        _, p_m = stats.wilcoxon(efc['reg_m6a'], efc['nonreg_m6a'])
        dp = efc['reg_polya'] - efc['nonreg_polya']
        dm = efc['reg_m6a'] - efc['nonreg_m6a']

        print(f"Paired Reg vs NonReg loci: {len(efc)}")
        print(f"Poly(A): Δ={dp.median():.1f}, P={p_p:.2e}")
        print(f"m6A/kb: Δ={dm.median():.2f}, P={p_m:.2e}")

# ── 10. Figures ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 10a. Per-state boxplot (all reads)
ax = axes[0, 0]
state_order = ['Quiescent', 'Transcribed', 'Enhancer', 'Promoter']
state_colors = {'Quiescent': '#bdbdbd', 'Transcribed': '#66c2a5', 'Enhancer': '#fc8d62', 'Promoter': '#e78ac3'}
bp_data = [df_anc[df_anc['chromhmm_group'] == s]['polya_length'].values for s in state_order]
bp = ax.boxplot(bp_data, labels=state_order, patch_artist=True, showfliers=False)
for patch, s in zip(bp['boxes'], state_order):
    patch.set_facecolor(state_colors[s])
for i, s in enumerate(state_order):
    n = len(bp_data[i])
    ax.text(i+1, ax.get_ylim()[0] + 2, f'n={n}', ha='center', fontsize=8)
ax.set_ylabel('Poly(A) length (nt)')
ax.set_title('a) Poly(A) by chromatin state\n(all ancient L1, normal)')

# 10b. Per-state boxplot m6A
ax = axes[0, 1]
bp_data_m = [df_anc[df_anc['chromhmm_group'] == s]['m6a_per_kb'].values for s in state_order]
bp = ax.boxplot(bp_data_m, labels=state_order, patch_artist=True, showfliers=False)
for patch, s in zip(bp['boxes'], state_order):
    patch.set_facecolor(state_colors[s])
for i, s in enumerate(state_order):
    n = len(bp_data_m[i])
    ax.text(i+1, ax.get_ylim()[0] + 0.05, f'n={n}', ha='center', fontsize=8)
ax.set_ylabel('m6A/kb')
ax.set_title('b) m6A/kb by chromatin state\n(all ancient L1, normal)')

# 10c. Switching loci paired plot (poly(A))
ax = axes[1, 0]
# Use whichever switching dataset has most loci
if 'eff2' in dir() and len(eff2) >= 5:
    plot_df = eff2
    title_suffix = '(2kb bins)'
elif 'eff_df' in dir() and len(eff_df) >= 3:
    plot_df = eff_df
    title_suffix = '(1kb bins)'
else:
    plot_df = None

if plot_df is not None and len(plot_df) >= 3:
    for _, row in plot_df.iterrows():
        ax.plot([0, 1], [row['nonreg_polya'], row['reg_polya']], 'o-', color='gray', alpha=0.3, markersize=3)
    ax.plot([0, 1], [plot_df['nonreg_polya'].median(), plot_df['reg_polya'].median()], 'o-', color='red', linewidth=2, markersize=8, zorder=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Regulatory', 'Regulatory'])
    ax.set_ylabel('Median poly(A) (nt)')
    # Add p-value
    try:
        dp_plot = plot_df['reg_polya'] - plot_df['nonreg_polya']
        _, p_plot = stats.wilcoxon(plot_df['reg_polya'], plot_df['nonreg_polya'])
        ax.set_title(f'c) Same-locus poly(A) switching {title_suffix}\nΔ={dp_plot.median():.1f}nt, P={p_plot:.2e}, n={len(plot_df)}')
    except:
        ax.set_title(f'c) Same-locus poly(A) switching {title_suffix}\nn={len(plot_df)}')
else:
    ax.text(0.5, 0.5, 'Insufficient\nswitching loci', transform=ax.transAxes, ha='center')
    ax.set_title('c) Same-locus poly(A) switching')

# 10d. Switching loci paired plot (m6A)
ax = axes[1, 1]
if plot_df is not None and len(plot_df) >= 3:
    for _, row in plot_df.iterrows():
        ax.plot([0, 1], [row['nonreg_m6a'], row['reg_m6a']], 'o-', color='gray', alpha=0.3, markersize=3)
    ax.plot([0, 1], [plot_df['nonreg_m6a'].median(), plot_df['reg_m6a'].median()], 'o-', color='red', linewidth=2, markersize=8, zorder=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Regulatory', 'Regulatory'])
    ax.set_ylabel('Median m6A/kb')
    try:
        dm_plot = plot_df['reg_m6a'] - plot_df['nonreg_m6a']
        _, p_m_plot = stats.wilcoxon(plot_df['reg_m6a'], plot_df['nonreg_m6a'])
        ax.set_title(f'd) Same-locus m6A switching {title_suffix}\nΔ={dm_plot.median():.2f}, P={p_m_plot:.2e}, n={len(plot_df)}')
    except:
        ax.set_title(f'd) Same-locus m6A switching {title_suffix}\nn={len(plot_df)}')
else:
    ax.text(0.5, 0.5, 'Insufficient\nswitching loci', transform=ax.transAxes, ha='center')
    ax.set_title('d) Same-locus m6A switching')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chromhmm_switching_analysis.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'chromhmm_switching_analysis.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: chromhmm_switching_analysis.pdf/png")

# ── 11. Summary table ────────────────────────────────────────
summary_rows = []

# All-reads per-state
for s in state_order:
    sub = df_anc[df_anc['chromhmm_group'] == s]
    summary_rows.append({
        'analysis': 'all_reads_per_state',
        'category': s,
        'n': len(sub),
        'polya_median': sub['polya_length'].median(),
        'm6a_median': sub['m6a_per_kb'].median(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT, 'summary_stats.tsv'), sep='\t', index=False)

# ── 12. Mixed-effects check: within-CL per-read regression ──
print(f"\n═══ WITHIN-CL REGRESSION (controlling for cell line) ═══")
# Simple approach: compute per-CL Spearman, then meta-analyze
cl_rhos = []
for cl in sorted(df_anc['cellline'].unique()):
    sub = df_anc[df_anc['cellline'] == cl].dropna(subset=['activity_score'])
    if len(sub) >= 30 and sub['activity_score'].nunique() >= 2:
        r_p, p_p = stats.spearmanr(sub['activity_score'], sub['polya_length'])
        r_m, p_m = stats.spearmanr(sub['activity_score'], sub['m6a_per_kb'])
        cl_rhos.append({'cellline': cl, 'n': len(sub),
                       'rho_polya': r_p, 'p_polya': p_p,
                       'rho_m6a': r_m, 'p_m6a': p_m})
        print(f"  {cl} (n={len(sub)}): polya rho={r_p:.3f} P={p_p:.2e}, m6a rho={r_m:.3f} P={p_m:.2e}")

if cl_rhos:
    cl_df = pd.DataFrame(cl_rhos)
    # Fisher's method for meta-analysis
    from scipy.stats import combine_pvalues
    _, meta_p_polya = combine_pvalues(cl_df['p_polya'].values, method='fisher')
    _, meta_p_m6a = combine_pvalues(cl_df['p_m6a'].values, method='fisher')
    mean_rho_polya = cl_df['rho_polya'].mean()
    mean_rho_m6a = cl_df['rho_m6a'].mean()
    print(f"\nMeta-analysis (Fisher):")
    print(f"  Poly(A) ~ activity: mean rho={mean_rho_polya:.4f}, meta P={meta_p_polya:.2e}")
    print(f"  m6A ~ activity:     mean rho={mean_rho_m6a:.4f}, meta P={meta_p_m6a:.2e}")

    cl_df.to_csv(os.path.join(OUT, 'within_cl_correlations.tsv'), sep='\t', index=False)

print("\n═══ ANALYSIS COMPLETE ═══")
