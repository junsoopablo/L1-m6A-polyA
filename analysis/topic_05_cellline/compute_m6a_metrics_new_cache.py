#!/usr/bin/env python3
"""
Comprehensive m6A metrics using NEW Part3 cache (threshold 0.80, ML>=204).
Computes per-CL m6A/kb, pooled stats, Young vs Ancient, per-site rate,
cross-CL consistency, HeLa-Ars dose-response, and decomposition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
L1_CACHE = BASE / "analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache"
CTRL_CACHE = BASE / "analysis/01_exploration/topic_05_cellline/part3_ctrl_per_read_cache"
RESULTS = BASE / "results_group"

# ── Cell line definitions ──
CL_GROUPS = {
    'A549':   ['A549_4', 'A549_5', 'A549_6'],
    'H9':     ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa':   ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2':  ['HepG2_5', 'HepG2_6'],
    'HEYA8':  ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562':   ['K562_4', 'K562_5', 'K562_6'],
    'MCF7':   ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}
BASE_CLS = [cl for cl in CL_GROUPS if cl != 'HeLa-Ars']
YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}


def load_cache(cache_dir, groups, suffix):
    """Load per-read cache files for a set of groups."""
    frames = []
    for g in groups:
        fpath = cache_dir / f"{g}_{suffix}_per_read.tsv"
        if fpath.exists():
            df = pd.read_csv(fpath, sep='\t')
            df['group'] = g
            frames.append(df)
        else:
            print(f"  WARNING: {fpath} not found")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_l1_summary(groups):
    """Load L1 summary files to get subfamily + poly(A) info."""
    frames = []
    for g in groups:
        fpath = RESULTS / g / "g_summary" / f"{g}_L1_summary.tsv"
        if fpath.exists():
            df = pd.read_csv(fpath, sep='\t')
            df['group'] = g
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def compute_m6a_per_kb(df):
    """Compute m6A sites per kilobase. Returns per-read m6a/kb and aggregate."""
    df = df[df['read_length'] > 0].copy()
    df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000.0)
    total_sites = df['m6a_sites_high'].sum()
    total_kb = df['read_length'].sum() / 1000.0
    agg_m6a_per_kb = total_sites / total_kb if total_kb > 0 else 0
    return df, agg_m6a_per_kb, total_sites, total_kb


# ═════════════════════════════════════════════════════════════
# SECTION 1: Per-CL m6A/kb for L1 and Ctrl
# ═════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 1: Per-CL m6A/kb (L1 vs Ctrl)")
print("=" * 80)

cl_results = {}
for cl, groups in CL_GROUPS.items():
    l1_df = load_cache(L1_CACHE, groups, 'l1')
    ctrl_df = load_cache(CTRL_CACHE, groups, 'ctrl')

    l1_df, l1_mkb, l1_sites, l1_kb = compute_m6a_per_kb(l1_df)
    ctrl_df, ctrl_mkb, ctrl_sites, ctrl_kb = compute_m6a_per_kb(ctrl_df)

    ratio = l1_mkb / ctrl_mkb if ctrl_mkb > 0 else float('nan')

    cl_results[cl] = {
        'l1_m6a_kb': l1_mkb, 'ctrl_m6a_kb': ctrl_mkb, 'ratio': ratio,
        'l1_reads': len(l1_df), 'ctrl_reads': len(ctrl_df),
        'l1_sites': l1_sites, 'ctrl_sites': ctrl_sites,
        'l1_total_kb': l1_kb, 'ctrl_total_kb': ctrl_kb,
        'l1_df': l1_df, 'ctrl_df': ctrl_df,
    }

print(f"\n{'CL':<12} {'L1 m6A/kb':>10} {'Ctrl m6A/kb':>12} {'Ratio':>7} {'L1 reads':>9} {'Ctrl reads':>11}")
print("-" * 65)
for cl in CL_GROUPS:
    r = cl_results[cl]
    print(f"{cl:<12} {r['l1_m6a_kb']:>10.3f} {r['ctrl_m6a_kb']:>12.3f} {r['ratio']:>7.3f} {r['l1_reads']:>9,} {r['ctrl_reads']:>11,}")


# ═════════════════════════════════════════════════════════════
# SECTION 2: Pooled across 9 base CL (exclude HeLa-Ars)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 2: Pooled across 9 base CL (excl. HeLa-Ars)")
print("=" * 80)

all_l1 = pd.concat([cl_results[cl]['l1_df'] for cl in BASE_CLS], ignore_index=True)
all_ctrl = pd.concat([cl_results[cl]['ctrl_df'] for cl in BASE_CLS], ignore_index=True)

all_l1['m6a_per_kb'] = all_l1['m6a_sites_high'] / (all_l1['read_length'] / 1000.0)
all_ctrl['m6a_per_kb'] = all_ctrl['m6a_sites_high'] / (all_ctrl['read_length'] / 1000.0)

pool_l1_mkb = all_l1['m6a_sites_high'].sum() / (all_l1['read_length'].sum() / 1000.0)
pool_ctrl_mkb = all_ctrl['m6a_sites_high'].sum() / (all_ctrl['read_length'].sum() / 1000.0)
pool_ratio = pool_l1_mkb / pool_ctrl_mkb

# MWU on per-read m6a/kb
mwu_stat, mwu_p = stats.mannwhitneyu(all_l1['m6a_per_kb'], all_ctrl['m6a_per_kb'], alternative='two-sided')

print(f"\nPooled L1 m6A/kb:   {pool_l1_mkb:.3f}  (N={len(all_l1):,} reads, {all_l1['m6a_sites_high'].sum():,} sites)")
print(f"Pooled Ctrl m6A/kb: {pool_ctrl_mkb:.3f}  (N={len(all_ctrl):,} reads, {all_ctrl['m6a_sites_high'].sum():,} sites)")
print(f"L1/Ctrl ratio:      {pool_ratio:.3f}")
print(f"MWU P-value:        {mwu_p:.2e}")
print(f"MWU U-stat:         {mwu_stat:.0f}")


# ═════════════════════════════════════════════════════════════
# SECTION 3: Young vs Ancient L1 m6A/kb (pooled 9 base CL)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 3: Young vs Ancient L1 m6A/kb (pooled 9 base CL)")
print("=" * 80)

# Load all L1 summaries for base CL to get subfamily
all_groups_base = []
for cl in BASE_CLS:
    all_groups_base.extend(CL_GROUPS[cl])

l1_summary = load_l1_summary(all_groups_base)
l1_summary['is_young'] = l1_summary['transcript_id'].apply(
    lambda x: any(x.startswith(sf) for sf in YOUNG_SUBFAMILIES) if isinstance(x, str) else False
)

# Merge L1 cache with summary to get subfamily
all_l1_with_sf = all_l1.merge(
    l1_summary[['read_id', 'is_young', 'transcript_id', 'polya_length', 'qc_tag']],
    on='read_id', how='left'
)

young_l1 = all_l1_with_sf[all_l1_with_sf['is_young'] == True]
ancient_l1 = all_l1_with_sf[all_l1_with_sf['is_young'] == False]

young_mkb = young_l1['m6a_sites_high'].sum() / (young_l1['read_length'].sum() / 1000.0) if len(young_l1) > 0 else 0
ancient_mkb = ancient_l1['m6a_sites_high'].sum() / (ancient_l1['read_length'].sum() / 1000.0) if len(ancient_l1) > 0 else 0
ya_ratio = young_mkb / ancient_mkb if ancient_mkb > 0 else float('nan')

print(f"\nYoung L1 m6A/kb:  {young_mkb:.3f}  (N={len(young_l1):,} reads)")
print(f"Ancient L1 m6A/kb: {ancient_mkb:.3f}  (N={len(ancient_l1):,} reads)")
print(f"Young/Ancient ratio: {ya_ratio:.3f}")

# Per-CL Young vs Ancient
print(f"\n{'CL':<12} {'Young m6A/kb':>12} {'Ancient m6A/kb':>14} {'Y/A ratio':>10} {'N_young':>8} {'N_ancient':>10}")
print("-" * 70)
cl_ya_ratios = []
for cl in BASE_CLS:
    groups = CL_GROUPS[cl]
    l1_cl = load_cache(L1_CACHE, groups, 'l1')
    l1_sum_cl = load_l1_summary(groups)
    l1_sum_cl['is_young'] = l1_sum_cl['transcript_id'].apply(
        lambda x: any(x.startswith(sf) for sf in YOUNG_SUBFAMILIES) if isinstance(x, str) else False
    )
    l1_cl_merged = l1_cl.merge(l1_sum_cl[['read_id', 'is_young']], on='read_id', how='left')

    y = l1_cl_merged[l1_cl_merged['is_young'] == True]
    a = l1_cl_merged[l1_cl_merged['is_young'] == False]

    y_mkb = y['m6a_sites_high'].sum() / (y['read_length'].sum() / 1000.0) if len(y) > 0 and y['read_length'].sum() > 0 else 0
    a_mkb = a['m6a_sites_high'].sum() / (a['read_length'].sum() / 1000.0) if len(a) > 0 and a['read_length'].sum() > 0 else 0
    r = y_mkb / a_mkb if a_mkb > 0 else float('nan')
    cl_ya_ratios.append(r)

    print(f"{cl:<12} {y_mkb:>12.3f} {a_mkb:>14.3f} {r:>10.3f} {len(y):>8,} {len(a):>10,}")


# ═════════════════════════════════════════════════════════════
# SECTION 4: Per-site rate (fraction of m6A candidate positions)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 4: Per-site m6A rate")
print("=" * 80)
print("(m6A sites / total possible sites proxy)")
print("NOTE: Using m6a_sites_high / (read_length * estimated_A_fraction) as proxy")
print("      Estimated A fraction ≈ 0.30 for human transcriptome\n")

# A more practical approach: total m6A sites / total read length gives m6a/kb
# Per-site rate is trickier without actual base composition.
# We use the MAFIA approach: total m6A sites detected / total DRACH motifs scanned
# Since we don't have motif counts, use m6a_sites / (read_length * A_freq) as proxy.
# Better: Just report m6A/kb and note that per-site rate requires motif scanning.
# Actually, the old analysis computed per-site rate differently. Let me use:
# total_m6a_sites / total_candidate_positions_scanned
# We can approximate: each A in the read is a candidate. A fraction ≈ 0.30

A_FRAC = 0.30  # approximate A fraction in reads

print(f"{'CL':<12} {'L1 rate%':>9} {'Ctrl rate%':>10} {'Ratio':>7} {'L1 A-pos':>10} {'Ctrl A-pos':>11}")
print("-" * 65)
cl_persite = {}
for cl in CL_GROUPS:
    r = cl_results[cl]
    l1_total_A = r['l1_total_kb'] * 1000 * A_FRAC
    ctrl_total_A = r['ctrl_total_kb'] * 1000 * A_FRAC

    l1_rate = (r['l1_sites'] / l1_total_A * 100) if l1_total_A > 0 else 0
    ctrl_rate = (r['ctrl_sites'] / ctrl_total_A * 100) if ctrl_total_A > 0 else 0
    ratio = l1_rate / ctrl_rate if ctrl_rate > 0 else float('nan')

    cl_persite[cl] = {'l1_rate': l1_rate, 'ctrl_rate': ctrl_rate, 'ratio': ratio}
    print(f"{cl:<12} {l1_rate:>9.3f} {ctrl_rate:>10.3f} {ratio:>7.3f} {l1_total_A:>10,.0f} {ctrl_total_A:>11,.0f}")

# Pooled per-site rate
pool_l1_A = all_l1['read_length'].sum() * A_FRAC
pool_ctrl_A = all_ctrl['read_length'].sum() * A_FRAC
pool_l1_rate = all_l1['m6a_sites_high'].sum() / pool_l1_A * 100
pool_ctrl_rate = all_ctrl['m6a_sites_high'].sum() / pool_ctrl_A * 100
pool_rate_ratio = pool_l1_rate / pool_ctrl_rate

print(f"\nPooled per-site rate:")
print(f"  L1:   {pool_l1_rate:.3f}%")
print(f"  Ctrl: {pool_ctrl_rate:.3f}%")
print(f"  Ratio: {pool_rate_ratio:.3f}")

# NOTE: per-site rate and m6A/kb should be proportional since both use same denominator base
# The real per-site rate uses actual detected candidate motifs. Since we don't have that,
# m6A/kb IS effectively the per-site rate metric (normalized by length).
print("\nNOTE: Per-site rate here is proportional to m6A/kb (same denominator).")
print("True per-site rate requires DRACH motif scanning (not in cache).")
print("The ratio L1/Ctrl is identical to m6A/kb ratio by construction.")


# ═════════════════════════════════════════════════════════════
# SECTION 5: Cross-CL consistency
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 5: Cross-CL consistency (9 base CL)")
print("=" * 80)

base_mkb = [cl_results[cl]['l1_m6a_kb'] for cl in BASE_CLS]
base_mkb_arr = np.array(base_mkb)

print(f"\nm6A/kb across 9 base CL:")
print(f"  Range:  {base_mkb_arr.min():.3f} - {base_mkb_arr.max():.3f}")
print(f"  Mean:   {base_mkb_arr.mean():.3f}")
print(f"  Median: {np.median(base_mkb_arr):.3f}")
print(f"  SD:     {base_mkb_arr.std():.3f}")
print(f"  CV:     {base_mkb_arr.std() / base_mkb_arr.mean():.3f}")

# Young/Ancient ratio CV
ya_arr = np.array([r for r in cl_ya_ratios if not np.isnan(r)])
print(f"\nYoung/Ancient ratio across 9 base CL:")
print(f"  Range:  {ya_arr.min():.3f} - {ya_arr.max():.3f}")
print(f"  Mean:   {ya_arr.mean():.3f}")
print(f"  Median: {np.median(ya_arr):.3f}")
print(f"  SD:     {ya_arr.std():.3f}")
print(f"  CV:     {ya_arr.std() / ya_arr.mean():.3f}")

# L1/Ctrl ratio CV
base_ratios = np.array([cl_results[cl]['ratio'] for cl in BASE_CLS])
print(f"\nL1/Ctrl m6A/kb ratio across 9 base CL:")
print(f"  Range:  {base_ratios.min():.3f} - {base_ratios.max():.3f}")
print(f"  Mean:   {base_ratios.mean():.3f}")
print(f"  Median: {np.median(base_ratios):.3f}")
print(f"  SD:     {base_ratios.std():.3f}")
print(f"  CV:     {base_ratios.std() / base_ratios.mean():.3f}")


# ═════════════════════════════════════════════════════════════
# SECTION 6: HeLa-Ars m6A-polyA dose-response
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 6: HeLa-Ars m6A-polyA dose-response")
print("=" * 80)

# Load HeLa-Ars L1 data with poly(A)
hela_ars_groups = CL_GROUPS['HeLa-Ars']
hela_ars_l1 = load_cache(L1_CACHE, hela_ars_groups, 'l1')
hela_ars_summary = load_l1_summary(hela_ars_groups)

# Merge to get poly(A)
ars_merged = hela_ars_l1.merge(
    hela_ars_summary[['read_id', 'polya_length', 'qc_tag', 'transcript_id']],
    on='read_id', how='inner'
)
ars_merged['m6a_per_kb'] = ars_merged['m6a_sites_high'] / (ars_merged['read_length'] / 1000.0)

# Filter PASS reads with valid poly(A)
ars_pass = ars_merged[(ars_merged['qc_tag'] == 'PASS') & (ars_merged['polya_length'] > 0)].copy()

print(f"\nHeLa-Ars PASS reads with poly(A): N={len(ars_pass):,}")

# Spearman correlation
if len(ars_pass) > 10:
    rho, pval = stats.spearmanr(ars_pass['m6a_per_kb'], ars_pass['polya_length'])
    print(f"Spearman rho (m6A/kb vs poly(A)): {rho:.4f}")
    print(f"Spearman P-value:                 {pval:.2e}")

    # Quartile analysis
    ars_pass['m6a_quartile'] = pd.qcut(ars_pass['m6a_per_kb'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    q_stats = ars_pass.groupby('m6a_quartile', observed=True)['polya_length'].agg(['mean', 'median', 'count', 'std'])
    print(f"\nQuartile analysis (m6A/kb → poly(A) length):")
    print(f"{'Quartile':<10} {'Mean polyA':>11} {'Median':>8} {'N':>7} {'SD':>8}")
    print("-" * 48)
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        if q in q_stats.index:
            row = q_stats.loc[q]
            print(f"{q:<10} {row['mean']:>11.1f} {row['median']:>8.1f} {row['count']:>7.0f} {row['std']:>8.1f}")

    if 'Q1' in q_stats.index and 'Q4' in q_stats.index:
        q1_mean = q_stats.loc['Q1', 'mean']
        q4_mean = q_stats.loc['Q4', 'mean']
        print(f"\nQ4 - Q1 range: {q4_mean - q1_mean:+.1f} nt")

    # OLS: poly(A) ~ stress * m6A/kb
    # For interaction, we need both HeLa (no stress) and HeLa-Ars (stress)
    print(f"\n--- OLS interaction: stress × m6A/kb → poly(A) ---")

    # Load HeLa baseline
    hela_groups = CL_GROUPS['HeLa']
    hela_l1 = load_cache(L1_CACHE, hela_groups, 'l1')
    hela_summary = load_l1_summary(hela_groups)
    hela_merged = hela_l1.merge(
        hela_summary[['read_id', 'polya_length', 'qc_tag']],
        on='read_id', how='inner'
    )
    hela_merged['m6a_per_kb'] = hela_merged['m6a_sites_high'] / (hela_merged['read_length'] / 1000.0)
    hela_pass = hela_merged[(hela_merged['qc_tag'] == 'PASS') & (hela_merged['polya_length'] > 0)].copy()

    # HeLa baseline correlation
    rho_hela, pval_hela = stats.spearmanr(hela_pass['m6a_per_kb'], hela_pass['polya_length'])
    print(f"\nHeLa baseline Spearman rho: {rho_hela:.4f}, P={pval_hela:.2e}")
    print(f"HeLa-Ars    Spearman rho:   {rho:.4f}, P={pval:.2e}")

    # OLS with interaction
    try:
        import statsmodels.api as sm

        hela_pass_ols = hela_pass[['m6a_per_kb', 'polya_length']].copy()
        hela_pass_ols['stress'] = 0
        ars_pass_ols = ars_pass[['m6a_per_kb', 'polya_length']].copy()
        ars_pass_ols['stress'] = 1

        combined = pd.concat([hela_pass_ols, ars_pass_ols], ignore_index=True)
        combined['stress_x_m6a'] = combined['stress'] * combined['m6a_per_kb']

        X = sm.add_constant(combined[['stress', 'm6a_per_kb', 'stress_x_m6a']])
        y = combined['polya_length']
        model = sm.OLS(y, X).fit()

        print(f"\nOLS: poly(A) ~ const + stress + m6A/kb + stress×m6A/kb")
        print(f"  N = {len(combined):,}")
        print(f"  R² = {model.rsquared:.4f}")
        print(f"  {'Variable':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'P':>12}")
        print(f"  {'-'*62}")
        for var in model.params.index:
            print(f"  {var:<20} {model.params[var]:>10.4f} {model.bse[var]:>10.4f} {model.tvalues[var]:>8.2f} {model.pvalues[var]:>12.2e}")
    except ImportError:
        print("  statsmodels not available, skipping OLS")


# ═════════════════════════════════════════════════════════════
# SECTION 7: Decomposition
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 7: Decomposition (m6A/kb ratio = motif density × per-site rate)")
print("=" * 80)
print("\nNOTE: True decomposition requires DRACH motif scanning per read.")
print("Without motif counts in cache, we cannot separate motif density from per-site rate.")
print("The m6A/kb ratio IS the combined effect.")
print(f"\nOverall L1/Ctrl m6A/kb ratio (pooled 9 CL): {pool_ratio:.3f}")
print("This ratio encapsulates both motif density and per-site modification rate differences.")


# ═════════════════════════════════════════════════════════════
# SECTION 8: Additional per-CL detail table
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 8: Summary comparison — old (0.50 threshold) vs new (0.80 threshold)")
print("=" * 80)
print("\nOLD values (from CLAUDE.md, 0.50 threshold):")
print(f"  L1 per-site rate:  27.2%   |  Ctrl: 20.4%  |  Ratio: 1.33x")
print(f"  m6A/kb ratio:      1.44x   (motif 1.22x × rate 1.33x)")
print(f"  Young: 29.4%  Ancient: 26.9%  Ctrl: 20.4%")

print(f"\nNEW values (0.80 threshold, ML>=204):")
print(f"  L1 m6A/kb (pooled):   {pool_l1_mkb:.3f}")
print(f"  Ctrl m6A/kb (pooled):  {pool_ctrl_mkb:.3f}")
print(f"  L1/Ctrl ratio:         {pool_ratio:.3f}")
print(f"  Young m6A/kb:          {young_mkb:.3f}")
print(f"  Ancient m6A/kb:        {ancient_mkb:.3f}")
print(f"  Young/Ancient ratio:   {ya_ratio:.3f}")
print(f"  Cross-CL m6A/kb CV:    {base_mkb_arr.std() / base_mkb_arr.mean():.3f}")
print(f"  Cross-CL Y/A ratio CV: {ya_arr.std() / ya_arr.mean():.3f}")
print(f"  HeLa-Ars rho:         {rho:.4f} (P={pval:.2e})")
print(f"  HeLa baseline rho:    {rho_hela:.4f} (P={pval_hela:.2e})")


# ═════════════════════════════════════════════════════════════
# SECTION 9: HeLa vs HeLa-Ars m6A level comparison
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 9: HeLa vs HeLa-Ars m6A level (stress effect on m6A itself)")
print("=" * 80)

hela_mkb = cl_results['HeLa']['l1_m6a_kb']
ars_mkb = cl_results['HeLa-Ars']['l1_m6a_kb']
print(f"\nHeLa L1 m6A/kb:     {hela_mkb:.3f}")
print(f"HeLa-Ars L1 m6A/kb: {ars_mkb:.3f}")
print(f"Ratio (Ars/HeLa):   {ars_mkb/hela_mkb:.3f}")

# MWU between HeLa and HeLa-Ars per-read m6a/kb
hela_pr = cl_results['HeLa']['l1_df']['m6a_per_kb'] if 'm6a_per_kb' in cl_results['HeLa']['l1_df'].columns else cl_results['HeLa']['l1_df']['m6a_sites_high'] / (cl_results['HeLa']['l1_df']['read_length'] / 1000.0)
ars_pr = cl_results['HeLa-Ars']['l1_df']['m6a_per_kb'] if 'm6a_per_kb' in cl_results['HeLa-Ars']['l1_df'].columns else cl_results['HeLa-Ars']['l1_df']['m6a_sites_high'] / (cl_results['HeLa-Ars']['l1_df']['read_length'] / 1000.0)
stat_ha, p_ha = stats.mannwhitneyu(hela_pr, ars_pr, alternative='two-sided')
print(f"MWU P (HeLa vs HeLa-Ars): {p_ha:.2e}")


# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY TABLE — m6A Metrics (NEW threshold 0.80, ML>=204)")
print("=" * 80)

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ METRIC                              │ VALUE                      │
├──────────────────────────────────────┼────────────────────────────┤
│ L1 m6A/kb (pooled 9 CL)            │ {pool_l1_mkb:>8.3f}                   │
│ Ctrl m6A/kb (pooled 9 CL)          │ {pool_ctrl_mkb:>8.3f}                   │
│ L1/Ctrl ratio                       │ {pool_ratio:>8.3f}x                  │
│ MWU P (L1 vs Ctrl)                  │ {mwu_p:>12.2e}              │
├──────────────────────────────────────┼────────────────────────────┤
│ Young L1 m6A/kb                     │ {young_mkb:>8.3f}                   │
│ Ancient L1 m6A/kb                   │ {ancient_mkb:>8.3f}                   │
│ Young/Ancient ratio                 │ {ya_ratio:>8.3f}x                  │
├──────────────────────────────────────┼────────────────────────────┤
│ Cross-CL m6A/kb range               │ {base_mkb_arr.min():.3f} - {base_mkb_arr.max():.3f}           │
│ Cross-CL m6A/kb CV                  │ {base_mkb_arr.std()/base_mkb_arr.mean():>8.3f}                   │
│ Cross-CL Y/A ratio range            │ {ya_arr.min():.3f} - {ya_arr.max():.3f}           │
│ Cross-CL Y/A ratio CV               │ {ya_arr.std()/ya_arr.mean():>8.3f}                   │
│ Cross-CL L1/Ctrl ratio range        │ {base_ratios.min():.3f} - {base_ratios.max():.3f}           │
│ Cross-CL L1/Ctrl ratio CV           │ {base_ratios.std()/base_ratios.mean():>8.3f}                   │
├──────────────────────────────────────┼────────────────────────────┤
│ HeLa vs HeLa-Ars m6A/kb             │ {hela_mkb:.3f} vs {ars_mkb:.3f}          │
│ Ars/HeLa ratio                      │ {ars_mkb/hela_mkb:>8.3f}x                  │
│ MWU P (HeLa vs HeLa-Ars)            │ {p_ha:>12.2e}              │
├──────────────────────────────────────┼────────────────────────────┤
│ HeLa-Ars rho (m6A/kb vs polyA)      │ {rho:>8.4f}                  │
│ HeLa-Ars rho P                      │ {pval:>12.2e}              │
│ HeLa baseline rho                   │ {rho_hela:>8.4f}                  │
│ HeLa baseline rho P                 │ {pval_hela:>12.2e}              │
│ Q1→Q4 polyA range (HeLa-Ars)        │ {q4_mean - q1_mean:>+8.1f} nt                │
└──────────────────────────────────────┴────────────────────────────┘
""")

print("DONE. All metrics computed with new threshold 0.80 (ML>=204).")
