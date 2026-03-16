#!/usr/bin/env python3
"""
Mixed Tail (Uridylation) Analysis Under Arsenite Stress
========================================================
L1 uridylation by TUT4/TUT7 → retrotransposition suppression (Warkocki 2018 Cell, 2024 NAR).
Arsenite poly(A) shortening may increase uridylation substrate (decay zone).

Analyses:
  1. Decorated tail rate × stress × m6A (HeLa vs HeLa-Ars)
  2. Non-A nucleotide position × stress
  3. Cross-cell-line decorated rate + Young vs Ancient

Caveats:
  - Ninetails detects internal non-A, not terminal oligo-U
  - U vs C discrimination unreliable → pyrimidine (U+C) as robust metric
  - Internal-A artifact possible for low est_nonA_pos
"""

import pandas as pd
import numpy as np
import glob as glob_module
import os
import sys
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
RESULTS_DIR = os.path.join(BASE_DIR, 'results_group')
PART3_CACHE = os.path.join(BASE_DIR, 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache')
OUT_DIR = os.path.join(BASE_DIR, 'analysis/01_exploration/topic_05_cellline/mixed_tail_analysis')

YOUNG_L1 = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CELL_LINES = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'Hek293T': ['Hek293T_3', 'Hek293T_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'MCF7-EV': ['MCF7-EV_1'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}

POLYA_BIN_SIZE = 50


# ── Helper functions ───────────────────────────────────────────────────────
def load_l1_summary(group_id):
    path = os.path.join(RESULTS_DIR, group_id, 'g_summary', f'{group_id}_L1_summary.tsv')
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t')
    df['group_id'] = group_id
    return df


def load_nonadenosine(group_id):
    pattern = os.path.join(RESULTS_DIR, group_id, 'f_ninetails',
                           f'*_{group_id}_nonadenosine_residues.tsv')
    files = glob_module.glob(pattern)
    if not files:
        print(f"  WARNING: nonadenosine file not found for {group_id}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(files[0], sep='\t')
    df['group_id'] = group_id
    return df


def load_part3_cache(group_id):
    path = os.path.join(PART3_CACHE, f'{group_id}_l1_per_read.tsv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    df['group_id'] = group_id
    return df


def get_cell_line(group_id):
    for cl, groups in CELL_LINES.items():
        if group_id in groups:
            return cl
    return group_id


def polya_bin(polya):
    return int(polya // POLYA_BIN_SIZE) * POLYA_BIN_SIZE


def fisher_safe(table_2x2):
    """Fisher exact test on a 2×2 array, return (OR, p)."""
    try:
        return stats.fisher_exact(table_2x2)
    except Exception:
        return (float('nan'), float('nan'))


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)


# ── ANALYSIS 1: Decorated × Stress × m6A ──────────────────────────────────
def analysis_1(df_pass, df_nona_l1):
    section("ANALYSIS 1: Decorated Rate × Stress × m6A")

    # ── 1a. Overall decorated rate ──
    print("\n[1a] Overall decorated rate (HeLa vs HeLa-Ars, PASS, blank/decorated):")
    results_1a = {}
    for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
        sub = df_pass[df_pass['is_stress'] == cond]
        n_dec = sub['is_decorated'].sum()
        n_tot = len(sub)
        rate = n_dec / n_tot if n_tot > 0 else 0
        results_1a[label] = {'n': n_tot, 'dec': n_dec, 'rate': rate}
        print(f"  {label}: {n_dec}/{n_tot} = {rate:.2%}")

    ct = pd.crosstab(df_pass['is_stress'], df_pass['is_decorated'])
    if ct.shape == (2, 2):
        odds, p = fisher_safe(ct.values)
        print(f"  Fisher exact: OR={odds:.3f}, p={p:.4e}")

    # ── 1b. Poly(A)-length-matched decorated rate ──
    print(f"\n[1b] Poly(A)-length-matched decorated rate ({POLYA_BIN_SIZE}nt bins):")
    print(f"  {'Bin':>10s}  {'HeLa_n':>7s} {'HeLa_%':>7s}  {'Ars_n':>7s} {'Ars_%':>7s}  {'OR':>7s} {'p':>10s}")

    bins_results = []
    for bval in sorted(df_pass['polya_bin'].unique()):
        sub = df_pass[df_pass['polya_bin'] == bval]
        row = {'polya_bin': bval}
        for cond, key in [(False, 'hela'), (True, 'ars')]:
            mask = sub['is_stress'] == cond
            n_dec = sub.loc[mask, 'is_decorated'].sum()
            n_tot = mask.sum()
            rate = n_dec / n_tot if n_tot > 0 else 0
            row[f'{key}_n'] = n_tot
            row[f'{key}_dec'] = int(n_dec)
            row[f'{key}_rate'] = rate

        if row['hela_n'] >= 5 and row['ars_n'] >= 5:
            tbl = [[row['hela_dec'], row['hela_n'] - row['hela_dec']],
                   [row['ars_dec'], row['ars_n'] - row['ars_dec']]]
            odds_b, p_b = fisher_safe(tbl)
        else:
            odds_b, p_b = float('nan'), float('nan')
        row['odds_ratio'] = odds_b
        row['p_value'] = p_b

        bstr = f"{bval}-{bval + POLYA_BIN_SIZE}"
        print(f"  {bstr:>10s}  {row['hela_n']:>7d} {row['hela_rate']:>7.1%}  "
              f"{row['ars_n']:>7d} {row['ars_rate']:>7.1%}  "
              f"{odds_b:>7.2f} {p_b:>10.3e}")
        bins_results.append(row)

    pd.DataFrame(bins_results).to_csv(
        os.path.join(OUT_DIR, 'polya_binned_decorated_rate.tsv'), sep='\t', index=False)

    # ── 1c. Logistic regression ──
    print("\n[1c] Logistic regression: decorated ~ stress + polya_length + read_length")
    try:
        import statsmodels.api as sm
        logit_df = df_pass[['is_decorated', 'is_stress', 'polya_length', 'read_length']].dropna().copy()
        logit_df['stress_int'] = logit_df['is_stress'].astype(int)
        logit_df['dec_int'] = logit_df['is_decorated'].astype(int)

        X = sm.add_constant(logit_df[['stress_int', 'polya_length', 'read_length']])
        y = logit_df['dec_int']

        model = sm.Logit(y, X).fit(disp=0)
        print(model.summary2().tables[1].to_string())
    except ImportError:
        print("  statsmodels not available, skipping")
    except Exception as e:
        print(f"  Logistic regression failed: {e}")

    # ── 1d. Decay zone ──
    print("\n[1d] Decay zone analysis (poly(A) < 30nt vs >= 30nt):")
    for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
        sub = df_pass[df_pass['is_stress'] == cond]
        for thresh, zone_label in [('<30nt', sub['polya_length'] < 30),
                                    ('>=30nt', sub['polya_length'] >= 30)]:
            z = sub[zone_label]
            n_dec = z['is_decorated'].sum()
            n_tot = len(z)
            rate = n_dec / n_tot if n_tot > 0 else 0
            print(f"  {label} {thresh}: {n_dec}/{n_tot} = {rate:.2%}")

    df_pass_loc = df_pass.copy()
    df_pass_loc['decay_zone'] = df_pass_loc['polya_length'] < 30
    for zone_val, zone_label in [(True, 'Decay (<30nt)'), (False, 'Normal (>=30nt)')]:
        sub = df_pass_loc[df_pass_loc['decay_zone'] == zone_val]
        ct2 = pd.crosstab(sub['is_stress'], sub['is_decorated'])
        if ct2.shape == (2, 2):
            odds2, p2 = fisher_safe(ct2.values)
            print(f"  {zone_label} stress effect: OR={odds2:.3f}, p={p2:.4e}")
        else:
            print(f"  {zone_label}: insufficient data (n={len(sub)})")

    # ── 1e. m6A × decorated ──
    print("\n[1e] m6A quartile × decorated rate:")
    df_m6a = df_pass[df_pass['read_length'] > 0].copy()
    df_m6a['m6a_per_kb'] = df_m6a['m6a_sites_high'] / (df_m6a['read_length'] / 1000)

    hela_m6a = df_m6a.loc[~df_m6a['is_stress'], 'm6a_per_kb'].dropna()
    if len(hela_m6a) >= 20:
        q25, q50, q75 = hela_m6a.quantile([0.25, 0.5, 0.75]).values
        q_cuts = [-0.01, q25, q50, q75, hela_m6a.max() + 1]
        q_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
        df_m6a['m6a_quartile'] = pd.cut(df_m6a['m6a_per_kb'], bins=q_cuts,
                                         labels=q_labels, include_lowest=True)

        print(f"  m6A/kb quartile boundaries: Q1<{q25:.2f}, Q2<{q50:.2f}, Q3<{q75:.2f}, Q4>{q75:.2f}")
        print(f"  {'Quartile':>12s}  {'HeLa_n':>7s} {'HeLa_%':>7s}  {'Ars_n':>7s} {'Ars_%':>7s}  {'OR':>7s} {'p':>10s}")

        m6a_results = []
        for q in q_labels:
            row = {'quartile': q}
            for cond, key in [(False, 'hela'), (True, 'ars')]:
                mask = (df_m6a['m6a_quartile'] == q) & (df_m6a['is_stress'] == cond)
                sub = df_m6a[mask]
                n_dec = sub['is_decorated'].sum()
                n_tot = len(sub)
                rate = n_dec / n_tot if n_tot > 0 else 0
                row[f'{key}_n'] = n_tot
                row[f'{key}_dec'] = int(n_dec)
                row[f'{key}_rate'] = rate

            if row['hela_n'] >= 5 and row['ars_n'] >= 5:
                tbl = [[row['hela_dec'], row['hela_n'] - row['hela_dec']],
                       [row['ars_dec'], row['ars_n'] - row['ars_dec']]]
                o, p = fisher_safe(tbl)
            else:
                o, p = float('nan'), float('nan')
            row['odds_ratio'] = o
            row['p_value'] = p

            print(f"  {q:>12s}  {row['hela_n']:>7d} {row['hela_rate']:>7.1%}  "
                  f"{row['ars_n']:>7d} {row['ars_rate']:>7.1%}  {o:>7.2f} {p:>10.3e}")
            m6a_results.append(row)

        pd.DataFrame(m6a_results).to_csv(
            os.path.join(OUT_DIR, 'm6a_quartile_decorated_rate.tsv'), sep='\t', index=False)
    else:
        print(f"  Insufficient HeLa m6A data (n={len(hela_m6a)})")

    # ── 1f. Non-A composition ──
    print("\n[1f] Non-A nucleotide composition:")
    if df_nona_l1 is not None and len(df_nona_l1) > 0:
        print(f"  Total non-A entries in L1 reads: {len(df_nona_l1)}")
        for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
            sub = df_nona_l1[df_nona_l1['is_stress'] == cond]
            comp = sub['prediction'].value_counts()
            total = len(sub)
            if total > 0:
                parts = [f"{k}={v} ({v/total:.1%})" for k, v in comp.items()]
                print(f"  {label} (n={total}): {', '.join(parts)}")

        df_nona_l1 = df_nona_l1.copy()
        df_nona_l1['is_pyrimidine'] = df_nona_l1['prediction'].isin(['U', 'C'])
        ct_nuc = pd.crosstab(df_nona_l1['is_stress'], df_nona_l1['is_pyrimidine'])
        if ct_nuc.shape == (2, 2):
            odds_n, p_n = fisher_safe(ct_nuc.values)
            print(f"  Pyrimidine(U+C) vs G ratio, stress: OR={odds_n:.3f}, p={p_n:.4e}")

        # Composition per stress and per prediction type
        comp_df = df_nona_l1.groupby(['is_stress', 'prediction']).size().reset_index(name='count')
        comp_df.to_csv(os.path.join(OUT_DIR, 'nona_composition.tsv'), sep='\t', index=False)
    else:
        print("  No non-adenosine data available")


# ── ANALYSIS 2: Non-A Position × Stress ────────────────────────────────────
def analysis_2(df_nona_l1, dist_map):
    section("ANALYSIS 2: Non-A Nucleotide Position × Stress")

    if df_nona_l1 is None or len(df_nona_l1) == 0:
        print("  No non-adenosine data available")
        return

    # ── 2a. Internal-A artifact check ──
    print("\n[2a] Internal-A artifact check:")
    df_nona_l1 = df_nona_l1.copy()
    df_nona_l1['dist_to_3prime'] = df_nona_l1['readname'].map(dist_map)

    low_pos = df_nona_l1[df_nona_l1['est_nonA_pos'] < 10]
    print(f"  est_nonA_pos < 10: {len(low_pos)} entries ({len(low_pos)/len(df_nona_l1):.1%})")
    if len(low_pos) > 0:
        low_dist = low_pos['dist_to_3prime'].dropna()
        if len(low_dist) > 0:
            print(f"    dist_to_3prime median: {low_dist.median():.1f}, "
                  f"mean: {low_dist.mean():.1f}")
            low_close = ((low_pos['est_nonA_pos'] < 10) &
                         (low_pos['dist_to_3prime'] < 50)).sum()
            print(f"    Potential artifacts (pos<10 & dist<50): {low_close}")

    # Filter artifacts
    artifact_mask = (df_nona_l1['est_nonA_pos'] < 10) & (df_nona_l1['dist_to_3prime'] < 50)
    n_artifact = artifact_mask.sum()
    df_clean = df_nona_l1[~artifact_mask].copy()
    print(f"  After artifact filter: {len(df_clean)} entries (removed {n_artifact})")

    # Sensitivity: also run without filter
    for tag, dset in [('filtered', df_clean), ('unfiltered', df_nona_l1)]:
        print(f"\n[2b] est_nonA_pos distribution ({tag}, n={len(dset)}):")
        for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
            sub = dset[dset['is_stress'] == cond]
            if len(sub) > 0:
                print(f"  {label} (n={len(sub)}): "
                      f"median={sub['est_nonA_pos'].median():.1f}, "
                      f"mean={sub['est_nonA_pos'].mean():.1f}, "
                      f"IQR=[{sub['est_nonA_pos'].quantile(0.25):.1f}, "
                      f"{sub['est_nonA_pos'].quantile(0.75):.1f}]")

        hela_pos = dset.loc[~dset['is_stress'], 'est_nonA_pos']
        ars_pos = dset.loc[dset['is_stress'], 'est_nonA_pos']
        if len(hela_pos) > 5 and len(ars_pos) > 5:
            u, p = stats.mannwhitneyu(hela_pos, ars_pos, alternative='two-sided')
            print(f"  Mann-Whitney U: U={u:.0f}, p={p:.4e}")

    # ── 2c. Position ratio ──
    print(f"\n[2c] Non-A position ratio (est_nonA_pos / polya_length):")
    df_clean['pos_ratio'] = (df_clean['est_nonA_pos'] / df_clean['polya_length']).clip(0, 1)

    for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
        sub = df_clean[df_clean['is_stress'] == cond]
        if len(sub) > 0:
            print(f"  {label}: median={sub['pos_ratio'].median():.3f}, "
                  f"mean={sub['pos_ratio'].mean():.3f}, "
                  f"IQR=[{sub['pos_ratio'].quantile(0.25):.3f}, "
                  f"{sub['pos_ratio'].quantile(0.75):.3f}]")

    hela_r = df_clean.loc[~df_clean['is_stress'], 'pos_ratio']
    ars_r = df_clean.loc[df_clean['is_stress'], 'pos_ratio']
    if len(hela_r) > 5 and len(ars_r) > 5:
        u2, p2 = stats.mannwhitneyu(hela_r, ars_r, alternative='two-sided')
        print(f"  Mann-Whitney U: U={u2:.0f}, p={p2:.4e}")

    # Save
    pos_summary = []
    for cond, label in [(False, 'HeLa'), (True, 'HeLa-Ars')]:
        sub = df_clean[df_clean['is_stress'] == cond]
        if len(sub) > 0:
            pos_summary.append({
                'condition': label, 'n': len(sub),
                'pos_median': sub['est_nonA_pos'].median(),
                'pos_mean': sub['est_nonA_pos'].mean(),
                'pos_q25': sub['est_nonA_pos'].quantile(0.25),
                'pos_q75': sub['est_nonA_pos'].quantile(0.75),
                'ratio_median': sub['pos_ratio'].median(),
                'ratio_mean': sub['pos_ratio'].mean(),
            })
    pd.DataFrame(pos_summary).to_csv(
        os.path.join(OUT_DIR, 'nona_position_summary.tsv'), sep='\t', index=False)


# ── ANALYSIS 3: Cross-CL Decorated Rate ───────────────────────────────────
def analysis_3():
    section("ANALYSIS 3: Cross-Cell-Line Decorated Rate + Young vs Ancient")

    print("\n[3a] Loading all cell lines...")
    all_data = []
    for cl, groups in CELL_LINES.items():
        for gid in groups:
            s = load_l1_summary(gid)
            if not s.empty:
                s['cell_line'] = cl
                all_data.append(s)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['is_young'] = df_all['gene_id'].apply(lambda x: x in YOUNG_L1)
    df_all['is_decorated'] = df_all['class'] == 'decorated'

    df_ap = df_all[
        (df_all['qc_tag'] == 'PASS') &
        (df_all['class'].isin(['blank', 'decorated']))
    ].copy()

    print(f"  Total reads (all CL, PASS, blank/dec): {len(df_ap)}")

    # ── 3a. Per-CL decorated rate ──
    print(f"\n  {'Cell Line':>12s}  {'N':>6s} {'Dec':>5s} {'Rate':>7s}  {'Med_pA':>7s}")
    cl_results = []
    for cl in sorted(df_ap['cell_line'].unique()):
        sub = df_ap[df_ap['cell_line'] == cl]
        n_dec = sub['is_decorated'].sum()
        n_tot = len(sub)
        rate = n_dec / n_tot if n_tot > 0 else 0
        med_pa = sub['polya_length'].median()
        print(f"  {cl:>12s}  {n_tot:>6d} {n_dec:>5d} {rate:>7.1%}  {med_pa:>7.1f}")
        cl_results.append({
            'cell_line': cl, 'n_total': n_tot, 'n_decorated': n_dec,
            'rate': rate, 'median_polya': med_pa
        })

    pd.DataFrame(cl_results).to_csv(
        os.path.join(OUT_DIR, 'cross_cl_decorated_rate.tsv'), sep='\t', index=False)

    # ── 3b. Decorated rate vs median poly(A) correlation ──
    cl_df = pd.DataFrame(cl_results)
    if len(cl_df) >= 5:
        r, p_corr = stats.spearmanr(cl_df['median_polya'], cl_df['rate'])
        print(f"\n  Decorated rate vs median poly(A): Spearman r={r:.3f}, p={p_corr:.4e}")
        print(f"  (Negative = shorter poly(A) → more decorated)")

    # ── 3c. Young vs Ancient ──
    print(f"\n[3b] Young vs Ancient L1 decorated rate (all CL pooled):")
    print(f"  {'Category':>10s}  {'N':>6s} {'Dec':>5s} {'Rate':>7s}")
    for is_y, label in [(True, 'Young'), (False, 'Ancient')]:
        sub = df_ap[df_ap['is_young'] == is_y]
        n_dec = sub['is_decorated'].sum()
        n_tot = len(sub)
        rate = n_dec / n_tot if n_tot > 0 else 0
        print(f"  {label:>10s}  {n_tot:>6d} {n_dec:>5d} {rate:>7.1%}")

    ct_age = pd.crosstab(df_ap['is_young'], df_ap['is_decorated'])
    if ct_age.shape == (2, 2):
        odds_a, p_a = fisher_safe(ct_age.values)
        print(f"  Young vs Ancient: OR={odds_a:.3f}, p={p_a:.4e}")

    # ── 3d. Young vs Ancient per CL ──
    print(f"\n[3c] Young vs Ancient decorated rate per cell line:")
    print(f"  {'Cell Line':>12s}  {'Young_n':>7s} {'Young_%':>7s}  {'Anc_n':>7s} {'Anc_%':>7s}")
    cl_age = []
    for cl in sorted(df_ap['cell_line'].unique()):
        sub = df_ap[df_ap['cell_line'] == cl]
        row = {'cell_line': cl}
        for is_y, key in [(True, 'young'), (False, 'ancient')]:
            z = sub[sub['is_young'] == is_y]
            n_dec = z['is_decorated'].sum()
            n_tot = len(z)
            rate = n_dec / n_tot if n_tot > 0 else 0
            row[f'{key}_n'] = n_tot
            row[f'{key}_dec'] = int(n_dec)
            row[f'{key}_rate'] = rate
        print(f"  {cl:>12s}  {row['young_n']:>7d} {row['young_rate']:>7.1%}  "
              f"{row['ancient_n']:>7d} {row['ancient_rate']:>7.1%}")
        cl_age.append(row)

    pd.DataFrame(cl_age).to_csv(
        os.path.join(OUT_DIR, 'cross_cl_young_ancient_rate.tsv'), sep='\t', index=False)

    # ── 3e. K562 vs A549 ──
    print(f"\n[3d] Special comparison: K562 (shortest poly(A)) vs A549 (longest):")
    for cl in ['K562', 'A549']:
        sub = df_ap[df_ap['cell_line'] == cl]
        if len(sub) > 0:
            med_pa = sub['polya_length'].median()
            n_dec = sub['is_decorated'].sum()
            n_tot = len(sub)
            rate = n_dec / n_tot if n_tot > 0 else 0
            print(f"  {cl}: median poly(A)={med_pa:.1f}nt, "
                  f"decorated={rate:.1%} ({n_dec}/{n_tot})")

    k562 = df_ap[df_ap['cell_line'] == 'K562']
    a549 = df_ap[df_ap['cell_line'] == 'A549']
    if len(k562) > 5 and len(a549) > 5:
        tbl = [[k562['is_decorated'].sum(), len(k562) - k562['is_decorated'].sum()],
               [a549['is_decorated'].sum(), len(a549) - a549['is_decorated'].sum()]]
        o, p = fisher_safe(tbl)
        print(f"  K562 vs A549: OR={o:.3f}, p={p:.4e}")

    # ── 3f. Length-matched K562 vs A549 ──
    print(f"\n[3e] Length-matched K562 vs A549 ({POLYA_BIN_SIZE}nt bins):")
    k562['polya_bin'] = k562['polya_length'].apply(polya_bin)
    a549['polya_bin'] = a549['polya_length'].apply(polya_bin)
    shared_bins = set(k562['polya_bin'].unique()) & set(a549['polya_bin'].unique())
    for bval in sorted(shared_bins):
        k_sub = k562[k562['polya_bin'] == bval]
        a_sub = a549[a549['polya_bin'] == bval]
        k_rate = k_sub['is_decorated'].mean() if len(k_sub) > 0 else 0
        a_rate = a_sub['is_decorated'].mean() if len(a_sub) > 0 else 0
        print(f"  {bval}-{bval+POLYA_BIN_SIZE}nt: K562 {k_rate:.1%}(n={len(k_sub)}) "
              f"vs A549 {a_rate:.1%}(n={len(a_sub)})")


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Mixed Tail (Uridylation) Analysis Under Arsenite Stress")
    print("=" * 70)

    # ── Load HeLa + HeLa-Ars data ──
    print("\n[Data Loading] HeLa + HeLa-Ars...")
    hela_groups = CELL_LINES['HeLa'] + CELL_LINES['HeLa-Ars']

    summaries, nonadenines, part3s = [], [], []
    for gid in hela_groups:
        s = load_l1_summary(gid)
        n = load_nonadenosine(gid)
        p = load_part3_cache(gid)
        if not s.empty:
            summaries.append(s)
        if not n.empty:
            nonadenines.append(n)
        if not p.empty:
            part3s.append(p)

    df_sum = pd.concat(summaries, ignore_index=True)
    df_nona = pd.concat(nonadenines, ignore_index=True) if nonadenines else pd.DataFrame()
    df_part3 = pd.concat(part3s, ignore_index=True) if part3s else pd.DataFrame()

    print(f"  L1 summary reads: {len(df_sum)}")
    print(f"  Non-adenosine entries: {len(df_nona)}")
    print(f"  Part3 cache reads: {len(df_part3)}")

    # ── Classify ──
    df_sum['cell_line'] = df_sum['group_id'].apply(get_cell_line)
    df_sum['is_stress'] = df_sum['cell_line'] == 'HeLa-Ars'
    df_sum['is_young'] = df_sum['gene_id'].apply(lambda x: x in YOUNG_L1)
    df_sum['is_decorated'] = df_sum['class'] == 'decorated'
    df_sum['polya_bin'] = df_sum['polya_length'].apply(polya_bin)

    # Merge Part3 cache for m6A
    if not df_part3.empty:
        # Part3 read_length is more accurate; drop summary read_length conflict
        df = df_sum.merge(df_part3[['read_id', 'm6a_sites_high', 'read_length']],
                          on='read_id', how='left', suffixes=('', '_p3'))
        # Use Part3 read_length if available
        if 'read_length_p3' in df.columns:
            df['read_length'] = df['read_length_p3'].fillna(df['read_length'])
            df.drop(columns=['read_length_p3'], inplace=True)
    else:
        df = df_sum.copy()
        df['m6a_sites_high'] = np.nan

    df['m6a_sites_high'] = df['m6a_sites_high'].fillna(0).astype(int)

    # PASS + blank/decorated filter
    df_pass = df[
        (df['qc_tag'] == 'PASS') &
        (df['class'].isin(['blank', 'decorated']))
    ].copy()

    n_total = len(df)
    n_pass = len(df_pass)
    part3_matched = df['m6a_sites_high'].notna().sum()
    print(f"\n  PASS + blank/decorated: {n_pass}/{n_total} = {n_pass/n_total:.1%}")

    # Per-group breakdown
    print("\n  Per-group read counts (PASS, blank/decorated):")
    for gid in hela_groups:
        n = len(df_pass[df_pass['group_id'] == gid])
        n_orig = len(df[df['group_id'] == gid])
        print(f"    {gid}: {n}/{n_orig}")

    # Prepare nonadenosine data for L1 reads
    df_nona_l1 = None
    dist_map = {}
    if not df_nona.empty:
        df_nona['cell_line'] = df_nona['group_id'].apply(get_cell_line)
        df_nona['is_stress'] = df_nona['cell_line'] == 'HeLa-Ars'
        df_nona_pass = df_nona[df_nona['qc_tag'] == 'PASS'].copy()

        l1_reads = set(df_pass['read_id'])
        df_nona_l1 = df_nona_pass[df_nona_pass['readname'].isin(l1_reads)].copy()
        print(f"  Non-adenosine entries in L1 PASS reads: {len(df_nona_l1)}")

        # Build dist_to_3prime map
        dist_map = df_pass.set_index('read_id')['dist_to_3prime'].to_dict()

    # ── Run analyses ──
    analysis_1(df_pass, df_nona_l1)
    analysis_2(df_nona_l1, dist_map)
    analysis_3()

    # ── Verification ──
    section("VERIFICATION")
    print(f"  Total HeLa+HeLa-Ars L1 reads: {n_total}")
    print(f"  After PASS+blank/dec filter: {n_pass} ({n_pass/n_total:.1%})")
    hela_pass = len(df_pass[~df_pass['is_stress']])
    ars_pass = len(df_pass[df_pass['is_stress']])
    print(f"  HeLa: {hela_pass}, HeLa-Ars: {ars_pass}")
    dec_total = df_pass['is_decorated'].sum()
    dec_rate = dec_total / n_pass if n_pass > 0 else 0
    print(f"  Overall decorated rate: {dec_total}/{n_pass} = {dec_rate:.2%}")
    decay_n = (df_pass['polya_length'] < 30).sum()
    print(f"  Decay zone (<30nt) reads: {decay_n}")

    section("DONE")
    print(f"  Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
