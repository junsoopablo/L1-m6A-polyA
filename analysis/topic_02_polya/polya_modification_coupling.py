#!/usr/bin/env python3
"""
Poly(A) tail length x m6A/psi modification coupling analysis.

Key questions:
  1. Do reads with m6A/psi modifications have different poly(A) lengths?
  2. Does this coupling change under arsenite stress?
  3. Is the coupling L1-specific or also seen in control transcripts?

Approach: merge MAFIA per-read data (m6A/psi site counts) with poly(A) data
from L1 summary and control summary files.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
ANALYSIS = PROJECT / 'analysis/01_exploration'
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load MAFIA per-read data (has m6A/psi site counts)
# =========================================================================
print("Loading MAFIA per-read data...")
mafia = pd.read_csv(ANALYSIS / 'topic_03_m6a_psi/hela_ars_comparison_per_read.tsv', sep='\t')
print(f"  MAFIA per-read: {len(mafia):,} rows")
print(f"  Sources: {mafia['source'].value_counts().to_dict()}")

# =========================================================================
# 2. Load poly(A) data and merge
# =========================================================================
print("\nLoading poly(A) data and merging...")

# --- L1 reads: poly(A) from L1 summary ---
l1_polya = {}
for g in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
    df = pd.read_csv(path, sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    for _, row in df.iterrows():
        l1_polya[row['read_id']] = {
            'polya_length': row['polya_length'],
            'class': row['class'],
            'gene_id': row['gene_id'],
        }
print(f"  L1 poly(A) records: {len(l1_polya):,}")

# --- Control reads: poly(A) from control summary ---
ctrl_polya = {}
for g in ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
    df = pd.read_csv(path, sep='\t')
    df = df[df['qc_tag'] == 'PASS']
    for _, row in df.iterrows():
        ctrl_polya[row['read_id']] = {
            'polya_length': row['polya_length'],
            'class': row['class'],
        }
print(f"  Ctrl poly(A) records: {len(ctrl_polya):,}")

# Merge: add poly(A) to MAFIA per-read data
polya_vals = []
class_vals = []
for _, row in mafia.iterrows():
    rid = row['read_id']
    if row['l1_age'] in ('young', 'ancient'):
        info = l1_polya.get(rid, {})
    else:
        info = ctrl_polya.get(rid, {})
    polya_vals.append(info.get('polya_length', np.nan))
    class_vals.append(info.get('class', 'unknown'))

mafia['polya_length'] = polya_vals
mafia['tail_class'] = class_vals

# Filter to reads with valid poly(A)
merged = mafia[mafia['polya_length'].notna() & (mafia['polya_length'] > 0)].copy()
merged['has_m6a'] = merged['m6a_sites_high'] > 0
merged['has_psi'] = merged['psi_sites_high'] > 0
merged['has_either'] = merged['has_m6a'] | merged['has_psi']
merged['has_both'] = merged['has_m6a'] & merged['has_psi']

print(f"\n  Merged (with valid poly(A)): {len(merged):,} / {len(mafia):,}")
print(f"  By source:")
for src, cnt in merged['source'].value_counts().items():
    print(f"    {src}: {cnt:,}")

# =========================================================================
# 3. Poly(A) by modification status
# =========================================================================
print(f"\n{'='*90}")
print("Poly(A) Tail Length by Modification Status")
print(f"{'='*90}")

conditions = [
    ('HeLa L1', merged[(merged['source'] == 'HeLa_L1')]),
    ('HeLa-Ars L1', merged[(merged['source'] == 'HeLa-Ars_L1')]),
    ('HeLa Ctrl', merged[(merged['source'] == 'HeLa_Ctrl')]),
    ('HeLa-Ars Ctrl', merged[(merged['source'] == 'HeLa-Ars_Ctrl')]),
]

print(f"\n{'Condition':<18} {'Mod':<15} {'N':>6} {'Median':>8} {'Mean':>8} {'p (vs no-mod)':>14}")
print("-" * 75)

for label, cond in conditions:
    for mod_name, mod_col in [('m6A+', 'has_m6a'), ('psi+', 'has_psi'),
                                ('either+', 'has_either'), ('both+', 'has_both')]:
        pos = cond[cond[mod_col]]
        neg = cond[~cond[mod_col]]
        if len(pos) >= 5 and len(neg) >= 5:
            _, p = stats.mannwhitneyu(pos['polya_length'], neg['polya_length'],
                                       alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            p_str = f"{p:.2e}({sig})"
        else:
            p_str = "n/a"
        pos_med = pos['polya_length'].median() if len(pos) > 0 else 0
        neg_med = neg['polya_length'].median() if len(neg) > 0 else 0
        n_pos = len(pos)
        # Only print the with-modification row; show neg median in p column context
        print(f"{label:<18} {mod_name:<15} {n_pos:>6} {pos_med:>8.1f} "
              f"{pos['polya_length'].mean():>8.1f} {p_str:>14}")
    # Print no-modification baseline
    neg_all = cond[~cond['has_either']]
    print(f"{label:<18} {'no mod':<15} {len(neg_all):>6} "
          f"{neg_all['polya_length'].median():>8.1f} {neg_all['polya_length'].mean():>8.1f}")
    print()

# =========================================================================
# 4. Correlation: poly(A) vs modification density
# =========================================================================
print(f"\n{'='*90}")
print("Correlation: Poly(A) Length vs Modification Density (sites/kb)")
print(f"{'='*90}")

print(f"\n{'Condition':<18} {'Mod':<8} {'r':>7} {'p':>12} {'N':>6}")
print("-" * 58)

for label, cond in conditions:
    for mod_name, col in [('m6A', 'm6a_per_kb'), ('psi', 'psi_per_kb')]:
        valid = cond[cond[col].notna() & np.isfinite(cond[col])]
        if len(valid) >= 10:
            r, p = stats.spearmanr(valid['polya_length'], valid[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{label:<18} {mod_name:<8} {r:>7.3f} {p:>11.2e}({sig}) {len(valid):>6}")

# =========================================================================
# 5. Poly(A) quartile analysis: modification density across tail length bins
# =========================================================================
print(f"\n{'='*90}")
print("Modification Density by Poly(A) Quartile")
print(f"{'='*90}")

polya_bins = [0, 50, 100, 150, 200, 10000]
polya_labels = ['0-50', '50-100', '100-150', '150-200', '200+']

for label, cond in conditions:
    if len(cond) < 20:
        continue
    cond = cond.copy()
    cond['pa_bin'] = pd.cut(cond['polya_length'], bins=polya_bins, labels=polya_labels)
    print(f"\n  {label}:")
    print(f"  {'PA bin':<10} {'N':>5} {'m6A/kb':>8} {'psi/kb':>8} {'m6A%':>6} {'psi%':>6}")
    print(f"  {'-'*48}")
    for b in polya_labels:
        bdata = cond[cond['pa_bin'] == b]
        if len(bdata) >= 5:
            m6a_density = bdata['m6a_per_kb'].mean()
            psi_density = bdata['psi_per_kb'].mean()
            m6a_rate = (bdata['has_m6a']).mean() * 100
            psi_rate = (bdata['has_psi']).mean() * 100
            print(f"  {b:<10} {len(bdata):>5} {m6a_density:>8.3f} {psi_density:>8.3f} "
                  f"{m6a_rate:>5.1f}% {psi_rate:>5.1f}%")

# =========================================================================
# 6. Young vs Ancient L1 coupling under arsenite
# =========================================================================
print(f"\n{'='*90}")
print("Young vs Ancient L1: Poly(A) x Modification Coupling")
print(f"{'='*90}")

l1_data = merged[merged['source'].isin(['HeLa_L1', 'HeLa-Ars_L1'])].copy()
l1_data['condition'] = l1_data['source'].map({
    'HeLa_L1': 'HeLa', 'HeLa-Ars_L1': 'HeLa-Ars'})

for cond_name in ['HeLa', 'HeLa-Ars']:
    print(f"\n  [{cond_name}]")
    for age in ['young', 'ancient']:
        subset = l1_data[(l1_data['condition'] == cond_name) & (l1_data['l1_age'] == age)]
        if len(subset) < 10:
            print(f"    {age}: n={len(subset)} (too few)")
            continue
        # Modified vs unmodified poly(A)
        psi_pos = subset[subset['has_psi']]
        psi_neg = subset[~subset['has_psi']]
        if len(psi_pos) >= 5 and len(psi_neg) >= 5:
            _, p = stats.mannwhitneyu(psi_pos['polya_length'], psi_neg['polya_length'],
                                       alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        else:
            p = np.nan
            sig = "n/a"
        print(f"    {age}: n={len(subset)}  "
              f"psi+ polyA={psi_pos['polya_length'].median():.1f} (n={len(psi_pos)})  "
              f"psi- polyA={psi_neg['polya_length'].median():.1f} (n={len(psi_neg)})  "
              f"p={p:.2e}({sig})" if not np.isnan(p) else
              f"    {age}: n={len(subset)}  too few modified reads")

        # Spearman
        valid = subset[subset['psi_per_kb'].notna() & np.isfinite(subset['psi_per_kb'])]
        if len(valid) >= 10:
            r, p_corr = stats.spearmanr(valid['polya_length'], valid['psi_per_kb'])
            print(f"           psi/kb ~ polyA: r={r:.3f}  p={p_corr:.2e}")

# =========================================================================
# 7. Key finding: decorated reads and modifications
# =========================================================================
print(f"\n{'='*90}")
print("Decorated (Mixed Tail) Reads: Modification Status")
print(f"{'='*90}")

for label, cond in conditions:
    if len(cond) < 20:
        continue
    dec = cond[cond['tail_class'] == 'decorated']
    non_dec = cond[cond['tail_class'] != 'decorated']
    if len(dec) < 5:
        continue
    print(f"\n  {label}:")
    print(f"    Decorated:     n={len(dec):<5} m6A/kb={dec['m6a_per_kb'].mean():.3f}  "
          f"psi/kb={dec['psi_per_kb'].mean():.3f}  polyA={dec['polya_length'].median():.1f}")
    print(f"    Non-decorated: n={len(non_dec):<5} m6A/kb={non_dec['m6a_per_kb'].mean():.3f}  "
          f"psi/kb={non_dec['psi_per_kb'].mean():.3f}  polyA={non_dec['polya_length'].median():.1f}")
    # Test
    for mod_name, col in [('m6A/kb', 'm6a_per_kb'), ('psi/kb', 'psi_per_kb')]:
        d_vals = dec[col].dropna()
        nd_vals = non_dec[col].dropna()
        d_vals = d_vals[np.isfinite(d_vals)]
        nd_vals = nd_vals[np.isfinite(nd_vals)]
        if len(d_vals) >= 5 and len(nd_vals) >= 5:
            _, p = stats.mannwhitneyu(d_vals, nd_vals, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {mod_name} decorated vs non: p={p:.2e} ({sig})")

# =========================================================================
# 8. Summary
# =========================================================================
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")

# Quick summary stats
hela_l1 = merged[merged['source'] == 'HeLa_L1']
ars_l1 = merged[merged['source'] == 'HeLa-Ars_L1']

for src_label, src_data in [('HeLa L1', hela_l1), ('HeLa-Ars L1', ars_l1)]:
    r_m6a, p_m6a = stats.spearmanr(src_data['polya_length'], src_data['m6a_per_kb'])
    r_psi, p_psi = stats.spearmanr(src_data['polya_length'], src_data['psi_per_kb'])
    print(f"\n  {src_label} (n={len(src_data):,}):")
    print(f"    polyA ~ m6A/kb: r={r_m6a:.3f} (p={p_m6a:.2e})")
    print(f"    polyA ~ psi/kb: r={r_psi:.3f} (p={p_psi:.2e})")

# Save merged data
merged.to_csv(ANALYSIS / 'topic_02_polya/polya_modification_coupled.tsv',
              sep='\t', index=False)
print(f"\nSaved: polya_modification_coupled.tsv")
