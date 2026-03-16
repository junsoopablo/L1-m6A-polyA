#!/usr/bin/env python3
"""
Verify new Part3 cache (m6A thr=0.80) vs old (thr=0.50).
Quick comparison of key metrics across all cell lines.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

TOPIC = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline'
NEW_L1 = f'{TOPIC}/part3_l1_per_read_cache'
NEW_CTRL = f'{TOPIC}/part3_ctrl_per_read_cache'
OLD_L1 = f'{TOPIC}/part3_l1_per_read_cache_thr128_backup'
OLD_CTRL = f'{TOPIC}/part3_ctrl_per_read_cache_thr128_backup'

CELL_LINES = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
}


def load_cache(cache_dir, groups, suffix):
    """Load and concatenate cache files."""
    dfs = []
    for g in groups:
        path = f'{cache_dir}/{g}_{suffix}_per_read.tsv'
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            df['group'] = g
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


print("=" * 80)
print("Verification: m6A threshold 0.50 (old) vs 0.80 (new)")
print("=" * 80)

# ── 1. Per-CL m6A/kb comparison ──
print(f"\n{'Cell Line':>12s}  {'L1 old':>8s}  {'L1 new':>8s}  {'Ctrl old':>9s}  {'Ctrl new':>9s}  "
      f"{'Ratio old':>10s}  {'Ratio new':>10s}  {'Δ ratio':>8s}")

all_ratios_old = []
all_ratios_new = []

for cl, groups in CELL_LINES.items():
    # L1
    l1_old = load_cache(OLD_L1, groups, 'l1')
    l1_new = load_cache(NEW_L1, groups, 'l1')
    # Ctrl
    ctrl_old = load_cache(OLD_CTRL, groups, 'ctrl')
    ctrl_new = load_cache(NEW_CTRL, groups, 'ctrl')

    if l1_old.empty or ctrl_old.empty:
        continue

    l1_old_d = (l1_old['m6a_sites_high'] / (l1_old['read_length'] / 1000.0)).mean()
    l1_new_d = (l1_new['m6a_sites_high'] / (l1_new['read_length'] / 1000.0)).mean()
    ctrl_old_d = (ctrl_old['m6a_sites_high'] / (ctrl_old['read_length'] / 1000.0)).mean()
    ctrl_new_d = (ctrl_new['m6a_sites_high'] / (ctrl_new['read_length'] / 1000.0)).mean()

    ratio_old = l1_old_d / ctrl_old_d if ctrl_old_d > 0 else float('nan')
    ratio_new = l1_new_d / ctrl_new_d if ctrl_new_d > 0 else float('nan')
    delta = ratio_new - ratio_old

    all_ratios_old.append(ratio_old)
    all_ratios_new.append(ratio_new)

    print(f"{cl:>12s}  {l1_old_d:>8.3f}  {l1_new_d:>8.3f}  {ctrl_old_d:>9.3f}  {ctrl_new_d:>9.3f}  "
          f"{ratio_old:>10.3f}  {ratio_new:>10.3f}  {delta:>+8.3f}")

print(f"\n{'MEAN':>12s}  {'':>8s}  {'':>8s}  {'':>9s}  {'':>9s}  "
      f"{np.mean(all_ratios_old):>10.3f}  {np.mean(all_ratios_new):>10.3f}  "
      f"{np.mean(all_ratios_new) - np.mean(all_ratios_old):>+8.3f}")

# ── 2. Per-site rate comparison (pooled across all CL) ──
print(f"\n{'='*80}")
print("Per-site rate (L1 vs Ctrl, pooled)")
print(f"{'='*80}")

# We can't compute per-site rate from the cache alone (need total candidate sites).
# But we can compare m6A/kb which is the key metric.

# ── 3. Psi sanity check (should be unchanged) ──
print(f"\n{'='*80}")
print("Psi sanity check (threshold unchanged at 0.50)")
print(f"{'='*80}")

psi_ok = True
for cl, groups in CELL_LINES.items():
    l1_old = load_cache(OLD_L1, groups, 'l1')
    l1_new = load_cache(NEW_L1, groups, 'l1')
    if l1_old.empty:
        continue

    old_psi = l1_old['psi_sites_high'].sum()
    new_psi = l1_new['psi_sites_high'].sum()
    match = "OK" if old_psi == new_psi else f"MISMATCH ({old_psi} vs {new_psi})"
    if old_psi != new_psi:
        psi_ok = False
    print(f"  {cl:>12s}: psi old={old_psi:,} new={new_psi:,} → {match}")

print(f"\n  Psi consistency: {'PASS' if psi_ok else 'FAIL'}")

# ── 4. HeLa-Ars m6A-polyA check ──
print(f"\n{'='*80}")
print("HeLa-Ars m6A-polyA dose-response")
print(f"{'='*80}")

PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
ars_groups = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

l1_new = load_cache(NEW_L1, ars_groups, 'l1')
l1_old = load_cache(OLD_L1, ars_groups, 'l1')

# Load poly(A) from L1 summary
polya_dfs = []
for g in ars_groups:
    path = f'{PROJECT}/results_group/{g}/g_summary/{g}_L1_summary.tsv'
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        if 'read_id' in df.columns and 'polya_length' in df.columns:
            polya_dfs.append(df[['read_id', 'polya_length']].drop_duplicates('read_id'))

if polya_dfs:
    polya = pd.concat(polya_dfs, ignore_index=True)

    for label, cache_df in [('Old (thr=0.50)', l1_old), ('New (thr=0.80)', l1_new)]:
        merged = cache_df.merge(polya, on='read_id', how='inner')
        merged = merged[merged['polya_length'].notna()]
        merged['m6a_kb'] = merged['m6a_sites_high'] / (merged['read_length'] / 1000.0)
        rho, p = stats.spearmanr(merged['m6a_kb'], merged['polya_length'])
        print(f"  {label}: N={len(merged):,}, rho={rho:.4f}, P={p:.2e}")

        # Quartiles
        try:
            merged['q'] = pd.qcut(merged['m6a_kb'].rank(method='first'), 4,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4'])
            q1 = merged[merged['q'] == 'Q1']['polya_length'].mean()
            q4 = merged[merged['q'] == 'Q4']['polya_length'].mean()
            print(f"    Q1 polyA={q1:.1f}nt, Q4 polyA={q4:.1f}nt, range={q4-q1:.1f}nt")
        except Exception:
            pass

print("\nDone.")
