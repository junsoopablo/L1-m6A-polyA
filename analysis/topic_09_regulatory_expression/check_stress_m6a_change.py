#!/usr/bin/env python3
"""
Check: does L1 m6A level change under arsenite stress?
Both absolute and relative to control.
"""
import pandas as pd
import numpy as np
from scipy import stats
import glob

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/'
TOPIC = 'analysis/01_exploration/topic_05_cellline/'

# --- 1. Load L1 Part3 cache ---
import os
l1_files = glob.glob(BASE + TOPIC + 'part3_l1_per_read_cache/*.tsv')
l1_dfs = []
for f in l1_files:
    df = pd.read_csv(f, sep='\t')
    # Extract group from filename: e.g. HeLa_1_l1_per_read.tsv -> HeLa_1
    fname = os.path.basename(f)
    group = fname.replace('_l1_per_read.tsv', '')
    df['group'] = group
    l1_dfs.append(df)
l1 = pd.concat(l1_dfs, ignore_index=True)
l1['m6a_kb'] = l1['m6a_sites_high'] / l1['read_length'] * 1000
l1['psi_kb'] = l1['psi_sites_high'] / l1['read_length'] * 1000

# --- 2. Load Control Part3 cache ---
ctrl_files = glob.glob(BASE + TOPIC + 'part3_ctrl_per_read_cache/*.tsv')
ctrl_dfs = []
for f in ctrl_files:
    df = pd.read_csv(f, sep='\t')
    fname = os.path.basename(f)
    group = fname.replace('_ctrl_per_read.tsv', '')
    df['group'] = group
    ctrl_dfs.append(df)
ctrl = pd.concat(ctrl_dfs, ignore_index=True)
ctrl['m6a_kb'] = ctrl['m6a_sites_high'] / ctrl['read_length'] * 1000
ctrl['psi_kb'] = ctrl['psi_sites_high'] / ctrl['read_length'] * 1000

# --- 3. Define groups ---
hela_normal = ['HeLa_1', 'HeLa_2', 'HeLa_3']
hela_ars = ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

print("=" * 70)
print("STRESS vs NORMAL: L1 and Control m6A/kb comparison")
print("=" * 70)

# --- 4. L1: HeLa vs HeLa-Ars ---
l1_normal = l1[l1['group'].isin(hela_normal)]
l1_stress = l1[l1['group'].isin(hela_ars)]

print(f"\n--- L1 reads ---")
print(f"  HeLa normal: n={len(l1_normal)}, m6A/kb mean={l1_normal['m6a_kb'].mean():.3f}, "
      f"median={l1_normal['m6a_kb'].median():.3f}")
print(f"  HeLa-Ars:    n={len(l1_stress)}, m6A/kb mean={l1_stress['m6a_kb'].mean():.3f}, "
      f"median={l1_stress['m6a_kb'].median():.3f}")
u, p = stats.mannwhitneyu(l1_normal['m6a_kb'], l1_stress['m6a_kb'], alternative='two-sided')
ks_stat, ks_p = stats.ks_2samp(l1_normal['m6a_kb'], l1_stress['m6a_kb'])
print(f"  MW P={p:.2e}, KS D={ks_stat:.3f} P={ks_p:.2e}")
print(f"  Ratio (Ars/Normal): {l1_stress['m6a_kb'].mean() / l1_normal['m6a_kb'].mean():.3f}")

# --- 5. Control: HeLa vs HeLa-Ars ---
ctrl_normal = ctrl[ctrl['group'].isin(hela_normal)]
ctrl_stress = ctrl[ctrl['group'].isin(hela_ars)]

print(f"\n--- Control reads ---")
print(f"  HeLa normal: n={len(ctrl_normal)}, m6A/kb mean={ctrl_normal['m6a_kb'].mean():.3f}, "
      f"median={ctrl_normal['m6a_kb'].median():.3f}")
print(f"  HeLa-Ars:    n={len(ctrl_stress)}, m6A/kb mean={ctrl_stress['m6a_kb'].mean():.3f}, "
      f"median={ctrl_stress['m6a_kb'].median():.3f}")
if len(ctrl_stress) > 0:
    u2, p2 = stats.mannwhitneyu(ctrl_normal['m6a_kb'], ctrl_stress['m6a_kb'], alternative='two-sided')
    ks2, ks2p = stats.ks_2samp(ctrl_normal['m6a_kb'], ctrl_stress['m6a_kb'])
    print(f"  MW P={p2:.2e}, KS D={ks2:.3f} P={ks2p:.2e}")
    print(f"  Ratio (Ars/Normal): {ctrl_stress['m6a_kb'].mean() / ctrl_normal['m6a_kb'].mean():.3f}")
else:
    print("  *** No HeLa-Ars control data found in Part3 cache ***")

# --- 6. L1/Control ratio change ---
print(f"\n--- L1/Control m6A/kb ratio ---")
ratio_normal = l1_normal['m6a_kb'].mean() / ctrl_normal['m6a_kb'].mean() if len(ctrl_normal) > 0 else None
print(f"  Normal: L1={l1_normal['m6a_kb'].mean():.3f} / Ctrl={ctrl_normal['m6a_kb'].mean():.3f} = {ratio_normal:.3f}x")
if len(ctrl_stress) > 0:
    ratio_stress = l1_stress['m6a_kb'].mean() / ctrl_stress['m6a_kb'].mean()
    print(f"  Stress: L1={l1_stress['m6a_kb'].mean():.3f} / Ctrl={ctrl_stress['m6a_kb'].mean():.3f} = {ratio_stress:.3f}x")
    print(f"  Ratio change: {ratio_stress/ratio_normal:.3f}x (stress/normal)")

# --- 7. Also check psi ---
print(f"\n--- psi/kb comparison ---")
print(f"  L1 Normal: {l1_normal['psi_kb'].mean():.3f}, L1 Stress: {l1_stress['psi_kb'].mean():.3f}, "
      f"ratio={l1_stress['psi_kb'].mean()/l1_normal['psi_kb'].mean():.3f}")
if len(ctrl_stress) > 0:
    print(f"  Ctrl Normal: {ctrl_normal['psi_kb'].mean():.3f}, Ctrl Stress: {ctrl_stress['psi_kb'].mean():.3f}, "
          f"ratio={ctrl_stress['psi_kb'].mean()/ctrl_normal['psi_kb'].mean():.3f}")

# --- 8. Per-replicate breakdown ---
print(f"\n--- Per-replicate L1 m6A/kb ---")
for grp in hela_normal + hela_ars:
    sub = l1[l1['group'] == grp]
    if len(sub) > 0:
        tag = "ARS" if "Ars" in grp else "   "
        print(f"  {tag} {grp:20s}: n={len(sub):5d}, m6A/kb={sub['m6a_kb'].mean():.3f} ± {sub['m6a_kb'].std():.3f}")

print(f"\n--- Per-replicate Control m6A/kb ---")
for grp in hela_normal + hela_ars:
    sub = ctrl[ctrl['group'] == grp]
    if len(sub) > 0:
        tag = "ARS" if "Ars" in grp else "   "
        print(f"  {tag} {grp:20s}: n={len(sub):5d}, m6A/kb={sub['m6a_kb'].mean():.3f} ± {sub['m6a_kb'].std():.3f}")

# --- 9. Cross all cell lines: L1 m6A/kb per group ---
print(f"\n--- All cell lines: L1 m6A/kb ---")
for grp in sorted(l1['group'].unique()):
    sub = l1[l1['group'] == grp]
    tag = "ARS" if "Ars" in grp else "EV " if "EV" in grp else "   "
    print(f"  {tag} {grp:25s}: n={len(sub):5d}, m6A/kb={sub['m6a_kb'].mean():.3f}")

# --- 10. m6a_sites_high per read (not normalized) ---
print(f"\n--- m6A sites per read (absolute, not /kb) ---")
print(f"  L1 Normal: {l1_normal['m6a_sites_high'].mean():.2f} sites/read (RL={l1_normal['read_length'].mean():.0f}bp)")
print(f"  L1 Stress: {l1_stress['m6a_sites_high'].mean():.2f} sites/read (RL={l1_stress['read_length'].mean():.0f}bp)")
print(f"  Sites ratio: {l1_stress['m6a_sites_high'].mean()/l1_normal['m6a_sites_high'].mean():.3f}")
print(f"  RL ratio: {l1_stress['read_length'].mean()/l1_normal['read_length'].mean():.3f}")
