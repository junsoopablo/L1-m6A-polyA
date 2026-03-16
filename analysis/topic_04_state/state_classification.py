#!/usr/bin/env python3
"""
Read-level state classification and arsenite-induced state rewiring.

Each L1 read is classified into a multi-feature state:
  - Poly(A) tail: short vs long (median-based threshold)
  - Pseudouridine: low vs high (has psi_sites_high > 0)
  - m6A: low vs high (has m6a_sites_high > 0)
  - Mixed tail: decorated vs non-decorated

States (simplified):
  A: long tail + low psi   ("stable, unmodified")
  B: long tail + high psi  ("stable, modified")
  C: short tail + high psi ("destabilized, modified")
  D: short tail + low psi  ("destabilized, unmodified")

Key question: Does arsenite shift L1 state occupancy?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
ANALYSIS = PROJECT / 'analysis/01_exploration'
OUTPUT_DIR = ANALYSIS / 'topic_04_state'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load coupled data
# =========================================================================
print("Loading coupled poly(A) + modification data...")
df = pd.read_csv(ANALYSIS / 'topic_02_polya/polya_modification_coupled.tsv', sep='\t')

# Focus on L1 reads
l1 = df[df['l1_age'].isin(['young', 'ancient'])].copy()
print(f"  L1 reads: {len(l1):,}")
print(f"  Sources: {l1['source'].value_counts().to_dict()}")

# Also load control for comparison
ctrl = df[df['source'].isin(['HeLa_Ctrl', 'HeLa-Ars_Ctrl'])].copy()
print(f"  Ctrl reads: {len(ctrl):,}")

# =========================================================================
# 2. Define state thresholds
# =========================================================================
# Use HeLa L1 median as reference (pre-stress baseline)
hela_l1 = l1[l1['source'] == 'HeLa_L1']
POLYA_THRESHOLD = hela_l1['polya_length'].median()
print(f"\n  Poly(A) threshold (HeLa L1 median): {POLYA_THRESHOLD:.1f} nt")

# State assignment
def assign_state(row, pa_thresh):
    """Assign read to a state based on poly(A) and psi."""
    long_tail = row['polya_length'] >= pa_thresh
    has_psi = row['psi_sites_high'] > 0
    has_m6a = row['m6a_sites_high'] > 0

    if long_tail and not has_psi:
        return 'A'  # long tail, low psi
    elif long_tail and has_psi:
        return 'B'  # long tail, high psi
    elif not long_tail and has_psi:
        return 'C'  # short tail, high psi
    else:
        return 'D'  # short tail, low psi

l1['state'] = l1.apply(lambda r: assign_state(r, POLYA_THRESHOLD), axis=1)
ctrl['state'] = ctrl.apply(lambda r: assign_state(r, POLYA_THRESHOLD), axis=1)

# Also add m6A sub-states
def assign_state_full(row, pa_thresh):
    """Assign 8-state classification (tail x psi x m6A)."""
    long_tail = 'long' if row['polya_length'] >= pa_thresh else 'short'
    psi = 'psi+' if row['psi_sites_high'] > 0 else 'psi-'
    m6a = 'm6A+' if row['m6a_sites_high'] > 0 else 'm6A-'
    return f"{long_tail}_{psi}_{m6a}"

l1['state_full'] = l1.apply(lambda r: assign_state_full(r, POLYA_THRESHOLD), axis=1)

# =========================================================================
# 3. State occupancy overview
# =========================================================================
print(f"\n{'='*90}")
print("State Occupancy: L1 Reads")
print(f"(A=long+psi-, B=long+psi+, C=short+psi+, D=short+psi-)")
print(f"{'='*90}")

state_labels = {
    'A': 'long tail, low psi',
    'B': 'long tail, high psi',
    'C': 'short tail, high psi',
    'D': 'short tail, low psi',
}

conditions = [
    ('HeLa L1 all',       l1[l1['source'] == 'HeLa_L1']),
    ('HeLa-Ars L1 all',   l1[l1['source'] == 'HeLa-Ars_L1']),
    ('HeLa L1 young',     l1[(l1['source'] == 'HeLa_L1') & (l1['l1_age'] == 'young')]),
    ('HeLa-Ars L1 young', l1[(l1['source'] == 'HeLa-Ars_L1') & (l1['l1_age'] == 'young')]),
    ('HeLa L1 ancient',   l1[(l1['source'] == 'HeLa_L1') & (l1['l1_age'] == 'ancient')]),
    ('HeLa-Ars L1 ancient',l1[(l1['source'] == 'HeLa-Ars_L1') & (l1['l1_age'] == 'ancient')]),
    ('HeLa Ctrl',         ctrl[ctrl['source'] == 'HeLa_Ctrl']),
    ('HeLa-Ars Ctrl',     ctrl[ctrl['source'] == 'HeLa-Ars_Ctrl']),
]

print(f"\n{'Condition':<24} {'N':>5} {'A%':>7} {'B%':>7} {'C%':>7} {'D%':>7}")
print("-" * 62)

occupancy_data = []
for label, cond in conditions:
    n = len(cond)
    state_counts = cond['state'].value_counts()
    pcts = {s: state_counts.get(s, 0) / n * 100 for s in 'ABCD'}
    print(f"{label:<24} {n:>5} {pcts['A']:>6.1f}% {pcts['B']:>6.1f}% "
          f"{pcts['C']:>6.1f}% {pcts['D']:>6.1f}%")
    occupancy_data.append({'condition': label, 'n': n, **pcts})

# =========================================================================
# 4. Arsenite state shift (chi-square test)
# =========================================================================
print(f"\n{'='*90}")
print("Arsenite State Shift (Chi-Square Test)")
print(f"{'='*90}")

shift_tests = [
    ('L1 all',     l1[l1['source'] == 'HeLa_L1'],     l1[l1['source'] == 'HeLa-Ars_L1']),
    ('L1 young',   l1[(l1['source'] == 'HeLa_L1') & (l1['l1_age'] == 'young')],
                   l1[(l1['source'] == 'HeLa-Ars_L1') & (l1['l1_age'] == 'young')]),
    ('L1 ancient', l1[(l1['source'] == 'HeLa_L1') & (l1['l1_age'] == 'ancient')],
                   l1[(l1['source'] == 'HeLa-Ars_L1') & (l1['l1_age'] == 'ancient')]),
    ('Control',    ctrl[ctrl['source'] == 'HeLa_Ctrl'], ctrl[ctrl['source'] == 'HeLa-Ars_Ctrl']),
]

for label, hela_grp, ars_grp in shift_tests:
    hela_counts = hela_grp['state'].value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    ars_counts = ars_grp['state'].value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)

    # Contingency table
    table = np.array([hela_counts.values, ars_counts.values])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"\n  {label}: chi2={chi2:.1f}, df={dof}, p={p:.2e} ({sig})")

    hela_n = len(hela_grp)
    ars_n = len(ars_grp)
    for s in 'ABCD':
        h_pct = hela_counts[s] / hela_n * 100
        a_pct = ars_counts[s] / ars_n * 100
        delta = a_pct - h_pct
        arrow = "↑" if delta > 2 else "↓" if delta < -2 else "→"
        print(f"    State {s} ({state_labels[s]}): "
              f"HeLa={h_pct:.1f}% → Ars={a_pct:.1f}% (Δ={delta:+.1f}%) {arrow}")

# =========================================================================
# 5. State transition flow: which states gain/lose
# =========================================================================
print(f"\n{'='*90}")
print("State Occupancy Change Summary (Ars - HeLa)")
print(f"{'='*90}")

hela_all = l1[l1['source'] == 'HeLa_L1']
ars_all = l1[l1['source'] == 'HeLa-Ars_L1']
hela_n = len(hela_all)
ars_n = len(ars_all)

print(f"\n  {'State':<8} {'HeLa':>8} {'HeLa-Ars':>10} {'Δ':>8} {'Direction'}")
print(f"  {'-'*45}")
for s in 'ABCD':
    h = (hela_all['state'] == s).sum() / hela_n * 100
    a = (ars_all['state'] == s).sum() / ars_n * 100
    d = a - h
    direction = "GAIN" if d > 2 else "LOSS" if d < -2 else "stable"
    print(f"  {s:<8} {h:>7.1f}% {a:>9.1f}% {d:>+7.1f}% {direction}")

print(f"\n  Interpretation:")
# Find biggest gainer and loser
gains = {}
for s in 'ABCD':
    h = (hela_all['state'] == s).sum() / hela_n * 100
    a = (ars_all['state'] == s).sum() / ars_n * 100
    gains[s] = a - h
biggest_gain = max(gains, key=gains.get)
biggest_loss = min(gains, key=gains.get)
print(f"    Biggest GAIN:  State {biggest_gain} ({state_labels[biggest_gain]}) +{gains[biggest_gain]:.1f}%")
print(f"    Biggest LOSS:  State {biggest_loss} ({state_labels[biggest_loss]}) {gains[biggest_loss]:.1f}%")

# =========================================================================
# 6. Full 8-state analysis (tail x psi x m6A)
# =========================================================================
print(f"\n{'='*90}")
print("Full 8-State Classification (tail x psi x m6A)")
print(f"{'='*90}")

full_states = sorted(l1['state_full'].unique())
print(f"\n{'State':<22} {'HeLa%':>7} {'Ars%':>7} {'Δ%':>7}")
print("-" * 48)

for fs in full_states:
    h = (hela_all['state_full'] == fs).sum() / hela_n * 100
    a = (ars_all['state_full'] == fs).sum() / ars_n * 100
    d = a - h
    marker = " <<<" if abs(d) > 3 else ""
    print(f"{fs:<22} {h:>6.1f}% {a:>6.1f}% {d:>+6.1f}%{marker}")

# =========================================================================
# 7. Per-state mean features
# =========================================================================
print(f"\n{'='*90}")
print("Per-State Feature Means (L1 all)")
print(f"{'='*90}")

print(f"\n{'State':<8} {'N':>6} {'polyA':>7} {'m6A/kb':>8} {'psi/kb':>8} {'rdLen':>7} {'dec%':>6}")
print("-" * 55)

for s in 'ABCD':
    sd = l1[l1['state'] == s]
    dec_rate = (sd['tail_class'] == 'decorated').mean() * 100
    print(f"{s:<8} {len(sd):>6,} {sd['polya_length'].median():>7.1f} "
          f"{sd['m6a_per_kb'].mean():>8.3f} {sd['psi_per_kb'].mean():>8.3f} "
          f"{sd['read_length'].mean():>7.0f} {dec_rate:>5.1f}%")

# =========================================================================
# 8. Young L1 state analysis
# =========================================================================
print(f"\n{'='*90}")
print("Young L1 State Occupancy: HeLa vs HeLa-Ars")
print(f"{'='*90}")

for age in ['young', 'ancient']:
    print(f"\n  [{age.upper()}]")
    hela_age = l1[(l1['source'] == 'HeLa_L1') & (l1['l1_age'] == age)]
    ars_age = l1[(l1['source'] == 'HeLa-Ars_L1') & (l1['l1_age'] == age)]
    h_n = len(hela_age)
    a_n = len(ars_age)

    if h_n < 10 or a_n < 10:
        print(f"    Too few reads: HeLa={h_n}, Ars={a_n}")
        continue

    print(f"    {'State':<8} {'HeLa':>8} {'Ars':>8} {'Δ':>8}")
    print(f"    {'-'*35}")
    for s in 'ABCD':
        h = (hela_age['state'] == s).sum() / h_n * 100
        a = (ars_age['state'] == s).sum() / a_n * 100
        d = a - h
        print(f"    {s:<8} {h:>7.1f}% {a:>7.1f}% {d:>+7.1f}%")

    # Chi-square
    hela_counts = hela_age['state'].value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    ars_counts = ars_age['state'].value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    table = np.array([hela_counts.values, ars_counts.values])
    # Only test if all cells > 0
    if (table > 0).all():
        chi2, p, _, _ = stats.chi2_contingency(table)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    Chi2={chi2:.1f} p={p:.2e} ({sig})")
    else:
        print(f"    Chi2: some cells=0, test not reliable")

# =========================================================================
# 9. Sensitivity analysis: different poly(A) thresholds
# =========================================================================
print(f"\n{'='*90}")
print("Sensitivity: State Shift Direction at Different Poly(A) Thresholds")
print(f"{'='*90}")

thresholds = [75, 100, 121, 150, 200]
hela_all_l1 = l1[l1['source'] == 'HeLa_L1']
ars_all_l1 = l1[l1['source'] == 'HeLa-Ars_L1']

print(f"\n{'Thresh':>7}", end="")
for s in 'ABCD':
    print(f"  {'Δ'+s:>7}", end="")
print(f"  {'chi2 p':>10}")
print("-" * 55)

for thresh in thresholds:
    hela_states = hela_all_l1.apply(lambda r: assign_state(r, thresh), axis=1)
    ars_states = ars_all_l1.apply(lambda r: assign_state(r, thresh), axis=1)

    print(f"{thresh:>5}nt", end="")
    deltas = {}
    for s in 'ABCD':
        h = (hela_states == s).sum() / len(hela_all_l1) * 100
        a = (ars_states == s).sum() / len(ars_all_l1) * 100
        d = a - h
        deltas[s] = d
        print(f"  {d:>+6.1f}%", end="")

    hela_c = hela_states.value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    ars_c = ars_states.value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    table = np.array([hela_c.values, ars_c.values])
    chi2, p, _, _ = stats.chi2_contingency(table)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {p:.2e}({sig})")

# =========================================================================
# 10. Summary
# =========================================================================
print(f"\n{'='*90}")
print("SUMMARY: State Rewiring under Arsenite")
print(f"{'='*90}")

print(f"""
  State definitions (poly(A) threshold = {POLYA_THRESHOLD:.0f}nt):
    A: long tail + low psi   ("stable, unmodified")
    B: long tail + high psi  ("stable, modified")
    C: short tail + high psi ("destabilized, modified")
    D: short tail + low psi  ("destabilized, unmodified")
""")

# Compute final delta
for s in 'ABCD':
    h = (hela_all['state'] == s).sum() / hela_n * 100
    a = (ars_all['state'] == s).sum() / ars_n * 100
    d = a - h
    print(f"  State {s}: {h:.1f}% → {a:.1f}% (Δ={d:+.1f}%)")

# Save
l1.to_csv(OUTPUT_DIR / 'l1_state_classification.tsv', sep='\t', index=False)
ctrl.to_csv(OUTPUT_DIR / 'ctrl_state_classification.tsv', sep='\t', index=False)
pd.DataFrame(occupancy_data).to_csv(OUTPUT_DIR / 'state_occupancy_summary.tsv',
                                     sep='\t', index=False)
print(f"\nSaved: l1_state_classification.tsv, ctrl_state_classification.tsv, state_occupancy_summary.tsv")
