#!/usr/bin/env python3
"""
Arsenite poly(A) tail analysis: L1 vs Control (non-L1 transcripts).

Key question: Is the arsenite-induced poly(A) shortening (~31nt in L1)
L1-specific or a global transcriptome effect?

Data sources:
  - L1: results_group/{group}/g_summary/{group}_L1_summary.tsv
  - Control: results_group/{group}/i_control/{group}_control_summary.tsv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUTPUT_DIR = PROJECT / 'analysis/01_exploration/topic_02_polya'

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# =========================================================================
# 1. Load data
# =========================================================================
print("Loading data...")

def load_l1(groups, label):
    """Load L1 summary for multiple replicate groups."""
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/g_summary/{g}_L1_summary.tsv'
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        df['group'] = g
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['source'] = label
    combined['l1_age'] = combined['gene_id'].apply(
        lambda x: 'young' if x in YOUNG else 'ancient')
    return combined

def load_ctrl(groups, label):
    """Load control summary for multiple replicate groups."""
    dfs = []
    for g in groups:
        path = PROJECT / f'results_group/{g}/i_control/{g}_control_summary.tsv'
        df = pd.read_csv(path, sep='\t')
        df = df[df['qc_tag'] == 'PASS'].copy()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined['source'] = label
    return combined

hela_l1 = load_l1(['HeLa_1', 'HeLa_2', 'HeLa_3'], 'HeLa')
ars_l1 = load_l1(['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'], 'HeLa-Ars')
hela_ctrl = load_ctrl(['HeLa_1', 'HeLa_2', 'HeLa_3'], 'HeLa')
ars_ctrl = load_ctrl(['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'], 'HeLa-Ars')

print(f"  HeLa L1:      {len(hela_l1):,} reads")
print(f"  HeLa-Ars L1:  {len(ars_l1):,} reads")
print(f"  HeLa Ctrl:    {len(hela_ctrl):,} reads")
print(f"  HeLa-Ars Ctrl:{len(ars_ctrl):,} reads")

# =========================================================================
# 2. Summary statistics
# =========================================================================
print(f"\n{'='*90}")
print("Poly(A) Tail Length: L1 vs Control under Arsenite")
print(f"{'='*90}")

categories = [
    ('HeLa L1 all',         hela_l1),
    ('HeLa-Ars L1 all',     ars_l1),
    ('HeLa L1 young',       hela_l1[hela_l1['l1_age'] == 'young']),
    ('HeLa-Ars L1 young',   ars_l1[ars_l1['l1_age'] == 'young']),
    ('HeLa L1 ancient',     hela_l1[hela_l1['l1_age'] == 'ancient']),
    ('HeLa-Ars L1 ancient', ars_l1[ars_l1['l1_age'] == 'ancient']),
    ('HeLa Control',        hela_ctrl),
    ('HeLa-Ars Control',    ars_ctrl),
]

print(f"\n{'Label':<24} {'N':>6} {'Mean':>7} {'Median':>7} {'Q25':>6} {'Q75':>6} {'IQR':>6}")
print("-" * 70)
for label, d in categories:
    n = len(d)
    med = d['polya_length'].median()
    mean = d['polya_length'].mean()
    q25 = d['polya_length'].quantile(0.25)
    q75 = d['polya_length'].quantile(0.75)
    iqr = q75 - q25
    print(f"{label:<24} {n:>6,} {mean:>7.1f} {med:>7.1f} {q25:>6.1f} {q75:>6.1f} {iqr:>6.1f}")

# =========================================================================
# 3. Pairwise comparisons: arsenite effect
# =========================================================================
print(f"\n{'='*90}")
print("Arsenite Effect on Poly(A) (HeLa vs HeLa-Ars)")
print(f"{'='*90}")

def compare_polya(label, df1, df2, name1='HeLa', name2='HeLa-Ars'):
    """Compare poly(A) between two groups."""
    med1 = df1['polya_length'].median()
    med2 = df2['polya_length'].median()
    delta = med2 - med1
    _, p = stats.mannwhitneyu(df1['polya_length'], df2['polya_length'],
                               alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    pct_change = (delta / med1) * 100
    print(f"  {label:<22} {name1}={med1:>7.1f}  {name2}={med2:>7.1f}  "
          f"Δ={delta:>+7.1f} ({pct_change:>+5.1f}%)  p={p:.2e} ({sig})")
    return {'label': label, 'med1': med1, 'med2': med2, 'delta': delta,
            'pct_change': pct_change, 'p': p, 'sig': sig}

results = []
results.append(compare_polya('L1 all', hela_l1, ars_l1))
results.append(compare_polya('L1 young',
                              hela_l1[hela_l1['l1_age'] == 'young'],
                              ars_l1[ars_l1['l1_age'] == 'young']))
results.append(compare_polya('L1 ancient',
                              hela_l1[hela_l1['l1_age'] == 'ancient'],
                              ars_l1[ars_l1['l1_age'] == 'ancient']))
results.append(compare_polya('Control (non-L1)', hela_ctrl, ars_ctrl))

# =========================================================================
# 4. Delta-delta: L1-specific vs global effect
# =========================================================================
print(f"\n{'='*90}")
print("Delta-Delta Analysis: Is L1 Shortening Specific?")
print(f"{'='*90}")

l1_delta = results[0]['delta']
ctrl_delta = results[3]['delta']
delta_delta = l1_delta - ctrl_delta

l1_pct = results[0]['pct_change']
ctrl_pct = results[3]['pct_change']
delta_pct = l1_pct - ctrl_pct

print(f"\n  L1 all Δ:     {l1_delta:>+.1f} nt ({l1_pct:>+.1f}%)")
print(f"  Control Δ:    {ctrl_delta:>+.1f} nt ({ctrl_pct:>+.1f}%)")
print(f"  ΔΔ (L1 - Ctrl): {delta_delta:>+.1f} nt ({delta_pct:>+.1f}%)")

# Young L1 vs Control
young_delta = results[1]['delta']
young_pct = results[1]['pct_change']
print(f"\n  L1 young Δ:   {young_delta:>+.1f} nt ({young_pct:>+.1f}%)")
print(f"  Control Δ:    {ctrl_delta:>+.1f} nt ({ctrl_pct:>+.1f}%)")
print(f"  ΔΔ (Young - Ctrl): {young_delta - ctrl_delta:>+.1f} nt ({young_pct - ctrl_pct:>+.1f}%)")

# Ancient L1 vs Control
anc_delta = results[2]['delta']
anc_pct = results[2]['pct_change']
print(f"\n  L1 ancient Δ: {anc_delta:>+.1f} nt ({anc_pct:>+.1f}%)")
print(f"  Control Δ:    {ctrl_delta:>+.1f} nt ({ctrl_pct:>+.1f}%)")
print(f"  ΔΔ (Ancient - Ctrl): {anc_delta - ctrl_delta:>+.1f} nt ({anc_pct - ctrl_pct:>+.1f}%)")

# Interpretation
print(f"\n  Interpretation:")
if abs(delta_pct) < 5:
    print(f"    → L1 and Control show similar arsenite-induced shortening")
    print(f"    → The effect is GLOBAL, not L1-specific")
elif delta_pct < -5:
    print(f"    → L1 shows MORE shortening than Control")
    print(f"    → L1 transcripts may be preferentially destabilized")
else:
    print(f"    → L1 shows LESS shortening than Control")
    print(f"    → L1 transcripts may be relatively protected")

# =========================================================================
# 5. Per-replicate consistency
# =========================================================================
print(f"\n{'='*90}")
print("Per-Replicate Poly(A) (Consistency Check)")
print(f"{'='*90}")

print(f"\n{'Group':<16} {'Type':<10} {'N':>6} {'Median':>8} {'Mean':>8}")
print("-" * 55)

for g in ['HeLa_1', 'HeLa_2', 'HeLa_3']:
    l1_d = hela_l1[hela_l1['group'] == g]
    ctrl_d = hela_ctrl[hela_ctrl['group'] == g]
    print(f"{g:<16} {'L1':<10} {len(l1_d):>6,} {l1_d['polya_length'].median():>8.1f} "
          f"{l1_d['polya_length'].mean():>8.1f}")
    print(f"{'':<16} {'Control':<10} {len(ctrl_d):>6,} {ctrl_d['polya_length'].median():>8.1f} "
          f"{ctrl_d['polya_length'].mean():>8.1f}")

for g in ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']:
    l1_d = ars_l1[ars_l1['group'] == g]
    ctrl_d = ars_ctrl[ars_ctrl['group'] == g]
    print(f"{g:<16} {'L1':<10} {len(l1_d):>6,} {l1_d['polya_length'].median():>8.1f} "
          f"{l1_d['polya_length'].mean():>8.1f}")
    print(f"{'':<16} {'Control':<10} {len(ctrl_d):>6,} {ctrl_d['polya_length'].median():>8.1f} "
          f"{ctrl_d['polya_length'].mean():>8.1f}")

# =========================================================================
# 6. Poly(A) length bins comparison
# =========================================================================
print(f"\n{'='*90}")
print("Poly(A) Length Bins: L1 vs Control under Arsenite")
print(f"{'='*90}")

bins = [0, 25, 50, 75, 100, 150, 200, 300, 10000]
bin_labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '200-300', '300+']

datasets = {
    'HeLa L1': hela_l1['polya_length'],
    'Ars L1': ars_l1['polya_length'],
    'HeLa Ctrl': hela_ctrl['polya_length'],
    'Ars Ctrl': ars_ctrl['polya_length'],
}

print(f"\n{'Bin':<10}", end="")
for name in datasets:
    print(f" {name:>12}", end="")
print()
print("-" * 62)

hists = {}
for name, vals in datasets.items():
    h = pd.cut(vals, bins=bins, labels=bin_labels).value_counts().sort_index()
    hists[name] = h / len(vals) * 100

for b in bin_labels:
    print(f"{b:<10}", end="")
    for name in datasets:
        pct = hists[name].get(b, 0)
        print(f" {pct:>11.1f}%", end="")
    print()

# =========================================================================
# 7. Decorated rate comparison: L1 vs Control under arsenite
# =========================================================================
print(f"\n{'='*90}")
print("Decorated Rate (Mixed Tail): L1 vs Control under Arsenite")
print(f"{'='*90}")

def decorated_stats(label, df):
    """Compute decorated rate for a group."""
    n = len(df)
    n_dec = (df['class'] == 'decorated').sum()
    rate = n_dec / n * 100 if n > 0 else 0
    return n, n_dec, rate

groups_dec = [
    ('HeLa L1 all',         hela_l1),
    ('HeLa-Ars L1 all',     ars_l1),
    ('HeLa L1 young',       hela_l1[hela_l1['l1_age'] == 'young']),
    ('HeLa-Ars L1 young',   ars_l1[ars_l1['l1_age'] == 'young']),
    ('HeLa L1 ancient',     hela_l1[hela_l1['l1_age'] == 'ancient']),
    ('HeLa-Ars L1 ancient', ars_l1[ars_l1['l1_age'] == 'ancient']),
    ('HeLa Control',        hela_ctrl),
    ('HeLa-Ars Control',    ars_ctrl),
]

print(f"\n{'Label':<24} {'N':>6} {'Decorated':>10} {'Rate':>8}")
print("-" * 55)
for label, d in groups_dec:
    n, n_dec, rate = decorated_stats(label, d)
    print(f"{label:<24} {n:>6,} {n_dec:>10,} {rate:>7.1f}%")

# Fisher's exact for decorated rate changes
print(f"\n  Arsenite effect on decorated rate:")
dec_comparisons = [
    ('L1 all',     hela_l1,    ars_l1),
    ('L1 young',   hela_l1[hela_l1['l1_age'] == 'young'],  ars_l1[ars_l1['l1_age'] == 'young']),
    ('L1 ancient', hela_l1[hela_l1['l1_age'] == 'ancient'],ars_l1[ars_l1['l1_age'] == 'ancient']),
    ('Control',    hela_ctrl,  ars_ctrl),
]

for label, d1, d2 in dec_comparisons:
    n1, dec1, rate1 = decorated_stats('', d1)
    n2, dec2, rate2 = decorated_stats('', d2)
    table = [[dec1, n1 - dec1], [dec2, n2 - dec2]]
    odds, p = stats.fisher_exact(table)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"    {label:<14} HeLa={rate1:.1f}%  Ars={rate2:.1f}%  "
          f"OR={odds:.2f}  p={p:.2e} ({sig})")

# =========================================================================
# 8. Poly(A) length controlled decorated rate
# =========================================================================
print(f"\n{'='*90}")
print("Decorated Rate by Poly(A) Bin (Length-Controlled)")
print(f"{'='*90}")

polya_bins_ctrl = [0, 50, 100, 150, 200, 10000]
polya_labels_ctrl = ['0-50', '50-100', '100-150', '150-200', '200+']

print(f"\n{'Bin':<10} {'HeLa L1':>10} {'Ars L1':>10} {'HeLa Ctrl':>11} {'Ars Ctrl':>10}")
print("-" * 55)

for ds_label, ds in [('HeLa L1', hela_l1), ('Ars L1', ars_l1),
                      ('HeLa Ctrl', hela_ctrl), ('Ars Ctrl', ars_ctrl)]:
    ds = ds.copy()
    ds['pa_bin'] = pd.cut(ds['polya_length'], bins=polya_bins_ctrl, labels=polya_labels_ctrl)

# Re-do with proper table format
for b in polya_labels_ctrl:
    row = f"{b:<10}"
    for ds_label, ds in [('HeLa L1', hela_l1), ('Ars L1', ars_l1),
                          ('HeLa Ctrl', hela_ctrl), ('Ars Ctrl', ars_ctrl)]:
        ds_copy = ds.copy()
        ds_copy['pa_bin'] = pd.cut(ds_copy['polya_length'], bins=polya_bins_ctrl,
                                    labels=polya_labels_ctrl)
        bin_data = ds_copy[ds_copy['pa_bin'] == b]
        if len(bin_data) >= 5:
            dec_rate = (bin_data['class'] == 'decorated').mean() * 100
            row += f" {dec_rate:>9.1f}%"
        else:
            row += f" {'n/a':>10}"
    print(row)

# =========================================================================
# 9. Baseline comparison: L1 vs Control poly(A) (same condition)
# =========================================================================
print(f"\n{'='*90}")
print("Baseline: L1 vs Control Poly(A) (Same Condition)")
print(f"{'='*90}")

baseline_tests = [
    ('HeLa: L1 vs Ctrl',     hela_l1, hela_ctrl),
    ('HeLa-Ars: L1 vs Ctrl', ars_l1, ars_ctrl),
]

for label, d1, d2 in baseline_tests:
    med1 = d1['polya_length'].median()
    med2 = d2['polya_length'].median()
    _, p = stats.mannwhitneyu(d1['polya_length'], d2['polya_length'],
                               alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label:<26} L1={med1:.1f}  Ctrl={med2:.1f}  "
          f"Δ={med1-med2:+.1f}  p={p:.2e} ({sig})")

# =========================================================================
# 10. Summary & conclusions
# =========================================================================
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")

l1_all_ars_effect = results[0]
ctrl_ars_effect = results[3]

print(f"""
  Arsenite poly(A) shortening:
    L1 all:     {l1_all_ars_effect['delta']:+.1f} nt ({l1_all_ars_effect['pct_change']:+.1f}%) p={l1_all_ars_effect['p']:.2e}
    Control:    {ctrl_ars_effect['delta']:+.1f} nt ({ctrl_ars_effect['pct_change']:+.1f}%) p={ctrl_ars_effect['p']:.2e}
    ΔΔ:         {l1_all_ars_effect['delta'] - ctrl_ars_effect['delta']:+.1f} nt

  Key question: Is arsenite poly(A) shortening L1-specific?
""")

if abs(l1_all_ars_effect['pct_change'] - ctrl_ars_effect['pct_change']) < 5:
    print("  → CONCLUSION: Arsenite-induced poly(A) shortening is a GLOBAL effect,")
    print("    affecting both L1 and non-L1 transcripts similarly.")
elif l1_all_ars_effect['pct_change'] < ctrl_ars_effect['pct_change'] - 5:
    print("  → CONCLUSION: L1 transcripts show GREATER shortening than Control,")
    print("    suggesting L1-specific vulnerability to arsenite-induced deadenylation.")
else:
    print("  → CONCLUSION: L1 transcripts show LESS shortening than Control,")
    print("    suggesting relative protection of L1 poly(A) under arsenite stress.")

# Save combined data for downstream
save_rows = []
for _, row in hela_l1[['polya_length', 'class', 'group', 'source', 'l1_age']].iterrows():
    save_rows.append({**row, 'type': 'L1'})
for _, row in ars_l1[['polya_length', 'class', 'group', 'source', 'l1_age']].iterrows():
    save_rows.append({**row, 'type': 'L1'})
for _, row in hela_ctrl[['polya_length', 'class', 'group', 'source']].iterrows():
    save_rows.append({**row, 'type': 'Control', 'l1_age': 'n/a'})
for _, row in ars_ctrl[['polya_length', 'class', 'group', 'source']].iterrows():
    save_rows.append({**row, 'type': 'Control', 'l1_age': 'n/a'})

save_df = pd.DataFrame(save_rows)
save_df.to_csv(OUTPUT_DIR / 'arsenite_l1_vs_control_polya.tsv', sep='\t', index=False)
print(f"\nSaved: arsenite_l1_vs_control_polya.tsv")
