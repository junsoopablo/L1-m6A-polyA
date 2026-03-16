#!/usr/bin/env python3
"""
B1 Analysis: METTL3 KO Control poly(A) specificity
Compare WT vs KO poly(A) for BOTH L1 and Control transcripts
to determine if poly(A) lengthening (+13.5nt) is L1-specific or global.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, gzip

BASE = "/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore"

# ── 1. Load nanopolish poly(A) for all 6 samples ──
def load_polya(sample_name):
    path = f"{BASE}/nanopolish/{sample_name}_guppy_polya.tsv.gz"
    df = pd.read_csv(path, sep='\t')
    df = df[df['qc_tag'] == 'PASS'][['readname', 'polya_length']].copy()
    df.rename(columns={'readname': 'read_id'}, inplace=True)
    return df

print("Loading nanopolish poly(A) data...")
polya_dfs = {}
for cond in ['WT', 'KO']:
    for rep in [1, 2, 3]:
        name = f"{cond}_rep{rep}"
        polya_dfs[name] = load_polya(name)
        print(f"  {name}: {len(polya_dfs[name])} PASS reads")

# ── 2. Load control read IDs ──
print("\nLoading control read IDs...")
ctrl = pd.read_csv(f"{BASE}/analysis/matched_guppy/ctrl_per_read_gene.tsv", sep='\t')
print(f"  Total control reads: {len(ctrl)}")
print(f"  WT: {(ctrl['condition']=='WT').sum()}, KO: {(ctrl['condition']=='KO').sum()}")

# ── 3. Load L1 read IDs ──
print("\nLoading L1 read IDs...")
l1 = pd.read_csv(f"{BASE}/analysis/matched_guppy/ml204_l1_per_read.tsv", sep='\t')
print(f"  Total L1 reads: {len(l1)}")
print(f"  WT: {(l1['condition']=='WT').sum()}, KO: {(l1['condition']=='KO').sum()}")

# ── 4. Concatenate all poly(A) ──
polya_all = pd.concat(polya_dfs.values(), ignore_index=True)
# Drop duplicates (a read shouldn't appear in multiple samples, but safety)
polya_all = polya_all.drop_duplicates(subset='read_id')
print(f"\nTotal poly(A) PASS reads: {len(polya_all)}")

# ── 5. Join control reads with poly(A) ──
ctrl_polya = ctrl.merge(polya_all, on='read_id', how='inner')
print(f"\nControl reads with poly(A): {len(ctrl_polya)}")
print(f"  WT: {(ctrl_polya['condition']=='WT').sum()}")
print(f"  KO: {(ctrl_polya['condition']=='KO').sum()}")

# ── 6. Join L1 reads with poly(A) ──
l1_polya = l1.merge(polya_all, on='read_id', how='inner')
print(f"\nL1 reads with poly(A): {len(l1_polya)}")
print(f"  WT: {(l1_polya['condition']=='WT').sum()}")
print(f"  KO: {(l1_polya['condition']=='KO').sum()}")

# ── 7. Compare WT vs KO poly(A) ──
print("\n" + "="*70)
print("RESULTS: METTL3 KO Poly(A) Specificity")
print("="*70)

for biotype, df in [("Control", ctrl_polya), ("L1", l1_polya)]:
    wt = df[df['condition'] == 'WT']['polya_length']
    ko = df[df['condition'] == 'KO']['polya_length']

    wt_med = wt.median()
    ko_med = ko.median()
    delta = ko_med - wt_med

    stat, pval = stats.mannwhitneyu(ko, wt, alternative='two-sided')

    print(f"\n--- {biotype} ---")
    print(f"  WT: n={len(wt)}, median={wt_med:.1f} nt, mean={wt.mean():.1f} nt")
    print(f"  KO: n={len(ko)}, median={ko_med:.1f} nt, mean={ko.mean():.1f} nt")
    print(f"  Δ (KO - WT): {delta:+.1f} nt")
    print(f"  Mann-Whitney P = {pval:.2e}")

# ── 8. Per-replicate breakdown ──
print("\n" + "="*70)
print("Per-Replicate Breakdown")
print("="*70)

for biotype, df in [("Control", ctrl_polya), ("L1", l1_polya)]:
    print(f"\n--- {biotype} ---")
    for cond in ['WT', 'KO']:
        for rep in ['rep1', 'rep2', 'rep3']:
            if 'rep' in df.columns:
                sub = df[(df['condition'] == cond) & (df['rep'] == rep)]
            else:
                sample_key = f"{cond}_{rep}"
                sub = df[df['sample'] == sample_key]
            if len(sub) > 0:
                print(f"  {cond}_{rep}: n={len(sub)}, median={sub['polya_length'].median():.1f} nt")

# ── 9. Specificity test ──
print("\n" + "="*70)
print("SPECIFICITY ASSESSMENT")
print("="*70)

ctrl_wt = ctrl_polya[ctrl_polya['condition'] == 'WT']['polya_length']
ctrl_ko = ctrl_polya[ctrl_polya['condition'] == 'KO']['polya_length']
l1_wt = l1_polya[l1_polya['condition'] == 'WT']['polya_length']
l1_ko = l1_polya[l1_polya['condition'] == 'KO']['polya_length']

ctrl_delta = ctrl_ko.median() - ctrl_wt.median()
l1_delta = l1_ko.median() - l1_wt.median()

print(f"\nControl Δ: {ctrl_delta:+.1f} nt")
print(f"L1 Δ:     {l1_delta:+.1f} nt")
print(f"ΔΔ (L1 - Control): {l1_delta - ctrl_delta:+.1f} nt")

if abs(ctrl_delta) < 5:
    print("\n→ CONCLUSION: Control poly(A) is UNCHANGED in METTL3 KO.")
    print("  L1 poly(A) lengthening is L1-SPECIFIC.")
    print("  → STRONG support for m6A-mediated L1 decay model.")
elif ctrl_delta > 0 and abs(ctrl_delta) >= 5:
    print(f"\n→ CONCLUSION: Control poly(A) is ALSO LENGTHENED in KO ({ctrl_delta:+.1f} nt).")
    if abs(l1_delta) > abs(ctrl_delta) * 1.5:
        print(f"  L1 lengthening ({l1_delta:+.1f} nt) exceeds Control ({ctrl_delta:+.1f} nt).")
        print("  → L1 shows DISPROPORTIONATE lengthening, consistent with enhanced m6A-dependent decay on L1.")
    else:
        print(f"  L1 ({l1_delta:+.1f} nt) and Control ({ctrl_delta:+.1f} nt) show SIMILAR lengthening.")
        print("  → Effect is GLOBAL, not L1-specific.")
        print("  → Must reframe: focus on L1/Ctrl enrichment ratio preservation instead.")
else:
    print(f"\n→ CONCLUSION: Unexpected pattern. Needs manual inspection.")

# ── 10. Gene biotype breakdown for controls ──
print("\n" + "="*70)
print("Control poly(A) by Gene Biotype (top biotypes)")
print("="*70)

for bt in ctrl_polya['gene_biotype'].value_counts().head(5).index:
    sub = ctrl_polya[ctrl_polya['gene_biotype'] == bt]
    wt = sub[sub['condition'] == 'WT']['polya_length']
    ko = sub[sub['condition'] == 'KO']['polya_length']
    if len(wt) >= 10 and len(ko) >= 10:
        delta = ko.median() - wt.median()
        _, p = stats.mannwhitneyu(ko, wt, alternative='two-sided')
        print(f"  {bt}: WT median={wt.median():.1f}, KO median={ko.median():.1f}, Δ={delta:+.1f}, P={p:.2e} (n_WT={len(wt)}, n_KO={len(ko)})")

# ── 11. Save summary for manuscript integration ──
outdir = os.path.dirname(os.path.abspath(__file__))
summary = {
    'metric': ['ctrl_wt_median', 'ctrl_ko_median', 'ctrl_delta', 'ctrl_p',
               'l1_wt_median', 'l1_ko_median', 'l1_delta', 'l1_p',
               'delta_delta', 'ctrl_wt_n', 'ctrl_ko_n', 'l1_wt_n', 'l1_ko_n'],
    'value': [
        ctrl_wt.median(), ctrl_ko.median(), ctrl_delta,
        stats.mannwhitneyu(ctrl_ko, ctrl_wt, alternative='two-sided')[1],
        l1_wt.median(), l1_ko.median(), l1_delta,
        stats.mannwhitneyu(l1_ko, l1_wt, alternative='two-sided')[1],
        l1_delta - ctrl_delta,
        len(ctrl_wt), len(ctrl_ko), len(l1_wt), len(l1_ko)
    ]
}
pd.DataFrame(summary).to_csv(f"{outdir}/mettl3ko_polya_specificity_summary.tsv",
                              sep='\t', index=False)
print(f"\nSummary saved to {outdir}/mettl3ko_polya_specificity_summary.tsv")
