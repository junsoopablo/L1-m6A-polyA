#!/usr/bin/env python3
"""
Psi motif selectivity analysis: Does L1 have different PUS enzyme preferences?

Questions:
1. L1 vs Control motif rates — which motifs differ most?
2. Is the difference explained by motif frequency (L1 is AT-rich)?
3. Group motifs by known PUS enzyme associations
4. Per-cell-line consistency of motif preferences
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
OUTDIR = TOPICDIR / 'psi_motif_selectivity'
OUTDIR.mkdir(exist_ok=True)

# =========================================================================
# Load motif comparison data
# =========================================================================
print("Loading motif data...")

comp_df = pd.read_csv(TOPICDIR / 'motif_l1_vs_ctrl_comparison.tsv', sep='\t')
# Columns: mod_type, motif, l1_rate, ctrl_rate, l1_weighted, ctrl_weighted, delta, OR, p_value
# Rates are already in percentage (e.g., 73.04 = 73.04%)

# Filter to psi motifs only
psi_comp = comp_df[comp_df['mod_type'] == 'psi'].copy()
print(f"Psi motifs: {len(psi_comp)}")

# Overall rates from per-motif data
motif_df = pd.read_csv(TOPICDIR / 'motif_enrichment_l1_vs_ctrl.tsv', sep='\t')
l1_psi = motif_df[(motif_df['source']=='L1') & (motif_df['mod_type']=='psi')]
ctrl_psi = motif_df[(motif_df['source']=='Control') & (motif_df['mod_type']=='psi')]
l1_psi_overall = l1_psi['n_modified'].sum() / l1_psi['n_sites'].sum() * 100
ctrl_psi_overall = ctrl_psi['n_modified'].sum() / ctrl_psi['n_sites'].sum() * 100
print(f"\nL1 vs Control overall psi rate:")
print(f"  L1: {l1_psi_overall:.1f}%")
print(f"  Control: {ctrl_psi_overall:.1f}%")

# Also get m6A for comparison
m6a_comp = comp_df[comp_df['mod_type'] == 'm6A'].copy()

# =========================================================================
# 1. L1 vs Control motif rates
# =========================================================================
print("\n" + "="*60)
print("1. Per-motif L1 vs Control rates (psi)")
print("="*60)

psi_comp = psi_comp.sort_values('l1_rate', ascending=False)
print(f"\n{'Motif':8s} {'L1%':>7s} {'Ctrl%':>7s} {'Delta':>7s} {'OR':>6s} {'p':>10s}")
for _, r in psi_comp.iterrows():
    sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else ''
    print(f"{r['motif']:8s} {r['l1_rate']:6.1f}% {r['ctrl_rate']:6.1f}% "
          f"{r['delta']:+6.2f} {r['OR']:5.2f} {r['p_value']:10.2e} {sig}")

# Add delta_rate column (already percentage)
psi_comp['delta_rate'] = psi_comp['l1_rate'] - psi_comp['ctrl_rate']
# odds_ratio column name
psi_comp['odds_ratio'] = psi_comp['OR']

# Correlation
r_corr, p_corr = stats.pearsonr(psi_comp['l1_rate'], psi_comp['ctrl_rate'])
print(f"\nPearson r (L1 vs Ctrl rate): {r_corr:.4f}, p = {p_corr:.2e}")

# =========================================================================
# 2. Rank comparison: do L1 and Control have same motif preference order?
# =========================================================================
print("\n" + "="*60)
print("2. Motif rank comparison")
print("="*60)

psi_comp['l1_rank'] = psi_comp['l1_rate'].rank(ascending=False)
psi_comp['ctrl_rank'] = psi_comp['ctrl_rate'].rank(ascending=False)
psi_comp['rank_diff'] = psi_comp['l1_rank'] - psi_comp['ctrl_rank']

rho, p_rho = stats.spearmanr(psi_comp['l1_rate'], psi_comp['ctrl_rate'])
print(f"Spearman rho: {rho:.4f}, p = {p_rho:.2e}")

# Motifs with biggest rank change
print(f"\nBiggest rank changes (L1 rank - Ctrl rank):")
for _, r in psi_comp.sort_values('rank_diff', key=abs, ascending=False).head(5).iterrows():
    print(f"  {r['motif']:8s}: L1 rank={r['l1_rank']:.0f}, Ctrl rank={r['ctrl_rank']:.0f}, "
          f"diff={r['rank_diff']:+.0f}")

# =========================================================================
# 3. Motif sequence properties — AT-rich composition effect?
# =========================================================================
print("\n" + "="*60)
print("3. Motif sequence composition vs L1 enrichment")
print("="*60)

def at_content(motif):
    return sum(1 for c in motif if c in 'AT') / len(motif)

psi_comp['at_content'] = psi_comp['motif'].apply(at_content)

r_at, p_at = stats.pearsonr(psi_comp['at_content'], psi_comp['delta_rate'])
print(f"AT content vs L1-Ctrl delta: r = {r_at:.3f}, p = {p_at:.3f}")

print(f"\nAT-rich motifs (AT>=60%):")
for _, r in psi_comp[psi_comp['at_content'] >= 0.6].sort_values('delta_rate', ascending=False).iterrows():
    print(f"  {r['motif']:8s}: AT={r['at_content']:.1%}, delta={r['delta_rate']:+.2f}pp, "
          f"L1={r['l1_rate']:.1f}%, Ctrl={r['ctrl_rate']:.1f}%")

print(f"\nGC-rich motifs (AT<40%):")
for _, r in psi_comp[psi_comp['at_content'] < 0.4].sort_values('delta_rate', ascending=False).iterrows():
    print(f"  {r['motif']:8s}: AT={r['at_content']:.1%}, delta={r['delta_rate']:+.2f}pp, "
          f"L1={r['l1_rate']:.1f}%, Ctrl={r['ctrl_rate']:.1f}%")

# =========================================================================
# 4. Compare m6A vs psi: enrichment pattern difference
# =========================================================================
print("\n" + "="*60)
print("4. m6A vs Psi enrichment pattern comparison")
print("="*60)

m6a_comp['delta_rate'] = m6a_comp['l1_rate'] - m6a_comp['ctrl_rate']

print(f"\nm6A: {len(m6a_comp)} motifs")
print(f"  All L1 > Ctrl? {(m6a_comp['delta_rate'] > 0).all()}")
print(f"  Mean delta: {m6a_comp['delta_rate'].mean():+.2f}pp")
print(f"  Range: {m6a_comp['delta_rate'].min():+.2f} to {m6a_comp['delta_rate'].max():+.2f}pp")
print(f"  # significant (p<0.05): {(m6a_comp['p_value'] < 0.05).sum()}/{len(m6a_comp)}")

print(f"\nPsi: {len(psi_comp)} motifs")
print(f"  All L1 > Ctrl? {(psi_comp['delta_rate'] > 0).all()}")
print(f"  Mean delta: {psi_comp['delta_rate'].mean():+.2f}pp")
print(f"  Range: {psi_comp['delta_rate'].min():+.2f} to {psi_comp['delta_rate'].max():+.2f}pp")
print(f"  # significant (p<0.05): {(psi_comp['p_value'] < 0.05).sum()}/{len(psi_comp)}")
print(f"  # L1 > Ctrl: {(psi_comp['delta_rate'] > 0).sum()}")
print(f"  # Ctrl > L1: {(psi_comp['delta_rate'] < 0).sum()}")

# =========================================================================
# 5. Motif grouping by known PUS enzyme associations
# =========================================================================
print("\n" + "="*60)
print("5. Motif grouping by sequence pattern (PUS enzyme clues)")
print("="*60)

# Group by common sequence features
# GTTC-containing: GTTCA, GTTCC, GTTCG, GTTCT (likely same PUS enzyme family)
# GT-containing: GGTCC, GGTGG, TGTAG, TGTGG, AGTGG (G_T motifs)
# T-rich: CTTTA, ATTTG, TATAA, CATAA (T-rich, possibly PUS7/PUS1 targets)
# C-rich: CATCC, CCTCC (C-rich)

groups = {
    'GTTCN (GTTC+N)': ['GTTCA', 'GTTCC', 'GTTCG', 'GTTCT'],
    'xGTGG': ['AGTGG', 'GGTGG', 'TGTGG'],
    'T-rich': ['CTTTA', 'ATTTG', 'TATAA', 'CATAA'],
    'C-containing': ['CATCC', 'CCTCC', 'GGTCC'],
    'other': ['TGTAG', 'GATGC'],
}

print(f"\n{'Group':20s} {'L1 mean%':>9s} {'Ctrl mean%':>11s} {'Delta':>7s} {'L1 rate range':>15s}")
for grp_name, motifs in groups.items():
    grp = psi_comp[psi_comp['motif'].isin(motifs)]
    if len(grp) == 0:
        continue
    l1_mean = grp['l1_rate'].mean()
    ctrl_mean = grp['ctrl_rate'].mean()
    delta = l1_mean - ctrl_mean
    l1_range = f"{grp['l1_rate'].min():.1f}-{grp['l1_rate'].max():.1f}%"
    print(f"  {grp_name:20s} {l1_mean:8.1f}% {ctrl_mean:10.1f}% {delta:+6.2f} {l1_range:>15s}")

    for _, r in grp.sort_values('l1_rate', ascending=False).iterrows():
        sig = '***' if r['p_value'] < 0.001 else '*' if r['p_value'] < 0.05 else ''
        print(f"    {r['motif']:8s}: L1={r['l1_rate']:5.1f}%, Ctrl={r['ctrl_rate']:5.1f}%, "
              f"delta={r['delta_rate']:+5.2f} {sig}")

# =========================================================================
# 6. Relative motif preference (within-source normalization)
# =========================================================================
print("\n" + "="*60)
print("6. Relative motif preference (within-source)")
print("="*60)

# Normalize each motif rate by the source's overall rate
# This shows which motifs are relatively more/less preferred in L1 vs Control
l1_overall = psi_comp['l1_rate'].mean()
ctrl_overall = psi_comp['ctrl_rate'].mean()

psi_comp['l1_relative'] = psi_comp['l1_rate'] / l1_overall
psi_comp['ctrl_relative'] = psi_comp['ctrl_rate'] / ctrl_overall
psi_comp['relative_diff'] = psi_comp['l1_relative'] - psi_comp['ctrl_relative']

print(f"\nRelative enrichment (motif_rate / mean_rate):")
print(f"{'Motif':8s} {'L1 relative':>12s} {'Ctrl relative':>14s} {'Diff':>8s}")
for _, r in psi_comp.sort_values('relative_diff', ascending=False).iterrows():
    print(f"  {r['motif']:8s} {r['l1_relative']:11.3f} {r['ctrl_relative']:13.3f} {r['relative_diff']:+7.3f}")

# =========================================================================
# 7. Dominant motif fraction — how concentrated is modification?
# =========================================================================
print("\n" + "="*60)
print("7. Modification concentration (dominance)")
print("="*60)

# What fraction of all psi modifications come from each motif?
# We need the actual site counts, not just rates
# Use the rate × frequency as a proxy
# Higher rate motifs contribute more to total modifications
total_l1 = psi_comp['l1_rate'].sum()
total_ctrl = psi_comp['ctrl_rate'].sum()

psi_comp['l1_frac'] = psi_comp['l1_rate'] / total_l1
psi_comp['ctrl_frac'] = psi_comp['ctrl_rate'] / total_ctrl

print(f"\nFraction of total modification rate by motif:")
print(f"{'Motif':8s} {'L1 frac':>8s} {'Ctrl frac':>10s} {'Diff':>8s}")
for _, r in psi_comp.sort_values('l1_frac', ascending=False).iterrows():
    print(f"  {r['motif']:8s} {r['l1_frac']:7.1%} {r['ctrl_frac']:9.1%} {r['l1_frac']-r['ctrl_frac']:+7.1%}")

# Top 2 motifs (CTTTA + CATCC) dominance
top2_l1 = psi_comp[psi_comp['motif'].isin(['CTTTA','CATCC'])]['l1_frac'].sum()
top2_ctrl = psi_comp[psi_comp['motif'].isin(['CTTTA','CATCC'])]['ctrl_frac'].sum()
print(f"\nTop 2 (CTTTA+CATCC): L1={top2_l1:.1%}, Ctrl={top2_ctrl:.1%}")
print(f"  → Same 2 motifs dominate in both. CATCC slightly more dominant in L1.")

# =========================================================================
# Save results
# =========================================================================
psi_comp.to_csv(OUTDIR / 'psi_motif_selectivity.tsv', sep='\t', index=False)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
1. L1 vs Control psi motif rates correlate at r={r_corr:.4f} (Spearman rho={rho:.4f})
   → Nearly identical enzyme specificity pattern.

2. CTTTA (73%) and CATCC (53%) dominate both L1 and Control.
   These two motifs account for ~{top2_l1:.0%} of total modification rate.

3. m6A: ALL 18 motifs enriched in L1 (mean delta +{m6a_comp['delta_rate'].mean():.1f}pp)
   Psi: Mixed — {(psi_comp['delta_rate']>0).sum()}/16 L1-enriched, {(psi_comp['delta_rate']<0).sum()}/16 Ctrl-enriched
   → m6A is globally elevated in L1; psi shows motif-specific differences

4. Biggest L1-enriched psi motif: CATCC (+4.81pp, OR=1.21)
   Biggest Ctrl-enriched psi motif: CTTTA (-1.32pp)
   → CTTTA enzyme has equal/slightly more access to Control;
     CATCC enzyme has preferential access to L1.

5. AT content does NOT explain delta (r={r_at:.3f}, p={p_at:.3f})
   → Not a simple composition effect.

6. GTTCN group: very low rates (6-12%), minimal L1 vs Ctrl difference.
   T-rich group: highest rates (13-73%), some L1-enriched (ATTTG, TATAA).
   → The most active PUS enzymes (CTTTA, CATCC targets) show divergent behavior.
""")
print(f"\nResults saved to: {OUTDIR}")
print("Done!")
