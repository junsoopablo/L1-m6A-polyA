#!/usr/bin/env python3
"""
Psi enrichment decomposition: How much of L1's higher psi/kb is explained by
substrate availability (more motif sites) vs per-site modification rate?

L1 psi/kb = genomic_motif_density × per_motif_modification_rate
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'

# =========================================================================
# 1. Load data
# =========================================================================
print("="*70)
print("PSI ENRICHMENT DECOMPOSITION: Substrate vs Rate")
print("="*70)

# Part 3 per-read data for actual psi/kb
# Load L1 and Control per-read caches
l1_dfs = []
ctrl_dfs = []
for f in sorted((TOPICDIR / 'part3_l1_per_read_cache').glob('*.tsv')):
    df = pd.read_csv(f, sep='\t')
    l1_dfs.append(df)
for f in sorted((TOPICDIR / 'part3_ctrl_per_read_cache').glob('*.tsv')):
    df = pd.read_csv(f, sep='\t')
    ctrl_dfs.append(df)
l1_all = pd.concat(l1_dfs, ignore_index=True)
ctrl_all = pd.concat(ctrl_dfs, ignore_index=True)

print(f"\nL1 reads: {len(l1_all)}, Control reads: {len(ctrl_all)}")

# psi/kb
l1_psi_per_kb = l1_all['psi_sites_high'].sum() / (l1_all['read_length'].sum() / 1000)
ctrl_psi_per_kb = ctrl_all['psi_sites_high'].sum() / (ctrl_all['read_length'].sum() / 1000)
print(f"\nActual psi/kb:")
print(f"  L1:      {l1_psi_per_kb:.3f}")
print(f"  Control: {ctrl_psi_per_kb:.3f}")
print(f"  Ratio:   {l1_psi_per_kb / ctrl_psi_per_kb:.3f}x")

# =========================================================================
# 2. Genomic motif frequency
# =========================================================================
# From l1_motif_conservation/individual_motif_frequency.tsv
motif_freq = pd.read_csv(TOPICDIR / 'l1_motif_conservation/individual_motif_frequency.tsv', sep='\t')
psi_freq = motif_freq[motif_freq['mod_type'] == 'psi'].copy()

l1_total_freq = psi_freq['l1_per_kb'].sum()
ctrl_total_freq = psi_freq['non_l1_per_kb'].sum()

print(f"\nGenomic psi motif frequency (reference sequence):")
print(f"  L1 regions:     {l1_total_freq:.2f} motifs/kb")
print(f"  Non-L1 regions: {ctrl_total_freq:.2f} motifs/kb")
print(f"  Ratio:          {l1_total_freq / ctrl_total_freq:.3f}x")

# =========================================================================
# 3. Per-motif modification rates (from motif_enrichment_l1_vs_ctrl.tsv)
# =========================================================================
motif_enrich = pd.read_csv(TOPICDIR / 'motif_enrichment_l1_vs_ctrl.tsv', sep='\t')
l1_motif = motif_enrich[(motif_enrich['source']=='L1') & (motif_enrich['mod_type']=='psi')]
ctrl_motif = motif_enrich[(motif_enrich['source']=='Control') & (motif_enrich['mod_type']=='psi')]

l1_overall_rate = l1_motif['n_modified'].sum() / l1_motif['n_sites'].sum()
ctrl_overall_rate = ctrl_motif['n_modified'].sum() / ctrl_motif['n_sites'].sum()

print(f"\nOverall per-site psi modification rate:")
print(f"  L1:      {l1_overall_rate:.4f} ({l1_overall_rate*100:.2f}%)")
print(f"  Control: {ctrl_overall_rate:.4f} ({ctrl_overall_rate*100:.2f}%)")
print(f"  Ratio:   {l1_overall_rate / ctrl_overall_rate:.3f}x")

# =========================================================================
# 4. Decomposition
# =========================================================================
print(f"\n{'='*70}")
print("DECOMPOSITION: psi/kb = motif_freq/kb × modification_rate")
print("="*70)

# Expected psi/kb = motif_freq × rate
l1_expected = l1_total_freq * l1_overall_rate
ctrl_expected = ctrl_total_freq * ctrl_overall_rate
print(f"\nExpected psi/kb (freq × rate):")
print(f"  L1:      {l1_total_freq:.2f} × {l1_overall_rate:.4f} = {l1_expected:.3f}")
print(f"  Control: {ctrl_total_freq:.2f} × {ctrl_overall_rate:.4f} = {ctrl_expected:.3f}")
print(f"  Expected ratio: {l1_expected / ctrl_expected:.3f}x")
print(f"  Actual ratio:   {l1_psi_per_kb / ctrl_psi_per_kb:.3f}x")

# Multiplicative decomposition
freq_ratio = l1_total_freq / ctrl_total_freq
rate_ratio = l1_overall_rate / ctrl_overall_rate
total_ratio = l1_psi_per_kb / ctrl_psi_per_kb

print(f"\nMultiplicative decomposition of {total_ratio:.3f}x enrichment:")
print(f"  Substrate (motif frequency): {freq_ratio:.3f}x")
print(f"  Enzyme (per-site rate):      {rate_ratio:.3f}x")
print(f"  Freq × Rate:                 {freq_ratio * rate_ratio:.3f}x")
print(f"  Residual (unexplained):      {total_ratio / (freq_ratio * rate_ratio):.3f}x")

# Percentage contribution (log decomposition)
import math
log_total = math.log(total_ratio)
log_freq = math.log(freq_ratio)
log_rate = math.log(rate_ratio)
log_residual = log_total - log_freq - log_rate

print(f"\nLog-scale contribution to enrichment:")
print(f"  Substrate availability: {log_freq/log_total*100:+.1f}%")
print(f"  Per-site rate:          {log_rate/log_total*100:+.1f}%")
print(f"  Residual:               {log_residual/log_total*100:+.1f}%")

# =========================================================================
# 5. L1 body vs flanking decomposition
# =========================================================================
print(f"\n{'='*70}")
print("L1 BODY vs FLANKING: per-motif comparison")
print("="*70)

body_flank = pd.read_csv(TOPICDIR / 'l1_region_mod/per_motif_l1_vs_flanking.tsv', sep='\t')
psi_bf = body_flank[body_flank['mod_type'] == 'psi'].copy()

print(f"\n{'Motif':8s} {'L1body%':>8s} {'Flank%':>8s} {'Delta':>8s} {'p':>10s}")
for _, r in psi_bf.sort_values('mean_l1', ascending=False).iterrows():
    sig = '***' if r['pval'] < 0.001 else '**' if r['pval'] < 0.01 else '*' if r['pval'] < 0.05 else ''
    print(f"  {r['motif']:8s} {r['mean_l1']:7.1f}% {r['mean_flank']:7.1f}% "
          f"{r['delta']:+7.2f} {r['pval']:10.2e} {sig}")

# Weighted average (by motif frequency in L1)
# Merge with genomic frequency
psi_bf_merged = psi_bf.merge(
    psi_freq[['motif','l1_per_kb']].rename(columns={'l1_per_kb':'freq'}),
    on='motif', how='left'
)
if psi_bf_merged['freq'].notna().sum() > 0:
    psi_bf_valid = psi_bf_merged.dropna(subset=['freq'])
    weighted_l1 = (psi_bf_valid['mean_l1'] * psi_bf_valid['freq']).sum() / psi_bf_valid['freq'].sum()
    weighted_flank = (psi_bf_valid['mean_flank'] * psi_bf_valid['freq']).sum() / psi_bf_valid['freq'].sum()
    print(f"\nFrequency-weighted mean rate:")
    print(f"  L1 body:    {weighted_l1:.2f}%")
    print(f"  Flanking:   {weighted_flank:.2f}%")
    print(f"  Delta:      {weighted_l1 - weighted_flank:+.2f}pp")

# =========================================================================
# 6. Per-motif contribution to L1 psi enrichment
# =========================================================================
print(f"\n{'='*70}")
print("PER-MOTIF CONTRIBUTION: frequency × rate → expected psi/kb")
print("="*70)

# For each motif: expected psi/kb contribution = genomic_freq × mod_rate
l1_motif_dict = dict(zip(l1_motif['motif'], l1_motif['rate']))
ctrl_motif_dict = dict(zip(ctrl_motif['motif'], ctrl_motif['rate']))

print(f"\n{'Motif':8s} {'L1 freq':>8s} {'Ctrl freq':>9s} {'L1 rate':>8s} {'Ctrl rate':>9s} "
      f"{'L1 psi/kb':>9s} {'Ctrl psi/kb':>10s} {'Delta':>8s}")

total_l1_contrib = 0
total_ctrl_contrib = 0
rows = []
for _, fr in psi_freq.iterrows():
    m = fr['motif']
    l1_f = fr['l1_per_kb']
    ctrl_f = fr['non_l1_per_kb']
    l1_r = l1_motif_dict.get(m, 0) / 100  # convert from percentage
    ctrl_r = ctrl_motif_dict.get(m, 0) / 100
    l1_contrib = l1_f * l1_r
    ctrl_contrib = ctrl_f * ctrl_r
    delta = l1_contrib - ctrl_contrib
    total_l1_contrib += l1_contrib
    total_ctrl_contrib += ctrl_contrib
    rows.append({'motif': m, 'l1_freq': l1_f, 'ctrl_freq': ctrl_f,
                 'l1_rate': l1_r, 'ctrl_rate': ctrl_r,
                 'l1_contrib': l1_contrib, 'ctrl_contrib': ctrl_contrib, 'delta': delta})
    print(f"  {m:8s} {l1_f:7.2f} {ctrl_f:8.2f} {l1_r:7.1%} {ctrl_r:8.1%} "
          f"{l1_contrib:8.3f} {ctrl_contrib:9.3f} {delta:+7.3f}")

print(f"\n  {'TOTAL':8s} {l1_total_freq:7.2f} {ctrl_total_freq:8.2f} "
      f"{'':>8s} {'':>9s} {total_l1_contrib:8.3f} {total_ctrl_contrib:9.3f} "
      f"{total_l1_contrib - total_ctrl_contrib:+7.3f}")

# Top contributors to delta
contrib_df = pd.DataFrame(rows).sort_values('delta', ascending=False)
print(f"\nTop contributors to L1 enrichment (expected psi/kb delta):")
for _, r in contrib_df.head(5).iterrows():
    freq_effect = (r['l1_freq'] - r['ctrl_freq']) * r['ctrl_rate']
    rate_effect = r['ctrl_freq'] * (r['l1_rate'] - r['ctrl_rate'])
    interaction = (r['l1_freq'] - r['ctrl_freq']) * (r['l1_rate'] - r['ctrl_rate'])
    print(f"  {r['motif']:8s}: delta={r['delta']:+.3f} "
          f"(freq_effect={freq_effect:+.3f}, rate_effect={rate_effect:+.3f}, interact={interaction:+.3f})")

# =========================================================================
# 7. Functional implication: psi sites per transcript
# =========================================================================
print(f"\n{'='*70}")
print("FUNCTIONAL IMPLICATION: Psi sites per transcript")
print("="*70)

l1_median_rdlen = l1_all['read_length'].median()
ctrl_median_rdlen = ctrl_all['read_length'].median()

# Expected psi sites per median-length transcript
l1_expected_sites = l1_psi_per_kb * l1_median_rdlen / 1000
ctrl_expected_sites = ctrl_psi_per_kb * ctrl_median_rdlen / 1000

# Fraction of reads with ≥1 psi
l1_frac_psi = (l1_all['psi_sites_high'] > 0).mean()
ctrl_frac_psi = (ctrl_all['psi_sites_high'] > 0).mean()

# Fraction with ≥3 psi (substantial modification)
l1_frac_3psi = (l1_all['psi_sites_high'] >= 3).mean()
ctrl_frac_3psi = (ctrl_all['psi_sites_high'] >= 3).mean()

print(f"\nMedian read length: L1={l1_median_rdlen:.0f}bp, Control={ctrl_median_rdlen:.0f}bp")
print(f"Expected psi sites per median-length read:")
print(f"  L1:      {l1_expected_sites:.2f} sites")
print(f"  Control: {ctrl_expected_sites:.2f} sites")
print(f"\nFraction of reads with ≥1 psi:")
print(f"  L1:      {l1_frac_psi:.1%}")
print(f"  Control: {ctrl_frac_psi:.1%}")
print(f"\nFraction of reads with ≥3 psi:")
print(f"  L1:      {l1_frac_3psi:.1%}")
print(f"  Control: {ctrl_frac_3psi:.1%}")

# =========================================================================
# 8. Summary
# =========================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print(f"""
L1 psi/kb enrichment = {total_ratio:.3f}x over Control

Decomposition:
  1. Substrate availability (genomic motif frequency): {freq_ratio:.3f}x
  2. Per-site modification rate:                       {rate_ratio:.3f}x
  3. Freq × Rate product:                              {freq_ratio * rate_ratio:.3f}x
  4. Residual (read length, context effects):           {total_ratio / (freq_ratio * rate_ratio):.3f}x

→ Substrate accounts for ~{log_freq/log_total*100:.0f}% of the enrichment (log-scale)
→ Per-site rate accounts for ~{log_rate/log_total*100:.0f}%
→ Residual ~{log_residual/log_total*100:.0f}%

L1 body vs flanking:
  Frequency-weighted mean: L1 body {weighted_l1:.1f}% vs flanking {weighted_flank:.1f}%
  → L1 body has {'LOWER' if weighted_l1 < weighted_flank else 'HIGHER'} per-site rate than flanking

Functional impact:
  {l1_frac_psi:.1%} of L1 reads have ≥1 psi (vs {ctrl_frac_psi:.1%} Control)
  {l1_frac_3psi:.1%} of L1 reads have ≥3 psi (vs {ctrl_frac_3psi:.1%} Control)
  → L1's AT-richness provides more substrate sites, leading to higher total psi load
  → This is biologically meaningful given psi's protective effect on poly(A) under stress
""")

# Save
contrib_df.to_csv(TOPICDIR / 'psi_motif_selectivity/psi_enrichment_decomposition.tsv',
                  sep='\t', index=False)
print("Done!")
