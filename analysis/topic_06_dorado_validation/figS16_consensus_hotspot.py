#!/usr/bin/env python3
"""
Supplementary Figure S16: Consensus Hotspot m6A Analysis
=========================================================
(a) Young vs Ancient bin-level m6A rate correlation scatter
(b) Ancient m6A rate by Hamming distance to Young flanking context

Usage:
  conda run -n research python figS16_consensus_hotspot.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from pathlib import Path

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/'
              'topic_06_dorado_validation/dorado_m6a_results')
FIGDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/manuscript/figures')

# ============================================================
# Load data
# ============================================================
print("Loading DRACH position data...")
drach_df = pd.read_csv(OUTDIR / 'consensus_drach_positions.tsv.gz', sep='\t', compression='gzip')
print(f"  Total DRACH positions: {len(drach_df):,}")

BIN_SIZE = 50

# ============================================================
# Panel (a): Young vs Ancient bin-level m6A rate scatter
# ============================================================
young = drach_df[drach_df['age_class'] == 'Young'].copy()
ancient = drach_df[drach_df['age_class'] == 'Ancient'].copy()

young = young[young['cons_pos'].notna()]
ancient = ancient[ancient['cons_pos'].notna()]

young['cons_bin'] = (young['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE
ancient['cons_bin'] = (ancient['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE

young_bin = young.groupby('cons_bin').agg(
    n=('is_methylated', 'count'),
    meth=('is_methylated', 'sum'),
).reset_index()
young_bin['rate'] = young_bin['meth'] / young_bin['n']
young_bin = young_bin[young_bin['n'] >= 5]

ancient_bin = ancient.groupby('cons_bin').agg(
    n=('is_methylated', 'count'),
    meth=('is_methylated', 'sum'),
).reset_index()
ancient_bin['rate'] = ancient_bin['meth'] / ancient_bin['n']
ancient_bin = ancient_bin[ancient_bin['n'] >= 10]

merged = young_bin.merge(ancient_bin, on='cons_bin', suffixes=('_young', '_ancient'))
rho, p_rho = stats.spearmanr(merged['rate_young'], merged['rate_ancient'])
print(f"  Panel (a): {len(merged)} bins, rho={rho:.3f}, P={p_rho:.2e}")

# ============================================================
# Panel (b): Hamming distance → m6A rate
# ============================================================
# Recompute Hamming distances for Ancient DRACH
from collections import Counter

young_meth = young[young['is_methylated'] & young['flanking_11mer'].notna()]['flanking_11mer']
young_11mer_counts = Counter(young_meth)
top_young_11mers = set(km for km, cnt in young_11mer_counts.most_common(100))

ancient_fl = ancient[ancient['flanking_11mer'].notna()].copy()

np.random.seed(42)
if len(ancient_fl) > 50000:
    sample_idx = np.random.choice(len(ancient_fl), 50000, replace=False)
    analysis_df = ancient_fl.iloc[sample_idx].copy()
else:
    analysis_df = ancient_fl.copy()

print(f"  Computing Hamming distances for {len(analysis_df):,} Ancient DRACH sites...")

def hamming_min(seq, ref_set):
    min_d = len(seq)
    for r in ref_set:
        if len(r) != len(seq):
            continue
        d = sum(1 for a, b in zip(seq, r) if a != b)
        if d < min_d:
            min_d = d
    return min_d

distances = [hamming_min(row['flanking_11mer'], top_young_11mers) for _, row in analysis_df.iterrows()]
analysis_df = analysis_df.copy()
analysis_df['hamming'] = distances

hamming_stats = []
for d in sorted(analysis_df['hamming'].unique()):
    sub = analysis_df[analysis_df['hamming'] == d]
    if len(sub) >= 10:
        rate = sub['is_methylated'].mean()
        ci = 1.96 * np.sqrt(rate * (1 - rate) / len(sub))
        hamming_stats.append({'hamming': d, 'n': len(sub), 'rate': rate, 'ci': ci})

hamming_df = pd.DataFrame(hamming_stats)
print(f"  Panel (b): Hamming bins: {len(hamming_df)}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# --- Panel (a): Scatter ---
ax = axes[0]
sizes = np.clip(merged['n_ancient'] / 5, 5, 100)
sc = ax.scatter(merged['rate_young'] * 100, merged['rate_ancient'] * 100,
                s=sizes, c='#2166ac', alpha=0.6, edgecolors='white', linewidths=0.3)

# Regression line
slope, intercept, r, p, se = stats.linregress(merged['rate_young'], merged['rate_ancient'])
x_fit = np.linspace(0, merged['rate_young'].max(), 100)
ax.plot(x_fit * 100, (slope * x_fit + intercept) * 100, 'k--', linewidth=0.8, alpha=0.7)

ax.set_xlabel('Young L1 m6A rate per bin (%)', fontsize=8)
ax.set_ylabel('Ancient L1 m6A rate per bin (%)', fontsize=8)
ax.set_title(f'ρ = {rho:.2f}, P = {p_rho:.1e}', fontsize=8, style='italic')
ax.tick_params(labelsize=7)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel (b): Bar chart ---
ax = axes[1]
colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(hamming_df)))
bars = ax.bar(hamming_df['hamming'], hamming_df['rate'] * 100,
              yerr=hamming_df['ci'] * 100, capsize=3,
              color=colors, edgecolor='black', linewidth=0.5)

# Add N labels on bars
for _, row in hamming_df.iterrows():
    ax.text(row['hamming'], row['rate'] * 100 + row['ci'] * 100 + 0.3,
            f"n={int(row['n']):,}", ha='center', va='bottom', fontsize=5.5, color='#333')

ax.set_xlabel('Hamming distance to Young L1 context', fontsize=8)
ax.set_ylabel('Ancient L1 per-DRACH m6A rate (%)', fontsize=8)
ax.set_xticks(hamming_df['hamming'])
ax.tick_params(labelsize=7)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Stats annotation
close_rate = analysis_df[analysis_df['hamming'] <= 2]['is_methylated'].mean()
far_rate = analysis_df[analysis_df['hamming'] > 2]['is_methylated'].mean()
ax.annotate(f'≤2 vs >2: {close_rate/far_rate:.1f}×\nP = 1.3×10⁻⁹⁶',
            xy=(3, far_rate * 100), xytext=(3.5, close_rate * 100),
            fontsize=6.5, ha='center',
            arrowprops=dict(arrowstyle='->', color='#666', lw=0.8))

plt.tight_layout(w_pad=2)

# Save
out_pdf = FIGDIR / 'figS16.pdf'
fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
print(f"\n  Saved: {out_pdf}")

out_local = OUTDIR / 'figS16_consensus_hotspot.pdf'
fig.savefig(out_local, bbox_inches='tight', dpi=300)
print(f"  Saved: {out_local}")

plt.close()
print("Done.")
