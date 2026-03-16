#!/usr/bin/env python3
"""
Poly(A) shortening → RNA stability: indirect evidence.

Three analyses:
  1. m6A/kb distribution shift: HeLa → HeLa-Ars
     - If low-m6A reads are selectively degraded, surviving pool should be m6A-enriched
  2. Critical poly(A) threshold: fraction of reads approaching decay zone (<20nt, <30nt)
     - Low-m6A reads should have more reads near the PABP dissociation threshold
  3. Per-locus read count × m6A: do low-m6A loci lose more reads under stress?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_L1 = TOPICDIR / 'part3_l1_per_read_cache'
RESULTS = PROJECT / 'results_group'
OUTDIR = TOPICDIR / 'polya_rna_stability'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
HELA_GROUPS = ['HeLa_1', 'HeLa_2', 'HeLa_3', 'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3']

# ==========================================================================
# Load data
# ==========================================================================
print("=== Loading data ===")

cache_dfs = []
for g in HELA_GROUPS:
    p = CACHE_L1 / f'{g}_l1_per_read.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df['group'] = g
        cache_dfs.append(df)
cache = pd.concat(cache_dfs, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)

summ_dfs = []
for g in HELA_GROUPS:
    p = RESULTS / f'{g}/g_summary/{g}_L1_summary.tsv'
    if p.exists():
        df = pd.read_csv(p, sep='\t')
        df = df[df['qc_tag'] == 'PASS']
        df['group'] = g
        summ_dfs.append(df[['read_id', 'polya_length', 'gene_id', 'TE_group', 'group']])
summ = pd.concat(summ_dfs, ignore_index=True)
summ['l1_age'] = summ['gene_id'].apply(
    lambda x: 'young' if x.split('_dup')[0] in YOUNG else 'ancient')

merged = cache.merge(summ, on=['read_id', 'group'], how='inner')
merged = merged[merged['polya_length'] > 0].copy()
merged['condition'] = merged['group'].apply(lambda x: 'HeLa-Ars' if 'Ars' in x else 'HeLa')

hela = merged[merged['condition'] == 'HeLa']
ars = merged[merged['condition'] == 'HeLa-Ars']
print(f"  HeLa: {len(hela):,} reads, HeLa-Ars: {len(ars):,} reads")

# Ancient only (main analysis target — young is immune)
ancient = merged[merged['l1_age'] == 'ancient']
anc_hela = ancient[ancient['condition'] == 'HeLa']
anc_ars = ancient[ancient['condition'] == 'HeLa-Ars']
print(f"  Ancient: HeLa={len(anc_hela):,}, HeLa-Ars={len(anc_ars):,}")

# ======================================================================
# Analysis 1: m6A/kb distribution shift
# ======================================================================
print("\n" + "="*70)
print("=== Analysis 1: m6A/kb Distribution Shift ===")
print("="*70)

# If low-m6A reads are selectively degraded, HeLa-Ars should have higher mean m6A/kb
print("\n--- All L1 ---")
print(f"  HeLa  m6A/kb: mean={hela['m6a_per_kb'].mean():.3f}, median={hela['m6a_per_kb'].median():.3f}")
print(f"  Ars   m6A/kb: mean={ars['m6a_per_kb'].mean():.3f}, median={ars['m6a_per_kb'].median():.3f}")
mw = stats.mannwhitneyu(hela['m6a_per_kb'], ars['m6a_per_kb'], alternative='two-sided')
print(f"  MW p={mw.pvalue:.4e}")

print("\n--- Ancient L1 ---")
print(f"  HeLa  m6A/kb: mean={anc_hela['m6a_per_kb'].mean():.3f}, median={anc_hela['m6a_per_kb'].median():.3f}")
print(f"  Ars   m6A/kb: mean={anc_ars['m6a_per_kb'].mean():.3f}, median={anc_ars['m6a_per_kb'].median():.3f}")
mw_anc = stats.mannwhitneyu(anc_hela['m6a_per_kb'], anc_ars['m6a_per_kb'], alternative='two-sided')
print(f"  MW p={mw_anc.pvalue:.4e}")

# KS test for distribution shift
ks_all = stats.ks_2samp(hela['m6a_per_kb'], ars['m6a_per_kb'])
ks_anc = stats.ks_2samp(anc_hela['m6a_per_kb'], anc_ars['m6a_per_kb'])
print(f"\n  KS test (all): D={ks_all.statistic:.4f}, p={ks_all.pvalue:.4e}")
print(f"  KS test (anc): D={ks_anc.statistic:.4f}, p={ks_anc.pvalue:.4e}")

# Bin-level comparison: proportion of reads in each m6A/kb bin
bins = [0, 2, 4, 6, 8, 10, 15, 25]
bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15+']

print("\n--- Ancient L1: m6A/kb bin proportions ---")
print(f"  {'Bin':>8s}  {'HeLa n':>7s}  {'HeLa %':>7s}  {'Ars n':>7s}  {'Ars %':>7s}  {'Δ%':>7s}")
bin_data = []
for i in range(len(bin_labels)):
    lo, hi = bins[i], bins[i+1]
    n_hela = ((anc_hela['m6a_per_kb'] >= lo) & (anc_hela['m6a_per_kb'] < hi)).sum()
    n_ars = ((anc_ars['m6a_per_kb'] >= lo) & (anc_ars['m6a_per_kb'] < hi)).sum()
    pct_hela = n_hela / len(anc_hela) * 100
    pct_ars = n_ars / len(anc_ars) * 100
    delta = pct_ars - pct_hela
    print(f"  {bin_labels[i]:>8s}  {n_hela:7d}  {pct_hela:6.1f}%  {n_ars:7d}  {pct_ars:6.1f}%  {delta:+6.1f}%")
    bin_data.append({
        'bin': bin_labels[i], 'n_hela': n_hela, 'n_ars': n_ars,
        'pct_hela': pct_hela, 'pct_ars': pct_ars, 'delta_pct': delta
    })

pd.DataFrame(bin_data).to_csv(OUTDIR / 'analysis1_m6a_distribution_bins.tsv', sep='\t', index=False)

# ======================================================================
# Analysis 2: Critical poly(A) threshold
# ======================================================================
print("\n" + "="*70)
print("=== Analysis 2: Critical Poly(A) Threshold ===")
print("="*70)

# PABP dissociation ~12-25nt. Use thresholds: <20nt, <30nt, <50nt
thresholds = [20, 30, 50, 75]

# By m6A quartile × condition
merged_anc = merged[merged['l1_age'] == 'ancient'].copy()
merged_anc['m6a_q'] = pd.qcut(merged_anc['m6a_per_kb'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print("\n--- Ancient L1: Fraction below poly(A) thresholds ---")
threshold_results = []
for cond in ['HeLa', 'HeLa-Ars']:
    print(f"\n  {cond}:")
    print(f"  {'Quartile':>8s}  {'n':>5s}  " + "  ".join([f'<{t}nt' for t in thresholds]))
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = merged_anc[(merged_anc['condition'] == cond) & (merged_anc['m6a_q'] == q)]
        n = len(sub)
        fracs = []
        for t in thresholds:
            frac = (sub['polya_length'] < t).mean() * 100
            fracs.append(frac)
            threshold_results.append({
                'condition': cond, 'quartile': q, 'threshold': t,
                'n': n, 'n_below': (sub['polya_length'] < t).sum(),
                'pct_below': frac
            })
        frac_strs = [f'{f:5.1f}%' for f in fracs]
        print(f"  {q:>8s}  {n:5d}  " + "  ".join(frac_strs))

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(OUTDIR / 'analysis2_polya_threshold.tsv', sep='\t', index=False)

# Key comparison: Q1 vs Q4 at <30nt threshold in HeLa-Ars
q1_ars = merged_anc[(merged_anc['condition'] == 'HeLa-Ars') & (merged_anc['m6a_q'] == 'Q1')]
q4_ars = merged_anc[(merged_anc['condition'] == 'HeLa-Ars') & (merged_anc['m6a_q'] == 'Q4')]
q1_below30 = (q1_ars['polya_length'] < 30).mean() * 100
q4_below30 = (q4_ars['polya_length'] < 30).mean() * 100
# Fisher's exact test
a = (q1_ars['polya_length'] < 30).sum()
b = (q1_ars['polya_length'] >= 30).sum()
c = (q4_ars['polya_length'] < 30).sum()
d = (q4_ars['polya_length'] >= 30).sum()
fisher_p = stats.fisher_exact([[a, b], [c, d]])[1]

print(f"\n  *** KEY: HeLa-Ars poly(A)<30nt ***")
print(f"  Q1 (low m6A):  {q1_below30:.1f}% ({a}/{a+b})")
print(f"  Q4 (high m6A): {q4_below30:.1f}% ({c}/{c+d})")
print(f"  Ratio: {q1_below30/q4_below30:.1f}x more decay-zone reads in low-m6A")
print(f"  Fisher p={fisher_p:.4e}")

# Same for HeLa (should be less dramatic)
q1_hela = merged_anc[(merged_anc['condition'] == 'HeLa') & (merged_anc['m6a_q'] == 'Q1')]
q4_hela = merged_anc[(merged_anc['condition'] == 'HeLa') & (merged_anc['m6a_q'] == 'Q4')]
q1h_below30 = (q1_hela['polya_length'] < 30).mean() * 100
q4h_below30 = (q4_hela['polya_length'] < 30).mean() * 100
ah = (q1_hela['polya_length'] < 30).sum()
bh = (q1_hela['polya_length'] >= 30).sum()
ch = (q4_hela['polya_length'] < 30).sum()
dh = (q4_hela['polya_length'] >= 30).sum()
fisher_h = stats.fisher_exact([[ah, bh], [ch, dh]])[1]

print(f"\n  HeLa (control) poly(A)<30nt:")
print(f"  Q1: {q1h_below30:.1f}% ({ah}/{ah+bh})")
print(f"  Q4: {q4h_below30:.1f}% ({ch}/{ch+dh})")
print(f"  Ratio: {q1h_below30/max(q4h_below30,0.01):.1f}x")
print(f"  Fisher p={fisher_h:.4e}")

# ======================================================================
# Analysis 3: Per-locus read count × m6A
# ======================================================================
print("\n" + "="*70)
print("=== Analysis 3: Per-Locus Read Count × m6A ===")
print("="*70)

# Get per-locus stats in each condition (ancient only)
def locus_stats(df, condition_label):
    sub = df[df['condition'] == condition_label]
    locus = sub.groupby('gene_id').agg(
        n_reads=('read_id', 'count'),
        m6a_kb_mean=('m6a_per_kb', 'mean'),
        m6a_kb_median=('m6a_per_kb', 'median'),
        polya_median=('polya_length', 'median'),
    ).reset_index()
    return locus

loci_hela = locus_stats(ancient, 'HeLa')
loci_ars = locus_stats(ancient, 'HeLa-Ars')

# Merge on gene_id to get shared loci
loci_merged = loci_hela.merge(loci_ars, on='gene_id', suffixes=('_hela', '_ars'))
print(f"\n  Shared ancient loci (detected in both): {len(loci_merged)}")
print(f"  HeLa-only loci: {len(loci_hela) - len(loci_merged)}")
print(f"  Ars-only loci: {len(loci_ars) - len(loci_merged)}")

# Read count change
loci_merged['read_ratio'] = loci_merged['n_reads_ars'] / loci_merged['n_reads_hela']
loci_merged['read_delta'] = loci_merged['n_reads_ars'] - loci_merged['n_reads_hela']
loci_merged['log2_ratio'] = np.log2(loci_merged['read_ratio'])

# Use HeLa m6A as baseline (pre-stress m6A level)
# Correlation: does higher HeLa m6A predict better read retention in Ars?
# Filter to loci with >=3 reads in HeLa (more reliable m6A estimate)
reliable = loci_merged[loci_merged['n_reads_hela'] >= 3].copy()
print(f"  Loci with >=3 HeLa reads: {len(reliable)}")

if len(reliable) >= 20:
    r, p = stats.spearmanr(reliable['m6a_kb_mean_hela'], reliable['log2_ratio'])
    print(f"\n  Spearman r(HeLa m6A/kb, log2 read ratio): r={r:.4f}, p={p:.4e}")

    # Also: m6A vs poly(A) delta
    reliable['polya_delta'] = reliable['polya_median_ars'] - reliable['polya_median_hela']
    r2, p2 = stats.spearmanr(reliable['m6a_kb_mean_hela'], reliable['polya_delta'])
    print(f"  Spearman r(HeLa m6A/kb, poly(A) delta): r={r2:.4f}, p={p2:.4e}")

    # Bin by m6A tertile
    reliable['m6a_tertile'] = pd.qcut(reliable['m6a_kb_mean_hela'], q=3,
                                       labels=['Low', 'Mid', 'High'])
    print(f"\n  Per m6A tertile (locus-level):")
    print(f"  {'Tertile':>8s}  {'n':>4s}  {'log2(Ars/HeLa)':>15s}  {'polyA Δ':>10s}  {'m6A/kb':>8s}")
    tertile_results = []
    for t in ['Low', 'Mid', 'High']:
        sub = reliable[reliable['m6a_tertile'] == t]
        med_ratio = sub['log2_ratio'].median()
        med_delta = sub['polya_delta'].median()
        med_m6a = sub['m6a_kb_mean_hela'].median()
        print(f"  {t:>8s}  {len(sub):4d}  {med_ratio:+15.3f}  {med_delta:+10.1f}nt  {med_m6a:8.1f}")
        tertile_results.append({
            'tertile': t, 'n': len(sub),
            'median_log2_ratio': med_ratio,
            'median_polya_delta': med_delta,
            'median_m6a_kb': med_m6a,
            'mean_log2_ratio': sub['log2_ratio'].mean(),
        })
    pd.DataFrame(tertile_results).to_csv(OUTDIR / 'analysis3_locus_tertile.tsv',
                                          sep='\t', index=False)

# Loci lost in Ars (detected in HeLa, absent in Ars) — potential degraded
hela_only = set(loci_hela['gene_id']) - set(loci_ars['gene_id'])
ars_only = set(loci_ars['gene_id']) - set(loci_hela['gene_id'])
shared = set(loci_hela['gene_id']) & set(loci_ars['gene_id'])

hela_only_m6a = loci_hela[loci_hela['gene_id'].isin(hela_only)]['m6a_kb_mean'].median()
shared_hela_m6a = loci_hela[loci_hela['gene_id'].isin(shared)]['m6a_kb_mean'].median()
ars_only_m6a = loci_ars[loci_ars['gene_id'].isin(ars_only)]['m6a_kb_mean'].median()

print(f"\n  --- Loci presence/absence ---")
print(f"  HeLa-only (lost in Ars): {len(hela_only)} loci, median m6A/kb={hela_only_m6a:.2f}")
print(f"  Shared:                   {len(shared)} loci, median m6A/kb={shared_hela_m6a:.2f}")
print(f"  Ars-only (new in Ars):    {len(ars_only)} loci, median m6A/kb={ars_only_m6a:.2f}")

if len(hela_only) > 10:
    hela_only_vals = loci_hela[loci_hela['gene_id'].isin(hela_only)]['m6a_kb_mean']
    shared_vals = loci_hela[loci_hela['gene_id'].isin(shared)]['m6a_kb_mean']
    mw_loci = stats.mannwhitneyu(hela_only_vals, shared_vals, alternative='less')
    print(f"  MW (hela-only < shared m6A): p={mw_loci.pvalue:.4e}")
    print(f"  → If p<0.05: loci that disappear under stress had lower m6A → selective degradation")

# But note singleton bias
hela_only_reads = loci_hela[loci_hela['gene_id'].isin(hela_only)]['n_reads']
shared_reads = loci_hela[loci_hela['gene_id'].isin(shared)]['n_reads']
print(f"\n  Singleton bias check:")
print(f"  HeLa-only loci: median reads = {hela_only_reads.median():.0f} "
      f"(singleton: {(hela_only_reads==1).mean()*100:.0f}%)")
print(f"  Shared loci:    median reads = {shared_reads.median():.0f} "
      f"(singleton: {(shared_reads==1).mean()*100:.0f}%)")

# Save locus data
loci_merged.to_csv(OUTDIR / 'analysis3_shared_loci.tsv', sep='\t', index=False)
reliable.to_csv(OUTDIR / 'analysis3_reliable_loci.tsv', sep='\t', index=False)

# ======================================================================
# Figures
# ======================================================================
print("\n=== Generating figures ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Panel A: m6A/kb distribution (CDF) ---
ax = axes[0, 0]
x_range = np.linspace(0, 20, 200)
for label, data, color, ls in [
    ('HeLa ancient', anc_hela['m6a_per_kb'], '#4C72B0', '-'),
    ('HeLa-Ars ancient', anc_ars['m6a_per_kb'], '#C44E52', '-'),
]:
    sorted_vals = np.sort(data.values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=2, label=label)

ax.set_xlabel('m6A/kb', fontsize=11)
ax.set_ylabel('Cumulative Fraction', fontsize=11)
ax.set_title('A. m6A/kb Distribution: HeLa vs HeLa-Ars\n(Ancient L1)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 18)
ax.axvline(x=anc_hela['m6a_per_kb'].median(), color='#4C72B0', linestyle=':', alpha=0.5)
ax.axvline(x=anc_ars['m6a_per_kb'].median(), color='#C44E52', linestyle=':', alpha=0.5)
ax.text(0.02, 0.98, f"KS D={ks_anc.statistic:.3f}\np={ks_anc.pvalue:.2e}",
        transform=ax.transAxes, va='top', fontsize=9, fontstyle='italic')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: Critical threshold by m6A quartile ---
ax = axes[0, 1]
threshold_t = 30  # PABP dissociation zone
q_labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
x = np.arange(4)
width = 0.35

hela_pcts = []
ars_pcts = []
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    hela_sub = threshold_df[(threshold_df['condition'] == 'HeLa') &
                             (threshold_df['quartile'] == q) &
                             (threshold_df['threshold'] == threshold_t)]
    ars_sub = threshold_df[(threshold_df['condition'] == 'HeLa-Ars') &
                            (threshold_df['quartile'] == q) &
                            (threshold_df['threshold'] == threshold_t)]
    hela_pcts.append(hela_sub['pct_below'].values[0] if len(hela_sub) else 0)
    ars_pcts.append(ars_sub['pct_below'].values[0] if len(ars_sub) else 0)

bars1 = ax.bar(x - width/2, hela_pcts, width, color='#4C72B0', alpha=0.8,
               edgecolor='black', linewidth=0.5, label='HeLa')
bars2 = ax.bar(x + width/2, ars_pcts, width, color='#C44E52', alpha=0.8,
               edgecolor='black', linewidth=0.5, label='HeLa-Ars')

for i in range(4):
    ax.text(i - width/2, hela_pcts[i] + 0.3, f'{hela_pcts[i]:.1f}%',
            ha='center', fontsize=8, color='#4C72B0')
    ax.text(i + width/2, ars_pcts[i] + 0.3, f'{ars_pcts[i]:.1f}%',
            ha='center', fontsize=8, color='#C44E52', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=10)
ax.set_ylabel(f'% Reads with poly(A) < {threshold_t}nt', fontsize=11)
ax.set_title(f'B. Reads Approaching Decay Zone (poly(A)<{threshold_t}nt)\n'
             f'by m6A/kb Quartile (Ancient L1)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel C: Per-locus log2 read ratio by m6A tertile ---
ax = axes[1, 0]
if len(reliable) >= 20:
    for i, t in enumerate(['Low', 'Mid', 'High']):
        sub = reliable[reliable['m6a_tertile'] == t]
        vals = sub['log2_ratio'].clip(-4, 4).values
        bp = ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                        showfliers=False)
        color = ['#C44E52', '#DD8452', '#4C72B0'][i]
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        bp['medians'][0].set_color('black')
        ax.text(i, sub['log2_ratio'].median() + 0.15,
                f'{sub["log2_ratio"].median():.2f}',
                ha='center', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Low m6A', 'Mid m6A', 'High m6A'], fontsize=11)
    ax.set_ylabel('log₂(Ars reads / HeLa reads)', fontsize=11)
    ax.set_title('C. Read Count Change by Locus m6A Level\n(Ancient L1, ≥3 HeLa reads)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Panel D: HeLa-only vs Shared loci m6A ---
ax = axes[1, 1]
if len(hela_only) > 5:
    hela_only_data = loci_hela[loci_hela['gene_id'].isin(hela_only)]['m6a_kb_mean']
    shared_data = loci_hela[loci_hela['gene_id'].isin(shared)]['m6a_kb_mean']
    ars_only_data = loci_ars[loci_ars['gene_id'].isin(ars_only)]['m6a_kb_mean']

    data_D = [hela_only_data.values, shared_data.values, ars_only_data.values]
    labels_D = [f'Lost in Ars\n(n={len(hela_only)})',
                f'Shared\n(n={len(shared)})',
                f'New in Ars\n(n={len(ars_only)})']
    colors_D = ['#C44E52', '#888888', '#4C72B0']

    parts = ax.violinplot(data_D, positions=[0, 1, 2], showmedians=True, showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors_D[i])
        body.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    for i, vals in enumerate(data_D):
        med = np.median(vals)
        ax.text(i, med + 0.3, f'{med:.1f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels_D, fontsize=10)
    ax.set_ylabel('m6A/kb (HeLa baseline)', fontsize=11)
    ax.set_title('D. m6A Level of Lost vs Retained Loci\n(Ancient L1)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate significance
    if len(hela_only) > 10:
        ax.text(0.02, 0.98, f'Lost vs Shared: MW p={mw_loci.pvalue:.2e}',
                transform=ax.transAxes, va='top', fontsize=9, fontstyle='italic')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig_polya_rna_stability.png', dpi=200)
plt.close()
print(f"  Saved: fig_polya_rna_stability.png")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "="*70)
print("=== SUMMARY: Poly(A) Shortening → RNA Stability Evidence ===")
print("="*70)

print(f"""
Analysis 1 - m6A/kb Distribution Shift:
  Ancient L1 mean m6A/kb: HeLa {anc_hela['m6a_per_kb'].mean():.3f} → Ars {anc_ars['m6a_per_kb'].mean():.3f}
  KS test: D={ks_anc.statistic:.3f}, p={ks_anc.pvalue:.2e}
  → {"Distribution shifted" if ks_anc.pvalue < 0.05 else "No significant shift"} under stress

Analysis 2 - Critical Threshold (poly(A)<30nt):
  HeLa-Ars Q1(low m6A): {q1_below30:.1f}% in decay zone
  HeLa-Ars Q4(high m6A): {q4_below30:.1f}% in decay zone
  Ratio: {q1_below30/max(q4_below30,0.01):.1f}x (Fisher p={fisher_p:.2e})
  → Low-m6A reads are {q1_below30/max(q4_below30,0.01):.1f}x more likely to approach decay threshold

Analysis 3 - Per-Locus Read Count:
  Shared loci with ≥3 reads: {len(reliable) if len(reliable) >= 20 else 'N/A'}
  Lost-in-Ars loci m6A/kb: {hela_only_m6a:.2f}
  Shared loci m6A/kb:       {shared_hela_m6a:.2f}
  → {"Lost loci had lower m6A" if hela_only_m6a < shared_hela_m6a else "No clear difference"}
""")

print(f"\nOutput: {OUTDIR}")
print(f"  fig_polya_rna_stability.png")
print(f"  analysis1_m6a_distribution_bins.tsv")
print(f"  analysis2_polya_threshold.tsv")
print(f"  analysis3_locus_tertile.tsv")
print(f"  analysis3_shared_loci.tsv")
print(f"  analysis3_reliable_loci.tsv")
