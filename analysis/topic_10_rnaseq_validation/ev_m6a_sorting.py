#!/usr/bin/env python3
"""
EV m6A Sorting Analysis: MCF7-EV vs MCF7 (cellular) L1 m6A comparison.

Question: Does m6A level predict EV sorting of L1 transcripts?

Context:
- MCF7-EV has young L1 enrichment (L1PA2-3: 2.7-2.8x, L1HS: 1.5x by count)
- MCF7-EV reads are longer (library-level artifact, gene-matched BG ratio=2.014)
- Previous analysis found m6A/kb similar, but used outdated data
- This script uses corrected Part3 cache (chemodCode swap fixed)

Data:
- Part3 cache: m6a_sites_high (count at prob>=128/255)
- L1 summary: subfamily (gene_id), poly(A), qc_tag
- MCF7 reps: MCF7_2, MCF7_3, MCF7_4
- MCF7-EV: MCF7-EV_1 (single replicate)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
SUMMARY_DIR = BASE / 'results_group'
OUT_DIR = BASE / 'analysis/01_exploration/topic_10_rnaseq_validation/ev_m6a_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

MCF7_REPS = ['MCF7_2', 'MCF7_3', 'MCF7_4']
EV_REPS = ['MCF7-EV_1']

# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("STEP 1: Loading Part3 cache and L1 summary")
print("=" * 70)

def load_cache(group):
    path = CACHE_DIR / f'{group}_l1_per_read.tsv'
    df = pd.read_csv(path, sep='\t')
    df['group'] = group
    return df

def load_summary(group):
    path = SUMMARY_DIR / group / 'g_summary' / f'{group}_L1_summary.tsv'
    df = pd.read_csv(path, sep='\t')
    df['group'] = group
    return df

# Load Part3 caches
cache_mcf7 = pd.concat([load_cache(g) for g in MCF7_REPS], ignore_index=True)
cache_ev = load_cache('MCF7-EV_1')

# Load summaries
summ_mcf7 = pd.concat([load_summary(g) for g in MCF7_REPS], ignore_index=True)
summ_ev = load_summary('MCF7-EV_1')

# Filter PASS only
summ_mcf7 = summ_mcf7[summ_mcf7['qc_tag'] == 'PASS'].copy()
summ_ev = summ_ev[summ_ev['qc_tag'] == 'PASS'].copy()

print(f"Part3 cache: MCF7 {len(cache_mcf7)} reads, MCF7-EV {len(cache_ev)} reads")
print(f"L1 summary (PASS): MCF7 {len(summ_mcf7)} reads, MCF7-EV {len(summ_ev)} reads")

# ============================================================
# 2. Merge Part3 + Summary
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Merging Part3 cache with L1 summary")
print("=" * 70)

# Select relevant columns from summary
summ_cols = ['read_id', 'gene_id', 'polya_length', 'chr', 'start', 'end',
             'te_chr', 'te_start', 'te_end', 'transcript_id', 'read_strand',
             'te_strand', 'dist_to_3prime', 'TE_group']

def merge_data(cache_df, summ_df, label):
    # Use available columns
    available = [c for c in summ_cols if c in summ_df.columns]
    merged = cache_df.merge(summ_df[available], on='read_id', how='inner')
    merged['source'] = label
    # Compute m6A/kb and psi/kb
    merged['m6a_per_kb'] = merged['m6a_sites_high'] / (merged['read_length'] / 1000)
    merged['psi_per_kb'] = merged['psi_sites_high'] / (merged['read_length'] / 1000)
    # Classify age
    merged['age'] = merged['gene_id'].apply(
        lambda x: 'Young' if x in YOUNG_SUBFAMILIES else 'Ancient')
    # Create locus ID for shared loci analysis
    merged['locus_id'] = merged['transcript_id']
    return merged

mcf7 = merge_data(cache_mcf7, summ_mcf7, 'MCF7')
ev = merge_data(cache_ev, summ_ev, 'MCF7-EV')

print(f"After merge: MCF7 {len(mcf7)} reads, MCF7-EV {len(ev)} reads")
print(f"MCF7 age: {mcf7['age'].value_counts().to_dict()}")
print(f"MCF7-EV age: {ev['age'].value_counts().to_dict()}")

# Combined for some analyses
combined = pd.concat([mcf7, ev], ignore_index=True)

# ============================================================
# 3. Basic statistics
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Basic statistics")
print("=" * 70)

for label, df in [('MCF7', mcf7), ('MCF7-EV', ev)]:
    print(f"\n--- {label} ---")
    print(f"  Total reads: {len(df)}")
    print(f"  Read length: median={df['read_length'].median():.0f}, "
          f"mean={df['read_length'].mean():.0f}")
    print(f"  m6A/kb: median={df['m6a_per_kb'].median():.2f}, "
          f"mean={df['m6a_per_kb'].mean():.2f}")
    print(f"  m6A sites/read: median={df['m6a_sites_high'].median():.0f}, "
          f"mean={df['m6a_sites_high'].mean():.1f}")
    print(f"  psi/kb: median={df['psi_per_kb'].median():.2f}, "
          f"mean={df['psi_per_kb'].mean():.2f}")
    if 'polya_length' in df.columns:
        valid_pa = df[df['polya_length'] > 0]
        print(f"  Poly(A): median={valid_pa['polya_length'].median():.1f}, "
              f"mean={valid_pa['polya_length'].mean():.1f} (n={len(valid_pa)})")
    for age_label in ['Young', 'Ancient']:
        sub = df[df['age'] == age_label]
        print(f"  {age_label}: n={len(sub)}, RL median={sub['read_length'].median():.0f}, "
              f"m6A/kb={sub['m6a_per_kb'].median():.2f}")

# ============================================================
# 4. Young L1 enrichment in EV (count-based, reproduced)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Young L1 enrichment in EV (count-based)")
print("=" * 70)

mcf7_young_frac = (mcf7['age'] == 'Young').mean()
ev_young_frac = (ev['age'] == 'Young').mean()
print(f"MCF7 young fraction: {mcf7_young_frac:.4f} ({(mcf7['age']=='Young').sum()} reads)")
print(f"MCF7-EV young fraction: {ev_young_frac:.4f} ({(ev['age']=='Young').sum()} reads)")
print(f"EV/MCF7 young enrichment: {ev_young_frac/mcf7_young_frac:.2f}x")

# Per-subfamily enrichment
print("\nPer-subfamily enrichment (top 15):")
mcf7_sub = mcf7['gene_id'].value_counts()
ev_sub = ev['gene_id'].value_counts()
mcf7_total = len(mcf7)
ev_total = len(ev)

all_subs = set(mcf7_sub.index) | set(ev_sub.index)
enrich_data = []
for sub in all_subs:
    n_mcf7 = mcf7_sub.get(sub, 0)
    n_ev = ev_sub.get(sub, 0)
    if n_mcf7 + n_ev < 20:
        continue
    frac_mcf7 = n_mcf7 / mcf7_total if mcf7_total > 0 else 0
    frac_ev = n_ev / ev_total if ev_total > 0 else 0
    enrichment = (frac_ev / frac_mcf7) if frac_mcf7 > 0 else float('inf')
    enrich_data.append({
        'subfamily': sub,
        'n_MCF7': n_mcf7,
        'n_EV': n_ev,
        'frac_MCF7': frac_mcf7,
        'frac_EV': frac_ev,
        'enrichment': enrichment,
        'age': 'Young' if sub in YOUNG_SUBFAMILIES else 'Ancient'
    })

enrich_df = pd.DataFrame(enrich_data).sort_values('enrichment', ascending=False)
print(f"{'Subfamily':<15} {'n_MCF7':>7} {'n_EV':>6} {'frac_MCF7':>10} {'frac_EV':>9} {'Enrich':>7} {'Age':<8}")
for _, row in enrich_df.head(15).iterrows():
    print(f"{row['subfamily']:<15} {row['n_MCF7']:>7} {row['n_EV']:>6} "
          f"{row['frac_MCF7']:>10.4f} {row['frac_EV']:>9.4f} {row['enrichment']:>7.2f} {row['age']:<8}")

# ============================================================
# 5a. Overall m6A/kb: EV vs MCF7
# ============================================================
print("\n" + "=" * 70)
print("STEP 5a: Overall m6A/kb comparison")
print("=" * 70)

u_stat, p_val = stats.mannwhitneyu(ev['m6a_per_kb'], mcf7['m6a_per_kb'], alternative='two-sided')
print(f"MCF7 m6A/kb: median={mcf7['m6a_per_kb'].median():.3f}, mean={mcf7['m6a_per_kb'].mean():.3f}")
print(f"MCF7-EV m6A/kb: median={ev['m6a_per_kb'].median():.3f}, mean={ev['m6a_per_kb'].mean():.3f}")
print(f"Ratio (EV/MCF7): {ev['m6a_per_kb'].median() / mcf7['m6a_per_kb'].median():.3f}")
print(f"Mann-Whitney U: U={u_stat:.0f}, P={p_val:.2e}")

# ============================================================
# 5b. Young L1 only
# ============================================================
print("\n" + "=" * 70)
print("STEP 5b: Young L1 m6A/kb comparison")
print("=" * 70)

mcf7_young = mcf7[mcf7['age'] == 'Young']
ev_young = ev[ev['age'] == 'Young']

if len(ev_young) > 0 and len(mcf7_young) > 0:
    u_stat, p_val = stats.mannwhitneyu(ev_young['m6a_per_kb'], mcf7_young['m6a_per_kb'],
                                        alternative='two-sided')
    print(f"MCF7 young m6A/kb: median={mcf7_young['m6a_per_kb'].median():.3f}, "
          f"mean={mcf7_young['m6a_per_kb'].mean():.3f}, n={len(mcf7_young)}")
    print(f"MCF7-EV young m6A/kb: median={ev_young['m6a_per_kb'].median():.3f}, "
          f"mean={ev_young['m6a_per_kb'].mean():.3f}, n={len(ev_young)}")
    print(f"Ratio: {ev_young['m6a_per_kb'].median() / mcf7_young['m6a_per_kb'].median():.3f}")
    print(f"Mann-Whitney P={p_val:.2e}")
else:
    print("Insufficient data for young L1 comparison")

# ============================================================
# 5c. Ancient L1 only
# ============================================================
print("\n" + "=" * 70)
print("STEP 5c: Ancient L1 m6A/kb comparison")
print("=" * 70)

mcf7_ancient = mcf7[mcf7['age'] == 'Ancient']
ev_ancient = ev[ev['age'] == 'Ancient']

u_stat, p_val = stats.mannwhitneyu(ev_ancient['m6a_per_kb'], mcf7_ancient['m6a_per_kb'],
                                    alternative='two-sided')
print(f"MCF7 ancient m6A/kb: median={mcf7_ancient['m6a_per_kb'].median():.3f}, "
      f"mean={mcf7_ancient['m6a_per_kb'].mean():.3f}, n={len(mcf7_ancient)}")
print(f"MCF7-EV ancient m6A/kb: median={ev_ancient['m6a_per_kb'].median():.3f}, "
      f"mean={ev_ancient['m6a_per_kb'].mean():.3f}, n={len(ev_ancient)}")
print(f"Ratio: {ev_ancient['m6a_per_kb'].median() / mcf7_ancient['m6a_per_kb'].median():.3f}")
print(f"Mann-Whitney P={p_val:.2e}")

# ============================================================
# 5d-e. Within-sample young vs ancient
# ============================================================
print("\n" + "=" * 70)
print("STEP 5d-e: Within-sample Young vs Ancient m6A/kb")
print("=" * 70)

for label, df in [('MCF7', mcf7), ('MCF7-EV', ev)]:
    young_sub = df[df['age'] == 'Young']
    ancient_sub = df[df['age'] == 'Ancient']
    if len(young_sub) > 0 and len(ancient_sub) > 0:
        u, p = stats.mannwhitneyu(young_sub['m6a_per_kb'], ancient_sub['m6a_per_kb'],
                                   alternative='two-sided')
        print(f"\n{label}: Young m6A/kb={young_sub['m6a_per_kb'].median():.3f} (n={len(young_sub)}) "
              f"vs Ancient={ancient_sub['m6a_per_kb'].median():.3f} (n={len(ancient_sub)})")
        print(f"  Ratio (Young/Ancient): {young_sub['m6a_per_kb'].median() / ancient_sub['m6a_per_kb'].median():.3f}")
        print(f"  Mann-Whitney P={p:.2e}")

# ============================================================
# 5f. Per-subfamily m6A comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 5f: Per-subfamily m6A/kb comparison (top subfamilies)")
print("=" * 70)

# Get subfamilies with >=10 reads in BOTH
sub_counts = {}
for sub in all_subs:
    n1 = mcf7[mcf7['gene_id'] == sub].shape[0]
    n2 = ev[ev['gene_id'] == sub].shape[0]
    if n1 >= 10 and n2 >= 10:
        sub_counts[sub] = n1 + n2

top_subs = sorted(sub_counts, key=sub_counts.get, reverse=True)[:20]

print(f"\n{'Subfamily':<15} {'MCF7_med':>9} {'EV_med':>8} {'Ratio':>7} {'P':>10} {'n_MCF7':>7} {'n_EV':>6}")
sub_scatter_data = []
for sub in top_subs:
    m = mcf7[mcf7['gene_id'] == sub]['m6a_per_kb']
    e = ev[ev['gene_id'] == sub]['m6a_per_kb']
    u, p = stats.mannwhitneyu(e, m, alternative='two-sided')
    ratio = e.median() / m.median() if m.median() > 0 else float('inf')
    age_label = 'Young' if sub in YOUNG_SUBFAMILIES else 'Ancient'
    print(f"{sub:<15} {m.median():>9.2f} {e.median():>8.2f} {ratio:>7.2f} {p:>10.2e} {len(m):>7} {len(e):>6}  {age_label}")
    sub_scatter_data.append({
        'subfamily': sub,
        'mcf7_m6a': m.median(),
        'ev_m6a': e.median(),
        'n_mcf7': len(m),
        'n_ev': len(e),
        'age': age_label
    })

sub_scatter_df = pd.DataFrame(sub_scatter_data)

# ============================================================
# 5g. Among ANCIENT L1 in MCF7: does high m6A predict EV enrichment?
# ============================================================
print("\n" + "=" * 70)
print("STEP 5g: Ancient L1 m6A-based EV enrichment prediction")
print("=" * 70)

# Use MCF7 ancient L1 to define m6A quartiles, then check EV proportions
mcf7_anc = mcf7[mcf7['age'] == 'Ancient'].copy()
ev_anc = ev[ev['age'] == 'Ancient'].copy()

# Combine for quartile analysis
all_ancient = pd.concat([mcf7_anc, ev_anc], ignore_index=True)
# Define quartiles based on MCF7 distribution
quartile_edges = mcf7_anc['m6a_per_kb'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
quartile_edges[0] = -0.01  # include 0
quartile_edges[-1] = all_ancient['m6a_per_kb'].max() + 0.01

quartile_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
all_ancient['m6a_quartile'] = pd.cut(all_ancient['m6a_per_kb'], bins=quartile_edges,
                                      labels=quartile_labels, include_lowest=True)

print(f"\nm6A/kb quartile edges (MCF7-defined): {[f'{x:.2f}' for x in quartile_edges]}")
print(f"\nEV fraction by m6A quartile (ancient L1):")
print(f"{'Quartile':<15} {'n_total':>8} {'n_EV':>6} {'frac_EV':>9} {'n_MCF7':>7}")

quartile_ev_fracs = []
for q in quartile_labels:
    sub = all_ancient[all_ancient['m6a_quartile'] == q]
    n_total = len(sub)
    n_ev = (sub['source'] == 'MCF7-EV').sum()
    n_mcf7 = (sub['source'] == 'MCF7').sum()
    frac = n_ev / n_total if n_total > 0 else 0
    quartile_ev_fracs.append(frac)
    print(f"{q:<15} {n_total:>8} {n_ev:>6} {frac:>9.4f} {n_mcf7:>7}")

# Chi-square test: is EV proportion different across quartiles?
contingency = []
for q in quartile_labels:
    sub = all_ancient[all_ancient['m6a_quartile'] == q]
    n_ev = (sub['source'] == 'MCF7-EV').sum()
    n_mcf7 = (sub['source'] == 'MCF7').sum()
    contingency.append([n_ev, n_mcf7])

contingency = np.array(contingency)
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test (EV proportion ~ m6A quartile): chi2={chi2:.2f}, P={p_chi:.2e}, dof={dof}")

# Trend test (Cochran-Armitage like): correlation between quartile rank and EV fraction
ranks = np.arange(1, 5)
r_trend, p_trend = stats.spearmanr(ranks, quartile_ev_fracs)
print(f"Spearman trend (quartile rank vs EV frac): rho={r_trend:.3f}, P={p_trend:.3f}")

# ============================================================
# 5h. Shared loci comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 5h: Shared loci m6A comparison")
print("=" * 70)

# Find loci present in both MCF7 and MCF7-EV
mcf7_loci = set(mcf7['locus_id'].unique())
ev_loci = set(ev['locus_id'].unique())
shared_loci = mcf7_loci & ev_loci

print(f"MCF7 loci: {len(mcf7_loci)}")
print(f"MCF7-EV loci: {len(ev_loci)}")
print(f"Shared loci: {len(shared_loci)}")
print(f"MCF7-only: {len(mcf7_loci - ev_loci)}")
print(f"EV-only: {len(ev_loci - mcf7_loci)}")

# Compare m6A/kb at shared loci
mcf7_shared = mcf7[mcf7['locus_id'].isin(shared_loci)]
ev_shared = ev[ev['locus_id'].isin(shared_loci)]

print(f"\nShared loci reads: MCF7 {len(mcf7_shared)}, MCF7-EV {len(ev_shared)}")

u, p = stats.mannwhitneyu(ev_shared['m6a_per_kb'], mcf7_shared['m6a_per_kb'],
                           alternative='two-sided')
print(f"MCF7 shared loci m6A/kb: median={mcf7_shared['m6a_per_kb'].median():.3f}")
print(f"EV shared loci m6A/kb: median={ev_shared['m6a_per_kb'].median():.3f}")
print(f"Ratio: {ev_shared['m6a_per_kb'].median() / mcf7_shared['m6a_per_kb'].median():.3f}")
print(f"Mann-Whitney P={p:.2e}")

# Per-locus mean comparison (loci with >=3 reads in both)
locus_m6a = {}
for locus in shared_loci:
    m_reads = mcf7[mcf7['locus_id'] == locus]
    e_reads = ev[ev['locus_id'] == locus]
    if len(m_reads) >= 3 and len(e_reads) >= 3:
        locus_m6a[locus] = {
            'mcf7_m6a': m_reads['m6a_per_kb'].mean(),
            'ev_m6a': e_reads['m6a_per_kb'].mean(),
            'n_mcf7': len(m_reads),
            'n_ev': len(e_reads)
        }

if len(locus_m6a) > 0:
    locus_df = pd.DataFrame(locus_m6a).T
    r, p_r = stats.spearmanr(locus_df['mcf7_m6a'], locus_df['ev_m6a'])
    print(f"\nPer-locus comparison (>=3 reads each): {len(locus_df)} loci")
    print(f"Spearman correlation (MCF7 vs EV m6A/kb per locus): r={r:.3f}, P={p_r:.2e}")
    paired_diff = locus_df['ev_m6a'] - locus_df['mcf7_m6a']
    t_stat, p_paired = stats.wilcoxon(paired_diff)
    print(f"Wilcoxon signed-rank (EV - MCF7 per locus): median diff={paired_diff.median():.3f}, P={p_paired:.2e}")
else:
    print("No loci with >=3 reads in both conditions")

# ============================================================
# 6. Read length matched comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Read-length-matched m6A comparison")
print("=" * 70)

rl_bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 10000)]
rl_labels = ['0-500', '500-1K', '1K-2K', '2K+']

rl_matched_results = []
print(f"\n{'RL bin':<10} {'MCF7_med':>9} {'EV_med':>8} {'Ratio':>7} {'P':>10} {'n_MCF7':>7} {'n_EV':>6}")

for (lo, hi), label in zip(rl_bins, rl_labels):
    m = mcf7[(mcf7['read_length'] >= lo) & (mcf7['read_length'] < hi)]
    e = ev[(ev['read_length'] >= lo) & (ev['read_length'] < hi)]
    if len(m) >= 10 and len(e) >= 10:
        u, p = stats.mannwhitneyu(e['m6a_per_kb'], m['m6a_per_kb'], alternative='two-sided')
        ratio = e['m6a_per_kb'].median() / m['m6a_per_kb'].median() if m['m6a_per_kb'].median() > 0 else np.nan
        print(f"{label:<10} {m['m6a_per_kb'].median():>9.3f} {e['m6a_per_kb'].median():>8.3f} "
              f"{ratio:>7.3f} {p:>10.2e} {len(m):>7} {len(e):>6}")
        rl_matched_results.append({
            'bin': label, 'mcf7_med': m['m6a_per_kb'].median(),
            'ev_med': e['m6a_per_kb'].median(), 'ratio': ratio,
            'p': p, 'n_mcf7': len(m), 'n_ev': len(e)
        })
    else:
        print(f"{label:<10} {'---':>9} {'---':>8} {'---':>7} {'---':>10} {len(m):>7} {len(e):>6}")

# Also do RL-matched by age
print(f"\n--- Read-length-matched, ANCIENT only ---")
print(f"{'RL bin':<10} {'MCF7_med':>9} {'EV_med':>8} {'Ratio':>7} {'P':>10} {'n_MCF7':>7} {'n_EV':>6}")
for (lo, hi), label in zip(rl_bins, rl_labels):
    m = mcf7_ancient[(mcf7_ancient['read_length'] >= lo) & (mcf7_ancient['read_length'] < hi)]
    e = ev_ancient[(ev_ancient['read_length'] >= lo) & (ev_ancient['read_length'] < hi)]
    if len(m) >= 10 and len(e) >= 10:
        u, p = stats.mannwhitneyu(e['m6a_per_kb'], m['m6a_per_kb'], alternative='two-sided')
        ratio = e['m6a_per_kb'].median() / m['m6a_per_kb'].median() if m['m6a_per_kb'].median() > 0 else np.nan
        print(f"{label:<10} {m['m6a_per_kb'].median():>9.3f} {e['m6a_per_kb'].median():>8.3f} "
              f"{ratio:>7.3f} {p:>10.2e} {len(m):>7} {len(e):>6}")
    else:
        print(f"{label:<10} {'---':>9} {'---':>8} {'---':>7} {'---':>10} {len(m):>7} {len(e):>6}")

print(f"\n--- Read-length-matched, YOUNG only ---")
print(f"{'RL bin':<10} {'MCF7_med':>9} {'EV_med':>8} {'Ratio':>7} {'P':>10} {'n_MCF7':>7} {'n_EV':>6}")
for (lo, hi), label in zip(rl_bins, rl_labels):
    m = mcf7_young[(mcf7_young['read_length'] >= lo) & (mcf7_young['read_length'] < hi)]
    e = ev_young[(ev_young['read_length'] >= lo) & (ev_young['read_length'] < hi)]
    if len(m) >= 10 and len(e) >= 10:
        u, p = stats.mannwhitneyu(e['m6a_per_kb'], m['m6a_per_kb'], alternative='two-sided')
        ratio = e['m6a_per_kb'].median() / m['m6a_per_kb'].median() if m['m6a_per_kb'].median() > 0 else np.nan
        print(f"{label:<10} {m['m6a_per_kb'].median():>9.3f} {e['m6a_per_kb'].median():>8.3f} "
              f"{ratio:>7.3f} {p:>10.2e} {len(m):>7} {len(e):>6}")
    else:
        print(f"{label:<10} {'---':>9} {'---':>8} {'---':>7} {'---':>10} {len(m):>7} {len(e):>6}")

# ============================================================
# 7. Poly(A) comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Poly(A) tail length comparison")
print("=" * 70)

mcf7_pa = mcf7[mcf7['polya_length'] > 0]
ev_pa = ev[ev['polya_length'] > 0]

for age_label in ['All', 'Young', 'Ancient']:
    if age_label == 'All':
        m_sub = mcf7_pa
        e_sub = ev_pa
    else:
        m_sub = mcf7_pa[mcf7_pa['age'] == age_label]
        e_sub = ev_pa[ev_pa['age'] == age_label]

    if len(m_sub) > 0 and len(e_sub) > 0:
        u, p = stats.mannwhitneyu(e_sub['polya_length'], m_sub['polya_length'],
                                   alternative='two-sided')
        print(f"\n{age_label}: MCF7 poly(A)={m_sub['polya_length'].median():.1f} (n={len(m_sub)}) "
              f"vs EV={e_sub['polya_length'].median():.1f} (n={len(e_sub)})")
        print(f"  Ratio: {e_sub['polya_length'].median() / m_sub['polya_length'].median():.3f}")
        print(f"  MW P={p:.2e}")

# ============================================================
# 8. EV-only loci: are they higher or lower m6A than shared?
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: EV-only vs shared loci m6A comparison")
print("=" * 70)

ev_only_loci = ev_loci - mcf7_loci
ev_in_shared = ev[ev['locus_id'].isin(shared_loci)]
ev_in_evonly = ev[ev['locus_id'].isin(ev_only_loci)]

print(f"EV reads at shared loci: {len(ev_in_shared)}")
print(f"EV reads at EV-only loci: {len(ev_in_evonly)}")

if len(ev_in_shared) > 0 and len(ev_in_evonly) > 0:
    u, p = stats.mannwhitneyu(ev_in_evonly['m6a_per_kb'], ev_in_shared['m6a_per_kb'],
                               alternative='two-sided')
    print(f"EV shared loci m6A/kb: median={ev_in_shared['m6a_per_kb'].median():.3f}")
    print(f"EV-only loci m6A/kb: median={ev_in_evonly['m6a_per_kb'].median():.3f}")
    print(f"Ratio (EV-only/shared): {ev_in_evonly['m6a_per_kb'].median() / ev_in_shared['m6a_per_kb'].median():.3f}")
    print(f"MW P={p:.2e}")

    # Also by age
    for age_label in ['Young', 'Ancient']:
        s = ev_in_shared[ev_in_shared['age'] == age_label]
        o = ev_in_evonly[ev_in_evonly['age'] == age_label]
        if len(s) >= 5 and len(o) >= 5:
            u, p = stats.mannwhitneyu(o['m6a_per_kb'], s['m6a_per_kb'], alternative='two-sided')
            print(f"  {age_label}: Shared m6A/kb={s['m6a_per_kb'].median():.3f} (n={len(s)}) "
                  f"vs EV-only={o['m6a_per_kb'].median():.3f} (n={len(o)}), P={p:.2e}")

# ============================================================
# 9. m6A × EV sorting: logistic regression
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Logistic regression — m6A predicting EV membership")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Binary outcome: 1=EV, 0=MCF7
combined['is_ev'] = (combined['source'] == 'MCF7-EV').astype(int)

# Model 1: m6A/kb only
X1 = combined[['m6a_per_kb']].values
y = combined['is_ev'].values
scaler1 = StandardScaler()
X1_sc = scaler1.fit_transform(X1)
lr1 = LogisticRegression()
lr1.fit(X1_sc, y)
print(f"\nModel 1 (m6A/kb only):")
print(f"  Coef: {lr1.coef_[0][0]:.4f}, Intercept: {lr1.intercept_[0]:.4f}")
print(f"  Accuracy: {lr1.score(X1_sc, y):.4f}")
print(f"  Direction: {'+ (higher m6A → more EV)' if lr1.coef_[0][0] > 0 else '- (higher m6A → less EV)'}")

# Model 2: m6A/kb + read_length + age
combined['is_young'] = (combined['age'] == 'Young').astype(int)
X2 = combined[['m6a_per_kb', 'read_length', 'is_young']].values
scaler2 = StandardScaler()
X2_sc = scaler2.fit_transform(X2)
lr2 = LogisticRegression()
lr2.fit(X2_sc, y)
feature_names = ['m6A/kb', 'read_length', 'is_young']
print(f"\nModel 2 (m6A/kb + read_length + age):")
for fn, coef in zip(feature_names, lr2.coef_[0]):
    print(f"  {fn}: coef={coef:.4f}")
print(f"  Intercept: {lr2.intercept_[0]:.4f}")
print(f"  Accuracy: {lr2.score(X2_sc, y):.4f}")

# Model 3: Ancient only, m6A/kb + read_length
ancient_combined = combined[combined['age'] == 'Ancient'].copy()
X3 = ancient_combined[['m6a_per_kb', 'read_length']].values
y3 = ancient_combined['is_ev'].values
scaler3 = StandardScaler()
X3_sc = scaler3.fit_transform(X3)
lr3 = LogisticRegression()
lr3.fit(X3_sc, y3)
print(f"\nModel 3 (Ancient only: m6A/kb + read_length):")
for fn, coef in zip(['m6A/kb', 'read_length'], lr3.coef_[0]):
    print(f"  {fn}: coef={coef:.4f}")
print(f"  Accuracy: {lr3.score(X3_sc, y3):.4f}")

# ============================================================
# 10. psi/kb comparison (for completeness)
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: psi/kb comparison (supplementary)")
print("=" * 70)

for age_label in ['All', 'Young', 'Ancient']:
    if age_label == 'All':
        m_sub = mcf7
        e_sub = ev
    else:
        m_sub = mcf7[mcf7['age'] == age_label]
        e_sub = ev[ev['age'] == age_label]

    if len(m_sub) > 0 and len(e_sub) > 0:
        u, p = stats.mannwhitneyu(e_sub['psi_per_kb'], m_sub['psi_per_kb'],
                                   alternative='two-sided')
        print(f"{age_label}: MCF7 psi/kb={m_sub['psi_per_kb'].median():.3f} (n={len(m_sub)}) "
              f"vs EV={e_sub['psi_per_kb'].median():.3f} (n={len(e_sub)}), "
              f"ratio={e_sub['psi_per_kb'].median() / m_sub['psi_per_kb'].median():.3f}, P={p:.2e}")

# ============================================================
# 11. Generate 4-panel figure
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Generating figure")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- Panel A: m6A/kb distribution by source and age ---
ax = axes[0, 0]
positions = [1, 2, 3.5, 4.5]
box_data = [
    mcf7_ancient['m6a_per_kb'].values,
    ev_ancient['m6a_per_kb'].values,
    mcf7_young['m6a_per_kb'].values if len(mcf7_young) > 0 else np.array([]),
    ev_young['m6a_per_kb'].values if len(ev_young) > 0 else np.array([])
]
colors = ['#4472C4', '#ED7D31', '#4472C4', '#ED7D31']
labels_box = ['MCF7\nAncient', 'EV\nAncient', 'MCF7\nYoung', 'EV\nYoung']

bp = ax.boxplot([d for d in box_data if len(d) > 0],
                positions=[p for p, d in zip(positions, box_data) if len(d) > 0],
                widths=0.6, showfliers=False, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], [c for c, d in zip(colors, box_data) if len(d) > 0]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks([p for p, d in zip(positions, box_data) if len(d) > 0])
ax.set_xticklabels([l for l, d in zip(labels_box, box_data) if len(d) > 0])
ax.set_ylabel('m6A/kb', fontsize=12)
ax.set_title('A. m6A/kb: MCF7 vs MCF7-EV', fontsize=13, fontweight='bold')
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

# Add significance stars
def add_sig(ax, x1, x2, y, p):
    sig = 'ns' if p >= 0.05 else ('*' if p >= 0.01 else ('**' if p >= 0.001 else '***'))
    ax.plot([x1, x1, x2, x2], [y, y*1.02, y*1.02, y], 'k-', linewidth=1)
    ax.text((x1+x2)/2, y*1.03, sig, ha='center', va='bottom', fontsize=10)

y_max = max(np.percentile(d, 75) + 1.5*(np.percentile(d, 75)-np.percentile(d, 25))
            for d in box_data if len(d) > 0)
_, p_anc = stats.mannwhitneyu(ev_ancient['m6a_per_kb'], mcf7_ancient['m6a_per_kb'])
add_sig(ax, 1, 2, y_max * 0.95, p_anc)
if len(mcf7_young) > 0 and len(ev_young) > 0:
    _, p_yng = stats.mannwhitneyu(ev_young['m6a_per_kb'], mcf7_young['m6a_per_kb'])
    add_sig(ax, 3.5, 4.5, y_max * 0.95, p_yng)

# Add n counts
for pos, d, lab in zip(positions, box_data, labels_box):
    if len(d) > 0:
        ax.text(pos, -0.3, f'n={len(d)}', ha='center', va='top', fontsize=8, color='gray')

# --- Panel B: Per-subfamily scatter ---
ax = axes[0, 1]

# Get data for all subfamilies with >=5 reads in both
all_sub_scatter = []
for sub in all_subs:
    m_vals = mcf7[mcf7['gene_id'] == sub]['m6a_per_kb']
    e_vals = ev[ev['gene_id'] == sub]['m6a_per_kb']
    if len(m_vals) >= 5 and len(e_vals) >= 5:
        all_sub_scatter.append({
            'subfamily': sub,
            'mcf7': m_vals.median(),
            'ev': e_vals.median(),
            'size': len(m_vals) + len(e_vals),
            'is_young': sub in YOUNG_SUBFAMILIES
        })

sdf = pd.DataFrame(all_sub_scatter)
if len(sdf) > 0:
    ancient_sdf = sdf[~sdf['is_young']]
    young_sdf = sdf[sdf['is_young']]

    ax.scatter(ancient_sdf['mcf7'], ancient_sdf['ev'],
               s=ancient_sdf['size']*0.5, alpha=0.6, c='#4472C4', label='Ancient', edgecolors='white')
    if len(young_sdf) > 0:
        ax.scatter(young_sdf['mcf7'], young_sdf['ev'],
                   s=young_sdf['size']*0.5, alpha=0.8, c='#ED7D31', label='Young',
                   edgecolors='white', marker='D')
        # Label young subfamilies
        for _, row in young_sdf.iterrows():
            ax.annotate(row['subfamily'], (row['mcf7'], row['ev']),
                       fontsize=7, ha='left', va='bottom', color='#ED7D31')

    # Diagonal
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, linewidth=1)

    # Correlation
    r, p = stats.spearmanr(sdf['mcf7'], sdf['ev'])
    ax.text(0.05, 0.95, f'Spearman r={r:.3f}\nP={p:.2e}\nn={len(sdf)} subfamilies',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('MCF7 m6A/kb (median)', fontsize=11)
ax.set_ylabel('MCF7-EV m6A/kb (median)', fontsize=11)
ax.set_title('B. Per-subfamily m6A/kb', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# --- Panel C: Ancient m6A quartile → EV enrichment ---
ax = axes[1, 0]
overall_ev_frac = len(ev_anc) / (len(mcf7_anc) + len(ev_anc))

bars = ax.bar(range(4), quartile_ev_fracs, color=['#2171B5', '#6BAED6', '#FD8D3C', '#D94701'],
              edgecolor='black', linewidth=0.8)
ax.axhline(overall_ev_frac, color='red', linestyle='--', alpha=0.7,
           label=f'Overall EV fraction ({overall_ev_frac:.3f})')
ax.set_xticks(range(4))
ax.set_xticklabels(quartile_labels, fontsize=10)
ax.set_ylabel('Fraction in MCF7-EV', fontsize=11)
ax.set_title('C. Ancient L1: EV fraction by m6A quartile', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# Annotate chi-square
ax.text(0.95, 0.95, f'Chi2={chi2:.1f}\nP={p_chi:.2e}',
        transform=ax.transAxes, va='top', ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add count labels on bars
for i, (frac, q) in enumerate(zip(quartile_ev_fracs, quartile_labels)):
    sub = all_ancient[all_ancient['m6a_quartile'] == q]
    n_ev_q = (sub['source'] == 'MCF7-EV').sum()
    ax.text(i, frac + 0.002, f'n={n_ev_q}', ha='center', va='bottom', fontsize=8)

# --- Panel D: Read-length-matched m6A comparison ---
ax = axes[1, 1]

if len(rl_matched_results) > 0:
    rl_df = pd.DataFrame(rl_matched_results)
    x = np.arange(len(rl_df))
    w = 0.35
    bars1 = ax.bar(x - w/2, rl_df['mcf7_med'], w, label='MCF7', color='#4472C4', alpha=0.8)
    bars2 = ax.bar(x + w/2, rl_df['ev_med'], w, label='MCF7-EV', color='#ED7D31', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(rl_df['bin'], fontsize=10)
    ax.set_xlabel('Read length bin', fontsize=11)
    ax.set_ylabel('Median m6A/kb', fontsize=11)
    ax.set_title('D. Read-length-matched m6A/kb', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Add significance
    for i, row in rl_df.iterrows():
        sig = 'ns' if row['p'] >= 0.05 else ('*' if row['p'] >= 0.01 else
              ('**' if row['p'] >= 0.001 else '***'))
        y_bar = max(row['mcf7_med'], row['ev_med'])
        ax.text(i, y_bar + 0.2, sig, ha='center', fontsize=10, fontweight='bold')

    # Add n labels
    for i, row in rl_df.iterrows():
        ax.text(i - w/2, -0.15, f'n={int(row["n_mcf7"])}', ha='center', fontsize=7, color='gray')
        ax.text(i + w/2, -0.15, f'n={int(row["n_ev"])}', ha='center', fontsize=7, color='gray')

plt.tight_layout()
fig_path = OUT_DIR / 'ev_m6a_sorting_figure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'ev_m6a_sorting_figure.pdf', bbox_inches='tight')
print(f"Figure saved: {fig_path}")
plt.close()

# ============================================================
# 12. Summary table
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Summary table")
print("=" * 70)

summary_rows = []

# Overall
u, p = stats.mannwhitneyu(ev['m6a_per_kb'], mcf7['m6a_per_kb'])
summary_rows.append({
    'Comparison': 'Overall m6A/kb',
    'MCF7': f"{mcf7['m6a_per_kb'].median():.2f}",
    'MCF7-EV': f"{ev['m6a_per_kb'].median():.2f}",
    'Ratio': f"{ev['m6a_per_kb'].median()/mcf7['m6a_per_kb'].median():.3f}",
    'P-value': f"{p:.2e}",
    'n_MCF7': len(mcf7),
    'n_EV': len(ev)
})

# Young
if len(mcf7_young) > 0 and len(ev_young) > 0:
    u, p = stats.mannwhitneyu(ev_young['m6a_per_kb'], mcf7_young['m6a_per_kb'])
    summary_rows.append({
        'Comparison': 'Young m6A/kb',
        'MCF7': f"{mcf7_young['m6a_per_kb'].median():.2f}",
        'MCF7-EV': f"{ev_young['m6a_per_kb'].median():.2f}",
        'Ratio': f"{ev_young['m6a_per_kb'].median()/mcf7_young['m6a_per_kb'].median():.3f}",
        'P-value': f"{p:.2e}",
        'n_MCF7': len(mcf7_young),
        'n_EV': len(ev_young)
    })

# Ancient
u, p = stats.mannwhitneyu(ev_ancient['m6a_per_kb'], mcf7_ancient['m6a_per_kb'])
summary_rows.append({
    'Comparison': 'Ancient m6A/kb',
    'MCF7': f"{mcf7_ancient['m6a_per_kb'].median():.2f}",
    'MCF7-EV': f"{ev_ancient['m6a_per_kb'].median():.2f}",
    'Ratio': f"{ev_ancient['m6a_per_kb'].median()/mcf7_ancient['m6a_per_kb'].median():.3f}",
    'P-value': f"{p:.2e}",
    'n_MCF7': len(mcf7_ancient),
    'n_EV': len(ev_ancient)
})

# Shared loci
u, p = stats.mannwhitneyu(ev_shared['m6a_per_kb'], mcf7_shared['m6a_per_kb'])
summary_rows.append({
    'Comparison': 'Shared loci m6A/kb',
    'MCF7': f"{mcf7_shared['m6a_per_kb'].median():.2f}",
    'MCF7-EV': f"{ev_shared['m6a_per_kb'].median():.2f}",
    'Ratio': f"{ev_shared['m6a_per_kb'].median()/mcf7_shared['m6a_per_kb'].median():.3f}",
    'P-value': f"{p:.2e}",
    'n_MCF7': len(mcf7_shared),
    'n_EV': len(ev_shared)
})

# Poly(A)
for age_label, m_sub, e_sub in [('All poly(A)', mcf7_pa, ev_pa),
                                  ('Young poly(A)', mcf7_pa[mcf7_pa['age']=='Young'],
                                   ev_pa[ev_pa['age']=='Young']),
                                  ('Ancient poly(A)', mcf7_pa[mcf7_pa['age']=='Ancient'],
                                   ev_pa[ev_pa['age']=='Ancient'])]:
    if len(m_sub) > 0 and len(e_sub) > 0:
        u, p = stats.mannwhitneyu(e_sub['polya_length'], m_sub['polya_length'])
        summary_rows.append({
            'Comparison': age_label,
            'MCF7': f"{m_sub['polya_length'].median():.1f}",
            'MCF7-EV': f"{e_sub['polya_length'].median():.1f}",
            'Ratio': f"{e_sub['polya_length'].median()/m_sub['polya_length'].median():.3f}",
            'P-value': f"{p:.2e}",
            'n_MCF7': len(m_sub),
            'n_EV': len(e_sub)
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / 'ev_m6a_summary.tsv'
summary_df.to_csv(summary_path, sep='\t', index=False)
print(f"\nSummary saved: {summary_path}")
print(summary_df.to_string(index=False))

# ============================================================
# 13. Final conclusions
# ============================================================
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("""
1. YOUNG L1 COUNT ENRICHMENT: Confirmed (EV young fraction vs MCF7 young fraction)
2. m6A/kb COMPARISON: See results above for overall, young, ancient, and RL-matched
3. m6A AS EV SORTING PREDICTOR: Chi-square test on m6A quartile-based EV enrichment
4. LOGISTIC REGRESSION: m6A/kb coefficient direction indicates whether higher m6A
   favors (positive) or disfavors (negative) EV sorting
5. SHARED LOCI: Same genomic loci comparison removes compositional confound
6. READ LENGTH MATCHING: Controls for known library-level artifact
""")

# Save per-read data for potential follow-up
combined.to_csv(OUT_DIR / 'ev_mcf7_combined_per_read.tsv', sep='\t', index=False)
print(f"Per-read data saved: {OUT_DIR / 'ev_mcf7_combined_per_read.tsv'}")

print("\nDone.")
