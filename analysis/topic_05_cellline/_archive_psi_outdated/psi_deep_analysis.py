#!/usr/bin/env python3
"""
Deep pseudouridine (psi) analysis for L1 retrotransposons.

4 analyses:
  1. Psi site consistency across reads at the same locus (stoichiometry)
  3. Psi site clustering within reads
  4. Gene context effect on psi (intronic vs intergenic)
  6. Stress-specific protection mechanism clues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
TOPICDIR = PROJECT / 'analysis/01_exploration'
CACHE_DIR = TOPICDIR / 'topic_05_cellline/part3_l1_per_read_cache'
OUTDIR = TOPICDIR / 'topic_05_cellline/psi_deep_analysis'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

CL_GROUPS = {
    'A549': ['A549_4','A549_5','A549_6'],
    'H9': ['H9_2','H9_3','H9_4'],
    'Hct116': ['Hct116_3','Hct116_4'],
    'HeLa': ['HeLa_1','HeLa_2','HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1','HeLa-Ars_2','HeLa-Ars_3'],
    'HepG2': ['HepG2_5','HepG2_6'],
    'HEYA8': ['HEYA8_1','HEYA8_2','HEYA8_3'],
    'K562': ['K562_4','K562_5','K562_6'],
    'MCF7': ['MCF7_2','MCF7_3','MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1','SHSY5Y_2','SHSY5Y_3'],
}

# =========================================================================
# Load data
# =========================================================================
print("Loading data...")

def load_group(group):
    """Load per-read cache + L1 summary, merge."""
    cache_path = CACHE_DIR / f'{group}_l1_per_read.tsv'
    sum_path = PROJECT / f'results_group/{group}/g_summary/{group}_L1_summary.tsv'
    if not cache_path.exists() or not sum_path.exists():
        return pd.DataFrame()

    mod = pd.read_csv(cache_path, sep='\t')
    mod['psi_positions'] = mod['psi_positions'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    mod['m6a_positions'] = mod['m6a_positions'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])

    summ = pd.read_csv(sum_path, sep='\t')
    summ = summ[summ['qc_tag'] == 'PASS'].copy()

    merged = mod.merge(summ[['read_id', 'transcript_id', 'gene_id', 'chr', 'start', 'end',
                              'read_strand', 'te_strand', 'polya_length', 'overlapping_genes',
                              'te_start', 'te_end']],
                        on='read_id', how='inner')
    merged['group'] = group
    return merged


# Load all cell lines
all_data = []
for cl, groups in CL_GROUPS.items():
    for g in groups:
        df = load_group(g)
        if len(df) > 0:
            df['cell_line'] = cl
            all_data.append(df)
            print(f"  {g}: {len(df)} reads")

all_df = pd.concat(all_data, ignore_index=True)
all_df['l1_age'] = all_df['gene_id'].apply(lambda x: 'young' if x in YOUNG else 'ancient')
all_df['psi_per_kb'] = all_df['psi_sites_high'] / all_df['read_length'] * 1000
all_df['m6a_per_kb'] = all_df['m6a_sites_high'] / all_df['read_length'] * 1000
all_df['is_intronic'] = all_df['overlapping_genes'].notna()
all_df['gene_context'] = all_df['is_intronic'].map({True: 'intronic', False: 'intergenic'})
all_df['condition'] = all_df['cell_line'].apply(
    lambda x: 'stressed' if x == 'HeLa-Ars' else 'unstressed')

print(f"\nTotal: {len(all_df):,} reads, {all_df['transcript_id'].nunique():,} loci")

# HeLa + HeLa-Ars subset
hela_df = all_df[all_df['cell_line'].isin(['HeLa', 'HeLa-Ars'])].copy()
print(f"HeLa/HeLa-Ars: {len(hela_df):,} reads")

# =========================================================================
# Analysis 1: Psi Site Consistency (Stoichiometry)
# =========================================================================
print("\n" + "="*60)
print("ANALYSIS 1: Psi Site Consistency Across Reads")
print("="*60)

# For multi-read loci, compute psi site consistency
# Use normalized position (fraction of read) binned into 20 bins
N_BINS = 20

# Focus on loci with ≥3 reads and reads with ≥1 psi site
locus_counts = all_df.groupby('transcript_id').size()
multi_loci = set(locus_counts[locus_counts >= 3].index)
print(f"  Loci with ≥3 reads: {len(multi_loci):,}")

multi_df = all_df[all_df['transcript_id'].isin(multi_loci)].copy()
print(f"  Reads at multi-read loci: {len(multi_df):,}")

# For each read, create a binary vector (N_BINS) of psi presence
def psi_bin_vector(row):
    """Convert psi positions to binned binary vector."""
    if row['psi_sites_high'] == 0:
        return np.zeros(N_BINS)
    positions = row['psi_positions']
    bins = np.zeros(N_BINS)
    for p in positions:
        frac = p / row['read_length']
        b = min(int(frac * N_BINS), N_BINS - 1)
        bins[b] = 1
    return bins

# Compute per-locus consistency
consistency_results = []
for locus_id, locus_grp in multi_df.groupby('transcript_id'):
    n_reads = len(locus_grp)
    if n_reads < 3:
        continue

    # Build bin matrix (n_reads x N_BINS)
    bin_matrix = np.array([psi_bin_vector(row) for _, row in locus_grp.iterrows()])

    # For each bin, compute fraction of reads with psi
    bin_occupancy = bin_matrix.mean(axis=0)  # shape (N_BINS,)

    # Bins that have ANY psi in any read
    active_bins = bin_occupancy > 0
    n_active = active_bins.sum()

    if n_active == 0:
        continue

    # Stoichiometry: for active bins, what's the mean occupancy?
    mean_stoich = bin_occupancy[active_bins].mean()

    # Classify: high consistency (>0.7), moderate (0.3-0.7), low (<0.3)
    consistency_results.append({
        'locus': locus_id,
        'n_reads': n_reads,
        'n_active_bins': int(n_active),
        'mean_stoich': mean_stoich,
        'l1_age': locus_grp['l1_age'].iloc[0],
    })

cons_df = pd.DataFrame(consistency_results)
print(f"\n  Loci analyzed: {len(cons_df)}")
print(f"  Mean stoichiometry: {cons_df['mean_stoich'].mean():.3f}")
print(f"  Median stoichiometry: {cons_df['mean_stoich'].median():.3f}")

# Distribution
for label, lo, hi in [('low (<0.3)', 0, 0.3), ('moderate (0.3-0.7)', 0.3, 0.7), ('high (>0.7)', 0.7, 1.01)]:
    n = ((cons_df['mean_stoich'] >= lo) & (cons_df['mean_stoich'] < hi)).sum()
    print(f"  {label}: {n} loci ({n/len(cons_df)*100:.1f}%)")

# By age
print("\n  By L1 age:")
for age in ['ancient', 'young']:
    sub = cons_df[cons_df['l1_age'] == age]
    if len(sub) > 0:
        print(f"    {age}: n={len(sub)}, mean stoich={sub['mean_stoich'].mean():.3f}, "
              f"median={sub['mean_stoich'].median():.3f}")

# By number of reads
print("\n  By read depth:")
for min_n, max_n, label in [(3, 5, '3-4'), (5, 10, '5-9'), (10, 999, '10+')]:
    sub = cons_df[(cons_df['n_reads'] >= min_n) & (cons_df['n_reads'] < max_n)]
    if len(sub) > 0:
        print(f"    {label} reads: n={len(sub)}, mean stoich={sub['mean_stoich'].mean():.3f}")

# Compare to random expectation
# If psi were randomly placed, expected stoichiometry = psi_rate_per_bin
# Actual psi rate across all reads
psi_rate = all_df['psi_sites_high'].sum() / (all_df['read_length'].sum() / 1000 * N_BINS)
print(f"\n  Global psi rate per bin: {psi_rate:.4f}")
print(f"  If random: expected stoichiometry at active bins ≈ {psi_rate:.3f}")
print(f"  Observed mean stoichiometry: {cons_df['mean_stoich'].mean():.3f}")
print(f"  => {'HIGHER' if cons_df['mean_stoich'].mean() > psi_rate * 2 else 'SIMILAR'} than random "
      f"(ratio = {cons_df['mean_stoich'].mean() / max(psi_rate, 1e-6):.1f}x)")

cons_df.to_csv(OUTDIR / 'analysis1_site_consistency.tsv', sep='\t', index=False)

# =========================================================================
# Analysis 3: Psi Site Clustering Within Reads
# =========================================================================
print("\n" + "="*60)
print("ANALYSIS 3: Psi Clustering Within Reads")
print("="*60)

# For reads with ≥2 psi sites, compute inter-site distances
multi_psi = all_df[all_df['psi_sites_high'] >= 2].copy()
print(f"  Reads with ≥2 psi sites: {len(multi_psi):,}")

def compute_inter_site_stats(positions, read_length):
    """Compute nearest-neighbor distance and gap variance."""
    if len(positions) < 2:
        return None
    pos = sorted(positions)
    gaps = np.diff(pos)
    nn_dist = gaps.min()
    mean_gap = gaps.mean()
    # Expected mean gap if uniform random: read_length / (n_sites + 1)
    expected_gap = read_length / (len(pos) + 1)
    # Coefficient of variation of gaps
    cv = gaps.std() / mean_gap if mean_gap > 0 else 0
    return {
        'n_sites': len(pos),
        'min_gap': nn_dist,
        'mean_gap': mean_gap,
        'max_gap': gaps.max(),
        'gap_cv': cv,
        'expected_gap': expected_gap,
        'gap_ratio': mean_gap / expected_gap if expected_gap > 0 else 0,
    }

cluster_results = []
for _, row in multi_psi.iterrows():
    r = compute_inter_site_stats(row['psi_positions'], row['read_length'])
    if r:
        r['read_id'] = row['read_id']
        r['read_length'] = row['read_length']
        r['cell_line'] = row['cell_line']
        r['l1_age'] = row['l1_age']
        cluster_results.append(r)

clust_df = pd.DataFrame(cluster_results)
print(f"  Analyzed: {len(clust_df):,} reads")

print(f"\n  Mean gap: {clust_df['mean_gap'].median():.0f} bp (median)")
print(f"  Expected gap (uniform): {clust_df['expected_gap'].median():.0f} bp")
print(f"  Gap ratio (obs/exp): {clust_df['gap_ratio'].median():.3f}")
print(f"  (ratio ~1.0 = uniform, <1.0 = clustered, >1.0 = dispersed)")

print(f"\n  Gap CV: {clust_df['gap_cv'].median():.3f} (uniform random CV ≈ 0.5-0.7)")
print(f"  Min nearest-neighbor distance: median = {clust_df['min_gap'].median():.0f} bp")

# Distribution of nearest-neighbor distances
nn_bins = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 500), (500, 9999)]
print("\n  Nearest-neighbor distance distribution:")
for lo, hi in nn_bins:
    n = ((clust_df['min_gap'] >= lo) & (clust_df['min_gap'] < hi)).sum()
    pct = n / len(clust_df) * 100
    print(f"    {lo:4d}-{hi:4d} bp: {n:5d} ({pct:5.1f}%)")

# Statistical test: are gaps more clustered than random?
# Compare observed min gap to simulated uniform random
print("\n  Monte Carlo test: observed vs random min gap...")
np.random.seed(42)
n_sim = 10000
sim_min_gaps = []
for _ in range(n_sim):
    # Sample a random read's n_sites and read_length
    idx = np.random.randint(len(clust_df))
    row = clust_df.iloc[idx]
    n_sites = int(row['n_sites'])
    rdlen = int(row['read_length'])
    # Generate random positions
    random_pos = np.sort(np.random.randint(0, rdlen, size=n_sites))
    if n_sites >= 2:
        sim_min_gaps.append(np.diff(random_pos).min())

obs_median_nn = clust_df['min_gap'].median()
sim_median_nn = np.median(sim_min_gaps)
print(f"  Observed median min gap: {obs_median_nn:.0f} bp")
print(f"  Simulated median min gap: {sim_median_nn:.0f} bp")
mw_nn = stats.mannwhitneyu(clust_df['min_gap'].values, sim_min_gaps, alternative='two-sided')
print(f"  MW test: p = {mw_nn.pvalue:.2e}")

# By age
print("\n  By L1 age:")
for age in ['ancient', 'young']:
    sub = clust_df[clust_df['l1_age'] == age]
    if len(sub) > 0:
        print(f"    {age}: n={len(sub)}, gap_ratio={sub['gap_ratio'].median():.3f}, "
              f"gap_cv={sub['gap_cv'].median():.3f}, min_gap={sub['min_gap'].median():.0f}bp")

# By number of sites
print("\n  By psi site count:")
for ns_lo, ns_hi, label in [(2, 3, '2'), (3, 5, '3-4'), (5, 10, '5-9'), (10, 999, '10+')]:
    sub = clust_df[(clust_df['n_sites'] >= ns_lo) & (clust_df['n_sites'] < ns_hi)]
    if len(sub) > 0:
        print(f"    {label} sites: n={len(sub)}, gap_ratio={sub['gap_ratio'].median():.3f}, "
              f"min_gap={sub['min_gap'].median():.0f}bp")

clust_df.to_csv(OUTDIR / 'analysis3_clustering.tsv', sep='\t', index=False)

# =========================================================================
# Analysis 4: Gene Context Effect on Psi
# =========================================================================
print("\n" + "="*60)
print("ANALYSIS 4: Gene Context Effect on Psi")
print("="*60)

# Intronic vs Intergenic
print("\n  Overall:")
for ctx in ['intronic', 'intergenic']:
    sub = all_df[all_df['gene_context'] == ctx]
    print(f"    {ctx:12s}: n={len(sub):5d}, psi/kb median={sub['psi_per_kb'].median():.2f}, "
          f"mean={sub['psi_per_kb'].mean():.2f}, m6a/kb mean={sub['m6a_per_kb'].mean():.2f}")

mw_ctx = stats.mannwhitneyu(
    all_df[all_df['gene_context']=='intronic']['psi_per_kb'],
    all_df[all_df['gene_context']=='intergenic']['psi_per_kb'],
    alternative='two-sided')
print(f"    MW test (psi/kb): p = {mw_ctx.pvalue:.2e}")

mw_ctx_m6a = stats.mannwhitneyu(
    all_df[all_df['gene_context']=='intronic']['m6a_per_kb'],
    all_df[all_df['gene_context']=='intergenic']['m6a_per_kb'],
    alternative='two-sided')
print(f"    MW test (m6a/kb): p = {mw_ctx_m6a.pvalue:.2e}")

# By cell line
print("\n  Per cell line (psi/kb mean, intronic vs intergenic):")
ctx_cl_results = []
for cl in sorted(CL_GROUPS.keys()):
    cl_sub = all_df[all_df['cell_line'] == cl]
    for ctx in ['intronic', 'intergenic']:
        sub = cl_sub[cl_sub['gene_context'] == ctx]
        if len(sub) >= 10:
            ctx_cl_results.append({
                'cell_line': cl, 'gene_context': ctx,
                'n': len(sub),
                'psi_per_kb_mean': sub['psi_per_kb'].mean(),
                'psi_per_kb_median': sub['psi_per_kb'].median(),
                'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
                'polya_median': sub['polya_length'].median(),
            })

ctx_cl_df = pd.DataFrame(ctx_cl_results)
for cl in sorted(CL_GROUPS.keys()):
    intr = ctx_cl_df[(ctx_cl_df['cell_line']==cl) & (ctx_cl_df['gene_context']=='intronic')]
    inter = ctx_cl_df[(ctx_cl_df['cell_line']==cl) & (ctx_cl_df['gene_context']=='intergenic')]
    if len(intr) > 0 and len(inter) > 0:
        psi_intr = intr['psi_per_kb_mean'].values[0]
        psi_inter = inter['psi_per_kb_mean'].values[0]
        n_intr = intr['n'].values[0]
        n_inter = inter['n'].values[0]
        delta = psi_intr - psi_inter
        print(f"    {cl:10s}: intronic={psi_intr:.2f} (n={n_intr:4d}), "
              f"intergenic={psi_inter:.2f} (n={n_inter:4d}), delta={delta:+.2f}")

# By age × context
print("\n  Age × Gene context:")
for age in ['ancient', 'young']:
    for ctx in ['intronic', 'intergenic']:
        sub = all_df[(all_df['l1_age']==age) & (all_df['gene_context']==ctx)]
        if len(sub) >= 10:
            print(f"    {age:8s} {ctx:12s}: n={len(sub):5d}, psi/kb={sub['psi_per_kb'].mean():.2f}, "
                  f"m6a/kb={sub['m6a_per_kb'].mean():.2f}")

# Read length as confound check
print("\n  Read length by context (confound check):")
for ctx in ['intronic', 'intergenic']:
    sub = all_df[all_df['gene_context'] == ctx]
    print(f"    {ctx:12s}: read_length median={sub['read_length'].median():.0f}, "
          f"mean={sub['read_length'].mean():.0f}")

# Read-length controlled comparison
print("\n  Read-length controlled (OLS psi/kb ~ context + rdLen_z):")
reg = all_df[['psi_per_kb', 'gene_context', 'read_length']].dropna().copy()
reg['is_intronic'] = (reg['gene_context'] == 'intronic').astype(float)
reg['rdlen_z'] = (reg['read_length'] - reg['read_length'].mean()) / reg['read_length'].std()
X = np.column_stack([np.ones(len(reg)), reg['is_intronic'].values, reg['rdlen_z'].values])
y = reg['psi_per_kb'].values
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta
ss_res = np.sum((y - y_hat)**2)
n, k = X.shape
mse = ss_res / (n - k)
cov_beta = mse * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov_beta))
t_stat = beta / se
p_vals = 2 * stats.t.sf(np.abs(t_stat), df=n-k)
print(f"    intercept (intergenic): {beta[0]:.3f}")
print(f"    intronic effect: {beta[1]:+.3f} (se={se[1]:.3f}, p={p_vals[1]:.2e})")
print(f"    read_length_z: {beta[2]:+.3f} (se={se[2]:.3f}, p={p_vals[2]:.2e})")

ctx_cl_df.to_csv(OUTDIR / 'analysis4_gene_context.tsv', sep='\t', index=False)

# =========================================================================
# Analysis 6: Stress-Specific Protection Mechanism
# =========================================================================
print("\n" + "="*60)
print("ANALYSIS 6: Stress-Specific Protection Mechanism")
print("="*60)

# Focus on HeLa-Ars reads
ars_df = all_df[all_df['cell_line'] == 'HeLa-Ars'].copy()
hela_base = all_df[all_df['cell_line'] == 'HeLa'].copy()

# Read-length correct poly(A)
for df_sub in [ars_df, hela_base]:
    rdz = (df_sub['read_length'] - df_sub['read_length'].mean()) / df_sub['read_length'].std()
    X_rd = np.column_stack([np.ones(len(df_sub)), rdz.values])
    b_rd = np.linalg.lstsq(X_rd, df_sub['polya_length'].values, rcond=None)[0]
    df_sub['polya_resid'] = df_sub['polya_length'] - X_rd @ b_rd

# Split HeLa-Ars by poly(A) residual tertile
ars_df['polya_group'] = pd.qcut(ars_df['polya_resid'].rank(method='first'), 3,
                                 labels=['short', 'mid', 'long'])

print(f"\n  HeLa-Ars reads: {len(ars_df):,}")
print(f"  Poly(A) residual tertiles:")
for grp in ['short', 'mid', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    print(f"    {grp:5s}: n={len(sub):4d}, polya_resid={sub['polya_resid'].median():+.1f}, "
          f"polya_raw={sub['polya_length'].median():.1f}")

# 6a: Psi density by protection group
print("\n  6a. Psi density by poly(A) protection group:")
for grp in ['short', 'mid', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    print(f"    {grp:5s}: psi/kb={sub['psi_per_kb'].mean():.2f}, "
          f"m6a/kb={sub['m6a_per_kb'].mean():.2f}, "
          f"psi_frac={sub['psi_sites_high'].gt(0).mean():.3f}")

mw_psi_prot = stats.mannwhitneyu(
    ars_df[ars_df['polya_group']=='long']['psi_per_kb'],
    ars_df[ars_df['polya_group']=='short']['psi_per_kb'],
    alternative='greater')
print(f"    Long vs Short psi/kb: MW p = {mw_psi_prot.pvalue:.2e} (one-sided greater)")

# 6b: Psi position bias in protected vs unprotected reads
print("\n  6b. Psi position bias (5' vs 3' half of read):")
for grp in ['short', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    five_prime = 0
    three_prime = 0
    total = 0
    for _, row in sub.iterrows():
        for p in row['psi_positions']:
            if p < row['read_length'] / 2:
                five_prime += 1
            else:
                three_prime += 1
            total += 1
    if total > 0:
        frac_5 = five_prime / total
        frac_3 = three_prime / total
        print(f"    {grp:5s}: 5'-half={frac_5:.3f}, 3'-half={frac_3:.3f} (n_sites={total})")

# More detailed: psi position distribution in quartiles of read
print("\n  Psi position by read quartile (normalized 0-1):")
for grp in ['short', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    all_norm_pos = []
    for _, row in sub.iterrows():
        for p in row['psi_positions']:
            all_norm_pos.append(p / row['read_length'])
    all_norm_pos = np.array(all_norm_pos)
    q_counts = np.histogram(all_norm_pos, bins=[0, 0.25, 0.5, 0.75, 1.0])[0]
    q_frac = q_counts / q_counts.sum()
    print(f"    {grp:5s}: Q1(5')={q_frac[0]:.3f}, Q2={q_frac[1]:.3f}, "
          f"Q3={q_frac[2]:.3f}, Q4(3')={q_frac[3]:.3f}")

# 6c: m6A co-occurrence in protected vs unprotected
print("\n  6c. m6A co-occurrence by protection group:")
for grp in ['short', 'mid', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    has_psi = sub['psi_sites_high'] > 0
    has_m6a = sub['m6a_sites_high'] > 0
    both = (has_psi & has_m6a).sum()
    neither = (~has_psi & ~has_m6a).sum()
    psi_only = (has_psi & ~has_m6a).sum()
    m6a_only = (~has_psi & has_m6a).sum()

    # OR
    a, b, c, d = both, m6a_only, psi_only, neither
    if b > 0 and c > 0:
        OR = (a * d) / (b * c) if (b * c) > 0 else float('inf')
    else:
        OR = float('inf')

    co_rate = both / len(sub)
    print(f"    {grp:5s}: co-occurrence={co_rate:.3f}, OR={OR:.2f} "
          f"(both={both}, psi_only={psi_only}, m6a_only={m6a_only}, neither={neither})")

# 6d: Compare protected reads in HeLa-Ars vs equivalent reads in HeLa
print("\n  6d. Are 'protected' reads in HeLa-Ars similar to normal HeLa reads?")

# High-psi reads in both conditions
for label, df_sub in [('HeLa', hela_base), ('HeLa-Ars', ars_df)]:
    high_psi = df_sub[df_sub['psi_per_kb'] > df_sub['psi_per_kb'].quantile(0.75)]
    low_psi = df_sub[df_sub['psi_per_kb'] <= df_sub['psi_per_kb'].quantile(0.25)]
    print(f"    {label:10s}: high-psi polyA={high_psi['polya_length'].median():.1f}, "
          f"low-psi polyA={low_psi['polya_length'].median():.1f}, "
          f"delta={high_psi['polya_length'].median() - low_psi['polya_length'].median():+.1f}")

# 6e: Psi site count (not just density) in protected vs unprotected
print("\n  6e. Absolute psi site count by protection group:")
for grp in ['short', 'mid', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    print(f"    {grp:5s}: psi_sites mean={sub['psi_sites_high'].mean():.2f}, "
          f"read_length mean={sub['read_length'].mean():.0f}, "
          f"sites_per_read={sub['psi_sites_high'].mean():.2f}")

# 6f: Is protection specific to ancient or also in young?
print("\n  6f. Psi-poly(A) correlation by L1 age in HeLa-Ars:")
for age in ['ancient', 'young']:
    sub = ars_df[ars_df['l1_age'] == age]
    if len(sub) >= 20:
        r, p = stats.pearsonr(sub['psi_per_kb'], sub['polya_resid'])
        print(f"    {age:8s}: n={len(sub):4d}, r(psi/kb, polya_resid)={r:+.3f}, p={p:.2e}")

# Save summary
prot_results = []
for grp in ['short', 'mid', 'long']:
    sub = ars_df[ars_df['polya_group'] == grp]
    prot_results.append({
        'group': grp, 'n': len(sub),
        'polya_median': sub['polya_length'].median(),
        'polya_resid_median': sub['polya_resid'].median(),
        'psi_per_kb_mean': sub['psi_per_kb'].mean(),
        'm6a_per_kb_mean': sub['m6a_per_kb'].mean(),
        'psi_frac': sub['psi_sites_high'].gt(0).mean(),
        'm6a_psi_co_rate': ((sub['psi_sites_high']>0)&(sub['m6a_sites_high']>0)).mean(),
    })
pd.DataFrame(prot_results).to_csv(OUTDIR / 'analysis6_protection_mechanism.tsv', sep='\t', index=False)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nAll results saved to: {OUTDIR}")
print("\nDone!")
