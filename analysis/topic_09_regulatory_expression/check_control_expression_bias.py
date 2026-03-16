#!/usr/bin/env python3
"""
Check whether control m6A rate varies with gene expression level.

If highly-expressed genes have lower m6A rates, then our control baseline
is artificially low, inflating the L1 vs Control enrichment ratio.

Steps:
1. Parse GTF -> gene intervals
2. Load control summary (read_id, chrom, position) + Part3 ctrl cache (m6a/psi/length)
3. Assign reads to genes via position overlap
4. Count reads per gene (expression proxy)
5. Stratify into quintiles, compute m6A/kb per quintile
6. Compare with L1 m6A/kb

Author: Claude
Date: 2026-02-14
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

# ── Paths ──
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
CTRL_CACHE_DIR = os.path.join(PROJECT, "analysis/01_exploration/topic_05_cellline/part3_ctrl_per_read_cache")
L1_CACHE_DIR = os.path.join(PROJECT, "analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache")
CTRL_SUMMARY_DIR = os.path.join(PROJECT, "results_group")
GTF_FILE = os.path.join(PROJECT, "reference/Human.gtf")
OUT_DIR = os.path.join(PROJECT, "analysis/01_exploration/topic_09_regulatory_expression")
OUT_FILE = os.path.join(OUT_DIR, "control_expression_bias_results.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ── BASE_GROUPS (exclude HeLa-Ars, MCF7-EV) ──
BASE_GROUPS = [
    'A549_4', 'A549_5', 'A549_6',
    'H9_2', 'H9_3', 'H9_4',
    'Hct116_3', 'Hct116_4',
    'HeLa_1', 'HeLa_2', 'HeLa_3',
    'HepG2_5', 'HepG2_6',
    'HEYA8_1', 'HEYA8_2', 'HEYA8_3',
    'K562_4', 'K562_5', 'K562_6',
    'MCF7_2', 'MCF7_3', 'MCF7_4',
    'SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3',
]

print("=" * 70)
print("CONTROL EXPRESSION BIAS ANALYSIS")
print("=" * 70)

# ── Step 1: Parse GTF for gene intervals ──
print("\n[1] Parsing GTF for gene intervals...")
gene_intervals = []  # list of (chrom, start, end, gene_name, gene_type)

with open(GTF_FILE, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        if fields[2] != 'gene':
            continue
        chrom = fields[0]
        start = int(fields[3]) - 1  # 0-based
        end = int(fields[4])
        attrs = fields[8]
        
        # Extract gene_name
        m = re.search(r'gene_name "([^"]+)"', attrs)
        gene_name = m.group(1) if m else "unknown"
        
        # Extract gene_type
        m2 = re.search(r'gene_type "([^"]+)"', attrs)
        gene_type = m2.group(1) if m2 else "unknown"
        
        gene_intervals.append((chrom, start, end, gene_name, gene_type))

print(f"  Loaded {len(gene_intervals)} genes from GTF")

# Build a simple interval lookup: for each chromosome, sorted list of (start, end, gene_name)
from bisect import bisect_right

chrom_genes = defaultdict(list)
for chrom, start, end, gene_name, gene_type in gene_intervals:
    chrom_genes[chrom].append((start, end, gene_name, gene_type))

# Sort by start position
for chrom in chrom_genes:
    chrom_genes[chrom].sort()

# Build index: for each chrom, array of start positions for binary search
chrom_starts = {}
for chrom in chrom_genes:
    chrom_starts[chrom] = np.array([g[0] for g in chrom_genes[chrom]])


def find_gene(chrom, pos):
    """Find gene overlapping a given position using binary search."""
    if chrom not in chrom_starts:
        return None, None
    starts = chrom_starts[chrom]
    genes = chrom_genes[chrom]
    # Find all genes that could overlap: start <= pos
    idx = bisect_right(starts, pos)
    best_gene = None
    best_type = None
    # Search in a window around the position (genes can overlap)
    lo = max(0, idx - 10)
    hi = min(len(genes), idx + 5)
    for i in range(lo, hi):
        s, e, gn, gt = genes[i]
        if s <= pos < e:
            # Prefer protein_coding genes
            if best_gene is None or gt == 'protein_coding':
                best_gene = gn
                best_type = gt
    return best_gene, best_type


# ── Step 2: Load control summaries (for position info) ──
print("\n[2] Loading control summaries...")
ctrl_positions = {}  # read_id -> (chrom, position, group)
for group in BASE_GROUPS:
    summary_file = os.path.join(CTRL_SUMMARY_DIR, group, "i_control", f"{group}_control_summary.tsv")
    if not os.path.exists(summary_file):
        print(f"  WARNING: {summary_file} not found, skipping")
        continue
    df = pd.read_csv(summary_file, sep='\t')
    for _, row in df.iterrows():
        ctrl_positions[row['read_id']] = (row['chrom'], int(row['position']), group)

print(f"  Loaded {len(ctrl_positions)} control read positions from {len(BASE_GROUPS)} groups")

# ── Step 3: Load Part3 ctrl cache (m6a/psi/length) ──
print("\n[3] Loading Part3 ctrl per-read cache...")
ctrl_mod = {}  # read_id -> (read_length, m6a_sites_high, psi_sites_high)
for group in BASE_GROUPS:
    cache_file = os.path.join(CTRL_CACHE_DIR, f"{group}_ctrl_per_read.tsv")
    if not os.path.exists(cache_file):
        print(f"  WARNING: {cache_file} not found, skipping")
        continue
    df = pd.read_csv(cache_file, sep='\t')
    for _, row in df.iterrows():
        ctrl_mod[row['read_id']] = (
            row['read_length'],
            row['m6a_sites_high'],
            row['psi_sites_high']
        )

print(f"  Loaded {len(ctrl_mod)} control reads with modification data")

# ── Step 4: Join and assign genes ──
print("\n[4] Assigning genes to control reads...")
common_reads = set(ctrl_positions.keys()) & set(ctrl_mod.keys())
print(f"  Common reads (in both summary and cache): {len(common_reads)}")

read_data = []
no_gene_count = 0
gene_counts = defaultdict(int)

for i, read_id in enumerate(common_reads):
    if i % 10000 == 0 and i > 0:
        print(f"  Processed {i}/{len(common_reads)} reads...")
    
    chrom, pos, group = ctrl_positions[read_id]
    rl, m6a, psi = ctrl_mod[read_id]
    
    gene_name, gene_type = find_gene(chrom, pos)
    
    if gene_name is None:
        no_gene_count += 1
        continue
    
    gene_counts[gene_name] += 1
    read_data.append({
        'read_id': read_id,
        'gene_name': gene_name,
        'gene_type': gene_type,
        'group': group,
        'read_length': rl,
        'm6a_sites': m6a,
        'psi_sites': psi,
        'm6a_per_kb': m6a / (rl / 1000) if rl > 0 else 0,
        'psi_per_kb': psi / (rl / 1000) if rl > 0 else 0,
    })

print(f"  Assigned genes: {len(read_data)} reads, {len(gene_counts)} unique genes")
print(f"  No gene overlap: {no_gene_count} reads ({no_gene_count/len(common_reads)*100:.1f}%)")

df_reads = pd.DataFrame(read_data)

# ── Step 5: Gene-level expression (read count) ──
print("\n[5] Computing gene expression levels...")
gene_expr = df_reads.groupby('gene_name').agg(
    n_reads=('read_id', 'count'),
    mean_m6a_per_kb=('m6a_per_kb', 'mean'),
    mean_psi_per_kb=('psi_per_kb', 'mean'),
    mean_rl=('read_length', 'mean'),
    gene_type=('gene_type', 'first'),
).reset_index()

print(f"  Gene expression range: {gene_expr['n_reads'].min()}-{gene_expr['n_reads'].max()} reads")
print(f"  Median reads/gene: {gene_expr['n_reads'].median():.0f}")
print(f"  Mean reads/gene: {gene_expr['n_reads'].mean():.1f}")

# ── Step 6: Expression-based binning ──
# Use BOTH rank-based quintiles AND intuitive read-count bins
print("\n[6] Stratifying genes by expression...")

# Method 1: Rank-based quintiles (equal number of genes per quintile)
gene_expr['rank'] = gene_expr['n_reads'].rank(method='first')
gene_expr['quintile'] = pd.qcut(
    gene_expr['rank'], q=5, 
    labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']
)

# Method 2: Read-count bins (intuitive grouping)
def assign_expr_bin(n):
    if n == 1:
        return 'Singleton (1)'
    elif n <= 3:
        return 'Low (2-3)'
    elif n <= 10:
        return 'Medium (4-10)'
    elif n <= 50:
        return 'High (11-50)'
    else:
        return 'Very High (>50)'

gene_expr['expr_bin'] = gene_expr['n_reads'].apply(assign_expr_bin)

# Merge back to reads
df_reads = df_reads.merge(
    gene_expr[['gene_name', 'quintile', 'expr_bin']], 
    on='gene_name', how='left'
)

# ── Step 7: Compute results ──
print("\n[7] Computing m6A/kb per expression stratum...\n")

results_lines = []
results_lines.append("=" * 80)
results_lines.append("CONTROL EXPRESSION BIAS ANALYSIS")
results_lines.append("Does control m6A rate vary with host gene expression level?")
results_lines.append("=" * 80)
results_lines.append(f"\nDate: 2026-02-14")
results_lines.append(f"Base groups used: {len(BASE_GROUPS)}")
results_lines.append(f"Total control reads with gene+modification data: {len(df_reads)}")
results_lines.append(f"Unique genes: {len(gene_counts)}")
results_lines.append(f"Reads without gene overlap: {no_gene_count} ({no_gene_count/len(common_reads)*100:.1f}%)")

# Gene expression distribution
results_lines.append("\n" + "-" * 80)
results_lines.append("GENE EXPRESSION DISTRIBUTION (reads per gene)")
results_lines.append("-" * 80)
pctiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
pvals = np.percentile(gene_expr['n_reads'], pctiles)
for p, v in zip(pctiles, pvals):
    results_lines.append(f"  P{p:>3}: {v:>6.0f} reads")
results_lines.append(f"  Singleton genes (1 read): {(gene_expr['n_reads'] == 1).sum()} ({(gene_expr['n_reads'] == 1).mean()*100:.1f}%)")

# ── A: Read-count bin analysis ──
results_lines.append("\n" + "-" * 80)
results_lines.append("A. READ-COUNT BIN ANALYSIS (gene-level)")
results_lines.append("-" * 80)
results_lines.append(f"{'Bin':<20} {'N genes':>8} {'N reads':>8} {'Gene m6A/kb':>12} {'Gene psi/kb':>12} {'Gene mean RL':>13}")
results_lines.append("-" * 80)

bin_order = ['Singleton (1)', 'Low (2-3)', 'Medium (4-10)', 'High (11-50)', 'Very High (>50)']
for b in bin_order:
    genes_sub = gene_expr[gene_expr['expr_bin'] == b]
    reads_sub = df_reads[df_reads['expr_bin'] == b]
    if len(genes_sub) == 0:
        continue
    results_lines.append(
        f"{b:<20} {len(genes_sub):>8} {len(reads_sub):>8} "
        f"{genes_sub['mean_m6a_per_kb'].mean():>12.3f} "
        f"{genes_sub['mean_psi_per_kb'].mean():>12.3f} "
        f"{genes_sub['mean_rl'].mean():>13.1f}"
    )

# Read-level version
results_lines.append("\n" + "-" * 80)
results_lines.append("A'. READ-COUNT BIN ANALYSIS (read-level, each read counts equally)")
results_lines.append("-" * 80)
results_lines.append(f"{'Bin':<20} {'N reads':>8} {'%reads':>7} {'m6A/kb mean':>12} {'m6A/kb med':>11} {'psi/kb mean':>12} {'Mean RL':>10}")
results_lines.append("-" * 80)

for b in bin_order:
    reads_sub = df_reads[df_reads['expr_bin'] == b]
    if len(reads_sub) == 0:
        continue
    results_lines.append(
        f"{b:<20} {len(reads_sub):>8} {len(reads_sub)/len(df_reads)*100:>6.1f}% "
        f"{reads_sub['m6a_per_kb'].mean():>12.3f} "
        f"{reads_sub['m6a_per_kb'].median():>11.3f} "
        f"{reads_sub['psi_per_kb'].mean():>12.3f} "
        f"{reads_sub['read_length'].mean():>10.1f}"
    )

results_lines.append("-" * 80)
results_lines.append(
    f"{'All Control':<20} {len(df_reads):>8} {'100.0%':>7} "
    f"{df_reads['m6a_per_kb'].mean():>12.3f} "
    f"{df_reads['m6a_per_kb'].median():>11.3f} "
    f"{df_reads['psi_per_kb'].mean():>12.3f} "
    f"{df_reads['read_length'].mean():>10.1f}"
)

# ── B: Rank-based quintile analysis ──
results_lines.append("\n" + "-" * 80)
results_lines.append("B. RANK-BASED QUINTILE ANALYSIS (gene-level, equal N genes per quintile)")
results_lines.append("-" * 80)
results_lines.append(f"{'Quintile':<16} {'N genes':>8} {'Reads range':>14} {'Gene m6A/kb':>12} {'Gene psi/kb':>12} {'Gene mean RL':>13}")
results_lines.append("-" * 80)

quintile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']
for q in quintile_labels:
    genes_sub = gene_expr[gene_expr['quintile'] == q]
    if len(genes_sub) == 0:
        continue
    rmin, rmax = genes_sub['n_reads'].min(), genes_sub['n_reads'].max()
    results_lines.append(
        f"{q:<16} {len(genes_sub):>8} {rmin:>5}-{rmax:<7} "
        f"{genes_sub['mean_m6a_per_kb'].mean():>12.3f} "
        f"{genes_sub['mean_psi_per_kb'].mean():>12.3f} "
        f"{genes_sub['mean_rl'].mean():>13.1f}"
    )

results_lines.append("\n  Read-level:")
results_lines.append(f"  {'Quintile':<16} {'N reads':>8} {'m6A/kb mean':>12} {'m6A/kb med':>11}")
results_lines.append("  " + "-" * 50)

quintile_m6a = {}
for q in quintile_labels:
    reads_sub = df_reads[df_reads['quintile'] == q]
    if len(reads_sub) == 0:
        continue
    quintile_m6a[q] = reads_sub['m6a_per_kb'].values
    results_lines.append(
        f"  {q:<16} {len(reads_sub):>8} {reads_sub['m6a_per_kb'].mean():>12.3f} "
        f"{reads_sub['m6a_per_kb'].median():>11.3f}"
    )

# ── Statistical tests ──
results_lines.append("\n" + "-" * 80)
results_lines.append("STATISTICAL TESTS")
results_lines.append("-" * 80)

# Kruskal-Wallis across quintiles (read-level)
quintile_values = [df_reads[df_reads['quintile'] == q]['m6a_per_kb'].values 
                   for q in quintile_labels
                   if len(df_reads[df_reads['quintile'] == q]) > 0]
if len(quintile_values) >= 2:
    kw_stat, kw_p = stats.kruskal(*quintile_values)
    results_lines.append(f"Kruskal-Wallis (read-level m6A/kb across quintiles): H={kw_stat:.2f}, p={kw_p:.2e}")

# Q1 vs Q5 Mann-Whitney (read-level)
q1_vals = df_reads[df_reads['quintile'] == 'Q1 (lowest)']['m6a_per_kb'].values
q5_vals = df_reads[df_reads['quintile'] == 'Q5 (highest)']['m6a_per_kb'].values
if len(q1_vals) > 0 and len(q5_vals) > 0:
    mw_stat, mw_p = stats.mannwhitneyu(q1_vals, q5_vals, alternative='two-sided')
    results_lines.append(f"Mann-Whitney Q1 vs Q5 (read-level m6A/kb): U={mw_stat:.0f}, p={mw_p:.2e}")
    results_lines.append(f"  Q1 mean: {q1_vals.mean():.3f}, Q5 mean: {q5_vals.mean():.3f}, Q1/Q5={q1_vals.mean()/q5_vals.mean():.3f}x")

# Kruskal-Wallis across read-count bins (read-level)
bin_values = [df_reads[df_reads['expr_bin'] == b]['m6a_per_kb'].values 
              for b in bin_order
              if len(df_reads[df_reads['expr_bin'] == b]) > 0]
if len(bin_values) >= 2:
    kw_stat2, kw_p2 = stats.kruskal(*bin_values)
    results_lines.append(f"Kruskal-Wallis (read-level m6A/kb across read-count bins): H={kw_stat2:.2f}, p={kw_p2:.2e}")

# Spearman: gene-level expression vs m6A/kb (exclude singletons for robustness)
multi_genes = gene_expr[gene_expr['n_reads'] >= 3]
if len(multi_genes) >= 20:
    log_expr = np.log10(multi_genes['n_reads'].values)
    gene_m6a = multi_genes['mean_m6a_per_kb'].values
    rho, rho_p = stats.spearmanr(log_expr, gene_m6a)
    results_lines.append(f"\nSpearman (log10 reads vs gene m6A/kb, genes with >=3 reads): rho={rho:.4f}, p={rho_p:.2e}")
else:
    rho, rho_p = np.nan, np.nan

# Also on ALL genes
log_expr_all = np.log10(gene_expr['n_reads'].values + 0.5)
gene_m6a_all = gene_expr['mean_m6a_per_kb'].values
rho_all, rho_p_all = stats.spearmanr(log_expr_all, gene_m6a_all)
results_lines.append(f"Spearman (log10 reads vs gene m6A/kb, ALL genes): rho={rho_all:.4f}, p={rho_p_all:.2e}")

# Singleton vs Very High (gene-level)
sg_genes = gene_expr[gene_expr['expr_bin'] == 'Singleton (1)']
vh_genes = gene_expr[gene_expr['expr_bin'] == 'Very High (>50)']
if len(sg_genes) > 0 and len(vh_genes) > 0:
    mw_s, mw_p3 = stats.mannwhitneyu(sg_genes['mean_m6a_per_kb'], vh_genes['mean_m6a_per_kb'], alternative='two-sided')
    results_lines.append(f"Mann-Whitney Singleton vs Very High genes (gene-level m6A/kb): p={mw_p3:.2e}")
    results_lines.append(f"  Singleton mean: {sg_genes['mean_m6a_per_kb'].mean():.3f}, VH mean: {vh_genes['mean_m6a_per_kb'].mean():.3f}")

# ── L1 comparison ──
results_lines.append("\n" + "-" * 80)
results_lines.append("L1 vs CONTROL ENRICHMENT (ORIGINAL AND EXPRESSION-CORRECTED)")
results_lines.append("-" * 80)

# Load L1 data
print("[8] Loading L1 Part3 cache for comparison...")
l1_data = []
for group in BASE_GROUPS:
    cache_file = os.path.join(L1_CACHE_DIR, f"{group}_l1_per_read.tsv")
    if not os.path.exists(cache_file):
        continue
    df = pd.read_csv(cache_file, sep='\t')
    df['m6a_per_kb'] = df['m6a_sites_high'] / (df['read_length'] / 1000)
    l1_data.append(df)

df_l1 = pd.concat(l1_data, ignore_index=True)
l1_m6a_mean = df_l1['m6a_per_kb'].mean()
l1_m6a_median = df_l1['m6a_per_kb'].median()

ctrl_all_mean = df_reads['m6a_per_kb'].mean()
ctrl_all_median = df_reads['m6a_per_kb'].median()

# Various "corrected" baselines
# 1. Gene-weighted (each gene counts equally)
gene_weighted_m6a = gene_expr['mean_m6a_per_kb'].mean()

# 2. Bottom quintile only (lowly-expressed genes)
ctrl_q1_reads = df_reads[df_reads['quintile'] == 'Q1 (lowest)']
ctrl_q1_mean = ctrl_q1_reads['m6a_per_kb'].mean() if len(ctrl_q1_reads) > 0 else np.nan

# 3. Singleton genes only
ctrl_singleton = df_reads[df_reads['expr_bin'] == 'Singleton (1)']
ctrl_sing_mean = ctrl_singleton['m6a_per_kb'].mean() if len(ctrl_singleton) > 0 else np.nan

# 4. Top quintile only (highly-expressed)
ctrl_q5_reads = df_reads[df_reads['quintile'] == 'Q5 (highest)']
ctrl_q5_mean = ctrl_q5_reads['m6a_per_kb'].mean() if len(ctrl_q5_reads) > 0 else np.nan

# 5. Very High only
ctrl_vh = df_reads[df_reads['expr_bin'] == 'Very High (>50)']
ctrl_vh_mean = ctrl_vh['m6a_per_kb'].mean() if len(ctrl_vh) > 0 else np.nan

results_lines.append(f"\nL1 m6A/kb: {l1_m6a_mean:.3f}  (n={len(df_l1)} reads)")
results_lines.append("")
results_lines.append(f"{'Control baseline':<40} {'m6A/kb':>8} {'L1/Ctrl ratio':>14} {'N reads':>8}")
results_lines.append("-" * 75)
results_lines.append(f"{'All control (unweighted, per-read)':<40} {ctrl_all_mean:>8.3f} {l1_m6a_mean/ctrl_all_mean:>14.3f}x {len(df_reads):>8}")
results_lines.append(f"{'Gene-weighted (each gene equal)':<40} {gene_weighted_m6a:>8.3f} {l1_m6a_mean/gene_weighted_m6a:>14.3f}x {len(gene_expr):>8} genes")
if not np.isnan(ctrl_q1_mean) and ctrl_q1_mean > 0:
    results_lines.append(f"{'Q1 only (lowest expression)':<40} {ctrl_q1_mean:>8.3f} {l1_m6a_mean/ctrl_q1_mean:>14.3f}x {len(ctrl_q1_reads):>8}")
if not np.isnan(ctrl_sing_mean) and ctrl_sing_mean > 0:
    results_lines.append(f"{'Singleton genes only (1 read)':<40} {ctrl_sing_mean:>8.3f} {l1_m6a_mean/ctrl_sing_mean:>14.3f}x {len(ctrl_singleton):>8}")
if not np.isnan(ctrl_q5_mean) and ctrl_q5_mean > 0:
    results_lines.append(f"{'Q5 only (highest expression)':<40} {ctrl_q5_mean:>8.3f} {l1_m6a_mean/ctrl_q5_mean:>14.3f}x {len(ctrl_q5_reads):>8}")
if not np.isnan(ctrl_vh_mean) and ctrl_vh_mean > 0:
    results_lines.append(f"{'Very High genes only (>50 reads)':<40} {ctrl_vh_mean:>8.3f} {l1_m6a_mean/ctrl_vh_mean:>14.3f}x {len(ctrl_vh):>8}")

# ── Read length control ──
results_lines.append("\n" + "-" * 80)
results_lines.append("READ LENGTH-CONTROLLED ANALYSIS")
results_lines.append("(Checking if expression-m6A relationship is confounded by read length)")
results_lines.append("-" * 80)

for rl_lo, rl_hi in [(200, 500), (500, 1000), (1000, 2000), (2000, 5000)]:
    subset_all = df_reads[(df_reads['read_length'] >= rl_lo) & (df_reads['read_length'] < rl_hi)]
    # Compare singleton vs very high within this RL bin
    sub_sing = subset_all[subset_all['expr_bin'] == 'Singleton (1)']
    sub_vh = subset_all[subset_all['expr_bin'] == 'Very High (>50)']
    if len(sub_sing) > 10 and len(sub_vh) > 10:
        ratio = sub_sing['m6a_per_kb'].mean() / sub_vh['m6a_per_kb'].mean() if sub_vh['m6a_per_kb'].mean() > 0 else np.nan
        mw_s, mw_p = stats.mannwhitneyu(sub_sing['m6a_per_kb'], sub_vh['m6a_per_kb'], alternative='two-sided')
        results_lines.append(
            f"  RL {rl_lo}-{rl_hi}bp: Singleton m6A/kb={sub_sing['m6a_per_kb'].mean():.3f} (n={len(sub_sing)}), "
            f"VeryHigh={sub_vh['m6a_per_kb'].mean():.3f} (n={len(sub_vh)}), "
            f"Sing/VH={ratio:.3f}x, p={mw_p:.2e}"
        )
    else:
        results_lines.append(
            f"  RL {rl_lo}-{rl_hi}bp: insufficient data (Singleton={len(sub_sing)}, VeryHigh={len(sub_vh)})"
        )

# ── Gene type breakdown ──
results_lines.append("\n" + "-" * 80)
results_lines.append("GENE TYPE BREAKDOWN (top 10 by read count)")
results_lines.append("-" * 80)
results_lines.append(f"{'Gene type':<40} {'N reads':>8} {'m6A/kb':>8} {'psi/kb':>8} {'Mean RL':>8}")
results_lines.append("-" * 80)

for gt in df_reads['gene_type'].value_counts().head(10).index:
    subset = df_reads[df_reads['gene_type'] == gt]
    results_lines.append(
        f"{gt:<40} {len(subset):>8} {subset['m6a_per_kb'].mean():>8.3f} "
        f"{subset['psi_per_kb'].mean():>8.3f} {subset['read_length'].mean():>8.0f}"
    )

# ── Protein-coding only ──
pc_reads = df_reads[df_reads['gene_type'] == 'protein_coding']
if len(pc_reads) > 100:
    results_lines.append("\n" + "-" * 80)
    results_lines.append("PROTEIN-CODING GENES ONLY (by expression bin)")
    results_lines.append("-" * 80)
    results_lines.append(f"{'Bin':<20} {'N reads':>8} {'m6A/kb mean':>12} {'m6A/kb med':>11}")
    results_lines.append("-" * 55)
    
    for b in bin_order:
        sub = pc_reads[pc_reads['expr_bin'] == b]
        if len(sub) == 0:
            continue
        results_lines.append(
            f"{b:<20} {len(sub):>8} {sub['m6a_per_kb'].mean():>12.3f} "
            f"{sub['m6a_per_kb'].median():>11.3f}"
        )
    
    pc_sing = pc_reads[pc_reads['expr_bin'] == 'Singleton (1)']
    pc_vh = pc_reads[pc_reads['expr_bin'] == 'Very High (>50)']
    if len(pc_sing) > 0 and len(pc_vh) > 0:
        pc_ratio = pc_sing['m6a_per_kb'].mean() / pc_vh['m6a_per_kb'].mean() if pc_vh['m6a_per_kb'].mean() > 0 else np.nan
        results_lines.append(f"\n  Singleton/VeryHigh ratio (protein-coding): {pc_ratio:.3f}x")
        results_lines.append(f"  L1 / PC-Singleton: {l1_m6a_mean/pc_sing['m6a_per_kb'].mean():.3f}x")
        results_lines.append(f"  L1 / PC-VeryHigh: {l1_m6a_mean/pc_vh['m6a_per_kb'].mean():.3f}x")

# ── Top expressed genes ──
results_lines.append("\n" + "-" * 80)
results_lines.append("TOP 20 MOST EXPRESSED CONTROL GENES")
results_lines.append("-" * 80)
results_lines.append(f"{'Gene':<20} {'Type':<30} {'Reads':>6} {'m6A/kb':>8} {'psi/kb':>8}")
results_lines.append("-" * 80)

gene_stats = df_reads.groupby(['gene_name', 'gene_type']).agg(
    n_reads=('read_id', 'count'),
    mean_m6a_kb=('m6a_per_kb', 'mean'),
    mean_psi_kb=('psi_per_kb', 'mean'),
).reset_index().sort_values('n_reads', ascending=False)

for _, row in gene_stats.head(20).iterrows():
    results_lines.append(
        f"{row['gene_name']:<20} {row['gene_type']:<30} {row['n_reads']:>6} "
        f"{row['mean_m6a_kb']:>8.2f} {row['mean_psi_kb']:>8.2f}"
    )

# ── Weighted vs unweighted ──
results_lines.append("\n" + "-" * 80)
results_lines.append("WEIGHTED vs UNWEIGHTED CONTROL BASELINE")
results_lines.append("-" * 80)
results_lines.append(f"Per-read (unweighted) control m6A/kb: {ctrl_all_mean:.3f}")
results_lines.append(f"Per-gene (gene-weighted) control m6A/kb: {gene_weighted_m6a:.3f}")
diff = gene_weighted_m6a - ctrl_all_mean
results_lines.append(f"Difference: {diff:+.3f} ({diff/ctrl_all_mean*100:+.1f}%)")
results_lines.append("")
if diff > 0:
    results_lines.append("Gene-weighted > Per-read: high-expr genes have LOWER m6A/kb than low-expr genes.")
    results_lines.append("The per-read control is dominated by high-expr reads, biasing it LOW.")
elif diff < 0:
    results_lines.append("Gene-weighted < Per-read: high-expr genes have HIGHER m6A/kb than low-expr genes.")
    results_lines.append("The per-read control is dominated by high-expr reads, biasing it HIGH.")
else:
    results_lines.append("No difference: expression level does not affect m6A/kb baseline.")

# ── CONCLUSION ──
results_lines.append("\n" + "=" * 80)
results_lines.append("CONCLUSION")
results_lines.append("=" * 80)

# Determine effect direction and size
if len(q1_vals) > 0 and len(q5_vals) > 0:
    q1_m = q1_vals.mean()
    q5_m = q5_vals.mean()
    bias_pct = abs(q1_m - q5_m) / ((q1_m + q5_m) / 2) * 100
    
    results_lines.append(f"\n1. DIRECTION: {'Low-expr genes have HIGHER m6A/kb' if q1_m > q5_m else 'High-expr genes have HIGHER m6A/kb' if q5_m > q1_m else 'No difference'}")
    results_lines.append(f"2. MAGNITUDE: Q1 vs Q5 difference = {bias_pct:.1f}% (Q1={q1_m:.3f}, Q5={q5_m:.3f})")
    results_lines.append(f"3. CORRELATION: Spearman rho = {rho_all:.4f} (p={rho_p_all:.2e})")
    
    if q1_m > q5_m:
        # Low-expression genes have higher m6A — control is biased low
        results_lines.append(f"\n=> BIAS DIRECTION: Control baseline is biased LOW by high-expression genes.")
        results_lines.append(f"   Our standard control m6A/kb ({ctrl_all_mean:.3f}) underestimates what L1")
        results_lines.append(f"   'should' have based on expression-matched genes.")
        results_lines.append(f"\n4. IMPACT ON L1 ENRICHMENT:")
        results_lines.append(f"   Original L1/Control ratio:           {l1_m6a_mean/ctrl_all_mean:.3f}x")
        results_lines.append(f"   Using low-expr control (Q1):         {l1_m6a_mean/q1_m:.3f}x")
        results_lines.append(f"   Using gene-weighted control:         {l1_m6a_mean/gene_weighted_m6a:.3f}x")
        if not np.isnan(ctrl_sing_mean) and ctrl_sing_mean > 0:
            results_lines.append(f"   Using singleton genes only:          {l1_m6a_mean/ctrl_sing_mean:.3f}x")
        results_lines.append(f"\n   L1 reads come from low-expression loci (median 1-2 reads/locus).")
        results_lines.append(f"   The APPROPRIATE comparison is with low-expression control genes.")
        if l1_m6a_mean / q1_m > 1.15:
            results_lines.append(f"\n   EVEN using the most conservative (low-expr) control baseline,")
            results_lines.append(f"   L1 m6A enrichment remains {l1_m6a_mean/q1_m:.2f}x — STILL GENUINE.")
        else:
            results_lines.append(f"\n   WARNING: Using expression-matched control, L1 enrichment drops to {l1_m6a_mean/q1_m:.2f}x.")
            results_lines.append(f"   The expression bias substantially explains the observed enrichment.")
    else:
        results_lines.append(f"\n=> No low-expression bias. Control baseline is not artificially low.")
        results_lines.append(f"   L1 m6A enrichment ({l1_m6a_mean/ctrl_all_mean:.2f}x) is ROBUST.")

results_lines.append(f"\n5. L1 CONTEXT: L1 loci are sparse (median ~2 reads/locus), comparable to")
results_lines.append(f"   low-expression control genes. The per-site rate validation (1.33x)")
results_lines.append(f"   already accounts for motif density differences, but NOT expression level.")

# ── Save ──
output_text = '\n'.join(results_lines)
with open(OUT_FILE, 'w') as f:
    f.write(output_text)

print(output_text)
print(f"\n\nResults saved to: {OUT_FILE}")
