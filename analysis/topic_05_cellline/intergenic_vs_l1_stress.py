#!/usr/bin/env python3
"""
Test: Is stress-induced poly(A) shortening L1-specific or general to intergenic transcripts?

Compare three groups under HeLa (normal) vs HeLa-Ars (stress):
  1. L1 reads (L1 element transcripts)
  2. Intergenic non-L1 reads (non-L1 transcripts outside annotated genes)
  3. Genic control reads (gene-associated transcripts)
"""
import pandas as pd
import numpy as np
import subprocess, os, glob, tempfile
from scipy import stats

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
ANALYSIS = f'{BASE}/analysis/01_exploration'
RESULTS = f'{BASE}/results_group'
REFERENCE = f'{BASE}/reference'
OUTDIR = f'{ANALYSIS}/topic_05_cellline'

# ══════════════════════════════════════════
# Step 1: Create merged gene BED from GENCODE
# ══════════════════════════════════════════
print("Step 1: Creating gene BED from GENCODE...")

gtf = f'{REFERENCE}/gencode.v38.annotation.gtf'
gene_bed = f'{OUTDIR}/gencode_genes_merged.bed'

if not os.path.exists(gene_bed):
    cmd = (
        f"grep -v '^#' {gtf} | "
        f"awk -F'\\t' '$3==\"gene\"{{print $1\"\\t\"$4-1\"\\t\"$5}}' | "
        f"sort -k1,1 -k2,2n | "
        f"bedtools merge > {gene_bed}"
    )
    subprocess.run(cmd, shell=True, check=True)

n_regions = sum(1 for _ in open(gene_bed))
print(f"  {n_regions:,} merged gene regions")

# ══════════════════════════════════════════
# Step 2: Classify control reads as genic vs intergenic
# ══════════════════════════════════════════
print("\nStep 2: Classifying control reads...")

cell_lines = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

intergenic_ids = {}  # cell_line -> set of read_ids
genic_ids = {}

for cl, groups in cell_lines.items():
    ig_all = set()
    ge_all = set()
    for group in groups:
        bam = f'{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam'
        if not os.path.exists(bam):
            print(f"  WARNING: {bam} not found, skipping")
            continue

        # Extract read coords as BED using bedtools bamtobed
        reads_bed = tempfile.mktemp(suffix='.bed')
        cmd = (
            f"samtools view -F 260 -b {bam} | "
            f"bedtools bamtobed -i stdin | "
            f"sort -k1,1 -k2,2n > {reads_bed}"
        )
        subprocess.run(cmd, shell=True, check=True)

        # Intergenic: -v (no overlap with genes)
        ig_bed = tempfile.mktemp(suffix='.bed')
        cmd = f"bedtools intersect -a {reads_bed} -b {gene_bed} -v > {ig_bed}"
        subprocess.run(cmd, shell=True, check=True)

        # Genic: -u (overlap with genes)
        ge_bed = tempfile.mktemp(suffix='.bed')
        cmd = f"bedtools intersect -a {reads_bed} -b {gene_bed} -u > {ge_bed}"
        subprocess.run(cmd, shell=True, check=True)

        # Read IDs (column 4 in BED from bamtobed)
        ig_reads = set()
        with open(ig_bed) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    ig_reads.add(parts[3])

        ge_reads = set()
        with open(ge_bed) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    ge_reads.add(parts[3])

        ig_all.update(ig_reads)
        ge_all.update(ge_reads)

        print(f"  {group}: {len(ge_reads):,} genic + {len(ig_reads):,} intergenic")

        for p in [reads_bed, ig_bed, ge_bed]:
            if os.path.exists(p):
                os.unlink(p)

    intergenic_ids[cl] = ig_all
    genic_ids[cl] = ge_all

total_ig = sum(len(v) for v in intergenic_ids.values())
total_ge = sum(len(v) for v in genic_ids.values())
print(f"  Total: {total_ge:,} genic + {total_ig:,} intergenic")

# ══════════════════════════════════════════
# Step 3: Load m6A and poly(A) data
# ══════════════════════════════════════════
print("\nStep 3: Loading m6A and poly(A) data...")

# Part3 L1 cache
l1_cache_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{ANALYSIS}/topic_05_cellline/part3_l1_per_read_cache/{group}_l1_per_read.tsv'
        if os.path.exists(f):
            tmp = pd.read_csv(f, sep='\t')
            tmp['cell_line'] = cl
            l1_cache_frames.append(tmp)
df_l1_cache = pd.concat(l1_cache_frames, ignore_index=True)
df_l1_cache['m6a_per_kb'] = df_l1_cache['m6a_sites_high'] / (df_l1_cache['read_length'] / 1000)
print(f"  L1 cache: {len(df_l1_cache):,} reads")

# Part3 Ctrl cache
ctrl_cache_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{ANALYSIS}/topic_05_cellline/part3_ctrl_per_read_cache/{group}_ctrl_per_read.tsv'
        if os.path.exists(f):
            tmp = pd.read_csv(f, sep='\t')
            tmp['cell_line'] = cl
            ctrl_cache_frames.append(tmp)
df_ctrl_cache = pd.concat(ctrl_cache_frames, ignore_index=True)
df_ctrl_cache['m6a_per_kb'] = df_ctrl_cache['m6a_sites_high'] / (df_ctrl_cache['read_length'] / 1000)
print(f"  Ctrl cache: {len(df_ctrl_cache):,} reads")

# L1 summary (poly(A))
l1_summary_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
        if os.path.exists(f):
            tmp = pd.read_csv(f, sep='\t', usecols=['read_id', 'polya_length', 'qc_tag', 'gene_id'])
            tmp = tmp[tmp['qc_tag'] == 'PASS']
            tmp['cell_line'] = cl
            l1_summary_frames.append(tmp)
df_l1_summary = pd.concat(l1_summary_frames, ignore_index=True)
print(f"  L1 summary (PASS): {len(df_l1_summary):,} reads")

# Control nanopolish poly(A)
ctrl_polya_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{RESULTS}/{group}/i_control/{group}.control.nanopolish.polya.tsv.gz'
        if os.path.exists(f):
            tmp = pd.read_csv(f, sep='\t', usecols=['readname', 'polya_length', 'qc_tag'])
            tmp = tmp[tmp['qc_tag'] == 'PASS']
            tmp['cell_line'] = cl
            tmp.rename(columns={'readname': 'read_id'}, inplace=True)
            ctrl_polya_frames.append(tmp)
df_ctrl_polya = pd.concat(ctrl_polya_frames, ignore_index=True)
print(f"  Ctrl poly(A) (PASS): {len(df_ctrl_polya):,} reads")

# ══════════════════════════════════════════
# Step 4: Build analysis dataframe
# ══════════════════════════════════════════
print("\nStep 4: Building analysis dataframe...")

# L1: merge summary (poly(A)) with cache (m6A)
df_l1 = df_l1_summary.merge(
    df_l1_cache[['read_id', 'm6a_per_kb']],
    on='read_id', how='inner'
)
df_l1['category'] = 'L1'
print(f"  L1 with poly(A) + m6A: {len(df_l1):,}")

# Ctrl: merge polya with cache, then classify
df_ctrl = df_ctrl_polya.merge(
    df_ctrl_cache[['read_id', 'm6a_per_kb', 'read_length']],
    on='read_id', how='inner'
)

# Classify ctrl reads
def classify_ctrl(row):
    cl = row['cell_line']
    rid = row['read_id']
    if rid in intergenic_ids.get(cl, set()):
        return 'Intergenic non-L1'
    elif rid in genic_ids.get(cl, set()):
        return 'Genic control'
    return 'Unknown'

df_ctrl['category'] = df_ctrl.apply(classify_ctrl, axis=1)
cat_counts = df_ctrl['category'].value_counts()
print(f"  Ctrl classified: {cat_counts.to_dict()}")

# Combine
cols = ['read_id', 'cell_line', 'polya_length', 'm6a_per_kb', 'category']
df_all = pd.concat([
    df_l1[cols],
    df_ctrl[cols],
], ignore_index=True)
df_all = df_all[df_all['category'] != 'Unknown']

# ══════════════════════════════════════════
# Step 5: Analysis
# ══════════════════════════════════════════
print("\n" + "=" * 65)
print("RESULTS: L1 vs Intergenic non-L1 vs Genic control")
print("=" * 65)

# 5a: Basic comparison
print("\n--- 5a: Poly(A) and m6A by category × condition ---")
print(f"{'Category':25s} {'Condition':10s} {'polyA_med':>9s} {'m6A/kb_med':>10s} {'n':>7s}")
print("-" * 65)
for cat in ['L1', 'Intergenic non-L1', 'Genic control']:
    for cl in ['HeLa', 'HeLa-Ars']:
        sub = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == cl)]
        if len(sub) > 0:
            med_pa = sub['polya_length'].median()
            med_m6a = sub['m6a_per_kb'].median()
            print(f"  {cat:25s} {cl:10s} {med_pa:8.1f} {med_m6a:9.2f} {len(sub):7,}")

# 5b: Stress-induced Δpoly(A)
print("\n--- 5b: Stress-induced Δpoly(A) ---")
for cat in ['L1', 'Intergenic non-L1', 'Genic control']:
    normal = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa')]['polya_length'].dropna()
    stress = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna()
    if len(normal) >= 10 and len(stress) >= 10:
        delta = stress.median() - normal.median()
        U, p = stats.mannwhitneyu(normal, stress, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {cat:25s}: Δ={delta:+.1f} nt, P={p:.2e} {sig}, "
              f"n={len(normal)}+{len(stress)}")
    else:
        print(f"  {cat:25s}: insufficient data (n={len(normal)}+{len(stress)})")

# 5c: m6A distribution
print("\n--- 5c: m6A/kb distribution by category ---")
for cat in ['L1', 'Intergenic non-L1', 'Genic control']:
    sub = df_all[df_all['category'] == cat]['m6a_per_kb']
    if len(sub) > 0:
        print(f"  {cat:25s}: median={sub.median():.2f}, mean={sub.mean():.2f}, "
              f"IQR=[{sub.quantile(0.25):.2f}, {sub.quantile(0.75):.2f}]")

# 5d: m6A-poly(A) correlation under stress
print("\n--- 5d: m6A-poly(A) correlation under stress ---")
for cat in ['L1', 'Intergenic non-L1', 'Genic control']:
    sub = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa-Ars')]
    sub = sub.dropna(subset=['m6a_per_kb', 'polya_length'])
    if len(sub) >= 20:
        rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
        print(f"  {cat:25s}: rho={rho:.3f}, P={p:.2e}, n={len(sub)}")
    else:
        print(f"  {cat:25s}: insufficient data (n={len(sub)})")

# 5e: OLS interaction — L1 vs Intergenic
print("\n--- 5e: OLS poly(A) ~ L1 + Stress + L1×Stress (L1 vs Intergenic only) ---")
df_test = df_all[df_all['category'].isin(['L1', 'Intergenic non-L1'])].dropna(subset=['polya_length']).copy()
n_ig = len(df_test[df_test['category'] == 'Intergenic non-L1'])
if n_ig >= 20:
    import statsmodels.api as sm
    df_test['is_l1'] = (df_test['category'] == 'L1').astype(int)
    df_test['is_stress'] = (df_test['cell_line'] == 'HeLa-Ars').astype(int)
    df_test['interaction'] = df_test['is_l1'] * df_test['is_stress']

    X = sm.add_constant(df_test[['is_l1', 'is_stress', 'interaction']])
    y = df_test['polya_length']
    model = sm.OLS(y, X).fit()

    print(f"  L1 effect:     β={model.params['is_l1']:+.2f}, P={model.pvalues['is_l1']:.2e}")
    print(f"  Stress effect: β={model.params['is_stress']:+.2f}, P={model.pvalues['is_stress']:.2e}")
    print(f"  L1×Stress:     β={model.params['interaction']:+.2f}, P={model.pvalues['interaction']:.2e}")
    print(f"  n={len(df_test):,}, R²={model.rsquared:.4f}")
else:
    print(f"  Insufficient intergenic reads ({n_ig}) for interaction test")

# 5f: m6A-matched comparison (if enough intergenic reads)
print("\n--- 5f: m6A-matched comparison (binned) ---")
ig_stress = df_all[(df_all['category'] == 'Intergenic non-L1') &
                   (df_all['cell_line'] == 'HeLa-Ars')].dropna(subset=['m6a_per_kb', 'polya_length'])
l1_stress = df_all[(df_all['category'] == 'L1') &
                   (df_all['cell_line'] == 'HeLa-Ars')].dropna(subset=['m6a_per_kb', 'polya_length'])

if len(ig_stress) >= 20:
    # Bin by m6A/kb using shared quartile boundaries
    combined_m6a = pd.concat([ig_stress['m6a_per_kb'], l1_stress['m6a_per_kb']])
    try:
        bins = pd.qcut(combined_m6a, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        bin_edges = pd.qcut(combined_m6a, 4, retbins=True, duplicates='drop')[1]
        print(f"  m6A/kb bin edges: {[f'{x:.2f}' for x in bin_edges]}")

        for q in bins.unique().dropna().sort_values():
            for cat, df_sub in [('L1', l1_stress), ('Intergenic', ig_stress)]:
                mask = pd.cut(df_sub['m6a_per_kb'], bins=bin_edges, labels=['Q1','Q2','Q3','Q4'][:len(bin_edges)-1], include_lowest=True) == q
                sub = df_sub[mask]
                if len(sub) >= 5:
                    print(f"  {q} × {cat:12s}: median_polyA={sub['polya_length'].median():.1f}, n={len(sub)}")
    except Exception as e:
        print(f"  Binning failed: {e}")
else:
    print(f"  Insufficient intergenic stressed reads ({len(ig_stress)})")

print("\nDone.")
