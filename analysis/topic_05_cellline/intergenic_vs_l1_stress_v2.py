#!/usr/bin/env python3
"""
Test v2: L1 sequence specificity in stress-induced poly(A) shortening.

Four groups:
  1. Intergenic L1 (L1 elements in intergenic regions)
  2. Intronic L1 (L1 elements within gene introns)
  3. Genic control (non-L1 gene-associated reads)
  4. Intergenic non-L1 (non-L1 intergenic reads)

Questions:
  A. Does L1 sequence itself drive stress vulnerability?
     → Compare intronic L1 vs genic control (same genomic context, different sequence)
  B. Does genomic context matter for L1?
     → Compare intronic L1 vs intergenic L1
  C. Do non-L1 intergenic transcripts show the same pattern?
     → Compare intergenic L1 vs intergenic non-L1
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

cell_lines = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
}

# ══════════════════════════════════════════
# Step 1: L1 reads — intronic vs intergenic
# ══════════════════════════════════════════
print("Step 1: Loading L1 data with intronic/intergenic classification...")

l1_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv'
        if not os.path.exists(f):
            continue
        tmp = pd.read_csv(f, sep='\t', usecols=[
            'read_id', 'polya_length', 'qc_tag', 'gene_id',
            'overlapping_genes', 'read_length'
        ])
        tmp = tmp[tmp['qc_tag'] == 'PASS']
        tmp['cell_line'] = cl
        tmp['group'] = group
        # Classify: intergenic if overlapping_genes is empty/dot/NaN
        tmp['l1_context'] = tmp['overlapping_genes'].apply(
            lambda x: 'intergenic' if (pd.isna(x) or str(x).strip() in ['', '.']) else 'intronic'
        )
        l1_frames.append(tmp)

df_l1_raw = pd.concat(l1_frames, ignore_index=True)
print(f"  L1 PASS reads: {len(df_l1_raw):,}")
print(f"  Context: {df_l1_raw['l1_context'].value_counts().to_dict()}")
for cl in ['HeLa', 'HeLa-Ars']:
    sub = df_l1_raw[df_l1_raw['cell_line'] == cl]
    ctx = sub['l1_context'].value_counts()
    print(f"    {cl}: intronic={ctx.get('intronic',0)}, intergenic={ctx.get('intergenic',0)}")

# Merge with L1 Part3 cache for m6A
l1_cache_frames = []
for cl, groups in cell_lines.items():
    for group in groups:
        f = f'{ANALYSIS}/topic_05_cellline/part3_l1_per_read_cache/{group}_l1_per_read.tsv'
        if os.path.exists(f):
            tmp = pd.read_csv(f, sep='\t')
            l1_cache_frames.append(tmp)
df_l1_cache = pd.concat(l1_cache_frames, ignore_index=True)
df_l1_cache['m6a_per_kb'] = df_l1_cache['m6a_sites_high'] / (df_l1_cache['read_length'] / 1000)

df_l1 = df_l1_raw.merge(df_l1_cache[['read_id', 'm6a_per_kb']], on='read_id', how='inner')
print(f"  L1 with poly(A) + m6A: {len(df_l1):,}")

# ══════════════════════════════════════════
# Step 2: Control reads — genic vs intergenic
# ══════════════════════════════════════════
print("\nStep 2: Classifying control reads (genic vs intergenic)...")

gene_bed = f'{OUTDIR}/gencode_genes_merged.bed'
if not os.path.exists(gene_bed):
    gtf = f'{REFERENCE}/gencode.v38.annotation.gtf'
    cmd = (
        f"grep -v '^#' {gtf} | "
        f"awk -F'\\t' '$3==\"gene\"{{print $1\"\\t\"$4-1\"\\t\"$5}}' | "
        f"sort -k1,1 -k2,2n | "
        f"bedtools merge > {gene_bed}"
    )
    subprocess.run(cmd, shell=True, check=True)

intergenic_ids = {}
genic_ids = {}

for cl, groups in cell_lines.items():
    ig_all = set()
    ge_all = set()
    for group in groups:
        bam = f'{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam'
        if not os.path.exists(bam):
            continue

        reads_bed = tempfile.mktemp(suffix='.bed')
        ig_bed = tempfile.mktemp(suffix='.bed')
        ge_bed = tempfile.mktemp(suffix='.bed')

        subprocess.run(
            f"module load samtools/1.23 bedtools/2.31.0 && "
            f"samtools view -F 260 -b {bam} | bedtools bamtobed -i stdin | "
            f"sort -k1,1 -k2,2n > {reads_bed}",
            shell=True, check=True, executable='/bin/bash'
        )
        subprocess.run(
            f"module load bedtools/2.31.0 && "
            f"bedtools intersect -a {reads_bed} -b {gene_bed} -v > {ig_bed}",
            shell=True, check=True, executable='/bin/bash'
        )
        subprocess.run(
            f"module load bedtools/2.31.0 && "
            f"bedtools intersect -a {reads_bed} -b {gene_bed} -u > {ge_bed}",
            shell=True, check=True, executable='/bin/bash'
        )

        for bed_file, target_set in [(ig_bed, ig_all), (ge_bed, ge_all)]:
            with open(bed_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        target_set.add(parts[3])

        n_ig = sum(1 for line in open(ig_bed))
        n_ge = sum(1 for line in open(ge_bed))
        print(f"  {group}: {n_ge:,} genic + {n_ig:,} intergenic (BAM level)")

        for p in [reads_bed, ig_bed, ge_bed]:
            if os.path.exists(p):
                os.unlink(p)

    intergenic_ids[cl] = ig_all
    genic_ids[cl] = ge_all

# ══════════════════════════════════════════
# Step 3: Load ctrl m6A + poly(A) and merge
# ══════════════════════════════════════════
print("\nStep 3: Building control dataframe...")

# Part3 ctrl cache
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

# Nanopolish poly(A)
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

# Merge
df_ctrl = df_ctrl_polya.merge(
    df_ctrl_cache[['read_id', 'm6a_per_kb', 'read_length']],
    on='read_id', how='inner'
)

# Classify
def classify_ctrl(row):
    cl = row['cell_line']
    rid = row['read_id']
    if rid in intergenic_ids.get(cl, set()):
        return 'Intergenic non-L1'
    elif rid in genic_ids.get(cl, set()):
        return 'Genic control'
    return 'Unknown'

df_ctrl['category'] = df_ctrl.apply(classify_ctrl, axis=1)

# Diagnostic: intergenic data loss
print("\n  === Intergenic count 진단 ===")
total_ig_bam = sum(len(v) for v in intergenic_ids.values())
n_ig_cache = len(df_ctrl_cache[df_ctrl_cache['read_id'].isin(
    intergenic_ids.get('HeLa', set()) | intergenic_ids.get('HeLa-Ars', set())
)])
n_ig_polya = len(df_ctrl_polya[df_ctrl_polya['read_id'].isin(
    intergenic_ids.get('HeLa', set()) | intergenic_ids.get('HeLa-Ars', set())
)])
n_ig_final = len(df_ctrl[df_ctrl['category'] == 'Intergenic non-L1'])
print(f"  BAM classified intergenic: {total_ig_bam}")
print(f"  In Part3 cache:           {n_ig_cache}")
print(f"  In nanopolish PASS:       {n_ig_polya}")
print(f"  Final (cache ∩ PASS):     {n_ig_final}")
print(f"  PASS rate intergenic:     {n_ig_polya/max(total_ig_bam,1)*100:.0f}%")
ge_bam = sum(len(v) for v in genic_ids.values())
ge_final = len(df_ctrl[df_ctrl['category'] == 'Genic control'])
print(f"  PASS rate genic:          {ge_final/max(ge_bam,1)*100:.0f}%")

# ══════════════════════════════════════════
# Step 4: Combine all groups
# ══════════════════════════════════════════
print("\nStep 4: Combining all groups...")

# L1 categories
df_l1_ig = df_l1[df_l1['l1_context'] == 'intergenic'].copy()
df_l1_ig['category'] = 'L1 intergenic'
df_l1_in = df_l1[df_l1['l1_context'] == 'intronic'].copy()
df_l1_in['category'] = 'L1 intronic'

cols = ['read_id', 'cell_line', 'polya_length', 'm6a_per_kb', 'category']
df_all = pd.concat([
    df_l1_ig[cols],
    df_l1_in[cols],
    df_ctrl[df_ctrl['category'] == 'Genic control'][cols],
    df_ctrl[df_ctrl['category'] == 'Intergenic non-L1'][cols],
], ignore_index=True)

print(f"  Total: {len(df_all):,}")
print(f"  Per group: {df_all['category'].value_counts().to_dict()}")

# ══════════════════════════════════════════
# Step 5: Analysis
# ══════════════════════════════════════════
print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

categories = ['L1 intergenic', 'L1 intronic', 'Genic control', 'Intergenic non-L1']

# 5a: Summary table
print("\n--- 5a: Poly(A) and m6A by group × condition ---")
print(f"{'Group':22s} {'Cond':10s} {'polyA':>7s} {'m6A/kb':>7s} {'n':>6s}")
print("-" * 60)
for cat in categories:
    for cl in ['HeLa', 'HeLa-Ars']:
        sub = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == cl)]
        if len(sub) > 0:
            print(f"  {cat:22s} {cl:10s} {sub['polya_length'].median():6.1f} "
                  f"{sub['m6a_per_kb'].median():6.2f} {len(sub):6,}")

# 5b: Δpoly(A) under stress
print("\n--- 5b: Stress-induced Δpoly(A) ---")
results = {}
for cat in categories:
    normal = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa')]['polya_length'].dropna()
    stress = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna()
    if len(normal) >= 5 and len(stress) >= 5:
        delta = stress.median() - normal.median()
        U, p = stats.mannwhitneyu(normal, stress, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        results[cat] = {'delta': delta, 'p': p, 'sig': sig, 'n_n': len(normal), 'n_s': len(stress)}
        print(f"  {cat:22s}: Δ={delta:+6.1f} nt, P={p:.2e} {sig:3s} "
              f"(n={len(normal)}+{len(stress)})")
    else:
        print(f"  {cat:22s}: insufficient (n={len(normal)}+{len(stress)})")

# 5c: m6A distribution
print("\n--- 5c: m6A/kb distribution ---")
for cat in categories:
    sub = df_all[df_all['category'] == cat]['m6a_per_kb']
    if len(sub) > 0:
        frac_zero = (sub == 0).mean() * 100
        print(f"  {cat:22s}: median={sub.median():.2f}, mean={sub.mean():.2f}, "
              f"IQR=[{sub.quantile(0.25):.2f},{sub.quantile(0.75):.2f}], "
              f"zero%={frac_zero:.0f}%")

# 5d: m6A-poly(A) correlation under stress
print("\n--- 5d: m6A-poly(A) Spearman rho under stress ---")
for cat in categories:
    sub = df_all[(df_all['category'] == cat) & (df_all['cell_line'] == 'HeLa-Ars')]
    sub = sub.dropna(subset=['m6a_per_kb', 'polya_length'])
    if len(sub) >= 15:
        rho, p = stats.spearmanr(sub['m6a_per_kb'], sub['polya_length'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {cat:22s}: rho={rho:+.3f}, P={p:.2e} {sig}, n={len(sub)}")
    else:
        print(f"  {cat:22s}: insufficient (n={len(sub)})")

# 5e: Key comparisons (pairwise MWU)
print("\n--- 5e: Key pairwise comparisons (stress condition only) ---")
comparisons = [
    ('L1 intronic', 'Genic control', 'Q: L1 sequence effect within genes?'),
    ('L1 intergenic', 'L1 intronic', 'Q: Genomic context effect for L1?'),
    ('L1 intergenic', 'Intergenic non-L1', 'Q: L1 vs non-L1 in intergenic?'),
]
for cat1, cat2, question in comparisons:
    s1 = df_all[(df_all['category'] == cat1) & (df_all['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna()
    s2 = df_all[(df_all['category'] == cat2) & (df_all['cell_line'] == 'HeLa-Ars')]['polya_length'].dropna()
    if len(s1) >= 5 and len(s2) >= 5:
        U, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
        print(f"  {question}")
        print(f"    {cat1:22s}: median={s1.median():.1f} (n={len(s1)})")
        print(f"    {cat2:22s}: median={s2.median():.1f} (n={len(s2)})")
        print(f"    MWU P={p:.2e}")

# 5f: OLS interaction — L1 intronic vs Genic control
print("\n--- 5f: OLS intronic L1 vs Genic control ---")
df_test = df_all[df_all['category'].isin(['L1 intronic', 'Genic control'])].dropna(subset=['polya_length']).copy()
if len(df_test) > 100:
    import statsmodels.api as sm
    df_test['is_l1'] = (df_test['category'] == 'L1 intronic').astype(int)
    df_test['is_stress'] = (df_test['cell_line'] == 'HeLa-Ars').astype(int)
    df_test['interaction'] = df_test['is_l1'] * df_test['is_stress']

    X = sm.add_constant(df_test[['is_l1', 'is_stress', 'interaction']])
    y = df_test['polya_length']
    model = sm.OLS(y, X).fit()

    print(f"  poly(A) ~ L1 + Stress + L1×Stress")
    print(f"    L1 effect:     β={model.params['is_l1']:+.2f} nt, P={model.pvalues['is_l1']:.2e}")
    print(f"    Stress effect: β={model.params['is_stress']:+.2f} nt, P={model.pvalues['is_stress']:.2e}")
    print(f"    L1×Stress:     β={model.params['interaction']:+.2f} nt, P={model.pvalues['interaction']:.2e}")
    print(f"    n={len(df_test):,}, R²={model.rsquared:.4f}")

# 5g: m6A-matched comparison (L1 intronic vs genic, stress only)
print("\n--- 5g: m6A-matched comparison (stress, L1 intronic vs Genic) ---")
l1i_stress = df_all[(df_all['category'] == 'L1 intronic') &
                    (df_all['cell_line'] == 'HeLa-Ars')].dropna(subset=['m6a_per_kb', 'polya_length'])
gc_stress = df_all[(df_all['category'] == 'Genic control') &
                   (df_all['cell_line'] == 'HeLa-Ars')].dropna(subset=['m6a_per_kb', 'polya_length'])

if len(l1i_stress) >= 50 and len(gc_stress) >= 50:
    combined = pd.concat([l1i_stress, gc_stress])
    try:
        _, bin_edges = pd.qcut(combined['m6a_per_kb'], 4, retbins=True, duplicates='drop')
        labels = [f'Q{i+1}' for i in range(len(bin_edges)-1)]
        print(f"  m6A/kb bin edges: {[f'{x:.2f}' for x in bin_edges]}")
        print(f"  {'Bin':4s} {'L1_intr polyA':>14s} {'L1_intr n':>10s} "
              f"{'Genic polyA':>12s} {'Genic n':>8s} {'MWU P':>10s}")
        for q in labels:
            for cat, df_sub in [('L1 intronic', l1i_stress), ('Genic', gc_stress)]:
                mask = pd.cut(df_sub['m6a_per_kb'], bins=bin_edges,
                             labels=labels, include_lowest=True) == q
                sub = df_sub[mask]
                if cat == 'L1 intronic':
                    l1_med, l1_n = (sub['polya_length'].median(), len(sub)) if len(sub) >= 3 else (np.nan, len(sub))
                else:
                    gc_med, gc_n = (sub['polya_length'].median(), len(sub)) if len(sub) >= 3 else (np.nan, len(sub))
            if l1_n >= 3 and gc_n >= 3:
                _, p = stats.mannwhitneyu(
                    l1i_stress[pd.cut(l1i_stress['m6a_per_kb'], bins=bin_edges, labels=labels, include_lowest=True) == q]['polya_length'],
                    gc_stress[pd.cut(gc_stress['m6a_per_kb'], bins=bin_edges, labels=labels, include_lowest=True) == q]['polya_length'],
                    alternative='two-sided'
                )
                print(f"  {q:4s} {l1_med:13.1f} {l1_n:10,} {gc_med:11.1f} {gc_n:8,} {p:10.2e}")
            else:
                print(f"  {q:4s} {'–':>14s} {l1_n:10,} {'–':>12s} {gc_n:8,}")
    except Exception as e:
        print(f"  Binning error: {e}")

print("\n" + "=" * 75)
print("INTERPRETATION")
print("=" * 75)
print("""
Q1: L1 서열이 stress 취약성을 결정하는가? (Intronic L1 vs Genic control)
    → 같은 gene body 안에 있어도 L1 sequence가 있으면 poly(A) shortening이
      다르게 나타나는지 확인. L1×Stress interaction이 유의하면 L1 서열 고유 효과.

Q2: Genomic context가 L1에 영향을 주는가? (Intergenic L1 vs Intronic L1)
    → 동일 L1이지만 intergenic vs intronic에서 stress 반응이 다른지.

Q3: Intergenic RNA 일반 성질인가? (Intergenic L1 vs Intergenic non-L1)
    → 동일 intergenic context에서 L1 유무에 따른 차이.
""")

print("Done.")
