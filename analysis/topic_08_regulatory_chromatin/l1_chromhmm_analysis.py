#!/usr/bin/env python3
"""
Ancient L1 × ENCODE ChromHMM × m6A × Arsenite Poly(A) Exploratory Analysis.

Steps:
1. Load L1 summary (per-read coords, gene_id, polya, qc_tag, TE_group)
2. Load Part3 cache (per-read m6A/kb)
3. Create L1 locus BED → bedtools intersect with ChromHMM (HeLa-S3 E117)
4. Assign chromatin state to each read
5. Compare:
   a) Chromatin state distribution: ancient L1 vs all, m6A-high vs m6A-low
   b) m6A/kb by chromatin state
   c) Poly(A) shortening (HeLa vs HeLa-Ars) by chromatin state
   d) m6A quartile × chromatin state × poly(A)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = BASE / 'results_group'
CACHE_DIR = BASE / 'analysis/01_exploration/topic_05_cellline/part3_l1_per_read_cache'
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'
OUT_DIR.mkdir(exist_ok=True)

CHROMHMM_BED = OUT_DIR / 'E117_15_coreMarks_hg38lift_mnemonics.bed.gz'  # HeLa-S3 (ENCODE)

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ChromHMM 15-state labels grouped into broader categories
STATE_GROUPS = {
    '1_TssA': 'Promoter',
    '2_TssAFlnk': 'Promoter',
    '3_TxFlnk': 'Transcribed',
    '4_Tx': 'Transcribed',
    '5_TxWk': 'Transcribed',
    '6_EnhG': 'Enhancer',
    '7_Enh': 'Enhancer',
    '8_ZNF/Rpts': 'ZNF/Repeats',
    '9_Het': 'Heterochromatin',
    '10_TssBiv': 'Bivalent',
    '11_BivFlnk': 'Bivalent',
    '12_EnhBiv': 'Bivalent',
    '13_ReprPC': 'Repressed',
    '14_ReprPCWk': 'Repressed',
    '15_Quies': 'Quiescent',
}

###############################################################################
# 1. Load L1 summary files
###############################################################################
print("=== Loading L1 summary files ===")

summary_rows = []
for f in sorted(RESULTS.glob('*/g_summary/*_L1_summary.tsv')):
    df = pd.read_csv(f, sep='\t')
    # Extract group from sample
    if 'sample' in df.columns:
        df['group'] = df['sample'].str.rsplit('_', n=1).str[0]
    summary_rows.append(df)

summary = pd.concat(summary_rows, ignore_index=True)
print(f"  Total L1 reads: {len(summary):,}")

# Filter PASS + polya > 0
summary = summary[summary['qc_tag'] == 'PASS'].copy()
summary = summary[summary['polya_length'] > 0].copy()
print(f"  After PASS + poly(A)>0: {len(summary):,}")

# Classify age
def classify_age(gene_id):
    subfamily = gene_id.split('_dup')[0] if '_dup' in gene_id else gene_id
    return 'young' if subfamily in YOUNG_SUBFAMILIES else 'ancient'

summary['l1_age'] = summary['gene_id'].apply(classify_age)

# Classify genomic context
def classify_context(tg):
    if pd.isna(tg):
        return 'other'
    tg = str(tg).lower()
    if 'intronic' in tg:
        return 'intronic'
    elif 'intergenic' in tg:
        return 'intergenic'
    return 'other'

summary['genomic_context'] = summary['TE_group'].apply(classify_context)

# Cell line & condition
def get_cellline(group):
    return group.rsplit('_', 1)[0]

summary['cellline'] = summary['group'].apply(get_cellline)
summary['condition'] = summary['cellline'].apply(
    lambda x: 'stress' if 'Ars' in x else 'normal'
)

print(f"  Age: {summary['l1_age'].value_counts().to_dict()}")
print(f"  Conditions: {summary['condition'].value_counts().to_dict()}")

###############################################################################
# 2. Load Part3 cache → m6A/kb
###############################################################################
print("\n=== Loading Part3 per-read cache ===")

cache_rows = []
for f in sorted(CACHE_DIR.glob('*_l1_per_read.tsv')):
    df = pd.read_csv(f, sep='\t', usecols=['read_id', 'read_length', 'm6a_sites_high'])
    cache_rows.append(df)

cache = pd.concat(cache_rows, ignore_index=True)
cache['m6a_per_kb'] = cache['m6a_sites_high'] / (cache['read_length'] / 1000)
print(f"  Cached reads: {len(cache):,}")

###############################################################################
# 3. Merge summary + cache
###############################################################################
print("\n=== Merging ===")
merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_per_kb']],
                       on='read_id', how='inner')
print(f"  Merged: {len(merged):,}")

###############################################################################
# 4. bedtools intersect: L1 read coords × ChromHMM
###############################################################################
print("\n=== bedtools intersect with ChromHMM ===")

# Create temp BED from L1 reads (use read alignment coords, not TE coords)
# Use midpoint of the read as the query point for ChromHMM assignment
merged['midpoint'] = ((merged['start'] + merged['end']) // 2).astype(int)

with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as tmp:
    tmp_bed = tmp.name
    for _, row in merged.iterrows():
        mid = row['midpoint']
        tmp.write(f"{row['chr']}\t{mid}\t{mid+1}\t{row['read_id']}\n")

print(f"  Wrote {len(merged):,} read midpoints to temp BED")

# Run bedtools intersect
intersect_out = OUT_DIR / 'l1_chromhmm_intersect.tsv'
cmd = (
    f"module load bedtools/2.31.0 && "
    f"bedtools intersect -a {tmp_bed} -b {CHROMHMM_BED} -wa -wb -sorted 2>/dev/null || "
    f"bedtools intersect -a {tmp_bed} -b {CHROMHMM_BED} -wa -wb"
)

# Sort the temp BED first (bedtools requires sorted for -sorted flag)
import os
sorted_tmp = tmp_bed + '.sorted'
os.system(f"sort -k1,1 -k2,2n {tmp_bed} > {sorted_tmp}")

cmd = (
    f"module load bedtools/2.31.0 2>/dev/null; "
    f"bedtools intersect -a {sorted_tmp} -b <(zcat {CHROMHMM_BED} | sort -k1,1 -k2,2n) -wa -wb"
)
# Simpler: just use unsorted intersect
cmd = (
    f"module load bedtools/2.31.0 2>/dev/null; "
    f"bedtools intersect -a {tmp_bed} -b {CHROMHMM_BED} -wa -wb"
)

result = subprocess.run(
    ['bash', '-c', cmd],
    capture_output=True, text=True, timeout=300
)

if result.returncode != 0:
    print(f"  bedtools error: {result.stderr[:500]}")
    # Try without module load
    cmd2 = f"bedtools intersect -a {tmp_bed} -b {CHROMHMM_BED} -wa -wb"
    result = subprocess.run(
        ['bash', '-c', cmd2],
        capture_output=True, text=True, timeout=300
    )

# Parse output
lines = result.stdout.strip().split('\n')
print(f"  Intersect results: {len(lines):,} lines")

chromhmm_map = {}
for line in lines:
    if not line.strip():
        continue
    parts = line.split('\t')
    if len(parts) >= 8:
        read_id = parts[3]
        state = parts[7]  # ChromHMM state is col 4 of B file (index 7)
        chromhmm_map[read_id] = state

print(f"  Reads with ChromHMM state: {len(chromhmm_map):,}")

# Clean up temp files
os.unlink(tmp_bed)
if os.path.exists(sorted_tmp):
    os.unlink(sorted_tmp)

###############################################################################
# 5. Add chromatin state to merged data
###############################################################################
merged['chromhmm_state'] = merged['read_id'].map(chromhmm_map)
merged['chromhmm_group'] = merged['chromhmm_state'].map(STATE_GROUPS)

assigned = merged.dropna(subset=['chromhmm_state'])
print(f"\n  Reads with ChromHMM assignment: {len(assigned):,} / {len(merged):,} "
      f"({100*len(assigned)/len(merged):.1f}%)")

###############################################################################
# 6. Analysis
###############################################################################
print("\n" + "="*70)
print("ANALYSIS RESULTS")
print("="*70)

# --- 6A. ChromHMM state distribution: All L1, Ancient only, Young only ---
print("\n--- 6A. ChromHMM State Distribution ---")
print("\nAll L1:")
state_counts = assigned['chromhmm_state'].value_counts()
total = len(assigned)
for state in sorted(state_counts.index):
    n = state_counts[state]
    grp = STATE_GROUPS.get(state, '?')
    print(f"  {state:20s} ({grp:15s}): {n:6,} ({100*n/total:5.1f}%)")

# By age
for age in ['ancient', 'young']:
    sub = assigned[assigned['l1_age'] == age]
    print(f"\n{age.capitalize()} L1 (n={len(sub):,}):")
    sc = sub['chromhmm_state'].value_counts()
    for state in sorted(sc.index):
        n = sc[state]
        grp = STATE_GROUPS.get(state, '?')
        print(f"  {state:20s} ({grp:15s}): {n:6,} ({100*n/len(sub):5.1f}%)")

# Group-level summary
print("\n\nGrouped ChromHMM states:")
for age in ['all', 'ancient', 'young']:
    sub = assigned if age == 'all' else assigned[assigned['l1_age'] == age]
    gc = sub['chromhmm_group'].value_counts()
    print(f"\n  {age.capitalize()} (n={len(sub):,}):")
    for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Bivalent', 'Repressed',
                 'Heterochromatin', 'Quiescent', 'ZNF/Repeats']:
        n = gc.get(grp, 0)
        print(f"    {grp:20s}: {n:6,} ({100*n/len(sub):5.1f}%)")

# --- 6B. m6A/kb by chromatin state ---
print("\n--- 6B. m6A/kb by Chromatin State ---")
print("\nAncient L1 only:")
ancient = assigned[assigned['l1_age'] == 'ancient'].copy()

for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Bivalent', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    sub = ancient[ancient['chromhmm_group'] == grp]
    if len(sub) < 10:
        continue
    med = sub['m6a_per_kb'].median()
    mean = sub['m6a_per_kb'].mean()
    print(f"  {grp:20s}: n={len(sub):5,}, median m6A/kb={med:.2f}, mean={mean:.2f}")

# KW test
groups_for_kw = []
labels_for_kw = []
for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Repressed', 'Heterochromatin', 'Quiescent']:
    sub = ancient[ancient['chromhmm_group'] == grp]
    if len(sub) >= 10:
        groups_for_kw.append(sub['m6a_per_kb'].values)
        labels_for_kw.append(grp)

if len(groups_for_kw) >= 2:
    kw_stat, kw_p = stats.kruskal(*groups_for_kw)
    print(f"\n  Kruskal-Wallis (m6A/kb across states): H={kw_stat:.1f}, p={kw_p:.2e}")

# Pairwise: Enhancer vs Quiescent, Promoter vs Quiescent
for grp1, grp2 in [('Enhancer', 'Quiescent'), ('Promoter', 'Quiescent'),
                     ('Enhancer', 'Repressed'), ('Transcribed', 'Quiescent')]:
    g1 = ancient[ancient['chromhmm_group'] == grp1]['m6a_per_kb']
    g2 = ancient[ancient['chromhmm_group'] == grp2]['m6a_per_kb']
    if len(g1) >= 10 and len(g2) >= 10:
        u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        ratio = g1.median() / g2.median() if g2.median() > 0 else np.nan
        print(f"  {grp1} vs {grp2}: median ratio={ratio:.2f}, MW p={u_p:.2e}")

# --- 6C. Arsenite poly(A) shortening by chromatin state ---
print("\n--- 6C. Arsenite Poly(A) Shortening by Chromatin State ---")
print("\nAncient L1, HeLa vs HeLa-Ars:")

# Filter to HeLa comparison only
hela = ancient[(ancient['cellline'] == 'HeLa')].copy()
hela_ars = ancient[(ancient['cellline'] == 'HeLa-Ars')].copy()

print(f"  HeLa reads: {len(hela):,}, HeLa-Ars reads: {len(hela_ars):,}")

for grp in ['Promoter', 'Enhancer', 'Transcribed', 'Bivalent', 'Repressed',
             'Heterochromatin', 'Quiescent']:
    h = hela[hela['chromhmm_group'] == grp]['polya_length']
    ha = hela_ars[hela_ars['chromhmm_group'] == grp]['polya_length']
    if len(h) >= 5 and len(ha) >= 5:
        delta = ha.median() - h.median()
        mw_stat, mw_p = stats.mannwhitneyu(h, ha, alternative='two-sided')
        print(f"  {grp:20s}: HeLa={h.median():.0f}nt (n={len(h)}), "
              f"Ars={ha.median():.0f}nt (n={len(ha)}), "
              f"Δ={delta:+.1f}nt, p={mw_p:.2e}")

# --- 6D. m6A quartile × chromatin state × poly(A) ---
print("\n--- 6D. m6A Quartile × Chromatin State × Poly(A) ---")
print("\nHeLa-Ars ancient L1:")

if len(hela_ars) >= 40:
    # m6A quartiles based on HeLa-Ars ancient
    hela_ars_q = hela_ars.copy()
    hela_ars_q['m6a_q'] = pd.qcut(hela_ars_q['m6a_per_kb'], q=4,
                                    labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                                    duplicates='drop')

    for grp in ['Enhancer', 'Transcribed', 'Quiescent', 'Repressed']:
        sub = hela_ars_q[hela_ars_q['chromhmm_group'] == grp]
        if len(sub) < 20:
            continue
        print(f"\n  {grp} (n={len(sub)}):")
        for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
            qs = sub[sub['m6a_q'] == q]['polya_length']
            if len(qs) >= 3:
                print(f"    {q}: median poly(A)={qs.median():.0f}nt (n={len(qs)})")

        # Q1 vs Q4
        q1 = sub[sub['m6a_q'] == 'Q1(low)']['polya_length']
        q4 = sub[sub['m6a_q'] == 'Q4(high)']['polya_length']
        if len(q1) >= 5 and len(q4) >= 5:
            delta = q4.median() - q1.median()
            _, p = stats.mannwhitneyu(q1, q4, alternative='two-sided')
            print(f"    Q4-Q1 Δ={delta:+.1f}nt, p={p:.2e}")

# --- 6E. Regulatory (Enhancer+Promoter) vs Non-regulatory comparison ---
print("\n--- 6E. Regulatory vs Non-regulatory L1 ---")

ancient['is_regulatory'] = ancient['chromhmm_group'].isin(['Enhancer', 'Promoter'])
reg = ancient[ancient['is_regulatory']]
nonreg = ancient[~ancient['is_regulatory']]

print(f"\n  Regulatory (Enhancer+Promoter): n={len(reg):,} ({100*len(reg)/len(ancient):.1f}%)")
print(f"  Non-regulatory: n={len(nonreg):,} ({100*len(nonreg)/len(ancient):.1f}%)")

# m6A comparison
print(f"\n  m6A/kb: Regulatory={reg['m6a_per_kb'].median():.2f} vs "
      f"Non-reg={nonreg['m6a_per_kb'].median():.2f}")
_, p = stats.mannwhitneyu(reg['m6a_per_kb'], nonreg['m6a_per_kb'], alternative='two-sided')
ratio = reg['m6a_per_kb'].median() / nonreg['m6a_per_kb'].median()
print(f"  Ratio={ratio:.2f}, MW p={p:.2e}")

# Poly(A) comparison (normal conditions, excluding Ars)
reg_normal = reg[reg['condition'] == 'normal']
nonreg_normal = nonreg[nonreg['condition'] == 'normal']
print(f"\n  Poly(A) (normal): Reg={reg_normal['polya_length'].median():.0f}nt vs "
      f"Non-reg={nonreg_normal['polya_length'].median():.0f}nt")

# Arsenite shortening: regulatory vs non-regulatory
for label, sub in [('Regulatory', reg), ('Non-regulatory', nonreg)]:
    h = sub[sub['cellline'] == 'HeLa']['polya_length']
    ha = sub[sub['cellline'] == 'HeLa-Ars']['polya_length']
    if len(h) >= 5 and len(ha) >= 5:
        delta = ha.median() - h.median()
        _, p = stats.mannwhitneyu(h, ha, alternative='two-sided')
        print(f"  {label}: HeLa={h.median():.0f}nt, Ars={ha.median():.0f}nt, "
              f"Δ={delta:+.1f}nt, p={p:.2e}")

# --- 6F. Enhancer L1: host gene annotation ---
print("\n--- 6F. Enhancer-associated L1 Loci ---")
enh_l1 = ancient[ancient['chromhmm_group'] == 'Enhancer'].copy()
print(f"\n  Enhancer L1 reads: {len(enh_l1):,}")

if 'overlapping_genes' in enh_l1.columns:
    enh_genes = enh_l1[enh_l1['overlapping_genes'].notna()]['overlapping_genes']
    # Flatten gene list
    all_genes = []
    for g in enh_genes:
        if pd.notna(g) and g != '.':
            all_genes.extend(str(g).split(','))
    gene_counts = pd.Series(all_genes).value_counts()
    print(f"  Unique host genes: {len(gene_counts)}")
    print(f"  Top 15 host genes:")
    for gene, cnt in gene_counts.head(15).items():
        print(f"    {gene}: {cnt} reads")

# Enhancer L1: intronic vs intergenic
enh_ctx = enh_l1['genomic_context'].value_counts()
for ctx, n in enh_ctx.items():
    print(f"  {ctx}: {n} ({100*n/len(enh_l1):.1f}%)")

###############################################################################
# 7. Save results
###############################################################################
print("\n=== Saving results ===")

# Save full annotated table
out_cols = ['read_id', 'chr', 'start', 'end', 'gene_id', 'l1_age',
            'polya_length', 'm6a_per_kb', 'm6a_sites_high',
            'cellline', 'condition', 'genomic_context',
            'chromhmm_state', 'chromhmm_group', 'group', 'sample']
out_cols = [c for c in out_cols if c in assigned.columns]
assigned[out_cols].to_csv(OUT_DIR / 'l1_chromhmm_annotated.tsv', sep='\t', index=False)
print(f"  Saved: l1_chromhmm_annotated.tsv ({len(assigned):,} reads)")

# Save summary stats
summary_stats = []
for age in ['ancient', 'young', 'all']:
    sub = assigned if age == 'all' else assigned[assigned['l1_age'] == age]
    for grp in sub['chromhmm_group'].dropna().unique():
        gs = sub[sub['chromhmm_group'] == grp]
        summary_stats.append({
            'l1_age': age,
            'chromhmm_group': grp,
            'n_reads': len(gs),
            'pct_reads': 100 * len(gs) / len(sub),
            'median_m6a_per_kb': gs['m6a_per_kb'].median(),
            'mean_m6a_per_kb': gs['m6a_per_kb'].mean(),
            'median_polya': gs['polya_length'].median(),
        })

pd.DataFrame(summary_stats).to_csv(OUT_DIR / 'chromhmm_summary_stats.tsv',
                                     sep='\t', index=False)
print(f"  Saved: chromhmm_summary_stats.tsv")

print("\n=== DONE ===")
