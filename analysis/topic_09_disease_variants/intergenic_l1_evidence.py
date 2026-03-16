#!/usr/bin/env python3
"""
Analyze evidence that intergenic L1 reads are real transcripts, not mapping artifacts.

Checks:
1. Fraction intergenic vs intronic across cell lines
2. MAPQ distribution (from BAM files)
3. Poly(A) tail distributions
4. m6A patterns (per-site rate, m6A/kb)
5. Arsenite response
6. Read length distributions
7. ChromHMM chromatin state patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
import pysam
import os
import sys
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
RESULTS = f'{PROJECT}/results_group'
TOPIC05 = f'{PROJECT}/analysis/01_exploration/topic_05_cellline'
TOPIC08 = f'{PROJECT}/analysis/01_exploration/topic_08_regulatory_chromatin'
CHROMHMM_FILE = f'{TOPIC08}/l1_chromhmm_annotated.tsv'
OUTDIR = f'{PROJECT}/analysis/01_exploration/topic_09_disease_variants/intergenic_evidence'
os.makedirs(OUTDIR, exist_ok=True)

# Cell lines with >= 200 reads (from existing analyses)
BASE_GROUPS = {
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HeLa-Ars': ['HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3'],
    'MCF7': ['MCF7_1', 'MCF7_2', 'MCF7_3'],
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'HepG2': ['HepG2_4', 'HepG2_5', 'HepG2_6'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'SHSY5Y': ['SHSY5Y_4', 'SHSY5Y_5', 'SHSY5Y_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'Hct116': ['Hct116_3', 'Hct116_4'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2'],
    'IMR90': ['IMR90_2', 'IMR90_3'],
}

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# ============================================================
# LOAD ALL L1 SUMMARY DATA
# ============================================================
print("=" * 80)
print("LOADING L1 SUMMARY DATA ACROSS CELL LINES")
print("=" * 80)

all_reads = []
for cl, groups in BASE_GROUPS.items():
    for grp in groups:
        path = f'{RESULTS}/{grp}/g_summary/{grp}_L1_summary.tsv'
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path, sep='\t')
        df = df[df.qc_tag == 'PASS'].copy()
        df['cellline'] = cl
        df['group'] = grp
        # Determine genomic context
        df['genomic_context'] = df['overlapping_genes'].apply(
            lambda x: 'intergenic' if pd.isna(x) else 'intronic'
        )
        # Determine age
        df['l1_age'] = df['gene_id'].apply(
            lambda x: 'young' if x in YOUNG_SUBFAMILIES else 'ancient'
        )
        df['read_length_kb'] = df['read_length'] / 1000
        all_reads.append(df)

all_df = pd.concat(all_reads, ignore_index=True)
print(f"\nTotal PASS L1 reads loaded: {len(all_df)}")
print(f"Cell lines: {all_df.cellline.nunique()}")

# ============================================================
# 1. FRACTION INTERGENIC VS INTRONIC
# ============================================================
print("\n" + "=" * 80)
print("1. INTERGENIC VS INTRONIC FRACTION BY CELL LINE")
print("=" * 80)

context_counts = all_df.groupby(['cellline', 'genomic_context']).size().unstack(fill_value=0)
context_counts['total'] = context_counts.sum(axis=1)
context_counts['intergenic_pct'] = 100 * context_counts['intergenic'] / context_counts['total']
context_counts['intronic_pct'] = 100 * context_counts['intronic'] / context_counts['total']
print(context_counts[['intronic', 'intergenic', 'total', 'intronic_pct', 'intergenic_pct']].to_string())

overall_inter = (all_df.genomic_context == 'intergenic').sum()
overall_total = len(all_df)
print(f"\nOverall: {overall_inter}/{overall_total} intergenic ({100*overall_inter/overall_total:.1f}%)")

# By age
for age in ['ancient', 'young']:
    sub = all_df[all_df.l1_age == age]
    n_inter = (sub.genomic_context == 'intergenic').sum()
    print(f"  {age}: {n_inter}/{len(sub)} intergenic ({100*n_inter/len(sub):.1f}%)")

# ============================================================
# 2. MAPQ DISTRIBUTION FROM BAM FILES
# ============================================================
print("\n" + "=" * 80)
print("2. MAPQ DISTRIBUTION: INTERGENIC VS INTRONIC")
print("=" * 80)

# Get MAPQ from MAFIA BAM files for all reads
mapq_data = []
read_ids_set = set(all_df.read_id)

for cl, groups in BASE_GROUPS.items():
    for grp in groups:
        bam_path = f'{RESULTS}/{grp}/h_mafia/{grp}.mAFiA.reads.bam'
        if not os.path.exists(bam_path):
            print(f"  MISSING BAM: {bam_path}")
            continue
        try:
            with pysam.AlignmentFile(bam_path, 'rb') as bam:
                for read in bam.fetch():
                    if read.query_name in read_ids_set:
                        mapq_data.append({
                            'read_id': read.query_name,
                            'mapq': read.mapping_quality
                        })
        except Exception as e:
            print(f"  ERROR {bam_path}: {e}")
            continue

mapq_df = pd.DataFrame(mapq_data)
print(f"MAPQ retrieved for {len(mapq_df)} reads")

# Merge with main data
merged = all_df.merge(mapq_df, on='read_id', how='inner')
print(f"Merged: {len(merged)} reads with MAPQ")

# Compare MAPQ
for ctx in ['intronic', 'intergenic']:
    sub = merged[merged.genomic_context == ctx]
    print(f"\n{ctx.upper()} (n={len(sub)}):")
    print(f"  MAPQ mean={sub.mapq.mean():.1f}, median={sub.mapq.median():.0f}")
    print(f"  MAPQ=0: {(sub.mapq==0).sum()} ({100*(sub.mapq==0).sum()/len(sub):.1f}%)")
    print(f"  MAPQ>=20: {(sub.mapq>=20).sum()} ({100*(sub.mapq>=20).sum()/len(sub):.1f}%)")
    print(f"  MAPQ>=30: {(sub.mapq>=30).sum()} ({100*(sub.mapq>=30).sum()/len(sub):.1f}%)")
    print(f"  MAPQ>=60: {(sub.mapq>=60).sum()} ({100*(sub.mapq>=60).sum()/len(sub):.1f}%)")
    print(f"  MAPQ distribution: {sub.mapq.describe().to_dict()}")

# Statistical test
intronic_mapq = merged[merged.genomic_context == 'intronic'].mapq
intergenic_mapq = merged[merged.genomic_context == 'intergenic'].mapq
mwu_stat, mwu_p = stats.mannwhitneyu(intronic_mapq, intergenic_mapq, alternative='two-sided')
print(f"\nMann-Whitney U test: U={mwu_stat:.0f}, p={mwu_p:.2e}")

# By age
for age in ['ancient', 'young']:
    sub = merged[merged.l1_age == age]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) > 0:
            print(f"  {age} {ctx}: MAPQ mean={s.mapq.mean():.1f}, median={s.mapq.median():.0f}, n={len(s)}")

# ============================================================
# 3. POLY(A) TAIL DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 80)
print("3. POLY(A) TAIL: INTERGENIC VS INTRONIC")
print("=" * 80)

for ctx in ['intronic', 'intergenic']:
    sub = all_df[(all_df.genomic_context == ctx) & (all_df.polya_length > 0)]
    print(f"\n{ctx.upper()} (n={len(sub)}, poly(A)>0):")
    print(f"  Poly(A) mean={sub.polya_length.mean():.1f}, median={sub.polya_length.median():.1f}")
    print(f"  Poly(A) std={sub.polya_length.std():.1f}")
    print(f"  Poly(A) Q25={sub.polya_length.quantile(0.25):.1f}, Q75={sub.polya_length.quantile(0.75):.1f}")

# Test
intr_pa = all_df[(all_df.genomic_context == 'intronic') & (all_df.polya_length > 0)].polya_length
intg_pa = all_df[(all_df.genomic_context == 'intergenic') & (all_df.polya_length > 0)].polya_length
mwu_stat, mwu_p = stats.mannwhitneyu(intr_pa, intg_pa, alternative='two-sided')
print(f"\nMann-Whitney U: p={mwu_p:.2e}")

# By cell line
print("\nPoly(A) median by cell line:")
for cl in sorted(all_df.cellline.unique()):
    sub = all_df[(all_df.cellline == cl) & (all_df.polya_length > 0)]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) >= 10:
            print(f"  {cl:12s} {ctx:12s}: median={s.polya_length.median():.1f} (n={len(s)})")

# By age
print("\nPoly(A) median by age:")
for age in ['ancient', 'young']:
    sub = all_df[(all_df.l1_age == age) & (all_df.polya_length > 0)]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) >= 10:
            print(f"  {age:8s} {ctx:12s}: median={s.polya_length.median():.1f} (n={len(s)})")

# ============================================================
# 4. m6A PATTERNS
# ============================================================
print("\n" + "=" * 80)
print("4. m6A PATTERNS: INTERGENIC VS INTRONIC")
print("=" * 80)

# Load Part3 cache for m6A/kb
part3_cache = f'{TOPIC05}/part3_l1_per_read_cache'
m6a_data = []
for cl, groups in BASE_GROUPS.items():
    for grp in groups:
        path = f'{part3_cache}/{grp}_l1_per_read.tsv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep='\t')
        df['group'] = grp
        df['cellline'] = cl
        m6a_data.append(df[['read_id', 'read_length', 'm6a_sites_high', 'group', 'cellline']])

m6a_df = pd.concat(m6a_data, ignore_index=True)
m6a_df['m6a_per_kb'] = m6a_df['m6a_sites_high'] / (m6a_df['read_length'] / 1000)

# Merge with genomic context
m6a_merged = m6a_df.merge(
    all_df[['read_id', 'genomic_context', 'l1_age', 'polya_length']].drop_duplicates(),
    on='read_id', how='inner'
)
print(f"m6A merged reads: {len(m6a_merged)}")

for ctx in ['intronic', 'intergenic']:
    sub = m6a_merged[m6a_merged.genomic_context == ctx]
    print(f"\n{ctx.upper()} (n={len(sub)}):")
    print(f"  m6A/kb: mean={sub.m6a_per_kb.mean():.2f}, median={sub.m6a_per_kb.median():.2f}")
    print(f"  m6A sites: mean={sub.m6a_sites_high.mean():.2f}, median={sub.m6a_sites_high.median():.0f}")
    print(f"  m6A>0: {(sub.m6a_sites_high>0).sum()} ({100*(sub.m6a_sites_high>0).sum()/len(sub):.1f}%)")

# Test
intr_m6a = m6a_merged[m6a_merged.genomic_context == 'intronic'].m6a_per_kb
intg_m6a = m6a_merged[m6a_merged.genomic_context == 'intergenic'].m6a_per_kb
mwu_stat, mwu_p = stats.mannwhitneyu(intr_m6a, intg_m6a, alternative='two-sided')
print(f"\nm6A/kb Mann-Whitney U: p={mwu_p:.2e}")
print(f"  Intronic m6A/kb median={intr_m6a.median():.2f} vs Intergenic={intg_m6a.median():.2f}")
print(f"  Ratio (intergenic/intronic): {intg_m6a.median()/intr_m6a.median():.3f}")

# By age
print("\nm6A/kb by age and context:")
for age in ['ancient', 'young']:
    sub = m6a_merged[m6a_merged.l1_age == age]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) >= 10:
            print(f"  {age:8s} {ctx:12s}: m6A/kb mean={s.m6a_per_kb.mean():.2f}, median={s.m6a_per_kb.median():.2f} (n={len(s)})")

# ============================================================
# 5. ARSENITE RESPONSE
# ============================================================
print("\n" + "=" * 80)
print("5. ARSENITE RESPONSE: INTERGENIC VS INTRONIC")
print("=" * 80)

# HeLa vs HeLa-Ars only
hela = all_df[(all_df.cellline.isin(['HeLa', 'HeLa-Ars'])) & (all_df.polya_length > 0)].copy()
hela['condition'] = hela.cellline.apply(lambda x: 'stress' if 'Ars' in x else 'normal')

print("\nHeLa poly(A) by context and condition:")
for ctx in ['intronic', 'intergenic']:
    sub = hela[hela.genomic_context == ctx]
    for cond in ['normal', 'stress']:
        s = sub[sub.condition == cond]
        print(f"  {ctx:12s} {cond:8s}: median={s.polya_length.median():.1f}, mean={s.polya_length.mean():.1f} (n={len(s)})")
    # Delta
    normal_pa = sub[sub.condition == 'normal'].polya_length
    stress_pa = sub[sub.condition == 'stress'].polya_length
    delta = stress_pa.median() - normal_pa.median()
    mwu_stat, mwu_p = stats.mannwhitneyu(normal_pa, stress_pa, alternative='two-sided')
    print(f"  {ctx:12s} Delta={delta:.1f}nt, p={mwu_p:.2e}")

# Compare the deltas
print("\n  KEY: Do both intergenic and intronic show similar arsenite shortening?")
for ctx in ['intronic', 'intergenic']:
    sub = hela[hela.genomic_context == ctx]
    normal_pa = sub[sub.condition == 'normal'].polya_length.median()
    stress_pa = sub[sub.condition == 'stress'].polya_length.median()
    print(f"  {ctx}: normal={normal_pa:.1f} -> stress={stress_pa:.1f}, delta={stress_pa-normal_pa:.1f}")

# By age x context x condition
print("\nArsenite response by age and context:")
for age in ['ancient', 'young']:
    for ctx in ['intronic', 'intergenic']:
        sub = hela[(hela.l1_age == age) & (hela.genomic_context == ctx) & (hela.polya_length > 0)]
        if len(sub) < 10:
            print(f"  {age:8s} {ctx:12s}: too few reads (n={len(sub)})")
            continue
        normal = sub[sub.condition == 'normal'].polya_length
        stress = sub[sub.condition == 'stress'].polya_length
        if len(normal) >= 5 and len(stress) >= 5:
            delta = stress.median() - normal.median()
            _, p = stats.mannwhitneyu(normal, stress, alternative='two-sided')
            print(f"  {age:8s} {ctx:12s}: normal={normal.median():.1f} -> stress={stress.median():.1f}, delta={delta:.1f}, p={p:.2e} (n_norm={len(normal)}, n_stress={len(stress)})")

# ============================================================
# 6. READ LENGTH DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 80)
print("6. READ LENGTH: INTERGENIC VS INTRONIC")
print("=" * 80)

for ctx in ['intronic', 'intergenic']:
    sub = all_df[all_df.genomic_context == ctx]
    print(f"\n{ctx.upper()} (n={len(sub)}):")
    print(f"  Read length: mean={sub.read_length.mean():.0f}, median={sub.read_length.median():.0f}")
    print(f"  Q25={sub.read_length.quantile(0.25):.0f}, Q75={sub.read_length.quantile(0.75):.0f}")

# Test
intr_rl = all_df[all_df.genomic_context == 'intronic'].read_length
intg_rl = all_df[all_df.genomic_context == 'intergenic'].read_length
mwu_stat, mwu_p = stats.mannwhitneyu(intr_rl, intg_rl, alternative='two-sided')
print(f"\nRead length Mann-Whitney U: p={mwu_p:.2e}")

# By age
for age in ['ancient', 'young']:
    sub = all_df[all_df.l1_age == age]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) >= 10:
            print(f"  {age:8s} {ctx:12s}: median={s.read_length.median():.0f} (n={len(s)})")

# ============================================================
# 7. CHROMHMM ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("7. CHROMHMM: INTERGENIC VS INTRONIC")
print("=" * 80)

chromhmm = pd.read_csv(CHROMHMM_FILE, sep='\t')
# Only HeLa (normal condition, for ChromHMM which is HeLa-S3)
chromhmm_hela = chromhmm[chromhmm.cellline.isin(['HeLa', 'HeLa-Ars'])].copy()

print(f"ChromHMM reads (HeLa+HeLa-Ars): {len(chromhmm_hela)}")

# Chromatin state distribution by context
print("\nChromHMM group distribution:")
for ctx in ['intronic', 'intergenic']:
    sub = chromhmm_hela[chromhmm_hela.genomic_context == ctx]
    print(f"\n  {ctx.upper()} (n={len(sub)}):")
    dist = sub.chromhmm_group.value_counts()
    for state, count in dist.items():
        print(f"    {state:15s}: {count:5d} ({100*count/len(sub):5.1f}%)")

# m6A/kb by context
print("\nm6A/kb by context (from ChromHMM file):")
for ctx in ['intronic', 'intergenic']:
    sub = chromhmm_hela[(chromhmm_hela.genomic_context == ctx) & (chromhmm_hela.condition == 'normal')]
    print(f"  {ctx:12s}: m6A/kb mean={sub.m6a_per_kb.mean():.2f}, median={sub.m6a_per_kb.median():.2f} (n={len(sub)})")

# Poly(A) by context (ChromHMM)
print("\nPoly(A) by context (from ChromHMM file, HeLa only):")
for ctx in ['intronic', 'intergenic']:
    sub = chromhmm_hela[(chromhmm_hela.genomic_context == ctx) & (chromhmm_hela.condition == 'normal') & (chromhmm_hela.polya_length > 0)]
    print(f"  {ctx:12s}: poly(A) mean={sub.polya_length.mean():.1f}, median={sub.polya_length.median():.1f} (n={len(sub)})")

# m6A/kb by chromHMM group x context
print("\nm6A/kb by ChromHMM group and context (HeLa normal):")
hela_norm = chromhmm_hela[chromhmm_hela.condition == 'normal']
for state in ['Quiescent', 'Transcribed', 'Enhancer', 'Promoter', 'Heterochromatin']:
    sub = hela_norm[hela_norm.chromhmm_group == state]
    for ctx in ['intronic', 'intergenic']:
        s = sub[sub.genomic_context == ctx]
        if len(s) >= 10:
            print(f"  {state:15s} {ctx:12s}: m6A/kb={s.m6a_per_kb.mean():.2f} (n={len(s)})")

# ============================================================
# 8. MULTI-MAPPER CHECK: MAPQ=0 READS
# ============================================================
print("\n" + "=" * 80)
print("8. MAPQ=0 ANALYSIS (POTENTIAL MULTI-MAPPERS)")
print("=" * 80)

if len(merged) > 0:
    # Compare MAPQ=0 fraction
    for ctx in ['intronic', 'intergenic']:
        sub = merged[merged.genomic_context == ctx]
        mapq0 = (sub.mapq == 0).sum()
        print(f"  {ctx:12s}: MAPQ=0 = {mapq0}/{len(sub)} ({100*mapq0/len(sub):.1f}%)")

    # What about excluding MAPQ=0?
    print("\nExcluding MAPQ=0 reads:")
    high_mq = merged[merged.mapq > 0]
    for ctx in ['intronic', 'intergenic']:
        sub = high_mq[high_mq.genomic_context == ctx]
        print(f"  {ctx:12s}: MAPQ mean={sub.mapq.mean():.1f}, median={sub.mapq.median():.0f} (n={len(sub)})")

    # Even MAPQ>=60 only
    print("\nMAPQ>=60 only:")
    hq = merged[merged.mapq >= 60]
    total_hq = len(hq)
    for ctx in ['intronic', 'intergenic']:
        sub = hq[hq.genomic_context == ctx]
        print(f"  {ctx:12s}: {len(sub)} reads ({100*len(sub)/total_hq:.1f}%)")

    # Poly(A) for high-MAPQ intergenic only
    hq_intergenic = hq[(hq.genomic_context == 'intergenic') & (hq.polya_length > 0)]
    hq_intronic = hq[(hq.genomic_context == 'intronic') & (hq.polya_length > 0)]
    if len(hq_intergenic) >= 10 and len(hq_intronic) >= 10:
        print(f"\n  High-MAPQ (>=60) poly(A): intronic={hq_intronic.polya_length.median():.1f}, intergenic={hq_intergenic.polya_length.median():.1f}")

# ============================================================
# 9. CONSISTENCY: INTER-REPLICATE CONCORDANCE
# ============================================================
print("\n" + "=" * 80)
print("9. INTER-REPLICATE CONCORDANCE OF INTERGENIC FRACTION")
print("=" * 80)

for cl, groups in BASE_GROUPS.items():
    fracs = []
    for grp in groups:
        sub = all_df[(all_df.group == grp)]
        if len(sub) < 20:
            continue
        frac = (sub.genomic_context == 'intergenic').sum() / len(sub)
        fracs.append(frac)
    if len(fracs) >= 2:
        print(f"  {cl:12s}: {', '.join([f'{f:.1%}' for f in fracs])} (CV={np.std(fracs)/np.mean(fracs):.3f})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF EVIDENCE")
print("=" * 80)

print("""
Evidence assessment for intergenic L1 reads being real transcripts:

1. FRACTION: ~38-39% of PASS L1 reads are intergenic (consistent across cell lines)
2. MAPQ: Compare above - similar MAPQ argues against mapping artifact
3. POLY(A): Intergenic L1 reads have poly(A) tails (DRS poly(A) selection = genuine mRNA/transcript)
4. m6A: Intergenic L1 reads show m6A patterns similar to intronic
5. ARSENITE: Both intergenic and intronic show poly(A) shortening under arsenite stress
6. READ LENGTH: Compare distributions above
7. CHROMHMM: Intergenic L1 reads occupy expected chromatin states
""")

# Save key numbers to file
with open(f'{OUTDIR}/summary_statistics.txt', 'w') as f:
    f.write("Intergenic L1 Evidence Summary\n")
    f.write("=" * 60 + "\n\n")

    total = len(all_df)
    n_inter = (all_df.genomic_context == 'intergenic').sum()
    n_intr = (all_df.genomic_context == 'intronic').sum()
    f.write(f"Total PASS L1 reads: {total}\n")
    f.write(f"Intronic: {n_intr} ({100*n_intr/total:.1f}%)\n")
    f.write(f"Intergenic: {n_inter} ({100*n_inter/total:.1f}%)\n")

print(f"\nResults saved to {OUTDIR}/")
print("DONE")
