#!/usr/bin/env python3
"""
Compare ChromHMM state assignments at L1 loci: E066 (Liver) vs E117 (HeLa-S3).

The existing L1 ChromHMM analysis used E066 (Liver). The correct epigenome for
HeLa-S3 is E117. This script compares the two annotations at the same L1 loci
to quantify how much the state assignments differ.

Steps:
1. Load the existing L1 annotated file (E066 states already assigned).
2. Compute midpoint for each read.
3. Create a BED file of midpoints, intersect with both E066 and E117.
4. Build confusion matrix and report agreement statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
OUT_DIR = BASE / 'analysis/01_exploration/topic_08_regulatory_chromatin'

ANNOTATED_FILE = OUT_DIR / 'l1_chromhmm_annotated.tsv'
E066_BED = OUT_DIR / 'E066_15_coreMarks_hg38lift_mnemonics.bed.gz'
E117_BED = OUT_DIR / 'E117_15_coreMarks_hg38lift_mnemonics.bed.gz'

# Same broad grouping as the original analysis
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
# 1. Load annotated L1 reads
###############################################################################
print("=== Loading L1 ChromHMM annotated file ===")
df = pd.read_csv(ANNOTATED_FILE, sep='\t')
print(f"  Total annotated reads: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# Compute midpoint
df['midpoint'] = ((df['start'] + df['end']) // 2).astype(int)

# Create a unique index for merging back
df['idx'] = range(len(df))

###############################################################################
# 2. Create midpoint BED and intersect with both epigenomes
###############################################################################
def intersect_with_chromhmm(df, chromhmm_bed, label):
    """Intersect midpoints with a ChromHMM BED file using bedtools."""
    print(f"\n=== Intersecting with {label} ({chromhmm_bed.name}) ===")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f_bed:
        bed_path = f_bed.name
        for _, row in df.iterrows():
            # BED: chr, start, end, name(idx)
            f_bed.write(f"{row['chr']}\t{row['midpoint']}\t{row['midpoint']+1}\t{row['idx']}\n")

    # bedtools intersect (unsorted -- BED files have different sort orders)
    result_path = bed_path + '.result'
    cmd = (
        f"bedtools intersect -a {bed_path} -b {chromhmm_bed} -wa -wb "
        f"> {result_path}"
    )
    subprocess.run(cmd, shell=True, check=True)

    # Parse results
    results = pd.read_csv(
        result_path, sep='\t', header=None,
        names=['chr_q', 'start_q', 'end_q', 'idx',
               'chr_s', 'start_s', 'end_s', 'state']
    )

    # Deduplicate: if a midpoint falls in multiple overlapping segments, take first
    results = results.drop_duplicates(subset='idx', keep='first')
    results['idx'] = results['idx'].astype(int)

    print(f"  Matched: {len(results):,} / {len(df):,} reads")

    # Clean up temp files
    import os
    for p in [bed_path, result_path]:
        if os.path.exists(p):
            os.remove(p)

    return results[['idx', 'state']].rename(columns={'state': f'state_{label}'})

# Run intersections
e066_results = intersect_with_chromhmm(df, E066_BED, 'E066')
e117_results = intersect_with_chromhmm(df, E117_BED, 'E117')

###############################################################################
# 3. Merge and compare
###############################################################################
print("\n=== Merging results ===")
merged = df.merge(e066_results, on='idx', how='inner')
merged = merged.merge(e117_results, on='idx', how='inner')
print(f"  Reads with both annotations: {len(merged):,}")

# Add broad groups
merged['group_E066'] = merged['state_E066'].map(STATE_GROUPS)
merged['group_E117'] = merged['state_E117'].map(STATE_GROUPS)

# Check for unmapped states
for col in ['group_E066', 'group_E117']:
    unmapped = merged[col].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} reads with unmapped {col}")
        print(f"    States: {merged.loc[merged[col].isna(), col.replace('group_','state_')].unique()}")

merged = merged.dropna(subset=['group_E066', 'group_E117'])
print(f"  After removing unmapped: {len(merged):,}")

###############################################################################
# 4. Overall agreement
###############################################################################
print("\n" + "="*80)
print("OVERALL AGREEMENT")
print("="*80)

# Fine-grained (15-state)
agree_fine = (merged['state_E066'] == merged['state_E117']).sum()
print(f"\n  15-state agreement: {agree_fine:,} / {len(merged):,} = {agree_fine/len(merged)*100:.1f}%")

# Broad (7-group)
agree_broad = (merged['group_E066'] == merged['group_E117']).sum()
print(f"  Broad group agreement: {agree_broad:,} / {len(merged):,} = {agree_broad/len(merged)*100:.1f}%")

###############################################################################
# 5. 15-state confusion matrix
###############################################################################
print("\n" + "="*80)
print("15-STATE CONFUSION MATRIX (E066 rows, E117 columns)")
print("="*80)

state_order = [
    '1_TssA', '2_TssAFlnk', '3_TxFlnk', '4_Tx', '5_TxWk',
    '6_EnhG', '7_Enh', '8_ZNF/Rpts', '9_Het',
    '10_TssBiv', '11_BivFlnk', '12_EnhBiv',
    '13_ReprPC', '14_ReprPCWk', '15_Quies'
]

# Filter to states that actually appear
states_present = sorted(
    set(merged['state_E066'].unique()) | set(merged['state_E117'].unique()),
    key=lambda x: state_order.index(x) if x in state_order else 99
)

conf = pd.crosstab(merged['state_E066'], merged['state_E117'], margins=True)
# Reorder
conf = conf.reindex(index=[s for s in states_present if s in conf.index] + ['All'],
                     columns=[s for s in states_present if s in conf.columns] + ['All'])
print(conf.to_string())

###############################################################################
# 6. Broad group confusion matrix
###############################################################################
print("\n" + "="*80)
print("BROAD GROUP CONFUSION MATRIX (E066 rows, E117 columns)")
print("="*80)

group_order = ['Promoter', 'Transcribed', 'Enhancer', 'ZNF/Repeats',
               'Heterochromatin', 'Bivalent', 'Repressed', 'Quiescent']
groups_present = [g for g in group_order if g in merged['group_E066'].unique()
                  or g in merged['group_E117'].unique()]

conf_broad = pd.crosstab(merged['group_E066'], merged['group_E117'], margins=True)
conf_broad = conf_broad.reindex(
    index=[g for g in groups_present if g in conf_broad.index] + ['All'],
    columns=[g for g in groups_present if g in conf_broad.columns] + ['All']
)
print(conf_broad.to_string())

# Print percentages (row-normalized: "of E066 state X, what % became Y in E117?")
print("\n--- Row-normalized (% of E066 group going to each E117 group) ---")
conf_broad_no_margins = pd.crosstab(merged['group_E066'], merged['group_E117'])
conf_pct = conf_broad_no_margins.div(conf_broad_no_margins.sum(axis=1), axis=0) * 100
conf_pct = conf_pct.reindex(
    index=[g for g in groups_present if g in conf_pct.index],
    columns=[g for g in groups_present if g in conf_pct.columns]
).fillna(0)
print(conf_pct.round(1).to_string())

###############################################################################
# 7. Key questions
###############################################################################
print("\n" + "="*80)
print("KEY COMPARISONS")
print("="*80)

# 7a. Regulatory (Enhancer + Promoter) stability
print("\n--- Regulatory L1 (Enhancer + Promoter in E066): where do they go in E117? ---")
regulatory_e066 = merged[merged['group_E066'].isin(['Enhancer', 'Promoter'])]
print(f"  Total Regulatory in E066: {len(regulatory_e066):,}")
reg_dest = regulatory_e066['group_E117'].value_counts()
for g, n in reg_dest.items():
    print(f"    → {g} in E117: {n:,} ({n/len(regulatory_e066)*100:.1f}%)")

stay_reg = regulatory_e066['group_E117'].isin(['Enhancer', 'Promoter']).sum()
print(f"  Stay Regulatory in E117: {stay_reg:,} / {len(regulatory_e066):,} = {stay_reg/len(regulatory_e066)*100:.1f}%")

# 7b. Heterochromatin stability
print("\n--- Heterochromatin in E066: where do they go in E117? ---")
het_e066 = merged[merged['group_E066'] == 'Heterochromatin']
print(f"  Total Het in E066: {len(het_e066):,}")
het_dest = het_e066['group_E117'].value_counts()
for g, n in het_dest.items():
    print(f"    → {g} in E117: {n:,} ({n/len(het_e066)*100:.1f}%)")
stay_het = (het_e066['group_E117'] == 'Heterochromatin').sum()
print(f"  Stay Het in E117: {stay_het:,} / {len(het_e066):,} = {stay_het/len(het_e066)*100:.1f}%")

# 7c. Quiescent stability
print("\n--- Quiescent in E066: where do they go in E117? ---")
quies_e066 = merged[merged['group_E066'] == 'Quiescent']
print(f"  Total Quiescent in E066: {len(quies_e066):,}")
quies_dest = quies_e066['group_E117'].value_counts()
for g, n in quies_dest.items():
    print(f"    → {g} in E117: {n:,} ({n/len(quies_e066)*100:.1f}%)")

# 7d. NEW Regulatory in E117 that were NOT Regulatory in E066
print("\n--- New Regulatory in E117 (not Regulatory in E066) ---")
reg_e117 = merged[merged['group_E117'].isin(['Enhancer', 'Promoter'])]
not_reg_e066 = reg_e117[~reg_e117['group_E066'].isin(['Enhancer', 'Promoter'])]
print(f"  Total Regulatory in E117: {len(reg_e117):,}")
print(f"  Of those, NOT Regulatory in E066: {len(not_reg_e066):,}")
print(f"  Their E066 origin:")
orig = not_reg_e066['group_E066'].value_counts()
for g, n in orig.items():
    print(f"    ← {g}: {n:,}")

###############################################################################
# 8. Impact on key findings: arsenite shortening by chromatin state
###############################################################################
print("\n" + "="*80)
print("IMPACT ON ARSENITE POLY(A) SHORTENING BY CHROMATIN STATE")
print("="*80)

from scipy import stats as sp_stats

# Filter to HeLa/HeLa-Ars only
hela_mask = merged['cellline'].isin(['HeLa', 'HeLa-Ars'])
hela = merged[hela_mask].copy()
hela['is_ars'] = hela['cellline'] == 'HeLa-Ars'
print(f"\n  HeLa + HeLa-Ars reads: {len(hela):,}")

# Classify age
YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
def get_subfamily(gene_id):
    return gene_id.split('_dup')[0] if '_dup' in gene_id else gene_id

hela['l1_age'] = hela['gene_id'].apply(lambda x: 'young' if get_subfamily(x) in YOUNG else 'ancient')
hela_ancient = hela[hela['l1_age'] == 'ancient']

print(f"  Ancient HeLa reads: {len(hela_ancient):,}")

for epigenome, state_col in [('E066', 'group_E066'), ('E117', 'group_E117')]:
    print(f"\n  --- Using {epigenome} states ---")

    for group_name in groups_present:
        sub = hela_ancient[hela_ancient[state_col] == group_name]
        if len(sub) < 20:
            continue
        normal = sub[~sub['is_ars']]['polya_length']
        ars = sub[sub['is_ars']]['polya_length']
        if len(normal) < 5 or len(ars) < 5:
            continue
        delta = ars.median() - normal.median()
        try:
            u_stat, p_val = sp_stats.mannwhitneyu(ars, normal, alternative='two-sided')
        except:
            p_val = 1.0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"    {group_name:15s}: N={len(normal):4d} Ars={len(ars):4d}  "
              f"med_N={normal.median():.1f}  med_A={ars.median():.1f}  "
              f"Δ={delta:+.1f}  p={p_val:.2e} {sig}")

###############################################################################
# 9. Distribution comparison: E066 vs E117 at L1 loci
###############################################################################
print("\n" + "="*80)
print("STATE DISTRIBUTION COMPARISON AT L1 LOCI")
print("="*80)

for epigenome, col in [('E066', 'group_E066'), ('E117', 'group_E117')]:
    print(f"\n  {epigenome} broad group distribution:")
    dist = merged[col].value_counts()
    total = len(merged)
    for g in groups_present:
        if g in dist.index:
            print(f"    {g:15s}: {dist[g]:6,} ({dist[g]/total*100:.1f}%)")

# Age-stratified
for age_label in ['ancient', 'young']:
    age_sub = merged[merged['l1_age'] == age_label]
    print(f"\n  --- {age_label.upper()} L1 ({len(age_sub):,} reads) ---")
    for epigenome, col in [('E066', 'group_E066'), ('E117', 'group_E117')]:
        print(f"    {epigenome}:")
        dist = age_sub[col].value_counts()
        total = len(age_sub)
        for g in groups_present:
            if g in dist.index:
                print(f"      {g:15s}: {dist[g]:5,} ({dist[g]/total*100:.1f}%)")

###############################################################################
# 10. Save detailed output
###############################################################################
out_file = OUT_DIR / 'e066_vs_e117_comparison.tsv'
merged[['read_id', 'chr', 'start', 'end', 'gene_id', 'l1_age', 'polya_length',
        'm6a_per_kb', 'cellline', 'condition',
        'state_E066', 'group_E066', 'state_E117', 'group_E117']].to_csv(
    out_file, sep='\t', index=False)
print(f"\n  Saved: {out_file}")

print("\n=== DONE ===")
