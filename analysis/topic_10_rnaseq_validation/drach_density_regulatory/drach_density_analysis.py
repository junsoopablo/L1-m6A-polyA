#!/usr/bin/env python3
"""
DRACH motif density analysis: Regulatory vs Non-regulatory ancient L1.
Decompose m6A/kb difference into DRACH density vs per-DRACH methylation rate.
"""

import pandas as pd
import numpy as np
import pysam
import re
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
CHROMHMM_FILE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_08_regulatory_chromatin/l1_chromhmm_annotated.tsv'
HISTONE_FILE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/histone_mark_analysis/l1_histone_annotated.tsv'
REF_GENOME = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta'
OUTDIR = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_10_rnaseq_validation/drach_density_regulatory/'

# DRACH motif: D=[AGT], R=[AG], A, C, H=[ACT]
DRACH_PATTERN = re.compile(r'[AGT][AG]AC[ACT]', re.IGNORECASE)
# PAS motif
PAS_PATTERN = re.compile(r'AATAAA', re.IGNORECASE)

# ── 1. Load and filter data ──
print("Loading ChromHMM data...")
df_chrom = pd.read_csv(CHROMHMM_FILE, sep='\t')
print(f"  Total reads: {len(df_chrom):,}")

# Filter: HeLa, normal condition, ancient L1
mask = (
    (df_chrom['cellline'] == 'HeLa') &
    (df_chrom['condition'] == 'normal') &
    (df_chrom['l1_age'] == 'ancient')
)
df = df_chrom[mask].copy()
print(f"  HeLa normal ancient L1: {len(df):,}")
print(f"  ChromHMM groups: {df['chromhmm_group'].value_counts().to_dict()}")

# Define regulatory vs non-regulatory
df['is_regulatory'] = df['chromhmm_group'].isin(['Enhancer', 'Promoter'])
df['reg_label'] = df['is_regulatory'].map({True: 'Regulatory', False: 'Non-regulatory'})

# Filter to Regulatory (Enhancer+Promoter) vs Non-regulatory (Transcribed+Quiescent)
df = df[df['chromhmm_group'].isin(['Enhancer', 'Promoter', 'Transcribed', 'Quiescent'])].copy()
print(f"  After filtering to Enh/Prom/Tx/Quiescent: {len(df):,}")
print(f"  Regulatory: {df['is_regulatory'].sum():,}, Non-regulatory: {(~df['is_regulatory']).sum():,}")

# ── 2. Extract genomic sequences and count DRACH motifs ──
print("\nExtracting genomic sequences and counting DRACH motifs...")
fasta = pysam.FastaFile(REF_GENOME)

drach_counts = []
pas_counts = []
seq_lengths = []

for _, row in df.iterrows():
    chrom = row['chr']
    start = int(row['start'])
    end = int(row['end'])
    try:
        seq = fasta.fetch(chrom, start, end)
        seq_len = len(seq)
        drach_n = len(DRACH_PATTERN.findall(seq))
        pas_n = len(PAS_PATTERN.findall(seq))
    except Exception:
        seq_len = end - start
        drach_n = np.nan
        pas_n = np.nan

    drach_counts.append(drach_n)
    pas_counts.append(pas_n)
    seq_lengths.append(seq_len)

df['drach_count'] = drach_counts
df['pas_count'] = pas_counts
df['seq_length'] = seq_lengths
df['drach_per_kb'] = df['drach_count'] / df['seq_length'] * 1000
df['pas_per_kb'] = df['pas_count'] / df['seq_length'] * 1000

# Per-DRACH methylation rate
df['per_drach_rate'] = np.where(df['drach_count'] > 0,
                                 df['m6a_sites_high'] / df['drach_count'],
                                 np.nan)

fasta.close()

# Drop any NaN rows
df = df.dropna(subset=['drach_count']).copy()
print(f"  Reads with valid sequences: {len(df):,}")

# ── 3. Main comparison: Regulatory vs Non-regulatory ──
print("\n" + "="*70)
print("DRACH DENSITY: Regulatory vs Non-regulatory Ancient L1 (HeLa normal)")
print("="*70)

reg = df[df['is_regulatory']]
nonreg = df[~df['is_regulatory']]

def compare_metric(reg_vals, nonreg_vals, metric_name):
    reg_med = np.nanmedian(reg_vals)
    nonreg_med = np.nanmedian(nonreg_vals)
    reg_mean = np.nanmean(reg_vals)
    nonreg_mean = np.nanmean(nonreg_vals)
    ratio = reg_med / nonreg_med if nonreg_med > 0 else np.inf
    stat, pval = stats.mannwhitneyu(
        reg_vals.dropna(), nonreg_vals.dropna(), alternative='two-sided'
    )
    print(f"\n{metric_name}:")
    print(f"  Regulatory    (n={len(reg_vals.dropna()):,}): median={reg_med:.3f}, mean={reg_mean:.3f}")
    print(f"  Non-regulatory(n={len(nonreg_vals.dropna()):,}): median={nonreg_med:.3f}, mean={nonreg_mean:.3f}")
    print(f"  Ratio (Reg/NonReg): {ratio:.3f}")
    print(f"  MWU P-value: {pval:.2e}")
    return reg_med, nonreg_med, ratio, pval

# 3a. DRACH/kb
r1 = compare_metric(reg['drach_per_kb'], nonreg['drach_per_kb'], 'DRACH motifs per kb')

# 3b. m6A/kb (verification)
r2 = compare_metric(reg['m6a_per_kb'], nonreg['m6a_per_kb'], 'm6A sites per kb')

# 3c. Per-DRACH methylation rate
r3 = compare_metric(reg['per_drach_rate'], nonreg['per_drach_rate'], 'Per-DRACH methylation rate')

# 3d. Sequence length
r4 = compare_metric(reg['seq_length'], nonreg['seq_length'], 'Sequence length (bp)')

# 3e. PAS/kb
r5 = compare_metric(reg['pas_per_kb'], nonreg['pas_per_kb'], 'PAS (AATAAA) motifs per kb')

# ── 4. Decomposition ──
print("\n" + "="*70)
print("DECOMPOSITION: m6A/kb = DRACH/kb × per-DRACH rate")
print("="*70)

reg_drach_kb = np.nanmean(reg['drach_per_kb'])
nonreg_drach_kb = np.nanmean(nonreg['drach_per_kb'])
reg_rate = np.nanmean(reg['per_drach_rate'].dropna())
nonreg_rate = np.nanmean(nonreg['per_drach_rate'].dropna())

drach_ratio = reg_drach_kb / nonreg_drach_kb
rate_ratio = reg_rate / nonreg_rate
product = drach_ratio * rate_ratio

print(f"\nReg DRACH/kb:     {reg_drach_kb:.3f}")
print(f"NonReg DRACH/kb:  {nonreg_drach_kb:.3f}")
print(f"DRACH ratio:      {drach_ratio:.3f}")
print(f"\nReg per-DRACH rate:     {reg_rate:.4f}")
print(f"NonReg per-DRACH rate:  {nonreg_rate:.4f}")
print(f"Rate ratio:             {rate_ratio:.3f}")
print(f"\nProduct (DRACH ratio × Rate ratio): {product:.3f}")
print(f"Observed m6A/kb ratio:              {np.nanmean(reg['m6a_per_kb']) / np.nanmean(nonreg['m6a_per_kb']):.3f}")

# Percent contribution
total_log = np.log(drach_ratio) + np.log(rate_ratio)
if abs(total_log) > 0:
    drach_contrib = np.log(drach_ratio) / total_log * 100
    rate_contrib = np.log(rate_ratio) / total_log * 100
else:
    drach_contrib = rate_contrib = 50.0
print(f"\nContribution to m6A/kb difference:")
print(f"  DRACH density:         {drach_contrib:.1f}%")
print(f"  Per-DRACH rate:        {rate_contrib:.1f}%")

# ── 5. Subgroup analysis by chromhmm_group ──
print("\n" + "="*70)
print("SUBGROUP ANALYSIS by ChromHMM group")
print("="*70)

for grp in ['Enhancer', 'Promoter', 'Transcribed', 'Quiescent']:
    sub = df[df['chromhmm_group'] == grp]
    if len(sub) == 0:
        continue
    print(f"\n{grp} (n={len(sub):,}):")
    print(f"  DRACH/kb: median={np.nanmedian(sub['drach_per_kb']):.3f}, mean={np.nanmean(sub['drach_per_kb']):.3f}")
    print(f"  m6A/kb:   median={np.nanmedian(sub['m6a_per_kb']):.3f}, mean={np.nanmean(sub['m6a_per_kb']):.3f}")
    rate_sub = sub['per_drach_rate'].dropna()
    print(f"  Per-DRACH rate: median={np.nanmedian(rate_sub):.4f}, mean={np.nanmean(rate_sub):.4f} (n={len(rate_sub):,})")
    print(f"  PAS/kb:   median={np.nanmedian(sub['pas_per_kb']):.3f}")
    print(f"  Length:   median={np.nanmedian(sub['seq_length']):.0f}")

# ── 6. Histone mark analysis ──
print("\n" + "="*70)
print("HISTONE MARK ANALYSIS: H3K27ac+ vs No-mark")
print("="*70)

df_hist = pd.read_csv(HISTONE_FILE, sep='\t')
# Filter: HeLa, normal, ancient
mask_h = (
    (df_hist['cellline'] == 'HeLa') &
    (df_hist['condition'] == 'normal') &
    (df_hist['l1_age'] == 'ancient')
)
df_h = df_hist[mask_h].copy()
print(f"HeLa normal ancient (histone): {len(df_h):,}")

if 'H3K27ac_overlap' in df_h.columns:
    # Extract sequences for histone-annotated reads
    fasta2 = pysam.FastaFile(REF_GENOME)
    drach_h = []
    for _, row in df_h.iterrows():
        try:
            seq = fasta2.fetch(row['chr'], int(row['start']), int(row['end']))
            drach_h.append(len(DRACH_PATTERN.findall(seq)))
        except:
            drach_h.append(np.nan)
    df_h['drach_count'] = drach_h
    df_h['seq_length'] = df_h['end'] - df_h['start']
    df_h['drach_per_kb'] = df_h['drach_count'] / df_h['seq_length'] * 1000
    df_h['per_drach_rate'] = np.where(df_h['drach_count'] > 0,
                                       df_h['m6a_sites_high'] / df_h['drach_count'],
                                       np.nan)
    fasta2.close()

    # H3K27ac+ vs No marks
    h3k27ac_pos = df_h[df_h['H3K27ac_overlap'] == 1]
    no_marks = df_h[df_h['histone_category'] == 'No_marks']
    print(f"  H3K27ac+ : {len(h3k27ac_pos):,}")
    print(f"  No_marks : {len(no_marks):,}")

    if len(h3k27ac_pos) > 10 and len(no_marks) > 10:
        compare_metric(h3k27ac_pos['drach_per_kb'], no_marks['drach_per_kb'],
                       'DRACH/kb: H3K27ac+ vs No-marks')
        compare_metric(h3k27ac_pos['m6a_per_kb'], no_marks['m6a_per_kb'],
                       'm6A/kb: H3K27ac+ vs No-marks')
        compare_metric(h3k27ac_pos['per_drach_rate'], no_marks['per_drach_rate'],
                       'Per-DRACH rate: H3K27ac+ vs No-marks')
    else:
        print("  Insufficient H3K27ac+ reads for comparison")
else:
    print("  H3K27ac column not found")

# ── 7. Length-matched analysis ──
print("\n" + "="*70)
print("LENGTH-MATCHED ANALYSIS (500-2000bp)")
print("="*70)

df_lm = df[(df['seq_length'] >= 500) & (df['seq_length'] <= 2000)].copy()
reg_lm = df_lm[df_lm['is_regulatory']]
nonreg_lm = df_lm[~df_lm['is_regulatory']]
print(f"  Length-matched: Reg={len(reg_lm):,}, NonReg={len(nonreg_lm):,}")

if len(reg_lm) > 10 and len(nonreg_lm) > 10:
    compare_metric(reg_lm['drach_per_kb'], nonreg_lm['drach_per_kb'],
                   'DRACH/kb (length-matched 500-2000bp)')
    compare_metric(reg_lm['per_drach_rate'], nonreg_lm['per_drach_rate'],
                   'Per-DRACH rate (length-matched 500-2000bp)')
    compare_metric(reg_lm['m6a_per_kb'], nonreg_lm['m6a_per_kb'],
                   'm6A/kb (length-matched 500-2000bp)')

# ── 8. Save results ──
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save per-read data
cols_out = ['read_id', 'chr', 'start', 'end', 'gene_id', 'l1_age',
            'chromhmm_group', 'reg_label', 'seq_length',
            'drach_count', 'drach_per_kb', 'pas_count', 'pas_per_kb',
            'm6a_per_kb', 'm6a_sites_high', 'per_drach_rate', 'polya_length']
outfile = os.path.join(OUTDIR, 'drach_density_per_read.tsv')
df[cols_out].to_csv(outfile, sep='\t', index=False)
print(f"  Saved: {outfile}")

# Summary table
summary_rows = []
for label, sub in [('Regulatory', reg), ('Non-regulatory', nonreg),
                    ('Enhancer', df[df['chromhmm_group']=='Enhancer']),
                    ('Promoter', df[df['chromhmm_group']=='Promoter']),
                    ('Transcribed', df[df['chromhmm_group']=='Transcribed']),
                    ('Quiescent', df[df['chromhmm_group']=='Quiescent'])]:
    rate_vals = sub['per_drach_rate'].dropna()
    summary_rows.append({
        'group': label,
        'n_reads': len(sub),
        'median_length': np.nanmedian(sub['seq_length']),
        'median_drach_per_kb': np.nanmedian(sub['drach_per_kb']),
        'mean_drach_per_kb': np.nanmean(sub['drach_per_kb']),
        'median_m6a_per_kb': np.nanmedian(sub['m6a_per_kb']),
        'mean_m6a_per_kb': np.nanmean(sub['m6a_per_kb']),
        'median_per_drach_rate': np.nanmedian(rate_vals),
        'mean_per_drach_rate': np.nanmean(rate_vals),
        'median_pas_per_kb': np.nanmedian(sub['pas_per_kb']),
    })

summary_df = pd.DataFrame(summary_rows)
sumfile = os.path.join(OUTDIR, 'drach_density_summary.tsv')
summary_df.to_csv(sumfile, sep='\t', index=False, float_format='%.4f')
print(f"  Saved: {sumfile}")

# ── 9. Final verdict ──
print("\n" + "="*70)
print("VERDICT")
print("="*70)

_, _, drach_r, drach_p = r1  # DRACH/kb ratio and p
_, _, rate_r, rate_p = r3     # per-DRACH rate ratio and p

if drach_p < 0.05 and rate_p >= 0.05:
    print("→ DRACH density differs significantly, per-DRACH rate does NOT.")
    print("→ Lower m6A at regulatory L1 is SEQUENCE-INTRINSIC (fewer DRACH motifs).")
elif drach_p >= 0.05 and rate_p < 0.05:
    print("→ DRACH density does NOT differ, but per-DRACH rate differs significantly.")
    print("→ Lower m6A at regulatory L1 is due to CHROMATIN ENVIRONMENT (lower per-motif rate).")
elif drach_p < 0.05 and rate_p < 0.05:
    print("→ BOTH DRACH density AND per-DRACH rate differ significantly.")
    print(f"→ Contribution: DRACH density {drach_contrib:.0f}%, per-DRACH rate {rate_contrib:.0f}%.")
    if abs(drach_contrib) > abs(rate_contrib):
        print("→ Primary driver: SEQUENCE-INTRINSIC (DRACH density).")
    else:
        print("→ Primary driver: CHROMATIN ENVIRONMENT (per-DRACH rate).")
else:
    print("→ Neither factor differs significantly despite m6A/kb difference.")
    print("→ Effect may be driven by read length or other factors.")

print("\nDone.")
