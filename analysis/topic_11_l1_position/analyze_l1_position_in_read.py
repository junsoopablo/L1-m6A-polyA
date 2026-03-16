#!/usr/bin/env python3
"""
Analyze where L1 elements sit within ONT DRS reads.
Key question: Is L1 near the 3' end (poly(A) tail) of reads?

ONT DRS reads 3'→5', so 3' end is the most reliable.
If L1 is at the 3' end → L1's own PAS is being used for polyadenylation.
"""
import pandas as pd
import numpy as np
import glob, os

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)

# ── Load all L1 summary data ──
frames = []
for f in sorted(glob.glob(f'{BASE}/results_group/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'chr', 'start', 'end', 'read_length', 'overlap_length',
        'te_start', 'te_end', 'transcript_id', 'gene_id', 'te_strand',
        'read_strand', 'dist_to_3prime', 'polya_length', 'qc_tag',
        'class', 'TE_group'
    ])
    group = os.path.basename(f).replace('_L1_summary.tsv', '')
    tmp['group'] = group
    tmp['cell_line'] = group.rsplit('_', 1)[0]
    frames.append(tmp)

df = pd.concat(frames, ignore_index=True)
df = df[df['qc_tag'] == 'PASS'].copy()
df['is_young'] = df['gene_id'].str.match(r'^L1HS$|^L1PA[1-3]$')
df['te_length'] = (df['te_end'] - df['te_start']).abs()
df['overlap_frac'] = df['overlap_length'] / df['read_length']

# Separate intergenic vs intronic
df_inter = df[df['TE_group'] == 'intergenic'].copy()
df_intra = df[df['TE_group'] == 'intronic'].copy()

print("=" * 70)
print("L1 POSITION WITHIN ONT DRS READS")
print("=" * 70)

# ── 1. dist_to_3prime distribution ──
print("\n[1] dist_to_3prime: L1 element distance from read's 3' end")
print(f"    ALL PASS reads (n={len(df):,}):")
print(f"      median = {df['dist_to_3prime'].median():.0f} bp")
print(f"      mean   = {df['dist_to_3prime'].mean():.0f} bp")
print(f"      Q25    = {df['dist_to_3prime'].quantile(0.25):.0f} bp")
print(f"      Q75    = {df['dist_to_3prime'].quantile(0.75):.0f} bp")
print(f"      <50bp  = {(df['dist_to_3prime'] < 50).mean()*100:.1f}%")
print(f"      <100bp = {(df['dist_to_3prime'] < 100).mean()*100:.1f}%")
print(f"      <200bp = {(df['dist_to_3prime'] < 200).mean()*100:.1f}%")

for label, sub in [("Intergenic", df_inter), ("Intronic", df_intra)]:
    print(f"\n    {label} (n={len(sub):,}):")
    print(f"      median = {sub['dist_to_3prime'].median():.0f} bp")
    print(f"      <50bp  = {(sub['dist_to_3prime'] < 50).mean()*100:.1f}%")
    print(f"      <100bp = {(sub['dist_to_3prime'] < 100).mean()*100:.1f}%")
    print(f"      <200bp = {(sub['dist_to_3prime'] < 200).mean()*100:.1f}%")

# ── 2. Young vs Ancient ──
print("\n[2] Young vs Ancient L1 dist_to_3prime")
for is_y, label in [(True, "Young"), (False, "Ancient")]:
    sub = df[df['is_young'] == is_y]
    print(f"    {label} (n={len(sub):,}):")
    print(f"      median dist_to_3prime = {sub['dist_to_3prime'].median():.0f} bp")
    print(f"      median read_length = {sub['read_length'].median():.0f} bp")
    print(f"      median te_length = {sub['te_length'].median():.0f} bp")
    print(f"      median overlap_frac = {sub['overlap_frac'].median():.3f}")
    print(f"      <100bp from 3' = {(sub['dist_to_3prime'] < 100).mean()*100:.1f}%")

# ── 3. Where L1 sits relative to read: fractional position ──
# Compute: L1's center position as fraction of read length (from 3' end)
# 0 = at 3' end, 1 = at 5' end
# For + strand reads: 3' end = highest genomic coordinate
# For - strand reads: 3' end = lowest genomic coordinate
print("\n[3] L1 fractional position within read (0=3' end, 1=5' end)")

# L1 center in genomic coords
df['l1_center'] = (df['te_start'] + df['te_end']) / 2
# Read extent
df['read_min'] = df[['start', 'end']].min(axis=1)
df['read_max'] = df[['start', 'end']].max(axis=1)
df['read_span'] = df['read_max'] - df['read_min']

# For + strand: 3' end = read_max; frac = (read_max - l1_center) / read_span
# For - strand: 3' end = read_min; frac = (l1_center - read_min) / read_span
df['l1_frac_from_3prime'] = np.where(
    df['read_strand'] == '+',
    (df['read_max'] - df['l1_center']) / df['read_span'].clip(lower=1),
    (df['l1_center'] - df['read_min']) / df['read_span'].clip(lower=1)
)
df['l1_frac_from_3prime'] = df['l1_frac_from_3prime'].clip(0, 1)

for label, sub in [("ALL", df), ("Intergenic", df_inter), ("Intronic", df_intra)]:
    # Recalculate for subsets
    sub = sub.copy()
    sub['l1_center'] = (sub['te_start'] + sub['te_end']) / 2
    sub['read_min'] = sub[['start', 'end']].min(axis=1)
    sub['read_max'] = sub[['start', 'end']].max(axis=1)
    sub['read_span'] = sub['read_max'] - sub['read_min']
    sub['l1_frac_from_3prime'] = np.where(
        sub['read_strand'] == '+',
        (sub['read_max'] - sub['l1_center']) / sub['read_span'].clip(lower=1),
        (sub['l1_center'] - sub['read_min']) / sub['read_span'].clip(lower=1)
    )
    sub['l1_frac_from_3prime'] = sub['l1_frac_from_3prime'].clip(0, 1)

    print(f"\n    {label} (n={len(sub):,}):")
    print(f"      median frac from 3' = {sub['l1_frac_from_3prime'].median():.3f}")
    print(f"      in 3' quarter (0-0.25) = {(sub['l1_frac_from_3prime'] < 0.25).mean()*100:.1f}%")
    print(f"      in 3' half (0-0.50)    = {(sub['l1_frac_from_3prime'] < 0.50).mean()*100:.1f}%")
    print(f"      in 5' quarter (0.75-1) = {(sub['l1_frac_from_3prime'] > 0.75).mean()*100:.1f}%")

# ── 4. Does L1 extend to read's 3' end? (PAS usage indicator) ──
print("\n[4] Does L1 extend to the 3' boundary of the read?")
print("    (dist_to_3prime ≤ 50bp → L1 PAS likely provides the poly(A) signal)")

for label, sub in [("ALL", df), ("Intergenic", df_inter), ("Intronic", df_intra)]:
    n_at_3end = (sub['dist_to_3prime'] <= 50).sum()
    frac = n_at_3end / len(sub) * 100
    print(f"    {label}: {n_at_3end:,}/{len(sub):,} = {frac:.1f}% L1 at read 3' end")

# ── 5. Overlap fraction vs dist_to_3prime ──
print("\n[5] Overlap fraction by dist_to_3prime bin")
bins = [(0, 50), (50, 200), (200, 500), (500, 2000)]
for lo, hi in bins:
    sub = df[(df['dist_to_3prime'] >= lo) & (df['dist_to_3prime'] < hi)]
    if len(sub) > 10:
        print(f"    dist {lo}-{hi}bp (n={len(sub):,}): "
              f"overlap_frac={sub['overlap_frac'].median():.3f}, "
              f"read_len={sub['read_length'].median():.0f}, "
              f"polya={sub['polya_length'].median():.1f}nt")

# ── 6. HeLa vs HeLa-Ars comparison ──
print("\n[6] HeLa vs HeLa-Ars: dist_to_3prime and positioning")
for cl in ['HeLa', 'HeLa-Ars']:
    sub = df[df['cell_line'] == cl]
    sub_ig = sub[sub['TE_group'] == 'intergenic']
    print(f"\n    {cl} (n={len(sub):,}, intergenic={len(sub_ig):,}):")
    print(f"      dist_to_3prime median = {sub['dist_to_3prime'].median():.0f} bp")
    print(f"      L1 at 3' end (<50bp) = {(sub['dist_to_3prime'] < 50).mean()*100:.1f}%")
    print(f"      overlap_frac median = {sub['overlap_frac'].median():.3f}")
    print(f"      IG dist_to_3prime median = {sub_ig['dist_to_3prime'].median():.0f} bp")

# ── 7. PAS analysis: intergenic L1 with PAS ──
print("\n[7] Intergenic L1: are reads using L1's own PAS?")
# If L1 is at the 3' end of the read AND the read has a poly(A) tail,
# then L1's internal PAS is likely providing the polyadenylation signal.
ig_at_3end = df_inter[df_inter['dist_to_3prime'] <= 50]
ig_not_3end = df_inter[df_inter['dist_to_3prime'] > 50]
print(f"    Intergenic L1 at 3' end (≤50bp): n={len(ig_at_3end):,}")
print(f"      median polya = {ig_at_3end['polya_length'].median():.1f} nt")
print(f"      median read_length = {ig_at_3end['read_length'].median():.0f} bp")
print(f"      median overlap_frac = {ig_at_3end['overlap_frac'].median():.3f}")
print(f"    Intergenic L1 NOT at 3' end (>50bp): n={len(ig_not_3end):,}")
print(f"      median polya = {ig_not_3end['polya_length'].median():.1f} nt")
print(f"      median read_length = {ig_not_3end['read_length'].median():.0f} bp")
print(f"      median overlap_frac = {ig_not_3end['overlap_frac'].median():.3f}")

# Young vs Ancient for intergenic at-3'-end
for is_y, label in [(True, "Young"), (False, "Ancient")]:
    sub = ig_at_3end[ig_at_3end['is_young'] == is_y]
    if len(sub) > 5:
        print(f"    {label} intergenic at 3' end: n={len(sub):,}, "
              f"polya={sub['polya_length'].median():.1f}nt, "
              f"overlap_frac={sub['overlap_frac'].median():.3f}")

# ── 8. L1 class (3UTR/ORF2/ORF1/5UTR) distribution ──
print("\n[8] L1 consensus region ('class' column) distribution")
if 'class' in df.columns:
    class_counts = df['class'].value_counts()
    total = len(df)
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt:,} ({cnt/total*100:.1f}%)")

# ── 9. Summary table per cell line ──
print("\n[9] Per-cell-line summary")
print(f"{'CL':>12} {'n':>7} {'dist3p_med':>10} {'at3end%':>8} {'ovlp_frac':>10} {'polya':>7}")
for cl in sorted(df['cell_line'].unique()):
    sub = df[df['cell_line'] == cl]
    print(f"{cl:>12} {len(sub):>7,} {sub['dist_to_3prime'].median():>10.0f} "
          f"{(sub['dist_to_3prime']<50).mean()*100:>7.1f}% "
          f"{sub['overlap_frac'].median():>10.3f} "
          f"{sub['polya_length'].median():>7.1f}")

print("\n" + "=" * 70)
print("DONE")
