#!/usr/bin/env python3
"""
Follow-up: L1 PAS usage and stress effects on L1 positioning.
"""
import pandas as pd
import numpy as np
import glob, os
from scipy import stats

BASE = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Load L1 summary ──
frames = []
for f in sorted(glob.glob(f'{BASE}/results_group/*/g_summary/*_L1_summary.tsv')):
    tmp = pd.read_csv(f, sep='\t', usecols=[
        'read_id', 'chr', 'start', 'end', 'read_length', 'overlap_length',
        'te_start', 'te_end', 'transcript_id', 'gene_id', 'te_strand',
        'read_strand', 'dist_to_3prime', 'polya_length', 'qc_tag',
        'TE_group'
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
df['l1_at_3end'] = df['dist_to_3prime'] <= 50

# ── Load sequence features (has PAS info) ──
feat = pd.read_csv(f'{BASE}/analysis/01_exploration/topic_08_sequence_features/ancient_l1_with_features.tsv',
                   sep='\t', usecols=['read_id', 'has_canonical_pas', 'has_any_pas',
                                       'a_fraction_3prime', 'condition'])

df_merged = df.merge(feat[['read_id', 'has_canonical_pas', 'has_any_pas', 'a_fraction_3prime']],
                     on='read_id', how='left')

print("=" * 70)
print("L1 PAS USAGE AND STRESS EFFECTS")
print("=" * 70)

# ── 1. PAS presence vs L1 at 3' end ──
print("\n[1] PAS presence vs L1 position (merged reads only)")
merged_valid = df_merged.dropna(subset=['has_canonical_pas'])
print(f"    Merged reads with PAS info: {len(merged_valid):,}")

for label, sub in [("L1 at 3' end (≤50bp)", merged_valid[merged_valid['l1_at_3end']]),
                   ("L1 NOT at 3' end", merged_valid[~merged_valid['l1_at_3end']])]:
    n = len(sub)
    canonical = sub['has_canonical_pas'].mean() * 100
    any_pas = sub['has_any_pas'].mean() * 100
    print(f"    {label} (n={n:,}):")
    print(f"      Canonical PAS: {canonical:.1f}%")
    print(f"      Any PAS: {any_pas:.1f}%")

# ── 2. HeLa vs HeLa-Ars: does stress change L1 positioning? ──
print("\n[2] HeLa vs HeLa-Ars: L1 positioning under stress")
hela = df[df['cell_line'] == 'HeLa']
hela_ars = df[df['cell_line'] == 'HeLa-Ars']

for label, sub in [("HeLa (normal)", hela), ("HeLa-Ars (stress)", hela_ars)]:
    n = len(sub)
    ig = sub[sub['TE_group'] == 'intergenic']
    it = sub[sub['TE_group'] == 'intronic']
    print(f"\n    {label} (n={n:,}):")
    print(f"      Intergenic: {len(ig):,} ({len(ig)/n*100:.1f}%)")
    print(f"      Intronic:   {len(it):,} ({len(it)/n*100:.1f}%)")
    print(f"      L1 at 3' end: {sub['l1_at_3end'].mean()*100:.1f}%")
    print(f"      IG at 3' end: {ig['l1_at_3end'].mean()*100:.1f}%")

    # Young vs Ancient
    young = sub[sub['is_young']]
    anc = sub[~sub['is_young']]
    print(f"      Young at 3' end: {young['l1_at_3end'].mean()*100:.1f}% (n={len(young):,})")
    print(f"      Ancient at 3' end: {anc['l1_at_3end'].mean()*100:.1f}% (n={len(anc):,})")

# Chi-squared test: does stress change L1 at 3' end proportion?
ct = pd.crosstab(
    pd.Categorical(df[df['cell_line'].isin(['HeLa', 'HeLa-Ars'])]['cell_line']),
    df[df['cell_line'].isin(['HeLa', 'HeLa-Ars'])]['l1_at_3end']
)
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"\n    Chi-squared (HeLa vs HeLa-Ars at-3'-end): χ²={chi2:.2f}, P={p:.3f}")

# ── 3. Poly(A) length by L1 position (HeLa vs HeLa-Ars) ──
print("\n[3] Poly(A) by L1 position × stress")
for cl in ['HeLa', 'HeLa-Ars']:
    sub = df[df['cell_line'] == cl]
    at3 = sub[sub['l1_at_3end']]
    not3 = sub[~sub['l1_at_3end']]
    print(f"    {cl}:")
    print(f"      L1 at 3': polya median = {at3['polya_length'].median():.1f} nt (n={len(at3):,})")
    print(f"      L1 away:  polya median = {not3['polya_length'].median():.1f} nt (n={len(not3):,})")

# ── 4. Intergenic L1 → is it providing PAS to the transcript? ──
print("\n[4] Intergenic L1 'PAS donor' analysis")
print("    Reads where L1 is at the 3' end of an intergenic read →")
print("    L1's own PAS (AATAAA or variant) terminates the transcript.")

ig = df[df['TE_group'] == 'intergenic'].copy()
ig_young = ig[ig['is_young']]
ig_anc = ig[~ig['is_young']]

print(f"\n    Young intergenic L1 (n={len(ig_young):,}):")
print(f"      At 3' end: {ig_young['l1_at_3end'].mean()*100:.1f}%")
print(f"      Median overlap_frac: {ig_young['overlap_frac'].median():.3f}")
print(f"      Median te_length: {ig_young['te_length'].median():.0f} bp")

print(f"\n    Ancient intergenic L1 (n={len(ig_anc):,}):")
print(f"      At 3' end: {ig_anc['l1_at_3end'].mean()*100:.1f}%")
print(f"      Median overlap_frac: {ig_anc['overlap_frac'].median():.3f}")
print(f"      Median te_length: {ig_anc['te_length'].median():.0f} bp")

# ── 5. What's downstream of L1 when L1 is NOT at 3' end? ──
print("\n[5] When L1 is NOT at 3' end of intergenic reads:")
print("    → downstream sequence provides PAS (L1 weak PAS → read-through)")
ig_not3 = ig[~ig['l1_at_3end']]
print(f"    n = {len(ig_not3):,}")
print(f"    dist_to_3prime: median = {ig_not3['dist_to_3prime'].median():.0f} bp")
print(f"    dist_to_3prime: mean = {ig_not3['dist_to_3prime'].mean():.0f} bp")
print(f"    dist_to_3prime: Q75 = {ig_not3['dist_to_3prime'].quantile(0.75):.0f} bp")
print(f"    → L1 reads through ~{ig_not3['dist_to_3prime'].median():.0f} bp downstream before polyadenylation")

# ── 6. High overlap fraction intergenic: true autonomous L1 transcripts ──
print("\n[6] High-overlap-fraction intergenic L1 (overlap_frac > 0.7)")
print("    These are likely autonomous L1 transcripts using their own promoter + PAS")
ig_hi = ig[ig['overlap_frac'] > 0.7]
ig_lo = ig[ig['overlap_frac'] <= 0.7]
print(f"    High-overlap (n={len(ig_hi):,}, {len(ig_hi)/len(ig)*100:.1f}%):")
print(f"      At 3' end: {ig_hi['l1_at_3end'].mean()*100:.1f}%")
print(f"      dist_to_3prime median: {ig_hi['dist_to_3prime'].median():.0f} bp")
print(f"      polya median: {ig_hi['polya_length'].median():.1f} nt")
print(f"      Young: {ig_hi['is_young'].mean()*100:.1f}%")
print(f"    Low-overlap (n={len(ig_lo):,}, {len(ig_lo)/len(ig)*100:.1f}%):")
print(f"      At 3' end: {ig_lo['l1_at_3end'].mean()*100:.1f}%")
print(f"      dist_to_3prime median: {ig_lo['dist_to_3prime'].median():.0f} bp")
print(f"      polya median: {ig_lo['polya_length'].median():.1f} nt")

# ── 7. HeLa stress effect stratified by L1 position ──
print("\n[7] Arsenite poly(A) shortening by L1 position")
hela_anc = df[(df['cell_line'] == 'HeLa') & (~df['is_young'])]
ars_anc = df[(df['cell_line'] == 'HeLa-Ars') & (~df['is_young'])]

for pos_label, at3_val in [("L1 at 3' end", True), ("L1 away from 3'", False)]:
    h = hela_anc[hela_anc['l1_at_3end'] == at3_val]['polya_length']
    a = ars_anc[ars_anc['l1_at_3end'] == at3_val]['polya_length']
    delta = a.median() - h.median()
    u_stat, p = stats.mannwhitneyu(h, a, alternative='two-sided')
    print(f"    {pos_label}:")
    print(f"      HeLa: {h.median():.1f} nt (n={len(h):,})")
    print(f"      HeLa-Ars: {a.median():.1f} nt (n={len(a):,})")
    print(f"      Δ = {delta:+.1f} nt (P = {p:.2e})")

# Same for intergenic only
print("\n    Intergenic only:")
hela_ig_anc = hela_anc[hela_anc['TE_group'] == 'intergenic']
ars_ig_anc = ars_anc[ars_anc['TE_group'] == 'intergenic']
for pos_label, at3_val in [("L1 at 3' end", True), ("L1 away from 3'", False)]:
    h = hela_ig_anc[hela_ig_anc['l1_at_3end'] == at3_val]['polya_length']
    a = ars_ig_anc[ars_ig_anc['l1_at_3end'] == at3_val]['polya_length']
    if len(h) > 10 and len(a) > 10:
        delta = a.median() - h.median()
        u_stat, p = stats.mannwhitneyu(h, a, alternative='two-sided')
        print(f"      {pos_label}: HeLa {h.median():.1f} → Ars {a.median():.1f}, Δ={delta:+.1f}nt, P={p:.2e}")

print("\n" + "=" * 70)
print("DONE")
