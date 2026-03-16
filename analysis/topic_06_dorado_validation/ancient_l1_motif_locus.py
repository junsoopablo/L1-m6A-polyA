#!/usr/bin/env python3
"""
Ancient L1 motif enrichment & locus-specific m6A reproducibility analysis.

Uses dorado RNA004 all-context m6A calls from HeLa to:
  Part A: Compare 5-mer motif frequencies between Ancient and Young L1
  Part B: Assess locus-specific m6A reproducibility (position-specific methylation)
  Part C: Map m6A to L1 consensus coordinates (relative position)
  Part D: Local secondary structure heuristic for top reproducible non-DRACH sites
"""

import sys, os, gzip, warnings
from collections import Counter, defaultdict
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
SITES_FILE = f"{BASE}/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results/all_m6a_sites.tsv.gz"
L1_BED = f"{BASE}/reference/L1_TE_L1_family.bed"
OUTDIR = f"{BASE}/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results"

YOUNG_SUBFAMS = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}
PROB_THR = 204  # High-confidence threshold

# DRACH definition: D={A,G,T}, R={A,G}, A=A, C=C, H={A,C,T}
DRACH_D = set("AGT")
DRACH_R = set("AG")
DRACH_H = set("ACT")

def is_drach(kmer5):
    """Check if a 5-mer (centered on A at pos 2) is DRACH."""
    if len(kmer5) != 5:
        return False
    return (kmer5[0] in DRACH_D and kmer5[1] in DRACH_R and
            kmer5[2] == 'A' and kmer5[3] == 'C' and kmer5[4] in DRACH_H)

def reverse_complement(seq):
    comp = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(comp)[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("Loading m6A site data...")
print("=" * 80)

df = pd.read_csv(SITES_FILE, sep='\t')
print(f"Total m6A sites: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Filter to L1 reads only
df_l1 = df[df['is_L1'] == True].copy()
print(f"L1 m6A sites: {len(df_l1):,}")

# Filter to high-confidence
df_hc = df_l1[df_l1['prob'] >= PROB_THR].copy()
print(f"L1 m6A sites (prob >= {PROB_THR}): {len(df_hc):,}")

# Classify Young vs Ancient
df_hc['age_class'] = df_hc['repname'].apply(
    lambda x: 'Young' if x in YOUNG_SUBFAMS else 'Ancient')
print(f"\nYoung L1 m6A sites: {(df_hc['age_class'] == 'Young').sum():,}")
print(f"Ancient L1 m6A sites: {(df_hc['age_class'] == 'Ancient').sum():,}")

# Extract 5-mer from 11-mer context (center 5 bases: positions 3-7 of 0-indexed 11-mer)
# The m6A 'A' is at position 5 (center of 11-mer), so 5-mer centered on A = positions 3:8
df_hc['context_5mer'] = df_hc['context_11mer'].apply(
    lambda x: x[3:8] if isinstance(x, str) and len(x) == 11 else 'NNNNN')

# Verify DRACH classification
df_hc['is_drach_check'] = df_hc['context_5mer'].apply(is_drach)
print(f"\nDRACH sites: {df_hc['is_drach_check'].sum():,} "
      f"({100*df_hc['is_drach_check'].mean():.1f}%)")
print(f"Non-DRACH sites: {(~df_hc['is_drach_check']).sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Ancient vs Young L1 motif comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART A: Ancient vs Young L1 5-mer motif enrichment")
print("=" * 80)

young_df = df_hc[df_hc['age_class'] == 'Young']
ancient_df = df_hc[df_hc['age_class'] == 'Ancient']

# 5-mer frequencies
young_5mer = Counter(young_df['context_5mer'])
ancient_5mer = Counter(ancient_df['context_5mer'])

n_young_total = sum(young_5mer.values())
n_ancient_total = sum(ancient_5mer.values())
print(f"Young total 5-mers: {n_young_total:,}")
print(f"Ancient total 5-mers: {n_ancient_total:,}")

# All possible 5-mers with A at center
all_5mers = set(young_5mer.keys()) | set(ancient_5mer.keys())

enrichment_rows = []
for kmer in sorted(all_5mers):
    if 'N' in kmer:
        continue
    n_y = young_5mer.get(kmer, 0)
    n_a = ancient_5mer.get(kmer, 0)
    freq_y = n_y / n_young_total if n_young_total > 0 else 0
    freq_a = n_a / n_ancient_total if n_ancient_total > 0 else 0

    # Fisher's exact: Ancient enrichment relative to Young
    # Table: [[n_a, n_ancient_total - n_a], [n_y, n_young_total - n_y]]
    table = [[n_a, n_ancient_total - n_a],
             [n_y, n_young_total - n_y]]
    try:
        odds_ratio, pval = stats.fisher_exact(table, alternative='two-sided')
    except:
        odds_ratio, pval = np.nan, 1.0

    fold_change = (freq_a / freq_y) if freq_y > 0 else np.inf

    enrichment_rows.append({
        '5mer': kmer,
        'is_drach': is_drach(kmer),
        'n_young': n_y,
        'n_ancient': n_a,
        'freq_young': freq_y,
        'freq_ancient': freq_a,
        'fold_change_anc_vs_young': fold_change,
        'fisher_OR': odds_ratio,
        'fisher_pval': pval,
    })

enrich_df = pd.DataFrame(enrichment_rows)
enrich_df['fdr'] = np.minimum(
    enrich_df['fisher_pval'] * len(enrich_df) /
    (enrich_df['fisher_pval'].rank(method='first')), 1.0)
enrich_df = enrich_df.sort_values('fold_change_anc_vs_young', ascending=False)

# Save
out_enrich = os.path.join(OUTDIR, 'ancient_vs_young_motif_enrichment.tsv')
enrich_df.to_csv(out_enrich, sep='\t', index=False, float_format='%.6g')
print(f"\nSaved: {out_enrich}")

# Top Ancient-enriched motifs (non-DRACH)
print("\n--- Top 20 Ancient-enriched NON-DRACH 5-mers (by fold-change) ---")
non_drach_enrich = enrich_df[
    (~enrich_df['is_drach']) & (enrich_df['n_ancient'] >= 50)
].head(20)
for _, row in non_drach_enrich.iterrows():
    sig = '***' if row['fdr'] < 0.001 else '**' if row['fdr'] < 0.01 else '*' if row['fdr'] < 0.05 else 'ns'
    print(f"  {row['5mer']}  Ancient: {row['n_ancient']:5d} ({row['freq_ancient']*100:.2f}%)  "
          f"Young: {row['n_young']:5d} ({row['freq_young']*100:.2f}%)  "
          f"FC={row['fold_change_anc_vs_young']:.2f}  {sig}")

# Top Young-enriched motifs
print("\n--- Top 20 Young-enriched 5-mers (lowest fold-change = Young enrichment) ---")
young_enrich = enrich_df[enrich_df['n_young'] >= 20].sort_values(
    'fold_change_anc_vs_young').head(20)
for _, row in young_enrich.iterrows():
    drach_tag = 'DRACH' if row['is_drach'] else 'non-DRACH'
    sig = '***' if row['fdr'] < 0.001 else '**' if row['fdr'] < 0.01 else '*' if row['fdr'] < 0.05 else 'ns'
    print(f"  {row['5mer']} [{drach_tag}]  Ancient: {row['n_ancient']:5d} ({row['freq_ancient']*100:.2f}%)  "
          f"Young: {row['n_young']:5d} ({row['freq_young']*100:.2f}%)  "
          f"FC={row['fold_change_anc_vs_young']:.2f}  {sig}")

# Summary statistics
print("\n--- DRACH vs non-DRACH breakdown by age class ---")
for cls in ['Young', 'Ancient']:
    sub = df_hc[df_hc['age_class'] == cls]
    n_drach = sub['is_drach_check'].sum()
    n_total = len(sub)
    print(f"  {cls}: {n_drach:,}/{n_total:,} DRACH ({100*n_drach/n_total:.1f}%), "
          f"{n_total - n_drach:,} non-DRACH ({100*(1-n_drach/n_total):.1f}%)")

# Check for known methyltransferase motifs
print("\n--- Known methyltransferase motif checks ---")
# METTL16 canonical: UACAG (in RNA = T replaced for DNA context)
# METTL16 core: xACx (positions 2-4 = AC)
# METTL3/14: DRACH
# PCIF1: m6Am at cap (not relevant here)
# Chen 2021 ABAG motif: A[CGT]AG
print("  Looking for METTL16-like and other motifs in Ancient non-DRACH top hits...")

mettl16_like = enrich_df[
    enrich_df['5mer'].str.match(r'^.AC..$') & (~enrich_df['is_drach']) &
    (enrich_df['n_ancient'] >= 30)
].sort_values('fold_change_anc_vs_young', ascending=False)
print(f"  xACxx motifs (METTL16-core-like): {len(mettl16_like)} found")
for _, row in mettl16_like.head(10).iterrows():
    print(f"    {row['5mer']}  FC={row['fold_change_anc_vs_young']:.2f}  "
          f"Ancient={row['n_ancient']}  FDR={row['fdr']:.2e}")

# ABAG motif (A[CGT]AG)
abag = enrich_df[
    enrich_df['5mer'].str.match(r'^..[ACG]AG') & (~enrich_df['is_drach']) &
    (enrich_df['n_ancient'] >= 30)
].sort_values('fold_change_anc_vs_young', ascending=False)
print(f"\n  xxBAG (ABAG-core) motifs: {len(abag)} found")
for _, row in abag.head(10).iterrows():
    print(f"    {row['5mer']}  FC={row['fold_change_anc_vs_young']:.2f}  "
          f"Ancient={row['n_ancient']}  FDR={row['fdr']:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Locus-specific m6A reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART B: Locus-specific m6A reproducibility")
print("=" * 80)

# Load L1 BED to define loci
print("Loading L1 annotation BED...")
l1_bed = pd.read_csv(L1_BED, sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'subfamily', 'locus_id'])
print(f"L1 loci in BED: {len(l1_bed):,}")

# Create interval index for fast lookup
# We need to assign each m6A site to an L1 locus
# Strategy: use the BAM data's read_id → we already know it's L1, and repname gives subfamily
# But we need the LOCUS (specific genomic element) for grouping reads

# Since reads are already classified as L1 with repname, we can group by
# genomic position directly. Two m6A sites at the same genomic position
# from different reads = reproducible.

# First, let's also load ALL L1 sites (not just high-confidence) to compute
# per-position coverage (how many reads cover each position)
print("Loading all L1 m6A sites (all prob levels) for coverage computation...")
df_l1_all = df[df['is_L1'] == True].copy()
df_l1_all['age_class'] = df_l1_all['repname'].apply(
    lambda x: 'Young' if x in YOUNG_SUBFAMS else 'Ancient')

# For reproducibility, we need:
# 1. Number of reads covering each genomic position (from all reads)
# 2. Number of reads with m6A call at that position (prob >= threshold)

# Group m6A calls by (chrom, ref_pos, strand)
# But coverage requires knowing which reads span each position - we don't have
# alignment coordinates in the sites file. We DO have all m6A sites though.

# APPROACH: Use read_id grouping to find unique reads per locus,
# then for each genomic position, count:
#   - Total reads that have ANY m6A call within ±5bp (as proxy for coverage)
#   - Reads with m6A at that exact position (prob >= thr)

# Better approach: group by L1 locus, then by genomic position
# A position is "covered" if the read has any site within the same L1 locus

# Actually, the simplest and most robust: for each (chrom, ref_pos, strand):
# - n_reads_with_m6a: count of unique read_ids with prob >= thr at this position
# - For coverage: count unique read_ids that have ANY m6A call (any prob) mapping
#   to the SAME L1 locus (chrom:locus_start:locus_end)

# Step 1: Assign each site to an L1 locus using bedtools-style overlap
# We'll do this with a merge approach: sort both by chrom+pos, use binary search

from bisect import bisect_left, bisect_right

# Build locus lookup: for each chrom, sorted list of (start, end, subfamily, locus_id)
print("Building locus lookup index...")
locus_lookup = defaultdict(list)
for _, row in l1_bed.iterrows():
    locus_lookup[row['chrom']].append(
        (row['start'], row['end'], row['subfamily'], row['locus_id']))

for chrom in locus_lookup:
    locus_lookup[chrom].sort()

# Extract starts for binary search
locus_starts = {chrom: np.array([x[0] for x in locs])
                for chrom, locs in locus_lookup.items()}

def find_locus(chrom, pos):
    """Find the L1 locus containing this position."""
    if chrom not in locus_starts:
        return None
    starts = locus_starts[chrom]
    locs = locus_lookup[chrom]
    # Find rightmost locus with start <= pos
    idx = bisect_right(starts, pos) - 1
    if idx < 0:
        return None
    # Check up to 3 candidates (overlapping elements possible)
    for i in range(max(0, idx - 2), min(len(locs), idx + 3)):
        s, e, subfam, lid = locs[i]
        if s <= pos < e:
            return lid
    return None

# Assign loci to high-confidence sites
print("Assigning m6A sites to L1 loci...")
df_hc['locus_id'] = [find_locus(r['chrom'], r['ref_pos'])
                      for _, r in df_hc.iterrows()]
n_assigned = df_hc['locus_id'].notna().sum()
print(f"Sites assigned to locus: {n_assigned:,} / {len(df_hc):,}")

# Also assign all-prob sites (for coverage estimation)
print("Assigning all-prob sites to loci (for coverage)...")
df_l1_all['locus_id'] = [find_locus(r['chrom'], r['ref_pos'])
                          for _, r in df_l1_all.iterrows()]

# Count unique reads per locus (as coverage proxy)
# A read "covers" a locus if it has any m6A call there (any prob)
locus_read_counts = df_l1_all.dropna(subset=['locus_id']).groupby(
    'locus_id')['read_id'].nunique().to_dict()

# For each genomic position (chrom, ref_pos, strand) within high-confidence:
# Count reads with m6A call
print("Computing per-position m6A counts...")
pos_key = df_hc.dropna(subset=['locus_id']).groupby(
    ['chrom', 'ref_pos', 'strand', 'locus_id']
).agg(
    n_reads_m6a=('read_id', 'nunique'),
    mean_prob=('prob', 'mean'),
    context_5mer=('context_5mer', 'first'),
    context_11mer=('context_11mer', 'first'),
    is_drach=('is_drach_check', 'first'),
    age_class=('age_class', 'first'),
    repname=('repname', 'first'),
).reset_index()

# Add locus coverage
pos_key['locus_coverage'] = pos_key['locus_id'].map(locus_read_counts)
pos_key['reproducibility'] = pos_key['n_reads_m6a'] / pos_key['locus_coverage']

# Filter to loci with >= 5 reads
pos_filt = pos_key[pos_key['locus_coverage'] >= 5].copy()
print(f"\nPositions in loci with ≥5 reads: {len(pos_filt):,}")

# Define reproducible: ≥ 50% of reads at that locus
pos_filt['is_reproducible'] = pos_filt['reproducibility'] >= 0.5
n_reprod = pos_filt['is_reproducible'].sum()
print(f"Reproducible positions (≥50% of reads): {n_reprod:,} / {len(pos_filt):,} "
      f"({100*n_reprod/len(pos_filt):.1f}%)")

# Also check with a stricter threshold
for thr_name, thr_val in [('≥30%', 0.3), ('≥50%', 0.5), ('≥70%', 0.7), ('≥90%', 0.9)]:
    n = (pos_filt['reproducibility'] >= thr_val).sum()
    print(f"  Positions with reproducibility {thr_name}: {n:,} ({100*n/len(pos_filt):.1f}%)")

# Compare Ancient vs Young reproducibility
print("\n--- Reproducibility by age class ---")
for cls in ['Young', 'Ancient']:
    sub = pos_filt[pos_filt['age_class'] == cls]
    if len(sub) == 0:
        print(f"  {cls}: no data")
        continue
    n_rep = sub['is_reproducible'].sum()
    print(f"  {cls}: {n_rep:,}/{len(sub):,} reproducible ({100*n_rep/len(sub):.1f}%)")
    # Median reproducibility
    print(f"    Median reproducibility: {sub['reproducibility'].median():.3f}")
    print(f"    Mean reproducibility: {sub['reproducibility'].mean():.3f}")

# Compare DRACH vs non-DRACH reproducibility
print("\n--- Reproducibility by DRACH status ---")
for drach_label, drach_val in [('DRACH', True), ('non-DRACH', False)]:
    sub = pos_filt[pos_filt['is_drach'] == drach_val]
    if len(sub) == 0:
        continue
    n_rep = sub['is_reproducible'].sum()
    print(f"  {drach_label}: {n_rep:,}/{len(sub):,} reproducible ({100*n_rep/len(sub):.1f}%)")
    print(f"    Median reproducibility: {sub['reproducibility'].median():.3f}")

# Cross-tabulation: age × DRACH × reproducibility
print("\n--- Cross-tabulation: age × DRACH × reproducibility ---")
for cls in ['Young', 'Ancient']:
    for drach_label, drach_val in [('DRACH', True), ('non-DRACH', False)]:
        sub = pos_filt[(pos_filt['age_class'] == cls) &
                       (pos_filt['is_drach'] == drach_val)]
        if len(sub) == 0:
            continue
        n_rep = sub['is_reproducible'].sum()
        print(f"  {cls} {drach_label}: {n_rep:,}/{len(sub):,} = "
              f"{100*n_rep/len(sub):.1f}% reproducible  "
              f"(median reprod={sub['reproducibility'].median():.3f})")

# Probability comparison: reproducible vs non-reproducible
print("\n--- Mean probability: reproducible vs non-reproducible ---")
rep_sites = pos_filt[pos_filt['is_reproducible']]
nonrep_sites = pos_filt[~pos_filt['is_reproducible']]
if len(rep_sites) > 0 and len(nonrep_sites) > 0:
    print(f"  Reproducible: mean prob = {rep_sites['mean_prob'].mean():.1f}")
    print(f"  Non-reproducible: mean prob = {nonrep_sites['mean_prob'].mean():.1f}")
    stat, pval = stats.mannwhitneyu(rep_sites['mean_prob'], nonrep_sites['mean_prob'],
                                     alternative='greater')
    print(f"  MWU P = {pval:.2e}")

# Save locus reproducibility
out_locus = os.path.join(OUTDIR, 'locus_reproducibility.tsv')
# Aggregate by locus
locus_stats = pos_filt.groupby(['locus_id', 'age_class']).agg(
    n_positions=('ref_pos', 'nunique'),
    n_reproducible=('is_reproducible', 'sum'),
    mean_reproducibility=('reproducibility', 'mean'),
    locus_coverage=('locus_coverage', 'first'),
    frac_drach=('is_drach', 'mean'),
).reset_index()
locus_stats['frac_reproducible'] = locus_stats['n_reproducible'] / locus_stats['n_positions']
locus_stats.to_csv(out_locus, sep='\t', index=False, float_format='%.4f')
print(f"\nSaved: {out_locus}")

# Save reproducible sites
out_reprod = os.path.join(OUTDIR, 'reproducible_sites.tsv')
rep_sites_save = pos_filt[pos_filt['is_reproducible']].sort_values(
    'reproducibility', ascending=False)
rep_sites_save.to_csv(out_reprod, sep='\t', index=False, float_format='%.4f')
print(f"Saved: {out_reprod}")

# Motif at reproducible sites
print("\n--- Motifs at reproducible sites ---")
print("  Top 20 5-mers at reproducible positions (all age classes):")
rep_motifs = Counter(rep_sites['context_5mer'])
for kmer, cnt in rep_motifs.most_common(20):
    drach_tag = 'DRACH' if is_drach(kmer) else 'non-DRACH'
    print(f"    {kmer} [{drach_tag}]: {cnt}")

print("\n  Top 20 5-mers at reproducible Ancient non-DRACH positions:")
anc_rep_nondrach = pos_filt[
    (pos_filt['is_reproducible']) &
    (pos_filt['age_class'] == 'Ancient') &
    (~pos_filt['is_drach'])
]
if len(anc_rep_nondrach) > 0:
    rep_nd_motifs = Counter(anc_rep_nondrach['context_5mer'])
    for kmer, cnt in rep_nd_motifs.most_common(20):
        print(f"    {kmer}: {cnt}")
else:
    print("    No reproducible Ancient non-DRACH sites found")


# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Position within L1 element (relative coordinate)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART C: m6A position within L1 element")
print("=" * 80)

# Merge L1 element coordinates with m6A sites
# Build locus → (start, end, length, subfamily) mapping
locus_info = {}
for _, row in l1_bed.iterrows():
    locus_info[row['locus_id']] = {
        'l1_start': row['start'],
        'l1_end': row['end'],
        'l1_len': row['end'] - row['start'],
        'subfamily': row['subfamily'],
    }

# For high-confidence sites with locus assigned
df_pos = df_hc.dropna(subset=['locus_id']).copy()
df_pos['l1_start'] = df_pos['locus_id'].map(lambda x: locus_info.get(x, {}).get('l1_start', np.nan))
df_pos['l1_end'] = df_pos['locus_id'].map(lambda x: locus_info.get(x, {}).get('l1_end', np.nan))
df_pos['l1_len'] = df_pos['locus_id'].map(lambda x: locus_info.get(x, {}).get('l1_len', np.nan))

# Relative position (0 = 5' end of element, 1 = 3' end)
# Need to account for strand: if element is on - strand, position is reversed
# But we don't have element strand here... L1 BED doesn't include strand
# Use simple (pos - start) / length which gives position from BED start
df_pos['rel_pos'] = (df_pos['ref_pos'] - df_pos['l1_start']) / df_pos['l1_len']

# Bin into 100bp windows for density analysis
# Use relative bins (0-1 range, 20 bins = 5% each)
n_bins = 20
df_pos['rel_bin'] = pd.cut(df_pos['rel_pos'], bins=n_bins, labels=False)

print("m6A density by relative L1 position (20 bins, 5% each):")
print("\n--- Ancient L1 ---")
ancient_pos = df_pos[df_pos['age_class'] == 'Ancient']
for drach_label, drach_val in [('DRACH', True), ('non-DRACH', False)]:
    sub = ancient_pos[ancient_pos['is_drach_check'] == drach_val]
    bin_counts = sub['rel_bin'].value_counts().sort_index()
    total = bin_counts.sum()
    print(f"\n  {drach_label} ({total:,} sites):")
    for b in range(n_bins):
        c = bin_counts.get(b, 0)
        bar = '█' * int(c / max(1, total) * 200)
        pct = 100 * c / total if total > 0 else 0
        print(f"    {b*5:3d}-{(b+1)*5:3d}%: {c:5d} ({pct:5.1f}%) {bar}")

print("\n--- Young L1 ---")
young_pos = df_pos[df_pos['age_class'] == 'Young']
for drach_label, drach_val in [('DRACH', True), ('non-DRACH', False)]:
    sub = young_pos[young_pos['is_drach_check'] == drach_val]
    bin_counts = sub['rel_bin'].value_counts().sort_index()
    total = bin_counts.sum()
    print(f"\n  {drach_label} ({total:,} sites):")
    for b in range(n_bins):
        c = bin_counts.get(b, 0)
        bar = '█' * int(c / max(1, total) * 200)
        pct = 100 * c / total if total > 0 else 0
        print(f"    {b*5:3d}-{(b+1)*5:3d}%: {c:5d} ({pct:5.1f}%) {bar}")

# Hotspot analysis: are there specific L1 positions enriched for non-DRACH?
# Compare DRACH vs non-DRACH positional distributions
print("\n--- KS test: DRACH vs non-DRACH positional distribution ---")
for cls in ['Young', 'Ancient']:
    sub_cls = df_pos[df_pos['age_class'] == cls]
    drach_pos = sub_cls[sub_cls['is_drach_check']]['rel_pos'].dropna()
    nondrach_pos = sub_cls[~sub_cls['is_drach_check']]['rel_pos'].dropna()
    if len(drach_pos) > 10 and len(nondrach_pos) > 10:
        ks_stat, ks_p = stats.ks_2samp(drach_pos, nondrach_pos)
        print(f"  {cls}: KS stat={ks_stat:.4f}, P={ks_p:.2e} "
              f"(DRACH median={drach_pos.median():.3f}, non-DRACH median={nondrach_pos.median():.3f})")

# For long Ancient L1 elements (>3kb), look at absolute position
print("\n--- Absolute position in long (>3kb) Ancient L1 elements ---")
long_ancient = df_pos[(df_pos['age_class'] == 'Ancient') & (df_pos['l1_len'] > 3000)]
if len(long_ancient) > 100:
    abs_pos = long_ancient['ref_pos'] - long_ancient['l1_start']
    # Bin into 100bp windows
    abs_bins = (abs_pos // 100) * 100
    abs_drach = long_ancient[long_ancient['is_drach_check']].copy()
    abs_nondrach = long_ancient[~long_ancient['is_drach_check']].copy()

    print(f"  Long Ancient L1 elements: {long_ancient['locus_id'].nunique()} loci, "
          f"{len(long_ancient):,} m6A sites")
    print(f"  DRACH: {len(abs_drach):,}, non-DRACH: {len(abs_nondrach):,}")

    # Top hotspot bins for non-DRACH
    nd_bins = ((abs_nondrach['ref_pos'] - abs_nondrach['l1_start']) // 200) * 200
    nd_bin_counts = nd_bins.value_counts().sort_index()
    print(f"\n  Top 10 non-DRACH hotspot bins (200bp windows) in long Ancient L1:")
    for pos, cnt in nd_bin_counts.nlargest(10).items():
        print(f"    Position {int(pos)}-{int(pos)+200}: {cnt} sites")
else:
    print(f"  Only {len(long_ancient)} sites in long Ancient L1 — skipping")


# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Local secondary structure heuristic
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART D: Secondary structure heuristic for reproducible non-DRACH sites")
print("=" * 80)

# Get top reproducible non-DRACH sites in Ancient L1
top_reprod_nd = pos_filt[
    (pos_filt['is_reproducible']) &
    (pos_filt['age_class'] == 'Ancient') &
    (~pos_filt['is_drach'])
].sort_values('reproducibility', ascending=False).head(30)

print(f"Top reproducible Ancient non-DRACH sites: {len(top_reprod_nd)}")

if len(top_reprod_nd) > 0:
    # Load reference genome for sequence extraction
    print("Loading reference genome for flanking sequence extraction...")
    import pysam

    ref_fa = pysam.FastaFile(f"{BASE}/reference/Human.fasta")

    stem_loop_candidates = []

    for idx, (_, site) in enumerate(top_reprod_nd.iterrows()):
        chrom = site['chrom']
        pos = int(site['ref_pos'])
        strand = site['strand']

        # Extract 101nt flanking (50nt each side)
        start = max(0, pos - 50)
        end = pos + 51
        try:
            seq = ref_fa.fetch(chrom, start, end).upper()
        except:
            continue

        if len(seq) < 101:
            continue

        center = 50  # m6A position in the extracted sequence

        # Check for potential stem-loop: look for reverse complement of
        # upstream 8-mer in downstream region (or vice versa)
        stem_found = False
        best_stem = None
        best_stem_len = 0

        for stem_len in range(8, 4, -1):
            # Check upstream stem → downstream complement
            upstream = seq[center - stem_len - 2 : center - 2]
            rc_upstream = reverse_complement(upstream)

            # Search in downstream region (allowing 3-15nt loop)
            for loop_start in range(center + 1, min(center + 16, len(seq) - stem_len)):
                downstream = seq[loop_start : loop_start + stem_len]
                if downstream == rc_upstream:
                    stem_found = True
                    loop_size = loop_start - center
                    best_stem = f"5'...{upstream}--[{seq[center-2:loop_start]}]--{downstream}...3'"
                    best_stem_len = stem_len
                    break

            # Also check the reverse: downstream stem → upstream complement
            if not stem_found:
                downstream = seq[center + 2 : center + 2 + stem_len]
                rc_downstream = reverse_complement(downstream)
                for loop_end in range(center - 1, max(center - 16, stem_len), -1):
                    upstream_check = seq[loop_end - stem_len : loop_end]
                    if upstream_check == rc_downstream:
                        stem_found = True
                        best_stem = f"Reverse stem at {chrom}:{pos}"
                        best_stem_len = stem_len
                        break

            if stem_found:
                break

        context = seq[center-5:center+6]
        stem_loop_candidates.append({
            'chrom': chrom,
            'pos': pos,
            'strand': strand,
            'reproducibility': site['reproducibility'],
            'n_reads_m6a': site['n_reads_m6a'],
            'locus_coverage': site['locus_coverage'],
            'context_11mer': context,
            'context_5mer': site['context_5mer'],
            'repname': site['repname'],
            'stem_loop_found': stem_found,
            'stem_length': best_stem_len if stem_found else 0,
            'stem_detail': best_stem if stem_found else '',
            'flanking_50nt': seq,
        })

    ref_fa.close()

    sl_df = pd.DataFrame(stem_loop_candidates)
    n_stem = sl_df['stem_loop_found'].sum()
    print(f"\nStem-loop candidates found: {n_stem}/{len(sl_df)} "
          f"({100*n_stem/max(1,len(sl_df)):.0f}%)")

    print("\n--- Top 15 reproducible Ancient non-DRACH sites ---")
    for _, row in sl_df.head(15).iterrows():
        sl_tag = f"STEM(len={row['stem_length']})" if row['stem_loop_found'] else "no-stem"
        print(f"  {row['chrom']}:{row['pos']} ({row['strand']})  "
              f"reprod={row['reproducibility']:.2f} ({row['n_reads_m6a']}/{row['locus_coverage']})  "
              f"5mer={row['context_5mer']}  subfam={row['repname']}  {sl_tag}")
        if row['stem_loop_found']:
            print(f"    {row['stem_detail']}")

    # Background rate: check random non-reproducible sites for stem-loop
    print("\n--- Background stem-loop rate (random non-reproducible sites) ---")
    bg_sites = pos_filt[
        (~pos_filt['is_reproducible']) &
        (pos_filt['age_class'] == 'Ancient') &
        (~pos_filt['is_drach'])
    ].sample(n=min(100, len(pos_filt[~pos_filt['is_reproducible']])), random_state=42)

    ref_fa = pysam.FastaFile(f"{BASE}/reference/Human.fasta")
    bg_stem_count = 0
    bg_total = 0

    for _, site in bg_sites.iterrows():
        chrom = site['chrom']
        pos = int(site['ref_pos'])
        start = max(0, pos - 50)
        end = pos + 51
        try:
            seq = ref_fa.fetch(chrom, start, end).upper()
        except:
            continue
        if len(seq) < 101:
            continue

        bg_total += 1
        center = 50

        for stem_len in range(8, 4, -1):
            upstream = seq[center - stem_len - 2 : center - 2]
            rc_upstream = reverse_complement(upstream)
            found = False
            for loop_start in range(center + 1, min(center + 16, len(seq) - stem_len)):
                if seq[loop_start : loop_start + stem_len] == rc_upstream:
                    found = True
                    break
            if not found:
                downstream = seq[center + 2 : center + 2 + stem_len]
                rc_downstream = reverse_complement(downstream)
                for loop_end in range(center - 1, max(center - 16, stem_len), -1):
                    if seq[loop_end - stem_len : loop_end] == rc_downstream:
                        found = True
                        break
            if found:
                bg_stem_count += 1
                break

    ref_fa.close()
    print(f"  Background: {bg_stem_count}/{bg_total} = "
          f"{100*bg_stem_count/max(1,bg_total):.0f}% with stem-loop")
    if n_stem > 0 and bg_total > 0:
        # Fisher's exact: reproducible stem vs background stem
        table = [[n_stem, len(sl_df) - n_stem],
                 [bg_stem_count, bg_total - bg_stem_count]]
        _, fisher_p = stats.fisher_exact(table)
        print(f"  Fisher's exact P = {fisher_p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  ANCIENT vs YOUNG L1 m6A MOTIF & LOCUS REPRODUCIBILITY ANALYSIS       ║
║  Dorado RNA004 all-context m6A | HeLa | prob ≥ 204                    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# Part A summary
print("PART A: Motif Enrichment")
print("-" * 60)
n_young_drach = young_df['is_drach_check'].sum()
n_young_nd = len(young_df) - n_young_drach
n_anc_drach = ancient_df['is_drach_check'].sum()
n_anc_nd = len(ancient_df) - n_anc_drach

print(f"  Young L1:  {len(young_df):,} sites  "
      f"(DRACH {n_young_drach:,} [{100*n_young_drach/len(young_df):.1f}%], "
      f"non-DRACH {n_young_nd:,} [{100*n_young_nd/len(young_df):.1f}%])")
print(f"  Ancient L1: {len(ancient_df):,} sites  "
      f"(DRACH {n_anc_drach:,} [{100*n_anc_drach/len(ancient_df):.1f}%], "
      f"non-DRACH {n_anc_nd:,} [{100*n_anc_nd/len(ancient_df):.1f}%])")
print(f"  → Ancient has LOWER DRACH fraction: "
      f"{100*n_anc_drach/len(ancient_df):.1f}% vs {100*n_young_drach/len(young_df):.1f}%")

# Top Ancient-specific non-DRACH
top5_anc = enrich_df[(~enrich_df['is_drach']) & (enrich_df['n_ancient'] >= 50)].head(5)
print(f"\n  Top 5 Ancient-enriched non-DRACH motifs:")
for _, r in top5_anc.iterrows():
    print(f"    {r['5mer']}  FC={r['fold_change_anc_vs_young']:.2f}  "
          f"(Ancient {r['n_ancient']:,}, Young {r['n_young']:,})")

# Part B summary
print(f"\nPART B: Locus-specific Reproducibility")
print("-" * 60)
for cls in ['Young', 'Ancient']:
    sub = pos_filt[pos_filt['age_class'] == cls]
    if len(sub) == 0:
        continue
    n_rep = sub['is_reproducible'].sum()
    print(f"  {cls}: {n_rep:,}/{len(sub):,} = {100*n_rep/len(sub):.1f}% reproducible (≥50%)")

# Part C summary
print(f"\nPART C: Positional Distribution")
print("-" * 60)
for cls in ['Young', 'Ancient']:
    sub_cls = df_pos[df_pos['age_class'] == cls]
    drach_pos = sub_cls[sub_cls['is_drach_check']]['rel_pos'].dropna()
    nondrach_pos = sub_cls[~sub_cls['is_drach_check']]['rel_pos'].dropna()
    if len(drach_pos) > 0 and len(nondrach_pos) > 0:
        print(f"  {cls}: DRACH median rel_pos={drach_pos.median():.3f}, "
              f"non-DRACH median={nondrach_pos.median():.3f}")

# Part D summary
if len(top_reprod_nd) > 0:
    print(f"\nPART D: Stem-loop Structure Heuristic")
    print("-" * 60)
    print(f"  Reproducible Ancient non-DRACH: {n_stem}/{len(sl_df)} "
          f"({100*n_stem/max(1,len(sl_df)):.0f}%) have potential stem-loop")
    print(f"  Background (non-reproducible): {bg_stem_count}/{bg_total} "
          f"({100*bg_stem_count/max(1,bg_total):.0f}%)")

print("\n" + "=" * 80)
print("Output files:")
print(f"  1. {out_enrich}")
print(f"  2. {out_locus}")
print(f"  3. {out_reprod}")
print("=" * 80)
