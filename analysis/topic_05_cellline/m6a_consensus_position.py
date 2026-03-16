#!/usr/bin/env python3
"""Map ancient L1 fragments to L1 consensus position and check m6A enrichment by region.

Hypothesis: If m6A enrichment is position-dependent within L1 consensus,
ancient fragments originating from m6A-rich regions should have higher m6A rates.

Uses UCSC hg38 rmsk table to get consensus coordinates (repStart/repEnd/repLeft).

UCSC rmsk consensus position convention:
  + strand: cons_start = repStart, cons_end = repEnd, cons_length = repEnd + abs(repLeft)
  - strand: cons_start = repLeft, cons_end = repEnd, cons_length = repEnd + abs(repStart)

L1HS consensus (~6064bp):
  5'UTR: 1-910 (0-15%), ORF1: 911-1924 (15-32%), ORF2: 1991-5817 (33-96%), 3'UTR: 5818-6064 (96-100%)
"""

import pandas as pd
import numpy as np
import gzip
import ast
from pathlib import Path
from scipy import stats

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_DIR = TOPIC_05 / 'part3_l1_per_read_cache'
RMSK = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/rmsk.txt.gz')
OUTDIR = TOPIC_05 / 'm6a_consensus_position'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Exclude special conditions
EXCLUDE_PREFIXES = ('HeLa-Ars', 'MCF7-EV')
ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# =====================================================================
# 1. Build rmsk L1 consensus position lookup
# =====================================================================
print("Loading UCSC rmsk L1 consensus positions...")
rmsk_lookup = {}  # (chrom, start, end, strand, repName) → (cons_start, cons_end, cons_length)

with gzip.open(RMSK, 'rt') as f:
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) < 16:
            continue
        if fields[11] != 'LINE' or fields[12] != 'L1':
            continue

        chrom = fields[5]
        geno_start = int(fields[6])
        geno_end = int(fields[7])
        strand = fields[9]
        rep_name = fields[10]
        rep_start = int(fields[13])
        rep_end = int(fields[14])
        rep_left = int(fields[15])

        if strand == '+':
            cons_start = rep_start
            cons_end = rep_end
            cons_length = rep_end + abs(rep_left)
        else:
            cons_start = rep_left
            cons_end = rep_end
            cons_length = rep_end + abs(rep_start)

        if cons_length <= 0:
            continue

        key = (chrom, geno_start, geno_end, strand, rep_name)
        rmsk_lookup[key] = (cons_start, cons_end, cons_length)

print(f"  Loaded {len(rmsk_lookup)} L1 entries with consensus positions")

# =====================================================================
# 2. Load L1 reads with m6A data
# =====================================================================
print("\nLoading L1 reads with m6A...")
all_reads = []

for grp in ALL_GROUPS:
    summary_f = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    cache_f = CACHE_DIR / f'{grp}_l1_per_read.tsv'
    if not summary_f.exists() or not cache_f.exists():
        continue

    summary = pd.read_csv(summary_f, sep='\t', usecols=[
        'read_id', 'chr', 'start', 'end', 'read_length',
        'te_start', 'te_end', 'gene_id', 'te_strand', 'read_strand',
        'transcript_id', 'qc_tag',
    ])
    summary = summary[summary['qc_tag'] == 'PASS'].copy()

    cache = pd.read_csv(cache_f, sep='\t', usecols=[
        'read_id', 'read_length', 'm6a_sites_high', 'm6a_positions'
    ])

    merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_positions']],
                           on='read_id', how='inner')
    merged['group'] = grp
    merged['is_young'] = merged['gene_id'].isin(YOUNG)
    merged['age'] = np.where(merged['is_young'], 'young', 'ancient')
    merged['te_length'] = merged['te_end'] - merged['te_start']

    all_reads.append(merged)

df = pd.concat(all_reads, ignore_index=True)
print(f"  Total PASS reads: {len(df)} (Young: {df['is_young'].sum()}, Ancient: {(~df['is_young']).sum()})")

# =====================================================================
# 3. Match reads to rmsk consensus positions
# =====================================================================
print("\nMatching reads to rmsk consensus positions...")

# Match by (chrom, te_start, te_end, te_strand, gene_id)
# Note: our BED uses 0-based start (same as UCSC rmsk genoStart)
cons_starts = []
cons_ends = []
cons_lengths = []
matched = 0
unmatched = 0

for _, row in df.iterrows():
    key = (row['chr'], row['te_start'], row['te_end'], row['te_strand'], row['gene_id'])
    if key in rmsk_lookup:
        cs, ce, cl = rmsk_lookup[key]
        cons_starts.append(cs)
        cons_ends.append(ce)
        cons_lengths.append(cl)
        matched += 1
    else:
        cons_starts.append(np.nan)
        cons_ends.append(np.nan)
        cons_lengths.append(np.nan)
        unmatched += 1

df['cons_start'] = cons_starts
df['cons_end'] = cons_ends
df['cons_length'] = cons_lengths

print(f"  Matched: {matched} ({matched/len(df)*100:.1f}%)")
print(f"  Unmatched: {unmatched} ({unmatched/len(df)*100:.1f}%)")

# Filter to matched only
df_m = df[df['cons_length'].notna()].copy()
print(f"  Using {len(df_m)} matched reads")

# Fractional position of L1 fragment within consensus
df_m['frac_start'] = df_m['cons_start'] / df_m['cons_length']
df_m['frac_end'] = df_m['cons_end'] / df_m['cons_length']
df_m['frac_center'] = (df_m['frac_start'] + df_m['frac_end']) / 2

# =====================================================================
# 4. Ancient L1: m6A rate by consensus region of origin
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 1: ANCIENT L1 — m6A BY CONSENSUS REGION OF ORIGIN")
print("=" * 70)

ancient = df_m[df_m['age'] == 'ancient'].copy()
ancient['m6a_per_kb'] = ancient['m6a_sites_high'] / ancient['read_length'] * 1000

# Define regions based on L1 consensus (normalized to 0-1)
regions = [
    ("5'UTR (0-15%)", 0, 0.15),
    ("ORF1 (15-32%)", 0.15, 0.32),
    ("ORF2-5' (32-60%)", 0.32, 0.60),
    ("ORF2-3' (60-96%)", 0.60, 0.96),
    ("3'UTR (96-100%)", 0.96, 1.0),
]

print(f"\nAncient L1: n={len(ancient)}")
print(f"\n{'Region':<25s} {'n':>6s} {'m6A/kb':>8s} {'m6A/read':>9s} {'med_RL':>8s} {'med_TE_len':>11s}")
print("-" * 70)

region_stats = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 0:
        m6a_kb = sub['m6a_per_kb'].median()
        m6a_read = sub['m6a_sites_high'].median()
        rl = sub['read_length'].median()
        te_len = sub['te_length'].median()
        print(f"  {label:<23s} {len(sub):>6d} {m6a_kb:>8.2f} {m6a_read:>9.1f} {rl:>8.0f} {te_len:>11.0f}")
        region_stats.append({
            'region': label,
            'frac_lo': lo,
            'frac_hi': hi,
            'n_reads': len(sub),
            'median_m6a_per_kb': m6a_kb,
            'median_m6a_per_read': m6a_read,
            'median_read_length': rl,
            'median_te_length': te_len,
        })

# Statistical test: is m6A/kb different across regions?
groups_for_kw = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 10:
        groups_for_kw.append(sub['m6a_per_kb'].values)

if len(groups_for_kw) >= 2:
    kw_stat, kw_p = stats.kruskal(*groups_for_kw)
    print(f"\n  Kruskal-Wallis test (m6A/kb across regions): H={kw_stat:.2f}, p={kw_p:.2e}")

# =====================================================================
# 5. Compare 5'UTR-origin vs ORF2-origin ancient fragments
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: 5'UTR-ORIGIN vs ORF2-ORIGIN ANCIENT L1")
print("=" * 70)

utr5 = ancient[(ancient['frac_center'] >= 0) & (ancient['frac_center'] < 0.15)]
orf2 = ancient[(ancient['frac_center'] >= 0.32) & (ancient['frac_center'] < 0.96)]

if len(utr5) > 10 and len(orf2) > 10:
    _, p = stats.mannwhitneyu(utr5['m6a_per_kb'], orf2['m6a_per_kb'], alternative='two-sided')
    print(f"\n  5'UTR-origin: n={len(utr5)}, m6A/kb median={utr5['m6a_per_kb'].median():.2f}")
    print(f"  ORF2-origin:  n={len(orf2)}, m6A/kb median={orf2['m6a_per_kb'].median():.2f}")
    print(f"  Ratio: {orf2['m6a_per_kb'].median() / utr5['m6a_per_kb'].median():.2f}x")
    print(f"  Mann-Whitney p={p:.2e}")

    # Read-length controlled comparison
    print("\n  Read-length-controlled comparison:")
    bins = [(200, 500), (500, 1000), (1000, 2000), (2000, 5000)]
    for lo, hi in bins:
        u = utr5[(utr5['read_length'] >= lo) & (utr5['read_length'] < hi)]['m6a_per_kb']
        o = orf2[(orf2['read_length'] >= lo) & (orf2['read_length'] < hi)]['m6a_per_kb']
        if len(u) > 5 and len(o) > 5:
            _, p = stats.mannwhitneyu(u, o, alternative='two-sided')
            print(f"    RL {lo}-{hi}: 5'UTR={u.median():.2f}(n={len(u)}), "
                  f"ORF2={o.median():.2f}(n={len(o)}), "
                  f"ratio={o.median()/u.median():.2f}x, p={p:.2e}")

# =====================================================================
# 6. Young L1 — same analysis for comparison
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: YOUNG L1 — m6A BY CONSENSUS REGION")
print("=" * 70)

young = df_m[df_m['age'] == 'young'].copy()
young['m6a_per_kb'] = young['m6a_sites_high'] / young['read_length'] * 1000

print(f"\nYoung L1: n={len(young)}")
print(f"  Consensus length: median={young['cons_length'].median():.0f}")
print(f"  Fragment covers: {young['frac_start'].median():.2f} to {young['frac_end'].median():.2f} of consensus")

# Most young L1 fragments are near the 3' end (DRS 3' bias)
print(f"\n  Fragment center distribution:")
for label, lo, hi in regions:
    sub = young[(young['frac_center'] >= lo) & (young['frac_center'] < hi)]
    print(f"    {label}: {len(sub)} reads ({len(sub)/len(young)*100:.1f}%)")

# =====================================================================
# 7. Ancient L1: finer resolution (10 bins)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: ANCIENT L1 — 10-BIN m6A PROFILE")
print("=" * 70)

print(f"\n{'Bin':<15s} {'n':>6s} {'m6A/kb':>8s} {'med_RL':>8s} {'med_TE':>8s}")
print("-" * 50)

fine_stats = []
for i in range(10):
    lo = i * 0.1
    hi = (i + 1) * 0.1
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 0:
        m6a_kb = sub['m6a_per_kb'].median()
        bar = '#' * int(m6a_kb / 0.5)
        print(f"  {lo:.1f}-{hi:.1f}  {len(sub):>6d} {m6a_kb:>8.2f} {sub['read_length'].median():>8.0f} {sub['te_length'].median():>8.0f} {bar}")
        fine_stats.append({
            'bin_start': lo,
            'bin_end': hi,
            'n_reads': len(sub),
            'median_m6a_per_kb': m6a_kb,
            'median_read_length': sub['read_length'].median(),
            'median_te_length': sub['te_length'].median(),
        })

# =====================================================================
# 8. Per-site m6A rate by consensus region (motif-normalized)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: m6A PER-SITE RATE BY REGION (READ-LEVEL)")
print("=" * 70)

# This uses m6a_sites_high / (expected motif sites based on read length)
# Approximate: DRACH density ≈ 4.2/kb for L1 (from m6a_validation)
DRACH_DENSITY = 4.22  # DRACH motifs per kb in L1

print(f"\n  Using DRACH density = {DRACH_DENSITY}/kb")
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 10:
        expected_motifs = sub['read_length'] / 1000 * DRACH_DENSITY
        # Per-site rate = m6A sites / expected motif sites
        per_site = sub['m6a_sites_high'] / expected_motifs
        rate = per_site.median()
        print(f"  {label:<23s}: n={len(sub):>5d}, per-site rate={rate:.3f}")

# =====================================================================
# 9. Summary and save
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6: SUMMARY")
print("=" * 70)

region_df = pd.DataFrame(region_stats)
print(region_df.to_string(index=False))
region_df.to_csv(OUTDIR / 'ancient_m6a_by_consensus_region.tsv', sep='\t', index=False)

fine_df = pd.DataFrame(fine_stats)
fine_df.to_csv(OUTDIR / 'ancient_m6a_10bin_profile.tsv', sep='\t', index=False)

# Young L1 summary
young_summary = []
for label, lo, hi in regions:
    sub = young[(young['frac_center'] >= lo) & (young['frac_center'] < hi)]
    if len(sub) > 0:
        young_summary.append({
            'region': label,
            'n_reads': len(sub),
            'median_m6a_per_kb': sub['m6a_per_kb'].median(),
        })
young_df = pd.DataFrame(young_summary)
if len(young_df) > 0:
    young_df.to_csv(OUTDIR / 'young_m6a_by_consensus_region.tsv', sep='\t', index=False)

print(f"\nResults saved to: {OUTDIR}")
print("Done!")
