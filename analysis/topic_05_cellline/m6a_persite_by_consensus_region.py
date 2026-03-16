#!/usr/bin/env python3
"""Per-site m6A rate by L1 consensus region — using actual fragment sequences.

Key question: ORF1-derived ancient fragments have higher m6A/kb (6.10) despite
lower DRACH density in the L1 consensus (24.6/kb vs ORF2 36.6/kb).

Hypothesis test:
  (A) Ancient mutations created more DRACH in ORF1 fragments → motif density ↑
  (B) Per-site m6A rate is higher in ORF1 fragments → enzymatic preference
  (C) High-rate motifs (TAACT, GGACT) are more common in ORF1 fragments

Method: Extract actual reference sequence for each L1 overlap region,
count DRACH motifs, compute per-site rate = m6A_in_L1 / DRACH_count.
"""

import pandas as pd
import numpy as np
import pysam
import gzip
import ast
import re
from pathlib import Path
from scipy import stats
from collections import Counter

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results_group'
TOPIC_05 = PROJECT / 'analysis/01_exploration/topic_05_cellline'
CACHE_DIR = TOPIC_05 / 'part3_l1_per_read_cache'
RMSK = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/rmsk.txt.gz')
GENOME = '/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/Human.fasta'
OUTDIR = TOPIC_05 / 'm6a_consensus_position'
OUTDIR.mkdir(exist_ok=True)

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}
EXCLUDE_PREFIXES = ('HeLa-Ars', 'MCF7-EV')

# DRACH = [AGT][AG]AC[ACT]
DRACH_PATTERN = re.compile(r'(?=([AGT][AG]AC[ACT]))')
# All 18 DRACH motifs
DRACH_BASES = {'D': 'AGT', 'R': 'AG', 'H': 'ACT'}

ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# =====================================================================
# 1. Build rmsk lookup
# =====================================================================
print("Loading rmsk...")
rmsk_lookup = {}
with gzip.open(RMSK, 'rt') as f:
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) < 16 or fields[11] != 'LINE' or fields[12] != 'L1':
            continue
        chrom = fields[5]
        gs, ge = int(fields[6]), int(fields[7])
        strand = fields[9]
        rn = fields[10]
        rs, re_val, rl = int(fields[13]), int(fields[14]), int(fields[15])

        if strand == '+':
            cs, ce, cl = rs, re_val, re_val + abs(rl)
        else:
            cs, ce, cl = rl, re_val, re_val + abs(rs)
        if cl <= 0:
            continue
        rmsk_lookup[(chrom, gs, ge, strand, rn)] = (cs, ce, cl)

print(f"  {len(rmsk_lookup)} entries")

# =====================================================================
# 2. Load reads
# =====================================================================
print("Loading reads...")
all_reads = []
for grp in ALL_GROUPS:
    sf = RESULTS / grp / 'g_summary' / f'{grp}_L1_summary.tsv'
    cf = CACHE_DIR / f'{grp}_l1_per_read.tsv'
    if not sf.exists() or not cf.exists():
        continue

    summary = pd.read_csv(sf, sep='\t', usecols=[
        'read_id', 'chr', 'start', 'end', 'read_length',
        'te_start', 'te_end', 'gene_id', 'te_strand', 'qc_tag',
    ])
    summary = summary[summary['qc_tag'] == 'PASS'].copy()

    cache = pd.read_csv(cf, sep='\t', usecols=[
        'read_id', 'read_length', 'm6a_sites_high', 'm6a_positions'
    ])

    merged = summary.merge(cache[['read_id', 'm6a_sites_high', 'm6a_positions']],
                           on='read_id', how='inner')
    merged['group'] = grp
    merged['age'] = np.where(merged['gene_id'].isin(YOUNG), 'young', 'ancient')
    merged['te_length'] = merged['te_end'] - merged['te_start']
    all_reads.append(merged)

df = pd.concat(all_reads, ignore_index=True)
print(f"  {len(df)} reads (Ancient: {(df['age']=='ancient').sum()})")

# Match rmsk
cons_data = []
for _, row in df.iterrows():
    key = (row['chr'], row['te_start'], row['te_end'], row['te_strand'], row['gene_id'])
    cons_data.append(rmsk_lookup.get(key, (np.nan, np.nan, np.nan)))

df['cons_start'] = [c[0] for c in cons_data]
df['cons_end'] = [c[1] for c in cons_data]
df['cons_length'] = [c[2] for c in cons_data]
df = df[df['cons_length'].notna()].copy()
df['frac_center'] = ((df['cons_start'] + df['cons_end']) / 2) / df['cons_length']

# Focus on ancient
ancient = df[df['age'] == 'ancient'].copy()
print(f"  Ancient matched: {len(ancient)}")

# =====================================================================
# 3. Extract reference sequences and count DRACH motifs
# =====================================================================
print("\nExtracting reference sequences and counting DRACH motifs...")

genome = pysam.FastaFile(GENOME)

def count_drach_in_region(chrom, start, end, strand):
    """Count DRACH motifs in the actual genomic sequence of L1 overlap."""
    try:
        seq = genome.fetch(chrom, max(0, start), end).upper()
    except (ValueError, KeyError):
        return 0, 0, Counter()

    if strand == '-':
        # Reverse complement for - strand L1
        comp = str.maketrans('ACGT', 'TGCA')
        seq = seq.translate(comp)[::-1]

    # Count DRACH on forward strand of the L1
    matches = DRACH_PATTERN.findall(seq)
    motif_counts = Counter(matches)
    seq_len = len(seq)
    at_count = seq.count('A') + seq.count('T')

    return len(matches), seq_len, motif_counts

def count_m6a_in_l1(row):
    """Count m6A sites within L1 element boundaries."""
    pos_str = row['m6a_positions']
    if pd.isna(pos_str) or pos_str == '[]':
        return 0
    try:
        positions = ast.literal_eval(pos_str)
    except:
        return 0
    if not positions:
        return 0

    read_start = row['start']
    te_s, te_e = row['te_start'], row['te_end']
    return sum(1 for p in positions if te_s <= read_start + p <= te_e)

# Process each read
print("  Processing reads (extracting sequence + counting DRACH)...")
drach_counts = []
seq_lengths = []
m6a_in_l1_counts = []
motif_counters = []

batch_size = 5000
for idx, (_, row) in enumerate(ancient.iterrows()):
    if idx % batch_size == 0 and idx > 0:
        print(f"    {idx}/{len(ancient)}...")

    # L1 overlap region
    ov_start = max(row['start'], row['te_start'])
    ov_end = min(row['end'], row['te_end'])

    if ov_end <= ov_start:
        drach_counts.append(0)
        seq_lengths.append(0)
        m6a_in_l1_counts.append(0)
        motif_counters.append(Counter())
        continue

    n_drach, seq_len, motif_cnt = count_drach_in_region(
        row['chr'], ov_start, ov_end, row['te_strand']
    )
    drach_counts.append(n_drach)
    seq_lengths.append(seq_len)
    motif_counters.append(motif_cnt)

    m6a_in = count_m6a_in_l1(row)
    m6a_in_l1_counts.append(m6a_in)

ancient = ancient.copy()
ancient['drach_count'] = drach_counts
ancient['seq_len'] = seq_lengths
ancient['m6a_in_l1'] = m6a_in_l1_counts
ancient['drach_per_kb'] = np.where(ancient['seq_len'] > 0,
                                    ancient['drach_count'] / ancient['seq_len'] * 1000, 0)
ancient['m6a_in_l1_per_kb'] = np.where(ancient['seq_len'] > 0,
                                        ancient['m6a_in_l1'] / ancient['seq_len'] * 1000, 0)
ancient['per_site_rate'] = np.where(ancient['drach_count'] > 0,
                                     ancient['m6a_in_l1'] / ancient['drach_count'], np.nan)

# Aggregate motif counts per region
region_motif_totals = {}

genome.close()
print("  Done.")

# =====================================================================
# 4. Results by consensus region
# =====================================================================
regions = [
    ("5'UTR (0-15%)", 0, 0.15),
    ("ORF1 (15-32%)", 0.15, 0.32),
    ("ORF2-5' (32-60%)", 0.32, 0.60),
    ("ORF2-3' (60-96%)", 0.60, 0.96),
    ("3'UTR (96-100%)", 0.96, 1.0),
]

print("\n" + "=" * 70)
print("SECTION 1: DECOMPOSITION — DRACH density vs per-site rate by region")
print("=" * 70)

print(f"\n{'Region':<25s} {'n':>6s} {'DRACH/kb':>9s} {'m6A/kb':>8s} {'rate':>7s} {'OL':>6s}")
print("-" * 65)

result_rows = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    sub = sub[sub['seq_len'] > 50]  # minimum 50bp
    if len(sub) > 0:
        drach_kb = sub['drach_per_kb'].median()
        m6a_kb = sub['m6a_in_l1_per_kb'].median()
        rate = sub[sub['drach_count'] > 0]['per_site_rate'].median()
        ol = sub['seq_len'].median()
        print(f"  {label:<23s} {len(sub):>6d} {drach_kb:>9.2f} {m6a_kb:>8.2f} {rate:>7.1%} {ol:>6.0f}")
        result_rows.append({
            'region': label,
            'n_reads': len(sub),
            'median_drach_per_kb': drach_kb,
            'median_m6a_per_kb': m6a_kb,
            'median_per_site_rate': rate,
            'median_overlap_len': ol,
        })

        # Aggregate motif counts for this region
        total_motifs = Counter()
        for i in sub.index:
            total_motifs += motif_counters[ancient.index.get_loc(i)]
        region_motif_totals[label] = total_motifs

# =====================================================================
# 5. Statistical tests on per-site rate
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: PER-SITE RATE COMPARISONS")
print("=" * 70)

# KW test on per-site rate
rate_groups = []
rate_labels = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi) &
                  (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]
    if len(sub) > 10:
        rate_groups.append(sub['per_site_rate'].dropna().values)
        rate_labels.append(label)

if len(rate_groups) >= 2:
    kw_stat, kw_p = stats.kruskal(*rate_groups)
    print(f"\n  Kruskal-Wallis (per-site rate across regions): H={kw_stat:.2f}, p={kw_p:.2e}")

# ORF1 vs ORF2 per-site rate
orf1_rate = ancient[(ancient['frac_center'] >= 0.15) & (ancient['frac_center'] < 0.32) &
                     (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]['per_site_rate'].dropna()
orf2_rate = ancient[(ancient['frac_center'] >= 0.32) & (ancient['frac_center'] < 0.96) &
                     (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]['per_site_rate'].dropna()
utr5_rate = ancient[(ancient['frac_center'] >= 0) & (ancient['frac_center'] < 0.15) &
                     (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]['per_site_rate'].dropna()
utr3_rate = ancient[(ancient['frac_center'] >= 0.96) & (ancient['frac_center'] < 1.0) &
                     (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]['per_site_rate'].dropna()

print(f"\n  ORF1 per-site rate: median={orf1_rate.median():.3f}, mean={orf1_rate.mean():.3f} (n={len(orf1_rate)})")
print(f"  ORF2 per-site rate: median={orf2_rate.median():.3f}, mean={orf2_rate.mean():.3f} (n={len(orf2_rate)})")
_, p = stats.mannwhitneyu(orf1_rate, orf2_rate, alternative='two-sided')
print(f"  ORF1 vs ORF2: {orf1_rate.median():.3f} vs {orf2_rate.median():.3f} = {orf1_rate.median()/orf2_rate.median():.2f}x, p={p:.2e}")

_, p2 = stats.mannwhitneyu(utr5_rate, orf2_rate, alternative='two-sided')
print(f"  5'UTR vs ORF2: {utr5_rate.median():.3f} vs {orf2_rate.median():.3f} = {utr5_rate.median()/orf2_rate.median():.2f}x, p={p2:.2e}")

# ORF1 vs ORF2 DRACH density
orf1_drach = ancient[(ancient['frac_center'] >= 0.15) & (ancient['frac_center'] < 0.32) &
                      (ancient['seq_len'] > 50)]['drach_per_kb']
orf2_drach = ancient[(ancient['frac_center'] >= 0.32) & (ancient['frac_center'] < 0.96) &
                      (ancient['seq_len'] > 50)]['drach_per_kb']
_, p3 = stats.mannwhitneyu(orf1_drach, orf2_drach, alternative='two-sided')
print(f"\n  ORF1 DRACH/kb: median={orf1_drach.median():.2f} (n={len(orf1_drach)})")
print(f"  ORF2 DRACH/kb: median={orf2_drach.median():.2f} (n={len(orf2_drach)})")
print(f"  ORF1 vs ORF2 DRACH: {orf1_drach.median():.2f} vs {orf2_drach.median():.2f} = {orf1_drach.median()/orf2_drach.median():.2f}x, p={p3:.2e}")

# =====================================================================
# 6. Motif composition by region
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: DRACH MOTIF COMPOSITION BY REGION")
print("=" * 70)

# Get all 18 DRACH motifs
all_motifs = sorted(set(
    d + r + 'AC' + h
    for d in 'AGT' for r in 'AG' for h in 'ACT'
))

print(f"\n{'Motif':<8s}", end='')
for label, _, _ in regions:
    print(f" {label[:12]:>12s}", end='')
print()
print("-" * 72)

for motif in all_motifs:
    print(f"  {motif:<6s}", end='')
    for label, _, _ in regions:
        if label in region_motif_totals:
            total = sum(region_motif_totals[label].values())
            cnt = region_motif_totals[label].get(motif, 0)
            pct = cnt / total * 100 if total > 0 else 0
            print(f" {pct:>11.1f}%", end='')
        else:
            print(f" {'N/A':>12s}", end='')
    print()

# Total DRACH counts per region
print(f"\n  {'Total':<6s}", end='')
for label, _, _ in regions:
    if label in region_motif_totals:
        total = sum(region_motif_totals[label].values())
        print(f" {total:>12d}", end='')
print()

# =====================================================================
# 7. Top high-rate vs low-rate motif enrichment
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: HIGH-RATE vs LOW-RATE MOTIF FRACTION BY REGION")
print("=" * 70)

# Known rates from motif_enrichment analysis
HIGH_RATE_MOTIFS = {'TAACT', 'GGACT', 'TAACA', 'GAACT', 'AAACT'}  # top 5 by rate
LOW_RATE_MOTIFS = {'TGACA', 'TGACC', 'TGACT', 'AGACC'}  # bottom 4 by rate

for label, _, _ in regions:
    if label not in region_motif_totals:
        continue
    mc = region_motif_totals[label]
    total = sum(mc.values())
    if total == 0:
        continue

    high = sum(mc.get(m, 0) for m in HIGH_RATE_MOTIFS)
    low = sum(mc.get(m, 0) for m in LOW_RATE_MOTIFS)
    print(f"  {label:<23s}: high-rate={high/total*100:.1f}%, low-rate={low/total*100:.1f}%, "
          f"H/L ratio={high/low:.2f}" if low > 0 else f"  {label:<23s}: high-rate={high/total*100:.1f}%, low-rate=0")

# =====================================================================
# 8. Overlap-length matched per-site rate
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: OVERLAP-LENGTH MATCHED PER-SITE RATE")
print("=" * 70)

orf1_all = ancient[(ancient['frac_center'] >= 0.15) & (ancient['frac_center'] < 0.32) &
                    (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]
orf2_all = ancient[(ancient['frac_center'] >= 0.32) & (ancient['frac_center'] < 0.96) &
                    (ancient['drach_count'] > 0) & (ancient['seq_len'] > 50)]

print("\n--- ORF1 vs ORF2 per-site rate (overlap-length matched) ---")
for lo, hi in [(100, 300), (300, 600), (600, 1000), (1000, 2000)]:
    o1 = orf1_all[(orf1_all['seq_len'] >= lo) & (orf1_all['seq_len'] < hi)]['per_site_rate'].dropna()
    o2 = orf2_all[(orf2_all['seq_len'] >= lo) & (orf2_all['seq_len'] < hi)]['per_site_rate'].dropna()
    if len(o1) > 5 and len(o2) > 5:
        _, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')
        print(f"  OL {lo}-{hi}: ORF1={o1.median():.3f}(n={len(o1)}), "
              f"ORF2={o2.median():.3f}(n={len(o2)}), "
              f"ratio={o1.median()/o2.median():.2f}x, p={p:.2e}")

print("\n--- ORF1 vs ORF2 DRACH/kb (overlap-length matched) ---")
for lo, hi in [(100, 300), (300, 600), (600, 1000), (1000, 2000)]:
    o1 = orf1_all[(orf1_all['seq_len'] >= lo) & (orf1_all['seq_len'] < hi)]['drach_per_kb']
    o2 = orf2_all[(orf2_all['seq_len'] >= lo) & (orf2_all['seq_len'] < hi)]['drach_per_kb']
    if len(o1) > 5 and len(o2) > 5:
        _, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')
        print(f"  OL {lo}-{hi}: ORF1={o1.median():.2f}(n={len(o1)}), "
              f"ORF2={o2.median():.2f}(n={len(o2)}), "
              f"ratio={o1.median()/o2.median():.2f}x, p={p:.2e}")

# =====================================================================
# 9. Summary table
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6: SUMMARY — m6A/kb DECOMPOSITION")
print("=" * 70)

print(f"\n  m6A/kb = DRACH/kb × per-site rate")
print(f"\n{'Region':<25s} {'DRACH/kb':>9s} {'rate':>7s} {'→ m6A/kb':>9s} {'observed':>9s}")
print("-" * 62)
for r in result_rows:
    predicted = r['median_drach_per_kb'] * r['median_per_site_rate']
    print(f"  {r['region']:<23s} {r['median_drach_per_kb']:>9.2f} {r['median_per_site_rate']:>6.1%} {predicted:>9.2f} {r['median_m6a_per_kb']:>9.2f}")

# Save
pd.DataFrame(result_rows).to_csv(OUTDIR / 'ancient_m6a_decomposition_by_region.tsv', sep='\t', index=False)
print(f"\nResults saved to: {OUTDIR}")
print("Done!")
