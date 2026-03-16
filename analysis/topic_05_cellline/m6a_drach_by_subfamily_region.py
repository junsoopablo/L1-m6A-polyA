#!/usr/bin/env python3
"""DRACH density by L1 subfamily × consensus region.

Question: Is the ORF1 > ORF2 DRACH density pattern in ancient fragments
an ancestral feature preserved from the original L1 subfamily consensus,
or an artifact of random mutation?

If ORF1 > ORF2 DRACH is consistent across diverse ancient subfamilies
(L1MC, L1ME, L1M, L1P etc.), it suggests an ancestral feature.
If it's subfamily-specific, it may reflect different consensus architectures.

Also: compare ancient fragment DRACH density vs L1HS consensus DRACH density
at the same consensus positions to assess whether mutations have changed
motif density.
"""

import pandas as pd
import numpy as np
import pysam
import gzip
import ast
import re
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

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

DRACH_PATTERN = re.compile(r'(?=([AGT][AG]AC[ACT]))')

ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# L1 consensus regions (fraction of consensus length)
REGIONS = [
    ("5'UTR", 0, 0.15),
    ("ORF1", 0.15, 0.32),
    ("ORF2", 0.32, 0.96),
    ("3'UTR", 0.96, 1.0),
]

# Major ancient subfamily groups
SUBFAMILY_GROUPS = {
    'L1PA': ['L1PA4', 'L1PA5', 'L1PA6', 'L1PA7', 'L1PA8', 'L1PA8A',
             'L1PA10', 'L1PA11', 'L1PA12', 'L1PA13', 'L1PA14',
             'L1PA15', 'L1PA16', 'L1PA17'],
    'L1PB': ['L1PB', 'L1PB1', 'L1PB2', 'L1PB3', 'L1PB4'],
    'L1MC': ['L1MC', 'L1MC1', 'L1MC2', 'L1MC3', 'L1MC4', 'L1MC4a',
             'L1MC5', 'L1MC5a', 'L1MCa', 'L1MCb'],
    'L1ME': ['L1ME1', 'L1ME2', 'L1ME2z', 'L1ME3', 'L1ME3A', 'L1ME3B',
             'L1ME3C', 'L1ME3Cz', 'L1ME3D', 'L1ME3E', 'L1ME3F',
             'L1ME3G', 'L1ME4a', 'L1ME4b', 'L1ME4c', 'L1ME5',
             'L1MEa', 'L1MEb', 'L1MEc', 'L1MEd', 'L1MEe', 'L1MEf',
             'L1MEg', 'L1MEg2', 'L1MEh', 'L1MEi'],
    'L1M': ['L1M', 'L1M1', 'L1M2', 'L1M2a', 'L1M2a1', 'L1M2b',
            'L1M2c', 'L1M3', 'L1M3a', 'L1M3b', 'L1M3c', 'L1M3d',
            'L1M3de', 'L1M3e', 'L1M3f', 'L1M4', 'L1M4a1', 'L1M4a2',
            'L1M4b', 'L1M4c', 'L1M5', 'L1M6', 'L1M6B', 'L1M7',
            'L1M8', 'L1Ma1', 'L1Ma2', 'L1Ma3', 'L1Ma4', 'L1Ma5',
            'L1Ma6', 'L1Ma7', 'L1Ma8', 'L1Ma9', 'L1Mb1', 'L1Mb2',
            'L1Mb3', 'L1Mb4', 'L1Mb5', 'L1Mb6', 'L1Mb7', 'L1Mb8'],
}

# =====================================================================
# 1. Build rmsk lookup
# =====================================================================
print("Loading rmsk...")
rmsk_lookup = {}

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

print(f"  {len(rmsk_lookup)} entries")

# =====================================================================
# 2. Load ancient L1 reads
# =====================================================================
print("Loading reads...")
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
    merged['te_length'] = merged['te_end'] - merged['te_start']
    all_reads.append(merged)

df = pd.concat(all_reads, ignore_index=True)
ancient = df[~df['is_young']].copy()
print(f"  Total: {len(df)}, Ancient: {len(ancient)}")

# Match to rmsk
cons_data = []
for _, row in ancient.iterrows():
    key = (row['chr'], row['te_start'], row['te_end'], row['te_strand'], row['gene_id'])
    if key in rmsk_lookup:
        cs, ce, cl = rmsk_lookup[key]
        cons_data.append((cs, ce, cl, (cs + ce) / 2 / cl))
    else:
        cons_data.append((np.nan, np.nan, np.nan, np.nan))

ancient['cons_start'] = [x[0] for x in cons_data]
ancient['cons_end'] = [x[1] for x in cons_data]
ancient['cons_length'] = [x[2] for x in cons_data]
ancient['frac_center'] = [x[3] for x in cons_data]

ancient = ancient[ancient['cons_length'].notna()].copy()
print(f"  Matched: {len(ancient)}")

# Assign region
def assign_region(fc):
    for label, lo, hi in REGIONS:
        if lo <= fc < hi:
            return label
    return 'unknown'

ancient['region'] = ancient['frac_center'].apply(assign_region)

# Assign subfamily group
def assign_subfam_group(gene_id):
    for grp_name, members in SUBFAMILY_GROUPS.items():
        if gene_id in members:
            return grp_name
    # Try prefix match for unlisted members
    for grp_name in ['L1PA', 'L1PB', 'L1MC', 'L1ME', 'L1M']:
        if gene_id.startswith(grp_name):
            return grp_name
    return 'Other'

ancient['subfam_group'] = ancient['gene_id'].apply(assign_subfam_group)

# =====================================================================
# 3. Extract reference sequences and count DRACH
# =====================================================================
print("\nExtracting reference sequences...")
genome = pysam.FastaFile(GENOME)

drach_counts = []
overlap_lengths = []

for idx, (_, row) in enumerate(ancient.iterrows()):
    if (idx + 1) % 5000 == 0:
        print(f"  {idx+1}/{len(ancient)}...")

    chrom = row['chr']
    te_start = int(row['te_start'])
    te_end = int(row['te_end'])
    read_start = int(row['start'])
    read_end = int(row['end'])

    # L1 overlap region
    ol_start = max(te_start, read_start)
    ol_end = min(te_end, read_end)
    ol_len = ol_end - ol_start

    if ol_len <= 10:
        drach_counts.append(0)
        overlap_lengths.append(ol_len)
        continue

    try:
        seq = genome.fetch(chrom, ol_start, ol_end).upper()
        matches = DRACH_PATTERN.findall(seq)
        drach_counts.append(len(matches))
        overlap_lengths.append(len(seq))
    except:
        drach_counts.append(0)
        overlap_lengths.append(ol_len)

ancient['drach_count'] = drach_counts
ancient['overlap_length'] = overlap_lengths
ancient['drach_per_kb'] = ancient['drach_count'] / ancient['overlap_length'] * 1000

# Count m6A within L1 only
def count_m6a_in_l1(row):
    pos_str = row['m6a_positions']
    if pd.isna(pos_str) or pos_str == '[]':
        return 0
    try:
        positions = ast.literal_eval(pos_str) if isinstance(pos_str, str) else []
    except:
        return []

    te_start = int(row['te_start'])
    te_end = int(row['te_end'])
    read_start = int(row['start'])

    count = 0
    for p in positions:
        genomic_pos = read_start + p
        if te_start <= genomic_pos <= te_end:
            count += 1
    return count

print("  Counting m6A within L1 boundaries...")
ancient['m6a_in_l1'] = ancient.apply(count_m6a_in_l1, axis=1)
ancient['m6a_in_l1_per_kb'] = ancient['m6a_in_l1'] / ancient['overlap_length'] * 1000

# Per-site rate
ancient['persite_rate'] = np.where(
    ancient['drach_count'] > 0,
    ancient['m6a_in_l1'] / ancient['drach_count'],
    np.nan
)

genome.close()

# =====================================================================
# 4. DRACH density by subfamily × region
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 1: DRACH/kb BY SUBFAMILY GROUP × CONSENSUS REGION")
print("=" * 80)

# Get subfamily group counts
subfam_counts = ancient['subfam_group'].value_counts()
print(f"\nSubfamily group sizes:")
for sg, cnt in subfam_counts.items():
    print(f"  {sg}: {cnt}")

print(f"\n{'SubfamGroup':<12s} {'Region':<8s} {'n':>6s} {'DRACH/kb':>9s} {'m6A/kb':>8s} {'rate':>7s}")
print("-" * 55)

results = []
for sg in ['L1PA', 'L1PB', 'L1MC', 'L1ME', 'L1M', 'Other']:
    sg_data = ancient[ancient['subfam_group'] == sg]
    if len(sg_data) < 50:
        continue
    for label, lo, hi in REGIONS:
        sub = sg_data[sg_data['region'] == label]
        if len(sub) < 10:
            continue
        drach_kb = sub['drach_per_kb'].median()
        m6a_kb = sub['m6a_in_l1_per_kb'].median()
        rate = sub['persite_rate'].median()
        print(f"  {sg:<12s} {label:<8s} {len(sub):>6d} {drach_kb:>9.1f} {m6a_kb:>8.2f} {rate:>7.1%}")
        results.append({
            'subfam_group': sg, 'region': label,
            'n': len(sub), 'drach_per_kb': drach_kb,
            'm6a_per_kb': m6a_kb, 'persite_rate': rate
        })

# =====================================================================
# 5. Is ORF1 > ORF2 DRACH universal across subfamilies?
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 2: ORF1 vs ORF2 DRACH/kb BY SUBFAMILY")
print("=" * 80)

print(f"\n{'SubfamGroup':<12s} {'ORF1_DRACH':>11s} {'ORF2_DRACH':>11s} {'ratio':>7s} {'p':>10s} {'ORF1_n':>7s} {'ORF2_n':>7s}")
print("-" * 75)

for sg in ['L1PA', 'L1PB', 'L1MC', 'L1ME', 'L1M', 'Other']:
    sg_data = ancient[ancient['subfam_group'] == sg]
    orf1 = sg_data[sg_data['region'] == 'ORF1']['drach_per_kb']
    orf2 = sg_data[sg_data['region'] == 'ORF2']['drach_per_kb']

    if len(orf1) >= 10 and len(orf2) >= 10:
        _, p = stats.mannwhitneyu(orf1, orf2, alternative='two-sided')
        ratio = orf1.median() / orf2.median()
        print(f"  {sg:<12s} {orf1.median():>11.1f} {orf2.median():>11.1f} {ratio:>7.2f}x {p:>10.2e} {len(orf1):>7d} {len(orf2):>7d}")

# =====================================================================
# 6. Per-site rate ORF1 vs ORF2 by subfamily
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 3: ORF1 vs ORF2 PER-SITE RATE BY SUBFAMILY")
print("=" * 80)

print(f"\n{'SubfamGroup':<12s} {'ORF1_rate':>10s} {'ORF2_rate':>10s} {'ratio':>7s} {'p':>10s}")
print("-" * 55)

for sg in ['L1PA', 'L1PB', 'L1MC', 'L1ME', 'L1M', 'Other']:
    sg_data = ancient[ancient['subfam_group'] == sg]
    orf1 = sg_data[(sg_data['region'] == 'ORF1') & sg_data['persite_rate'].notna()]['persite_rate']
    orf2 = sg_data[(sg_data['region'] == 'ORF2') & sg_data['persite_rate'].notna()]['persite_rate']

    if len(orf1) >= 10 and len(orf2) >= 10:
        _, p = stats.mannwhitneyu(orf1, orf2, alternative='two-sided')
        ratio = orf1.median() / orf2.median()
        print(f"  {sg:<12s} {orf1.median():>10.1%} {orf2.median():>10.1%} {ratio:>7.2f}x {p:>10.2e}")

# =====================================================================
# 7. Overall: age gradient within subfamilies
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 4: DRACH/kb BY EVOLUTIONARY AGE (SUBFAMILY GROUP)")
print("=" * 80)

# Approximate evolutionary order: L1PA (youngest ancient) → L1PB → L1MC → L1ME → L1M (oldest)
age_order = ['L1PA', 'L1PB', 'L1MC', 'L1ME', 'L1M']

print(f"\n{'SubfamGroup':<12s} {'n':>6s} {'DRACH/kb':>9s} {'m6A/kb':>8s} {'rate':>7s} {'age_approx':>12s}")
print("-" * 60)

for sg in age_order:
    sg_data = ancient[ancient['subfam_group'] == sg]
    if len(sg_data) < 50:
        continue
    valid = sg_data[sg_data['persite_rate'].notna()]
    print(f"  {sg:<12s} {len(sg_data):>6d} {sg_data['drach_per_kb'].median():>9.1f} "
          f"{sg_data['m6a_in_l1_per_kb'].median():>8.2f} {valid['persite_rate'].median():>7.1%} "
          f"{'~20 Mya':>12s}" if sg == 'L1PA' else
          f"  {sg:<12s} {len(sg_data):>6d} {sg_data['drach_per_kb'].median():>9.1f} "
          f"{sg_data['m6a_in_l1_per_kb'].median():>8.2f} {valid['persite_rate'].median():>7.1%} "
          f"{'~40 Mya':>12s}" if sg == 'L1PB' else
          f"  {sg:<12s} {len(sg_data):>6d} {sg_data['drach_per_kb'].median():>9.1f} "
          f"{sg_data['m6a_in_l1_per_kb'].median():>8.2f} {valid['persite_rate'].median():>7.1%} "
          f"{'~80 Mya':>12s}" if sg == 'L1MC' else
          f"  {sg:<12s} {len(sg_data):>6d} {sg_data['drach_per_kb'].median():>9.1f} "
          f"{sg_data['m6a_in_l1_per_kb'].median():>8.2f} {valid['persite_rate'].median():>7.1%} "
          f"{'~100 Mya':>12s}" if sg == 'L1ME' else
          f"  {sg:<12s} {len(sg_data):>6d} {sg_data['drach_per_kb'].median():>9.1f} "
          f"{sg_data['m6a_in_l1_per_kb'].median():>8.2f} {valid['persite_rate'].median():>7.1%} "
          f"{'~150 Mya':>12s}")

# =====================================================================
# 8. Key comparison: Do mutations change DRACH density over evolutionary time?
# =====================================================================
print("\n" + "=" * 80)
print("SECTION 5: DRACH/kb vs EVOLUTIONARY AGE — Neutral expectation")
print("=" * 80)

# If mutations are random, AT→GC and GC→AT roughly balance, so DRACH density
# should drift randomly. Over very long time (L1M ~150 Mya), DRACH density
# should converge toward random expectation.
# Random DRACH density for human genome (40% GC): ~2.8/kb from DRACH probability
# = P(AGT) × P(AG) × P(A) × P(C) × P(ACT) = 0.6 × 0.4 × 0.3 × 0.2 × 0.6 / 2 ≈ 4.3/kb
# (per strand, with overlapping search)

print("\nDRACH/kb should converge toward genome-average with evolutionary time")
print("if mutations are neutral w.r.t. DRACH creation/destruction.\n")

for sg in age_order:
    sg_data = ancient[ancient['subfam_group'] == sg]
    if len(sg_data) < 50:
        continue
    for label in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
        sub = sg_data[sg_data['region'] == label]
        if len(sub) >= 10:
            d = sub['drach_per_kb'].median()
            bar = '█' * int(d / 2)
            print(f"  {sg:<6s} {label:<6s} {d:>6.1f} {bar}")
    print()

# =====================================================================
# 9. Save results
# =====================================================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTDIR / 'drach_by_subfamily_region.tsv', sep='\t', index=False)

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Question: Is ORF1 > ORF2 DRACH density an ancestral feature or mutation artifact?

If ORF1 > ORF2 is consistent across all ancient subfamilies (L1PA→L1M),
it's likely an ancestral feature preserved through fragmentation.

If only specific subfamilies show this pattern, it may reflect
different consensus architectures across L1 subfamilies.
""")

print(f"\nResults saved to: {OUTDIR}")
print("Done!")
