#!/usr/bin/env python3
"""m6A by L1 consensus region — CORRECTED: count only m6A sites WITHIN L1 element.

Previous analysis (m6a_consensus_position.py) used m6A/kb of the full read,
which includes flanking non-L1 sequence. This inflates/deflates estimates
depending on flanking region m6A density.

Correction: only count m6A sites whose genomic position falls within
[te_start, te_end], and normalize by L1 overlap length (not read length).

Also decompose into: (1) DRACH motif density within L1, (2) per-site rate.
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
EXCLUDE_PREFIXES = ('HeLa-Ars', 'MCF7-EV')

ALL_GROUPS = sorted([
    f.stem.replace('_l1_per_read', '')
    for f in CACHE_DIR.glob('*_l1_per_read.tsv')
    if not any(f.stem.startswith(ex) for ex in EXCLUDE_PREFIXES)
])

# =====================================================================
# 1. Build rmsk lookup
# =====================================================================
print("Loading UCSC rmsk L1 consensus positions...")
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

print(f"  Loaded {len(rmsk_lookup)} L1 entries")

# =====================================================================
# 2. Load and merge data
# =====================================================================
print("\nLoading L1 reads...")
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
# 3. Match rmsk consensus positions
# =====================================================================
print("\nMatching rmsk consensus positions...")
cons_data = []
for _, row in df.iterrows():
    key = (row['chr'], row['te_start'], row['te_end'], row['te_strand'], row['gene_id'])
    if key in rmsk_lookup:
        cons_data.append(rmsk_lookup[key])
    else:
        cons_data.append((np.nan, np.nan, np.nan))

df['cons_start'] = [c[0] for c in cons_data]
df['cons_end'] = [c[1] for c in cons_data]
df['cons_length'] = [c[2] for c in cons_data]

df_m = df[df['cons_length'].notna()].copy()
df_m['frac_center'] = ((df_m['cons_start'] + df_m['cons_end']) / 2) / df_m['cons_length']
print(f"  Matched: {len(df_m)}")

# =====================================================================
# 4. Count m6A sites WITHIN L1 element only
# =====================================================================
print("\nCounting m6A sites within L1 boundaries...")

def count_m6a_within_l1(row):
    """Count m6A sites that fall within [te_start, te_end]."""
    pos_str = row['m6a_positions']
    if pd.isna(pos_str) or pos_str == '[]':
        return 0

    try:
        positions = ast.literal_eval(pos_str)
    except (ValueError, SyntaxError):
        return 0

    if not positions:
        return 0

    read_start = row['start']  # genomic start of alignment
    te_start = row['te_start']
    te_end = row['te_end']

    count = 0
    for p in positions:
        genomic_pos = read_start + p
        if te_start <= genomic_pos <= te_end:
            count += 1
    return count

df_m['m6a_in_l1'] = df_m.apply(count_m6a_within_l1, axis=1)

# L1-internal m6A/kb (normalized by TE overlap length, not read length)
# Use overlap = min(read_end, te_end) - max(read_start, te_start)
df_m['overlap_start'] = df_m[['start', 'te_start']].max(axis=1)
df_m['overlap_end'] = df_m[['end', 'te_end']].min(axis=1)
df_m['overlap_len'] = (df_m['overlap_end'] - df_m['overlap_start']).clip(lower=1)
df_m['m6a_in_l1_per_kb'] = df_m['m6a_in_l1'] / df_m['overlap_len'] * 1000

# Also compute full-read m6A/kb for comparison
df_m['m6a_full_per_kb'] = df_m['m6a_sites_high'] / df_m['read_length'] * 1000

# =====================================================================
# 5. Ancient L1: m6A within L1 by consensus region
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 1: ANCIENT L1 — m6A WITHIN L1 BY CONSENSUS REGION")
print("=" * 70)

ancient = df_m[df_m['age'] == 'ancient'].copy()

regions = [
    ("5'UTR (0-15%)", 0, 0.15),
    ("ORF1 (15-32%)", 0.15, 0.32),
    ("ORF2-5' (32-60%)", 0.32, 0.60),
    ("ORF2-3' (60-96%)", 0.60, 0.96),
    ("3'UTR (96-100%)", 0.96, 1.0),
]

print(f"\nAncient L1: n={len(ancient)}")
print(f"\n{'Region':<25s} {'n':>6s} {'m6A_L1/kb':>10s} {'m6A_full/kb':>12s} {'med_OL':>7s} {'med_RL':>7s} {'med_TE':>7s} {'m6A_in':>6s} {'m6A_all':>7s}")
print("-" * 95)

region_stats = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 0:
        m6a_l1_kb = sub['m6a_in_l1_per_kb'].median()
        m6a_full_kb = sub['m6a_full_per_kb'].median()
        ol = sub['overlap_len'].median()
        rl = sub['read_length'].median()
        te = sub['te_length'].median()
        m6a_in = sub['m6a_in_l1'].median()
        m6a_all = sub['m6a_sites_high'].median()
        print(f"  {label:<23s} {len(sub):>6d} {m6a_l1_kb:>10.2f} {m6a_full_kb:>12.2f} {ol:>7.0f} {rl:>7.0f} {te:>7.0f} {m6a_in:>6.1f} {m6a_all:>7.1f}")
        region_stats.append({
            'region': label,
            'n_reads': len(sub),
            'median_m6a_in_l1_per_kb': m6a_l1_kb,
            'median_m6a_full_per_kb': m6a_full_kb,
            'median_overlap_len': ol,
            'median_read_length': rl,
            'median_te_length': te,
            'median_m6a_in_l1': m6a_in,
            'median_m6a_total': m6a_all,
        })

# KW test on L1-internal m6A/kb
groups_kw = []
labels_kw = []
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 10:
        groups_kw.append(sub['m6a_in_l1_per_kb'].values)
        labels_kw.append(label)

if len(groups_kw) >= 2:
    kw_stat, kw_p = stats.kruskal(*groups_kw)
    print(f"\n  Kruskal-Wallis (m6A within L1 per kb): H={kw_stat:.2f}, p={kw_p:.2e}")

# =====================================================================
# 6. Pairwise comparisons
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: PAIRWISE COMPARISONS (L1-internal m6A/kb)")
print("=" * 70)

for i in range(len(regions)):
    for j in range(i+1, len(regions)):
        li, loi, hii = regions[i]
        lj, loj, hij = regions[j]
        si = ancient[(ancient['frac_center'] >= loi) & (ancient['frac_center'] < hii)]['m6a_in_l1_per_kb']
        sj = ancient[(ancient['frac_center'] >= loj) & (ancient['frac_center'] < hij)]['m6a_in_l1_per_kb']
        if len(si) > 10 and len(sj) > 10:
            _, p = stats.mannwhitneyu(si, sj, alternative='two-sided')
            ratio = sj.median() / si.median() if si.median() > 0 else np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {li} vs {lj}: {si.median():.2f} vs {sj.median():.2f} = {ratio:.2f}x, p={p:.2e} {sig}")

# =====================================================================
# 7. Read-length and overlap-length controlled
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: OVERLAP-LENGTH CONTROLLED COMPARISON")
print("=" * 70)

# Compare ORF1 vs ORF2 within matched overlap-length bins
orf1 = ancient[(ancient['frac_center'] >= 0.15) & (ancient['frac_center'] < 0.32)]
orf2 = ancient[(ancient['frac_center'] >= 0.32) & (ancient['frac_center'] < 0.96)]
utr5 = ancient[(ancient['frac_center'] >= 0) & (ancient['frac_center'] < 0.15)]
utr3 = ancient[(ancient['frac_center'] >= 0.96) & (ancient['frac_center'] < 1.0)]

print("\n--- ORF1 vs ORF2 (overlap-length matched) ---")
for lo, hi in [(50, 200), (200, 500), (500, 1000), (1000, 2000)]:
    o1 = orf1[(orf1['overlap_len'] >= lo) & (orf1['overlap_len'] < hi)]['m6a_in_l1_per_kb']
    o2 = orf2[(orf2['overlap_len'] >= lo) & (orf2['overlap_len'] < hi)]['m6a_in_l1_per_kb']
    if len(o1) > 5 and len(o2) > 5:
        _, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')
        ratio = o1.median() / o2.median() if o2.median() > 0 else np.nan
        print(f"  OL {lo}-{hi}: ORF1={o1.median():.2f}(n={len(o1)}), "
              f"ORF2={o2.median():.2f}(n={len(o2)}), "
              f"ratio={ratio:.2f}x, p={p:.2e}")

print("\n--- 5'UTR vs ORF2 (overlap-length matched) ---")
for lo, hi in [(50, 200), (200, 500), (500, 1000)]:
    u5 = utr5[(utr5['overlap_len'] >= lo) & (utr5['overlap_len'] < hi)]['m6a_in_l1_per_kb']
    o2 = orf2[(orf2['overlap_len'] >= lo) & (orf2['overlap_len'] < hi)]['m6a_in_l1_per_kb']
    if len(u5) > 5 and len(o2) > 5:
        _, p = stats.mannwhitneyu(u5, o2, alternative='two-sided')
        ratio = u5.median() / o2.median() if o2.median() > 0 else np.nan
        print(f"  OL {lo}-{hi}: 5'UTR={u5.median():.2f}(n={len(u5)}), "
              f"ORF2={o2.median():.2f}(n={len(o2)}), "
              f"ratio={ratio:.2f}x, p={p:.2e}")

# =====================================================================
# 8. 10-bin fine resolution (L1-internal only)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: 10-BIN PROFILE (L1-internal m6A/kb)")
print("=" * 70)

print(f"\n{'Bin':<12s} {'n':>6s} {'m6A_L1/kb':>10s} {'m6A_full/kb':>12s} {'OL':>6s}")
print("-" * 50)

fine_stats = []
for i in range(10):
    lo = i * 0.1
    hi = (i + 1) * 0.1
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 0:
        m6a_l1 = sub['m6a_in_l1_per_kb'].median()
        m6a_full = sub['m6a_full_per_kb'].median()
        ol = sub['overlap_len'].median()
        bar = '#' * int(m6a_l1 / 0.5)
        print(f"  {lo:.1f}-{hi:.1f} {len(sub):>6d} {m6a_l1:>10.2f} {m6a_full:>12.2f} {ol:>6.0f} {bar}")
        fine_stats.append({
            'bin': f'{lo:.1f}-{hi:.1f}',
            'n_reads': len(sub),
            'median_m6a_in_l1_per_kb': m6a_l1,
            'median_m6a_full_per_kb': m6a_full,
            'median_overlap_len': ol,
        })

# =====================================================================
# 9. Decomposition: is it motif density or per-site rate?
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: FRACTION OF READ COVERED BY L1")
print("=" * 70)

# Check if L1 fraction of read differs by region
for label, lo, hi in regions:
    sub = ancient[(ancient['frac_center'] >= lo) & (ancient['frac_center'] < hi)]
    if len(sub) > 0:
        l1_frac = (sub['overlap_len'] / sub['read_length']).median()
        flank_m6a = (sub['m6a_sites_high'] - sub['m6a_in_l1']) / (sub['read_length'] - sub['overlap_len']).clip(lower=1) * 1000
        print(f"  {label:<23s}: L1 frac={l1_frac:.2f}, "
              f"L1 m6A/kb={sub['m6a_in_l1_per_kb'].median():.2f}, "
              f"flanking m6A/kb={flank_m6a.median():.2f}")

# =====================================================================
# 10. Save results
# =====================================================================
pd.DataFrame(region_stats).to_csv(OUTDIR / 'ancient_m6a_corrected_by_region.tsv', sep='\t', index=False)
pd.DataFrame(fine_stats).to_csv(OUTDIR / 'ancient_m6a_corrected_10bin.tsv', sep='\t', index=False)

print(f"\nResults saved to: {OUTDIR}")
print("Done!")
