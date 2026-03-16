#!/usr/bin/env python3
"""
L1 Consensus Hotspot m6A Analysis
===================================
Tests whether Ancient L1 DRACH motifs at Young-L1-hotspot consensus positions
show higher m6A rates than other DRACH positions.

Decomposition:
  Part A: Map dorado m6A calls to L1 consensus coordinates
  Part B: Identify Young L1 m6A hotspot positions
  Part C: Ancient L1 m6A rate at hotspot vs non-hotspot DRACH
  Part D: Flanking sequence conservation → m6A rate correlation

Usage:
  conda run -n research python consensus_hotspot_m6a_analysis.py
"""

import os
import re
import gzip
import sys
from pathlib import Path
from collections import defaultdict, Counter
from bisect import bisect_left

import numpy as np
import pandas as pd
import pysam
from scipy import stats

# ============================================================
# Configuration
# ============================================================
PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
REF_PATH = str(PROJECT / 'reference/Human.fasta')
RMSK_PATH = str(PROJECT / 'reference/rmsk.txt.gz')
L1_BED = str(PROJECT / 'reference/L1_TE_L1_family.bed')
BAM_PATH = "/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/HeLa_1_1.dorado.m6A.sorted.bam"
TE_GTF = str(PROJECT / 'reference/hg38_rmsk_TE.gtf')

PERREAD_TSV = PROJECT / 'analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results/dorado_per_read_m6a.tsv.gz'
OUTDIR = PROJECT / 'analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results'
OUTDIR.mkdir(parents=True, exist_ok=True)

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}
M6A_PROB_THRESHOLD = 204
DRACH_RE = re.compile(r"^[AGT][AG]AC[ACT]$")

# L1HS consensus regions (1-based)
L1HS_REGIONS = {
    "5'UTR": (1, 910),
    "ORF1": (911, 1924),
    "ORF2": (1991, 5817),
    "3'UTR": (5818, 6064),
}
L1HS_LENGTH = 6064

np.random.seed(42)


# ============================================================
# 1. Load rmsk consensus coordinate lookup
# ============================================================
def load_rmsk_l1():
    """Load L1 entries from rmsk.txt.gz with consensus coordinates."""
    print("Loading rmsk.txt.gz for L1 consensus coordinates...")
    lookup = {}  # (chrom, geno_start, geno_end) -> dict
    n = 0

    with gzip.open(RMSK_PATH, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 17:
                continue
            rep_class = fields[11]
            rep_family = fields[12]
            if rep_class != 'LINE' or rep_family != 'L1':
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

            key = (chrom, geno_start, geno_end)
            lookup[key] = {
                'strand': strand,
                'rep_name': rep_name,
                'cons_start': cons_start,
                'cons_end': cons_end,
                'cons_length': cons_length,
            }
            n += 1

    print(f"  Loaded {n:,} L1 entries")
    return lookup


# ============================================================
# 2. Build L1 element interval index for read-to-element matching
# ============================================================
def build_l1_intervals():
    """Build sorted interval list for matching reads to L1 elements."""
    print("Building L1 interval index...")
    intervals = defaultdict(list)  # chrom -> [(start, end, repname)]

    with open(L1_BED) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            chrom, start, end, repname = parts[0], int(parts[1]), int(parts[2]), parts[3]
            intervals[chrom].append((start, end, repname))

    for chrom in intervals:
        intervals[chrom].sort()

    n = sum(len(v) for v in intervals.values())
    print(f"  {n:,} L1 elements indexed")
    return intervals


def find_best_l1(intervals, chrom, read_start, read_end):
    """Find the best-overlapping L1 element for a read."""
    if chrom not in intervals:
        return None
    elems = intervals[chrom]
    idx = bisect_left(elems, (read_start,))
    start_idx = max(0, idx - 1)

    best_overlap, best_elem = 0, None
    for i in range(start_idx, len(elems)):
        es, ee, rn = elems[i]
        if es >= read_end:
            break
        ov = min(read_end, ee) - max(read_start, es)
        if ov > best_overlap:
            best_overlap = ov
            best_elem = (es, ee, rn)

    return best_elem


# ============================================================
# 3. Parse m6A from dorado MM/ML (with correct reverse-strand handling)
# ============================================================
def parse_m6a_positions(read):
    """
    Parse A+a modification from dorado MM/ML.
    Returns: dict {read_seq_position: ml_probability}
    """
    if not read.has_tag("MM") or not read.has_tag("ML"):
        return {}

    mm_str = read.get_tag("MM")
    ml_arr = list(read.get_tag("ML"))
    seq = read.query_sequence
    if seq is None:
        return {}

    is_reverse = read.is_reverse
    mod_sections = [s.strip() for s in mm_str.rstrip(";").split(";") if s.strip()]
    ml_offset = 0
    m6a_map = {}

    for section in mod_sections:
        tokens = section.split(",")
        header = tokens[0]
        skip_values = [int(x) for x in tokens[1:]] if len(tokens) > 1 else []
        n_mods = len(skip_values)

        if header.startswith("A+a"):
            target_base = "T" if is_reverse else "A"
            base_positions = [i for i, base in enumerate(seq) if base == target_base]
            a_idx = 0
            for mod_i, skip in enumerate(skip_values):
                a_idx += skip
                if a_idx < len(base_positions):
                    read_pos = base_positions[a_idx]
                    prob = ml_arr[ml_offset + mod_i]
                    m6a_map[read_pos] = prob
                a_idx += 1

        ml_offset += n_mods

    return m6a_map


# ============================================================
# 4. Map genomic m6A position to consensus position
# ============================================================
def genomic_to_consensus(genomic_pos, l1_geno_start, l1_geno_end, l1_strand,
                         cons_start, cons_end, cons_length):
    """
    Convert a genomic position within an L1 element to consensus coordinate.
    Returns consensus position (1-based) or None if outside.
    """
    if genomic_pos < l1_geno_start or genomic_pos >= l1_geno_end:
        return None

    if l1_strand == '+':
        # Same direction: offset from geno_start maps to cons_start
        offset = genomic_pos - l1_geno_start
        cons_pos = cons_start + offset
    else:
        # Reverse: offset from geno_end maps to cons_start
        offset = l1_geno_end - 1 - genomic_pos
        cons_pos = cons_start + offset

    if cons_pos < 0 or cons_pos > cons_length:
        return None

    return cons_pos


# ============================================================
# 5. Main analysis
# ============================================================
def main():
    print("L1 Consensus Hotspot m6A Analysis")
    print("=" * 70)

    # Load resources
    rmsk = load_rmsk_l1()
    l1_intervals = build_l1_intervals()
    ref_fasta = pysam.FastaFile(REF_PATH)

    # Load per-read data for read IDs and categories
    print("\nLoading per-read data...")
    perread = pd.read_csv(PERREAD_TSV, sep='\t', compression='gzip')
    l1_reads = perread[perread['category'] == 'L1']
    target_ids = set(l1_reads['read_id'].values)
    read_info = l1_reads.set_index('read_id')[['age_class', 'subfamily']].to_dict('index')
    print(f"  L1 reads to process: {len(target_ids):,}")

    # ============================================================
    # Scan BAM: collect per-position m6A data
    # ============================================================
    print("\nScanning BAM for per-position m6A data...")

    # Store: list of (read_id, age_class, subfamily, cons_pos, cons_length,
    #                 ref_5mer, is_drach, m6a_prob, is_methylated,
    #                 flanking_11mer)
    site_records = []
    # Also collect per-read ALL DRACH positions (including non-methylated)
    drach_records = []

    n_found = 0
    n_total = 0

    with pysam.AlignmentFile(BAM_PATH, 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            n_total += 1
            if n_total % 500000 == 0:
                print(f"  ... {n_total:,} reads scanned, {n_found:,}/{len(target_ids):,} found")

            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            rid = read.query_name
            if rid not in target_ids:
                continue

            n_found += 1
            info = read_info[rid]
            age = info['age_class']
            subfam = info['subfamily']

            chrom = read.reference_name
            ref_start = read.reference_start
            ref_end = read.reference_end
            is_reverse = read.is_reverse

            if ref_end is None or ref_start is None:
                continue

            # Find best L1 element
            best = find_best_l1(l1_intervals, chrom, ref_start, ref_end)
            if best is None:
                continue
            l1_start, l1_end, l1_repname = best

            # Get consensus coordinates from rmsk
            rmsk_key = (chrom, l1_start, l1_end)
            if rmsk_key not in rmsk:
                continue
            rmsk_info = rmsk[rmsk_key]
            l1_strand = rmsk_info['strand']
            cons_start = rmsk_info['cons_start']
            cons_end = rmsk_info['cons_end']
            cons_length = rmsk_info['cons_length']

            # Get aligned pairs for read→ref mapping
            pairs = read.get_aligned_pairs(matches_only=False)
            read_to_ref = {}
            for qpos, rpos in pairs:
                if qpos is not None and rpos is not None:
                    read_to_ref[qpos] = rpos

            # Parse m6A
            m6a_map = parse_m6a_positions(read)

            # Get reference sequence for the L1 element region
            try:
                l1_ref_seq = ref_fasta.fetch(chrom, l1_start, l1_end).upper()
            except (ValueError, KeyError):
                continue

            # Orient to RNA strand
            if l1_strand == '-':
                comp_table = str.maketrans("ACGT", "TGCA")
                l1_ref_seq_oriented = l1_ref_seq.translate(comp_table)[::-1]
            else:
                l1_ref_seq_oriented = l1_ref_seq

            # For each m6A called position, map to consensus
            for read_pos, prob in m6a_map.items():
                ref_pos = read_to_ref.get(read_pos)
                if ref_pos is None:
                    continue

                # Only consider positions within the L1 element
                if ref_pos < l1_start or ref_pos >= l1_end:
                    continue

                # Get 5-mer context (oriented to RNA strand)
                ctx = None
                try:
                    ctx_start = ref_pos - 2
                    ctx_end = ref_pos + 3
                    if ctx_start >= 0:
                        raw_ctx = ref_fasta.fetch(chrom, ctx_start, ctx_end).upper()
                        if len(raw_ctx) == 5 and 'N' not in raw_ctx:
                            if is_reverse:
                                comp_table = str.maketrans("ACGT", "TGCA")
                                ctx = raw_ctx.translate(comp_table)[::-1]
                            else:
                                ctx = raw_ctx
                except (ValueError, KeyError):
                    pass

                # Get 11-mer flanking context
                flanking = None
                try:
                    fl_start = ref_pos - 5
                    fl_end = ref_pos + 6
                    if fl_start >= 0:
                        raw_fl = ref_fasta.fetch(chrom, fl_start, fl_end).upper()
                        if len(raw_fl) == 11 and 'N' not in raw_fl:
                            if is_reverse:
                                comp_table = str.maketrans("ACGT", "TGCA")
                                flanking = raw_fl.translate(comp_table)[::-1]
                            else:
                                flanking = raw_fl
                except (ValueError, KeyError):
                    pass

                is_drach = bool(ctx and DRACH_RE.match(ctx))
                is_methylated = prob >= M6A_PROB_THRESHOLD

                # Map to consensus
                cpos = genomic_to_consensus(ref_pos, l1_start, l1_end, l1_strand,
                                           cons_start, cons_end, cons_length)

                site_records.append({
                    'read_id': rid,
                    'age_class': age,
                    'subfamily': subfam,
                    'genomic_pos': ref_pos,
                    'cons_pos': cpos,
                    'cons_length': cons_length,
                    'cons_frac': cpos / cons_length if cpos and cons_length > 0 else None,
                    'ref_5mer': ctx,
                    'flanking_11mer': flanking,
                    'is_drach': is_drach,
                    'm6a_prob': prob,
                    'is_methylated': is_methylated,
                })

            # Now scan ALL DRACH motifs in the L1 reference region covered by this read
            # to get the denominator (total DRACH regardless of methylation)
            for read_pos, ref_pos in read_to_ref.items():
                if ref_pos is None or ref_pos < l1_start or ref_pos >= l1_end:
                    continue

                # Get 5-mer context
                try:
                    ctx_start = ref_pos - 2
                    ctx_end = ref_pos + 3
                    if ctx_start < 0:
                        continue
                    raw_ctx = ref_fasta.fetch(chrom, ctx_start, ctx_end).upper()
                    if len(raw_ctx) != 5 or 'N' in raw_ctx:
                        continue
                    if is_reverse:
                        comp_table = str.maketrans("ACGT", "TGCA")
                        ctx = raw_ctx.translate(comp_table)[::-1]
                    else:
                        ctx = raw_ctx
                except (ValueError, KeyError):
                    continue

                # Only process if center base is A (potential m6A target)
                if ctx[2] != 'A':
                    continue

                if not DRACH_RE.match(ctx):
                    continue

                # This is a DRACH A position — check if methylated
                is_meth = False
                if read_pos in m6a_map and m6a_map[read_pos] >= M6A_PROB_THRESHOLD:
                    is_meth = True

                cpos = genomic_to_consensus(ref_pos, l1_start, l1_end, l1_strand,
                                           cons_start, cons_end, cons_length)

                # 11-mer flanking
                flanking = None
                try:
                    fl_start = ref_pos - 5
                    fl_end = ref_pos + 6
                    if fl_start >= 0:
                        raw_fl = ref_fasta.fetch(chrom, fl_start, fl_end).upper()
                        if len(raw_fl) == 11 and 'N' not in raw_fl:
                            if is_reverse:
                                comp_table = str.maketrans("ACGT", "TGCA")
                                flanking = raw_fl.translate(comp_table)[::-1]
                            else:
                                flanking = raw_fl
                except (ValueError, KeyError):
                    pass

                drach_records.append({
                    'read_id': rid,
                    'age_class': age,
                    'subfamily': subfam,
                    'genomic_pos': ref_pos,
                    'cons_pos': cpos,
                    'cons_length': cons_length,
                    'cons_frac': cpos / cons_length if cpos and cons_length > 0 else None,
                    'ref_5mer': ctx,
                    'flanking_11mer': flanking,
                    'is_methylated': is_meth,
                })

            if n_found >= len(target_ids):
                break

    ref_fasta.close()

    print(f"\n  m6A called sites in L1: {len(site_records):,}")
    print(f"  All DRACH positions in L1: {len(drach_records):,}")

    sites_df = pd.DataFrame(site_records)
    drach_df = pd.DataFrame(drach_records)

    # Save intermediate
    drach_df.to_csv(OUTDIR / 'consensus_drach_positions.tsv.gz', sep='\t',
                    index=False, compression='gzip')

    # ============================================================
    # Part A: Young L1 consensus position m6A landscape
    # ============================================================
    print("\n" + "=" * 70)
    print("PART A: Young L1 Consensus m6A Hotspots")
    print("=" * 70)

    young_drach = drach_df[drach_df['age_class'] == 'Young'].copy()
    young_drach = young_drach[young_drach['cons_pos'].notna()].copy()
    print(f"  Young L1 DRACH positions: {len(young_drach):,}")

    # Bin consensus positions (50bp bins)
    BIN_SIZE = 50
    young_drach['cons_bin'] = (young_drach['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE

    # Per-bin methylation rate
    bin_stats = young_drach.groupby('cons_bin').agg(
        n_total=('is_methylated', 'count'),
        n_meth=('is_methylated', 'sum'),
    ).reset_index()
    bin_stats['meth_rate'] = bin_stats['n_meth'] / bin_stats['n_total']
    bin_stats = bin_stats[bin_stats['n_total'] >= 5]  # require >= 5 observations

    # Top hotspots
    hotspot_bins = bin_stats.nlargest(20, 'meth_rate')
    print(f"\n  Top 20 Young L1 m6A hotspot bins (50bp, ≥5 obs):")
    print(f"  {'Bin':>6s} {'Region':>12s} {'N':>5s} {'Meth':>5s} {'Rate':>7s}")
    print("  " + "-" * 40)

    hotspot_set = set()
    for _, row in hotspot_bins.iterrows():
        bin_pos = int(row['cons_bin'])
        frac = bin_pos / L1HS_LENGTH
        region = "unknown"
        for rname, (rs, re) in L1HS_REGIONS.items():
            if rs <= bin_pos <= re:
                region = rname
                break
        print(f"  {bin_pos:>6d} {region:>12s} {int(row['n_total']):>5d} "
              f"{int(row['n_meth']):>5d} {row['meth_rate']:>7.3f}")
        hotspot_set.add(bin_pos)

    # Overall Young rate
    y_rate = young_drach['is_methylated'].mean()
    print(f"\n  Young L1 overall DRACH methylation rate: {y_rate:.4f}")

    # ============================================================
    # Part B: Ancient L1 — hotspot vs non-hotspot DRACH
    # ============================================================
    print("\n" + "=" * 70)
    print("PART B: Ancient L1 — Hotspot vs Non-Hotspot DRACH Positions")
    print("=" * 70)

    ancient_drach = drach_df[drach_df['age_class'] == 'Ancient'].copy()
    ancient_drach = ancient_drach[ancient_drach['cons_pos'].notna()].copy()
    ancient_drach['cons_bin'] = (ancient_drach['cons_pos'] // BIN_SIZE).astype(int) * BIN_SIZE
    print(f"  Ancient L1 DRACH positions: {len(ancient_drach):,}")

    # Classify as hotspot or not
    ancient_drach['is_hotspot'] = ancient_drach['cons_bin'].isin(hotspot_set)

    hotspot_sub = ancient_drach[ancient_drach['is_hotspot']]
    nonhot_sub = ancient_drach[~ancient_drach['is_hotspot']]

    print(f"\n  At Young-hotspot positions: {len(hotspot_sub):,} DRACH sites "
          f"({hotspot_sub['is_methylated'].sum():,} methylated)")
    print(f"  At non-hotspot positions:  {len(nonhot_sub):,} DRACH sites "
          f"({nonhot_sub['is_methylated'].sum():,} methylated)")

    if len(hotspot_sub) > 0 and len(nonhot_sub) > 0:
        hot_rate = hotspot_sub['is_methylated'].mean()
        non_rate = nonhot_sub['is_methylated'].mean()
        ratio = hot_rate / non_rate if non_rate > 0 else float('inf')

        # Fisher's exact test
        a = hotspot_sub['is_methylated'].sum()
        b = len(hotspot_sub) - a
        c = nonhot_sub['is_methylated'].sum()
        d = len(nonhot_sub) - c
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

        print(f"\n  Hotspot m6A rate:     {hot_rate:.4f}")
        print(f"  Non-hotspot m6A rate: {non_rate:.4f}")
        print(f"  Ratio: {ratio:.2f}x  (Fisher P={fisher_p:.2e}, OR={odds_ratio:.2f})")

    # Bin-level comparison: Ancient meth rate at each Young hotspot bin
    ancient_bin_stats = ancient_drach.groupby('cons_bin').agg(
        n_total=('is_methylated', 'count'),
        n_meth=('is_methylated', 'sum'),
    ).reset_index()
    ancient_bin_stats['meth_rate'] = ancient_bin_stats['n_meth'] / ancient_bin_stats['n_total']
    ancient_bin_stats = ancient_bin_stats[ancient_bin_stats['n_total'] >= 10]

    # Correlation: Young bin rate vs Ancient bin rate
    merged_bins = bin_stats.merge(ancient_bin_stats, on='cons_bin', suffixes=('_young', '_ancient'))
    if len(merged_bins) >= 5:
        rho, p_rho = stats.spearmanr(merged_bins['meth_rate_young'], merged_bins['meth_rate_ancient'])
        print(f"\n  Young vs Ancient bin-level m6A rate correlation:")
        print(f"    Spearman rho = {rho:.3f} (P = {p_rho:.2e}), N bins = {len(merged_bins)}")

    # ============================================================
    # Part C: Flanking sequence conservation analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("PART C: Flanking Sequence Conservation → m6A Rate")
    print("=" * 70)

    # Get L1HS consensus-derived 11-mers at DRACH positions
    # First, extract L1HS reference from a full-length Young L1 element
    print("  Building L1HS consensus DRACH 11-mers from reference Young L1 reads...")

    young_flanking = young_drach[young_drach['flanking_11mer'].notna()].copy()
    # Most common 11-mers at methylated Young DRACH sites
    young_meth = young_flanking[young_flanking['is_methylated']]['flanking_11mer']
    young_11mer_counts = Counter(young_meth)
    top_young_11mers = set(km for km, cnt in young_11mer_counts.most_common(100))
    print(f"  Top 100 Young m6A 11-mers collected")

    # For Ancient DRACH sites, compute similarity to nearest Young 11-mer
    ancient_with_flanking = ancient_drach[ancient_drach['flanking_11mer'].notna()].copy()
    print(f"  Ancient DRACH sites with flanking: {len(ancient_with_flanking):,}")

    def hamming_min(seq, reference_set):
        """Min Hamming distance to any sequence in reference set."""
        min_dist = len(seq)
        for ref_seq in reference_set:
            if len(ref_seq) != len(seq):
                continue
            dist = sum(1 for a, b in zip(seq, ref_seq) if a != b)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # Subsample for speed (compute Hamming for each Ancient DRACH site)
    if len(ancient_with_flanking) > 50000:
        sample_idx = np.random.choice(len(ancient_with_flanking), 50000, replace=False)
        analysis_df = ancient_with_flanking.iloc[sample_idx].copy()
    else:
        analysis_df = ancient_with_flanking.copy()

    print(f"  Computing Hamming distances for {len(analysis_df):,} sites...")
    distances = []
    for _, row in analysis_df.iterrows():
        d = hamming_min(row['flanking_11mer'], top_young_11mers)
        distances.append(d)

    analysis_df = analysis_df.copy()
    analysis_df['hamming_dist'] = distances

    # Group by distance and compute methylation rate
    print(f"\n  {'Hamming':>8s} {'N':>8s} {'Meth':>6s} {'Rate':>8s}")
    print("  " + "-" * 35)

    for d in sorted(analysis_df['hamming_dist'].unique()):
        sub = analysis_df[analysis_df['hamming_dist'] == d]
        if len(sub) < 10:
            continue
        rate = sub['is_methylated'].mean()
        print(f"  {d:>8d} {len(sub):>8,} {sub['is_methylated'].sum():>6,} {rate:>8.4f}")

    # Correlation
    rho_h, p_h = stats.spearmanr(analysis_df['hamming_dist'], analysis_df['is_methylated'].astype(float))
    print(f"\n  Hamming distance vs methylation: Spearman rho = {rho_h:.3f} (P = {p_h:.2e})")

    # Binary: close (<=2 mismatches) vs far (>2)
    close = analysis_df[analysis_df['hamming_dist'] <= 2]
    far = analysis_df[analysis_df['hamming_dist'] > 2]
    if len(close) > 0 and len(far) > 0:
        close_rate = close['is_methylated'].mean()
        far_rate = far['is_methylated'].mean()
        ratio = close_rate / far_rate if far_rate > 0 else float('inf')
        a = close['is_methylated'].sum()
        b = len(close) - a
        c = far['is_methylated'].sum()
        d_val = len(far) - c
        or_val, fp = stats.fisher_exact([[a, b], [c, d_val]])
        print(f"\n  Close (≤2 mismatches): rate={close_rate:.4f} (N={len(close):,})")
        print(f"  Far (>2 mismatches):   rate={far_rate:.4f} (N={len(far):,})")
        print(f"  Ratio: {ratio:.2f}x (Fisher P={fp:.2e}, OR={or_val:.2f})")

    # ============================================================
    # Part D: L1HS region-specific m6A rate (Ancient fragments)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART D: Ancient L1 m6A Rate by Consensus Region")
    print("=" * 70)

    ancient_with_pos = ancient_drach[ancient_drach['cons_pos'].notna()].copy()
    ancient_with_pos['cons_frac'] = ancient_with_pos['cons_pos'] / ancient_with_pos['cons_length']

    print(f"\n  {'Region':<20s} {'N_DRACH':>8s} {'N_meth':>7s} {'Rate':>8s}")
    print("  " + "-" * 48)

    regions_list = [
        ("5'UTR (0-15%)", 0, 0.15),
        ("ORF1 (15-32%)", 0.15, 0.32),
        ("ORF2-5' (32-60%)", 0.32, 0.60),
        ("ORF2-3' (60-96%)", 0.60, 0.96),
        ("3'UTR (96-100%)", 0.96, 1.01),
    ]

    region_data = []
    for label, lo, hi in regions_list:
        sub = ancient_with_pos[(ancient_with_pos['cons_frac'] >= lo) &
                               (ancient_with_pos['cons_frac'] < hi)]
        if len(sub) > 0:
            rate = sub['is_methylated'].mean()
            print(f"  {label:<20s} {len(sub):>8,} {sub['is_methylated'].sum():>7,} {rate:>8.4f}")
            region_data.append({'region': label, 'n': len(sub),
                               'n_meth': sub['is_methylated'].sum(), 'rate': rate})

    # Save all results
    results = {
        'analysis_df': analysis_df,
        'drach_df': drach_df,
        'ancient_bin_stats': ancient_bin_stats if 'ancient_bin_stats' in dir() else None,
        'bin_stats': bin_stats,
    }

    # Save summary
    summary_out = OUTDIR / 'consensus_hotspot_analysis_summary.tsv'
    summary_rows = []
    if len(hotspot_sub) > 0:
        summary_rows.append({
            'test': 'Ancient_hotspot_vs_nonhotspot',
            'hotspot_rate': hotspot_sub['is_methylated'].mean(),
            'nonhotspot_rate': nonhot_sub['is_methylated'].mean() if len(nonhot_sub) > 0 else np.nan,
            'ratio': (hotspot_sub['is_methylated'].mean() /
                     nonhot_sub['is_methylated'].mean()
                     if len(nonhot_sub) > 0 and nonhot_sub['is_methylated'].mean() > 0
                     else np.nan),
            'fisher_p': fisher_p if 'fisher_p' in dir() else np.nan,
        })
    if len(close) > 0 and len(far) > 0:
        summary_rows.append({
            'test': 'Ancient_close_vs_far_flanking',
            'hotspot_rate': close['is_methylated'].mean(),
            'nonhotspot_rate': far['is_methylated'].mean(),
            'ratio': close['is_methylated'].mean() / far['is_methylated'].mean() if far['is_methylated'].mean() > 0 else np.nan,
            'fisher_p': fp,
        })
    pd.DataFrame(summary_rows).to_csv(summary_out, sep='\t', index=False)
    print(f"\n  Summary saved: {summary_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
