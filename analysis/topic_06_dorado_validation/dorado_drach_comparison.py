#!/usr/bin/env python3
"""
Dorado DRACH m6A comparison: L1 vs protein-coding mRNA.

Compares m6A landscape between L1 transposons and non-L1 reads:
  Part A: DRACH m6A probability comparison (ALL A positions as denominator)
  Part B: Motif enrichment analysis at high-confidence thresholds
  Part C: Per-read m6A rate at DRACH vs non-DRACH

Input: dorado BAM with A+a modification tags (MM suffix '.', only called sites)
Output: Summary tables, motif rankings, per-read rates

Note: dorado uses MM suffix '.' (unmodified bases NOT reported), so we parse
called m6A sites from MM/ML, then enumerate ALL A positions from read sequence
+ aligned pairs to get the denominator (total DRACH-A and non-DRACH-A counts).
"""

import pysam
import numpy as np
import re
import sys
import os
import gzip
from collections import defaultdict, Counter
from scipy import stats
import random

random.seed(42)
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
BAM_PATH = "/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/HeLa_1_1.dorado.m6A.sorted.bam"
REF_PATH = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta"
L1_BED = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/L1_TE_L1_family.bed"
OUTDIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation/dorado_m6a_results"

MIN_L1_OVERLAP = 0.10
CONTEXT_FLANK = 2  # for 5-mer (2 + center + 2)
CONTEXT_FLANK_11 = 5  # for 11-mer used in DRACH check

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# Sampling: ALL L1 reads, subsample non-L1
MAX_NONL1_READS = 10000
# For Part A (all A positions), cap at 5000 each to limit memory
MAX_READS_PART_A = 5000

# DRACH: D=[AGT], R=[AG], A=A, C=C, H=[ACT]
# Applied on the 5-mer centered on the A of interest
DRACH_RE = re.compile(r"^[AGT][AG]A[C][ACT]$")

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# L1 annotation loading
# ============================================================
def load_l1_intervals(bed_path):
    """Load L1 BED into sorted interval lists per chromosome."""
    print("Loading L1 annotations...")
    l1_data = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, start, end, repname = parts[0], int(parts[1]), int(parts[2]), parts[3]
            l1_data[chrom].append((start, end, repname))
    for chrom in l1_data:
        l1_data[chrom].sort()
    n_total = sum(len(v) for v in l1_data.values())
    print(f"  Loaded {n_total:,} L1 elements across {len(l1_data)} chromosomes")
    return l1_data


def find_l1_overlap(l1_data, chrom, read_start, read_end):
    """Find L1 overlap for a read. Returns (overlap_frac, best_repname)."""
    if chrom not in l1_data:
        return 0.0, None
    intervals = l1_data[chrom]
    read_len = read_end - read_start
    if read_len <= 0:
        return 0.0, None
    from bisect import bisect_left
    left = bisect_left(intervals, (read_start,))
    start_idx = max(0, left - 1)
    best_overlap, best_repname = 0, None
    for i in range(start_idx, len(intervals)):
        istart, iend, repname = intervals[i]
        if istart >= read_end:
            break
        ov = min(read_end, iend) - max(read_start, istart)
        if ov > best_overlap:
            best_overlap = ov
            best_repname = repname
    return best_overlap / read_len, best_repname


# ============================================================
# MM/ML parsing
# ============================================================
def parse_m6a_from_read(read):
    """
    Parse A+a modification from MM/ML tags.
    Returns dict: {read_pos: ml_probability} for each called m6A A.
    """
    if not read.has_tag("MM") or not read.has_tag("ML"):
        return {}

    mm_str = read.get_tag("MM")
    ml_arr = list(read.get_tag("ML"))
    seq = read.query_sequence
    if seq is None:
        return {}

    mod_sections = [s.strip() for s in mm_str.rstrip(";").split(";") if s.strip()]
    ml_offset = 0
    m6a_map = {}

    for section in mod_sections:
        tokens = section.split(",")
        header = tokens[0]
        skip_values = [int(x) for x in tokens[1:]] if len(tokens) > 1 else []
        n_mods = len(skip_values)

        if header.startswith("A+a"):
            a_positions = [i for i, base in enumerate(seq) if base == "A"]
            a_idx = 0
            for mod_i, skip in enumerate(skip_values):
                a_idx += skip
                if a_idx < len(a_positions):
                    read_pos = a_positions[a_idx]
                    prob = ml_arr[ml_offset + mod_i]
                    m6a_map[read_pos] = prob
                a_idx += 1

        ml_offset += n_mods

    return m6a_map


# ============================================================
# Reference context helpers
# ============================================================
def is_drach_5mer(fivemer):
    """Check if a 5-mer (centered on A at position 2) matches DRACH."""
    if len(fivemer) != 5:
        return False
    return bool(DRACH_RE.match(fivemer.upper()))


def get_ref_5mer(ref, chrom, ref_pos, is_reverse):
    """
    Get 5-mer centered on the given ref_pos, oriented to the RNA strand.
    For forward reads: A on ref fwd strand -> 5-mer from fwd strand
    For reverse reads: A in stored SEQ = A on ref fwd strand,
      but the RNA is on the reverse strand, so reverse-complement the context.
    """
    start = ref_pos - 2
    end = ref_pos + 3
    if start < 0:
        return None
    try:
        ctx = ref.fetch(chrom, start, end).upper()
    except (ValueError, KeyError):
        return None
    if len(ctx) != 5 or "N" in ctx:
        return None

    if is_reverse:
        # Reverse complement to get RNA-strand context
        comp = str.maketrans("ACGT", "TGCA")
        ctx = ctx.translate(comp)[::-1]

    return ctx


# ============================================================
# Main analysis
# ============================================================
def main():
    print("=" * 70)
    print("Dorado DRACH m6A Comparison: L1 vs non-L1")
    print("=" * 70)

    l1_data = load_l1_intervals(L1_BED)
    bam = pysam.AlignmentFile(BAM_PATH, "rb")
    ref = pysam.FastaFile(REF_PATH)

    # ---- Phase 1: Classify all reads as L1 or non-L1 ----
    # Collect read info: parse m6A calls, enumerate ALL A positions
    # We store per-read data for Parts A, B, C

    # Data structures for results
    # Per-read: (read_id, is_l1, repname, aligned_len, n_total_A,
    #            n_drach_A, n_nondrach_A, n_m6a_drach_{thr}, n_m6a_nondrach_{thr},
    #            list_of_site_info)
    # Per-site: (prob, is_drach, 5mer)

    # To manage memory, we process reads in streaming fashion and accumulate:
    # Part A: probability distributions (all called sites, by DRACH class)
    # Part B: motif counts at threshold
    # Part C: per-read rates

    # Part A accumulators
    # For ALL A positions (with or without m6A call), classify DRACH
    # Then get the m6A probability (0 if uncalled, or the actual value if called)
    part_a_l1_drach_probs = []
    part_a_l1_nondrach_probs = []
    part_a_nonl1_drach_probs = []
    part_a_nonl1_nondrach_probs = []

    # Part B accumulators: (5mer, prob) for called m6A sites
    part_b_l1_sites = []     # (5mer, prob)
    part_b_nonl1_sites = []

    # Part C accumulators: per-read dict
    part_c_rows = []  # (read_id, is_l1, repname, aligned_len, n_A_drach, n_A_nondrach,
    #                   n_m6a_drach_204, n_m6a_nondrach_204, n_m6a_drach_128, n_m6a_nondrach_128)

    n_processed = 0
    n_unmapped = 0
    n_no_mm = 0
    n_l1 = 0
    n_nonl1 = 0
    n_l1_partA = 0
    n_nonl1_partA = 0
    n_nonl1_selected = 0

    # For Part A subsampling, we use reservoir sampling
    # Collect indices of non-L1 reads, then decide at end
    # Actually simpler: process first MAX_NONL1_READS non-L1 reads
    # For Part A: first MAX_READS_PART_A of each class

    print("\nProcessing BAM reads (streaming)...")
    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            n_unmapped += 1
            continue
        if not read.has_tag("MM") or not read.has_tag("ML"):
            n_no_mm += 1
            continue
        seq = read.query_sequence
        if seq is None:
            continue

        n_processed += 1
        if n_processed % 50000 == 0:
            print(f"  Processed {n_processed:,} reads (L1={n_l1}, non-L1 selected={n_nonl1_selected})")

        chrom = read.reference_name
        read_start = read.reference_start
        read_end = read.reference_end
        if read_end is None:
            continue
        is_reverse = read.is_reverse
        aligned_len = read_end - read_start

        # Classify L1
        overlap_frac, repname = find_l1_overlap(l1_data, chrom, read_start, read_end)
        is_l1 = overlap_frac >= MIN_L1_OVERLAP

        if is_l1:
            n_l1 += 1
        else:
            n_nonl1 += 1
            if n_nonl1_selected >= MAX_NONL1_READS:
                continue  # skip excess non-L1
            n_nonl1_selected += 1

        # Parse m6A calls
        m6a_map = parse_m6a_from_read(read)

        # Decide if this read contributes to Part A (all-A analysis)
        do_part_a = False
        if is_l1 and n_l1_partA < MAX_READS_PART_A:
            do_part_a = True
            n_l1_partA += 1
        elif not is_l1 and n_nonl1_partA < MAX_READS_PART_A:
            do_part_a = True
            n_nonl1_partA += 1

        # Get aligned pairs for mapping read positions to reference
        pairs = read.get_aligned_pairs()
        read2ref = {}
        for rp, rfp in pairs:
            if rp is not None and rfp is not None:
                read2ref[rp] = rfp

        # Find all A positions in the read
        a_positions = [i for i, base in enumerate(seq) if base == "A"]

        # Per-read counters
        n_drach_a = 0
        n_nondrach_a = 0
        n_m6a_drach_204 = 0
        n_m6a_nondrach_204 = 0
        n_m6a_drach_128 = 0
        n_m6a_nondrach_128 = 0

        for rp in a_positions:
            ref_pos = read2ref.get(rp)
            if ref_pos is None:
                continue

            fivemer = get_ref_5mer(ref, chrom, ref_pos, is_reverse)
            if fivemer is None:
                continue

            drach = is_drach_5mer(fivemer)

            # Get m6A probability for this position (0 if not called)
            prob = m6a_map.get(rp, 0)

            if drach:
                n_drach_a += 1
                if prob >= 204:
                    n_m6a_drach_204 += 1
                if prob >= 128:
                    n_m6a_drach_128 += 1
            else:
                n_nondrach_a += 1
                if prob >= 204:
                    n_m6a_nondrach_204 += 1
                if prob >= 128:
                    n_m6a_nondrach_128 += 1

            # Part A: accumulate probability for ALL A positions
            if do_part_a:
                if is_l1:
                    if drach:
                        part_a_l1_drach_probs.append(prob)
                    else:
                        part_a_l1_nondrach_probs.append(prob)
                else:
                    if drach:
                        part_a_nonl1_drach_probs.append(prob)
                    else:
                        part_a_nonl1_nondrach_probs.append(prob)

            # Part B: accumulate called m6A sites (prob > 0 means dorado reported it)
            if prob > 0:
                if is_l1:
                    part_b_l1_sites.append((fivemer, prob))
                else:
                    part_b_nonl1_sites.append((fivemer, prob))

        # Part C: per-read
        part_c_rows.append((
            read.query_name, is_l1, repname if is_l1 else ".",
            aligned_len, n_drach_a, n_nondrach_a,
            n_m6a_drach_204, n_m6a_nondrach_204,
            n_m6a_drach_128, n_m6a_nondrach_128
        ))

    bam.close()
    ref.close()

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total reads processed: {n_processed:,}")
    print(f"L1 reads: {n_l1:,}")
    print(f"Non-L1 reads selected: {n_nonl1_selected:,} (of {n_nonl1:,} total)")
    print(f"Part A reads: L1={n_l1_partA}, non-L1={n_nonl1_partA}")

    # ============================================================
    # PART A: DRACH m6A probability comparison
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART A: m6A PROBABILITY DISTRIBUTIONS (ALL A positions)")
    print(f"{'=' * 70}")

    def prob_stats(arr, label):
        arr = np.array(arr)
        n = len(arr)
        if n == 0:
            print(f"  {label}: n=0")
            return {}
        n_zero = np.sum(arr == 0)
        n_above_128 = np.sum(arr >= 128)
        n_above_204 = np.sum(arr >= 204)
        n_at_255 = np.sum(arr == 255)
        nonzero = arr[arr > 0]
        print(f"  {label}:")
        print(f"    Total A positions: {n:,}")
        print(f"    Uncalled (prob=0): {n_zero:,} ({n_zero/n:.1%})")
        print(f"    Called (prob>0):    {n - n_zero:,} ({(n - n_zero)/n:.1%})")
        print(f"    Above 128 (50%):   {n_above_128:,} ({n_above_128/n:.4%})")
        print(f"    Above 204 (80%):   {n_above_204:,} ({n_above_204/n:.4%})")
        print(f"    At 255 (max):      {n_at_255:,} ({n_at_255/n:.4%})")
        if len(nonzero) > 0:
            print(f"    Called prob: median={np.median(nonzero):.0f}, mean={np.mean(nonzero):.1f}")
        return {
            "n": n, "n_zero": int(n_zero), "n_above_128": int(n_above_128),
            "n_above_204": int(n_above_204), "n_at_255": int(n_at_255),
            "pct_called": (n - n_zero) / n if n > 0 else 0,
            "pct_above_128": n_above_128 / n if n > 0 else 0,
            "pct_above_204": n_above_204 / n if n > 0 else 0,
            "pct_at_255": n_at_255 / n if n > 0 else 0,
            "median_called": float(np.median(nonzero)) if len(nonzero) > 0 else 0,
        }

    stats_l1_drach = prob_stats(part_a_l1_drach_probs, "L1 DRACH-A")
    stats_l1_nondrach = prob_stats(part_a_l1_nondrach_probs, "L1 non-DRACH-A")
    stats_nonl1_drach = prob_stats(part_a_nonl1_drach_probs, "non-L1 DRACH-A")
    stats_nonl1_nondrach = prob_stats(part_a_nonl1_nondrach_probs, "non-L1 non-DRACH-A")

    # Statistical tests
    print(f"\n  --- Statistical tests (Mann-Whitney) ---")

    def safe_mwu(a, b, label):
        a, b = np.array(a), np.array(b)
        if len(a) < 10 or len(b) < 10:
            print(f"  {label}: insufficient data")
            return None
        try:
            u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            print(f"  {label}: U={u:.0f}, P={p:.2e}")
            return p
        except Exception as e:
            print(f"  {label}: error - {e}")
            return None

    # L1 vs non-L1 within DRACH
    safe_mwu(part_a_l1_drach_probs, part_a_nonl1_drach_probs,
             "L1 DRACH vs non-L1 DRACH")
    # L1 vs non-L1 within non-DRACH
    safe_mwu(part_a_l1_nondrach_probs, part_a_nonl1_nondrach_probs,
             "L1 non-DRACH vs non-L1 non-DRACH")
    # DRACH vs non-DRACH within L1
    safe_mwu(part_a_l1_drach_probs, part_a_l1_nondrach_probs,
             "L1: DRACH vs non-DRACH")
    # DRACH vs non-DRACH within non-L1
    safe_mwu(part_a_nonl1_drach_probs, part_a_nonl1_nondrach_probs,
             "non-L1: DRACH vs non-DRACH")

    # Enrichment: what fraction of m6A>=204 falls at DRACH sites?
    print(f"\n  --- DRACH fraction among high-confidence m6A ---")
    for label, drach_arr, nondrach_arr in [
        ("L1", part_a_l1_drach_probs, part_a_l1_nondrach_probs),
        ("non-L1", part_a_nonl1_drach_probs, part_a_nonl1_nondrach_probs),
    ]:
        drach_arr_np = np.array(drach_arr)
        nondrach_arr_np = np.array(nondrach_arr)
        n_drach_total = len(drach_arr_np)
        n_nondrach_total = len(nondrach_arr_np)
        drach_frac_bg = n_drach_total / (n_drach_total + n_nondrach_total) if (n_drach_total + n_nondrach_total) > 0 else 0

        for thr_name, thr in [(">=128", 128), (">=204", 204), ("=255", 255)]:
            nd = int(np.sum(drach_arr_np >= thr)) if thr_name != "=255" else int(np.sum(drach_arr_np == 255))
            nn = int(np.sum(nondrach_arr_np >= thr)) if thr_name != "=255" else int(np.sum(nondrach_arr_np == 255))
            tot = nd + nn
            frac = nd / tot if tot > 0 else 0
            enrichment = frac / drach_frac_bg if drach_frac_bg > 0 else 0
            print(f"    {label} thr{thr_name}: DRACH={nd}, non-DRACH={nn}, "
                  f"DRACH_frac={frac:.3f} (bg={drach_frac_bg:.3f}, enrichment={enrichment:.2f}x)")

    # ============================================================
    # PART B: Motif enrichment analysis
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART B: MOTIF ENRICHMENT (5-mer centered on m6A)")
    print(f"{'=' * 70}")

    def motif_analysis(sites, label, threshold):
        """Analyze 5-mer motifs at m6A sites above threshold."""
        filtered = [(fivemer, prob) for fivemer, prob in sites if prob >= threshold]
        if not filtered:
            print(f"  {label} (thr>={threshold}): no sites")
            return Counter(), 0, 0

        fivemers = [f for f, p in filtered]
        n_total = len(fivemers)
        n_drach = sum(1 for f in fivemers if is_drach_5mer(f))
        drach_frac = n_drach / n_total
        print(f"\n  {label} (thr>={threshold}): {n_total:,} sites, "
              f"DRACH={n_drach:,} ({drach_frac:.1%}), non-DRACH={n_total - n_drach:,} ({1 - drach_frac:.1%})")

        kmer_counts = Counter(fivemers)
        print(f"  Top 20 5-mers:")
        for rank, (kmer, count) in enumerate(kmer_counts.most_common(20), 1):
            pct = count / n_total * 100
            drach_flag = " [DRACH]" if is_drach_5mer(kmer) else ""
            print(f"    {rank:2d}. {kmer}: {count:,} ({pct:.1f}%){drach_flag}")
        return kmer_counts, n_drach, n_total

    l1_motifs_204, l1_ndr_204, l1_tot_204 = motif_analysis(part_b_l1_sites, "L1", 204)
    nonl1_motifs_204, nonl1_ndr_204, nonl1_tot_204 = motif_analysis(part_b_nonl1_sites, "non-L1", 204)

    l1_motifs_255, l1_ndr_255, l1_tot_255 = motif_analysis(part_b_l1_sites, "L1", 255)
    nonl1_motifs_255, nonl1_ndr_255, nonl1_tot_255 = motif_analysis(part_b_nonl1_sites, "non-L1", 255)

    # Also at threshold 128 for completeness
    l1_motifs_128, l1_ndr_128, l1_tot_128 = motif_analysis(part_b_l1_sites, "L1", 128)
    nonl1_motifs_128, nonl1_ndr_128, nonl1_tot_128 = motif_analysis(part_b_nonl1_sites, "non-L1", 128)

    # Overlap analysis: top 20 5-mers at thr>=204
    print(f"\n  --- Top-20 motif overlap (thr>=204) ---")
    l1_top20 = set(k for k, _ in l1_motifs_204.most_common(20))
    nonl1_top20 = set(k for k, _ in nonl1_motifs_204.most_common(20))
    overlap = l1_top20 & nonl1_top20
    l1_only = l1_top20 - nonl1_top20
    nonl1_only = nonl1_top20 - l1_top20
    print(f"    Shared: {len(overlap)} ({', '.join(sorted(overlap))})")
    print(f"    L1-only: {len(l1_only)} ({', '.join(sorted(l1_only))})")
    print(f"    non-L1-only: {len(nonl1_only)} ({', '.join(sorted(nonl1_only))})")

    # ============================================================
    # PART C: Per-read m6A rate at DRACH vs non-DRACH
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART C: PER-READ m6A RATE (DRACH vs non-DRACH)")
    print(f"{'=' * 70}")

    l1_rows = [r for r in part_c_rows if r[1]]
    nonl1_rows = [r for r in part_c_rows if not r[1]]

    def compute_per_read_rates(rows, label):
        """Compute per-read m6A rates at DRACH vs non-DRACH."""
        drach_rates = []
        nondrach_rates = []
        ratio_vals = []
        for row in rows:
            # row: (read_id, is_l1, repname, aligned_len, n_drach_a, n_nondrach_a,
            #        n_m6a_drach_204, n_m6a_nondrach_204, n_m6a_drach_128, n_m6a_nondrach_128)
            n_drach_a = row[4]
            n_nondrach_a = row[5]
            n_m6a_drach = row[6]  # threshold 204
            n_m6a_nondrach = row[7]
            if n_drach_a >= 5:  # require minimum 5 DRACH-A sites
                dr = n_m6a_drach / n_drach_a
                drach_rates.append(dr)
            if n_nondrach_a >= 5:
                ndr = n_m6a_nondrach / n_nondrach_a
                nondrach_rates.append(ndr)
            if n_drach_a >= 5 and n_nondrach_a >= 5:
                dr = n_m6a_drach / n_drach_a
                ndr = n_m6a_nondrach / n_nondrach_a
                if ndr > 0:
                    ratio_vals.append(dr / ndr)

        dr_arr = np.array(drach_rates)
        ndr_arr = np.array(nondrach_rates)
        print(f"\n  {label}:")
        print(f"    Reads with >=5 DRACH-A: {len(dr_arr):,}")
        if len(dr_arr) > 0:
            print(f"    DRACH m6A rate (thr>=204): median={np.median(dr_arr):.4f}, "
                  f"mean={np.mean(dr_arr):.4f}")
        print(f"    Reads with >=5 non-DRACH-A: {len(ndr_arr):,}")
        if len(ndr_arr) > 0:
            print(f"    non-DRACH m6A rate (thr>=204): median={np.median(ndr_arr):.4f}, "
                  f"mean={np.mean(ndr_arr):.4f}")
        if len(ratio_vals) > 0:
            ratio_arr = np.array(ratio_vals)
            print(f"    DRACH/non-DRACH rate ratio: median={np.median(ratio_arr):.2f}, "
                  f"mean={np.mean(ratio_arr):.2f}")
        return dr_arr, ndr_arr

    l1_dr, l1_ndr = compute_per_read_rates(l1_rows, "L1")
    nonl1_dr, nonl1_ndr = compute_per_read_rates(nonl1_rows, "non-L1")

    # Compare L1 vs non-L1 within DRACH
    print(f"\n  --- Per-read comparisons (Mann-Whitney) ---")
    safe_mwu(l1_dr, nonl1_dr, "L1 vs non-L1: DRACH m6A rate")
    safe_mwu(l1_ndr, nonl1_ndr, "L1 vs non-L1: non-DRACH m6A rate")

    # Within each class: DRACH vs non-DRACH
    safe_mwu(l1_dr, l1_ndr, "L1: DRACH vs non-DRACH rate")
    safe_mwu(nonl1_dr, nonl1_ndr, "non-L1: DRACH vs non-DRACH rate")

    # Young vs Ancient per-read rates
    print(f"\n  --- Young vs Ancient L1 per-read rates ---")
    young_rows = [r for r in l1_rows if r[2] in YOUNG_SUBFAMILIES]
    ancient_rows = [r for r in l1_rows if r[2] not in YOUNG_SUBFAMILIES and r[2] != "."]
    y_dr, y_ndr = compute_per_read_rates(young_rows, "Young L1")
    a_dr, a_ndr = compute_per_read_rates(ancient_rows, "Ancient L1")
    if len(y_dr) > 0 and len(a_dr) > 0:
        safe_mwu(y_dr, a_dr, "Young vs Ancient: DRACH m6A rate")
    if len(y_ndr) > 0 and len(a_ndr) > 0:
        safe_mwu(y_ndr, a_ndr, "Young vs Ancient: non-DRACH m6A rate")

    # ============================================================
    # Save outputs
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print(f"{'=' * 70}")

    # 1. drach_comparison_summary.tsv
    summary_path = os.path.join(OUTDIR, "drach_comparison_summary.tsv")
    with open(summary_path, "w") as f:
        f.write("category\tn_A_positions\tn_uncalled\tpct_called\t"
                "n_above_128\tpct_above_128\tn_above_204\tpct_above_204\t"
                "n_at_255\tpct_at_255\tmedian_called_prob\n")
        for label, st in [
            ("L1_DRACH", stats_l1_drach),
            ("L1_nonDRACH", stats_l1_nondrach),
            ("nonL1_DRACH", stats_nonl1_drach),
            ("nonL1_nonDRACH", stats_nonl1_nondrach),
        ]:
            if not st:
                continue
            f.write(f"{label}\t{st['n']}\t{st['n_zero']}\t{st['pct_called']:.6f}\t"
                    f"{st['n_above_128']}\t{st['pct_above_128']:.6f}\t"
                    f"{st['n_above_204']}\t{st['pct_above_204']:.6f}\t"
                    f"{st['n_at_255']}\t{st['pct_at_255']:.6f}\t"
                    f"{st['median_called']:.0f}\n")
    print(f"  Saved: {summary_path}")

    # 2. motif_top20_L1.tsv and motif_top20_nonL1.tsv
    for label, motifs, fname in [
        ("L1", l1_motifs_204, "motif_top20_L1.tsv"),
        ("nonL1", nonl1_motifs_204, "motif_top20_nonL1.tsv"),
    ]:
        path = os.path.join(OUTDIR, fname)
        total = sum(motifs.values())
        with open(path, "w") as f:
            f.write("rank\tfivemer\tcount\tfraction\tis_drach\n")
            for rank, (kmer, count) in enumerate(motifs.most_common(20), 1):
                frac = count / total if total > 0 else 0
                f.write(f"{rank}\t{kmer}\t{count}\t{frac:.6f}\t{is_drach_5mer(kmer)}\n")
        print(f"  Saved: {path}")

    # 3. perread_drach_rates.tsv
    perread_path = os.path.join(OUTDIR, "perread_drach_rates.tsv")
    with open(perread_path, "w") as f:
        f.write("read_id\tis_L1\trepname\taligned_len\t"
                "n_drach_A\tn_nondrach_A\t"
                "n_m6a_drach_204\tn_m6a_nondrach_204\t"
                "n_m6a_drach_128\tn_m6a_nondrach_128\t"
                "drach_rate_204\tnondrach_rate_204\n")
        for row in part_c_rows:
            rid, is_l1, repname, alen, nd_a, nnd_a, nd_204, nnd_204, nd_128, nnd_128 = row
            dr_rate = nd_204 / nd_a if nd_a > 0 else "NA"
            ndr_rate = nnd_204 / nnd_a if nnd_a > 0 else "NA"
            f.write(f"{rid}\t{is_l1}\t{repname}\t{alen}\t"
                    f"{nd_a}\t{nnd_a}\t{nd_204}\t{nnd_204}\t{nd_128}\t{nnd_128}\t"
                    f"{dr_rate}\t{ndr_rate}\n")
    print(f"  Saved: {perread_path}")

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n1. DRACH fraction of all A positions (background):")
    for label, d, nd in [
        ("L1", len(part_a_l1_drach_probs), len(part_a_l1_nondrach_probs)),
        ("non-L1", len(part_a_nonl1_drach_probs), len(part_a_nonl1_nondrach_probs)),
    ]:
        frac = d / (d + nd) if (d + nd) > 0 else 0
        print(f"  {label}: DRACH={d:,}, non-DRACH={nd:,}, DRACH_frac={frac:.3f}")

    print(f"\n2. DRACH fraction among m6A calls (enrichment):")
    for label, sites in [("L1", part_b_l1_sites), ("non-L1", part_b_nonl1_sites)]:
        for thr in [128, 204, 255]:
            filtered = [f for f, p in sites if (p >= thr if thr < 255 else p == 255)]
            n_dr = sum(1 for f in filtered if is_drach_5mer(f))
            n_tot = len(filtered)
            frac = n_dr / n_tot if n_tot > 0 else 0
            print(f"  {label} thr>={thr}: DRACH={n_dr:,}/{n_tot:,} ({frac:.1%})")

    print(f"\n3. Per-read DRACH m6A rate (thr>=204):")
    for label, dr_arr in [("L1", l1_dr), ("non-L1", nonl1_dr)]:
        if len(dr_arr) > 0:
            print(f"  {label}: median={np.median(dr_arr):.4f}, mean={np.mean(dr_arr):.4f}")
    print(f"\n4. Per-read non-DRACH m6A rate (thr>=204):")
    for label, ndr_arr in [("L1", l1_ndr), ("non-L1", nonl1_ndr)]:
        if len(ndr_arr) > 0:
            print(f"  {label}: median={np.median(ndr_arr):.4f}, mean={np.mean(ndr_arr):.4f}")

    print(f"\n5. DRACH/non-DRACH rate ratio (enrichment of DRACH over non-DRACH):")
    for label, dr_arr, ndr_arr in [("L1", l1_dr, l1_ndr), ("non-L1", nonl1_dr, nonl1_ndr)]:
        if len(dr_arr) > 0 and len(ndr_arr) > 0:
            dr_med = np.median(dr_arr)
            ndr_med = np.median(ndr_arr)
            ratio = dr_med / ndr_med if ndr_med > 0 else float("inf")
            print(f"  {label}: DRACH_median/nonDRACH_median = {dr_med:.4f}/{ndr_med:.4f} = {ratio:.2f}x")

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
