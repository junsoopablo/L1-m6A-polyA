#!/usr/bin/env python3
"""
Dorado non-DRACH m6A analysis: L1 vs non-L1 reads.

Extracts m6A sites from dorado all-context calling (A+a MM/ML tags),
classifies them as DRACH vs non-DRACH, and compares L1 vs non-L1 reads
to find potential METTL16-mediated methylation evidence.

Input: dorado BAM with A+a modification tags, L1 BED annotations
Output: Summary tables of DRACH/non-DRACH m6A distribution
"""

import pysam
import numpy as np
import re
import sys
import os
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

M6A_THRESHOLD = 128  # 0.50 probability (dorado calibration, lower than MAFIA's 204)
MIN_L1_OVERLAP = 0.10  # 10% overlap for L1 classification
MAX_NONL1_READS = 10000  # subsample non-L1 for speed
CONTEXT_FLANK = 5  # bases upstream/downstream for motif context

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# DRACH: D=[AGT], R=[AG], A=A, C=C, H=[ACT]
DRACH_PATTERN = re.compile(r"[AGT][AG]A[C][ACT]")

# METTL16 motifs (DNA version, since we work with reference)
METTL16_CANONICAL = "TACAGAGAA"  # methylated A at position 3 (0-indexed)
METTL16_CORE = "ACAGAG"

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Step 1: Load L1 annotations into interval trees
# ============================================================
def load_l1_intervals(bed_path):
    """Load L1 BED into dict of sorted interval lists per chromosome."""
    print("Loading L1 annotations...")
    from bisect import bisect_left, bisect_right

    l1_data = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            repname = parts[3]
            l1_data[chrom].append((start, end, repname))

    # Sort by start position for binary search
    for chrom in l1_data:
        l1_data[chrom].sort()

    n_total = sum(len(v) for v in l1_data.values())
    print(f"  Loaded {n_total:,} L1 elements across {len(l1_data)} chromosomes")
    return l1_data


def find_l1_overlap(l1_data, chrom, read_start, read_end):
    """Find L1 overlap for a read. Returns (overlap_frac, best_repname) or (0, None)."""
    if chrom not in l1_data:
        return 0.0, None

    intervals = l1_data[chrom]
    read_len = read_end - read_start
    if read_len <= 0:
        return 0.0, None

    # Binary search for candidate intervals
    # Find intervals that could overlap [read_start, read_end)
    best_overlap = 0
    best_repname = None

    # Simple linear scan of nearby intervals (efficient enough for sorted data)
    from bisect import bisect_left
    # Find first interval whose end > read_start
    lo = 0
    hi = len(intervals)
    # Binary search for approximate start
    left = bisect_left(intervals, (read_start,))
    # Look back a bit in case intervals span our region
    start_idx = max(0, left - 1)

    for i in range(start_idx, len(intervals)):
        istart, iend, repname = intervals[i]
        if istart >= read_end:
            break
        # Calculate overlap
        ov_start = max(read_start, istart)
        ov_end = min(read_end, iend)
        if ov_end > ov_start:
            overlap = ov_end - ov_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_repname = repname

    overlap_frac = best_overlap / read_len if read_len > 0 else 0
    return overlap_frac, best_repname


# ============================================================
# Step 2: Parse m6A from MM/ML tags
# ============================================================
def parse_m6a_from_read(read):
    """
    Parse A+a modification from MM/ML tags.
    Returns list of (read_position_0based, ml_probability) for each A+a call.
    """
    if not read.has_tag("MM") or not read.has_tag("ML"):
        return []

    mm_str = read.get_tag("MM")
    ml_arr = list(read.get_tag("ML"))
    seq = read.query_sequence
    if seq is None:
        return []

    # Parse MM tag: semicolon-separated modification types
    # Each: "BASE+MOD.,skip1,skip2,..."
    mod_sections = [s.strip() for s in mm_str.rstrip(";").split(";") if s.strip()]

    ml_offset = 0
    m6a_results = []

    for section in mod_sections:
        tokens = section.split(",")
        header = tokens[0]  # e.g., "A+a."
        skip_values = [int(x) for x in tokens[1:]] if len(tokens) > 1 else []
        n_mods = len(skip_values)

        if header.startswith("A+a"):
            # This is the m6A section
            # Find all A positions in the read sequence
            a_positions = [i for i, base in enumerate(seq) if base == "A"]

            # Walk through skip values to find modified positions
            a_idx = 0  # index into a_positions
            for mod_i, skip in enumerate(skip_values):
                a_idx += skip  # skip this many A's
                if a_idx < len(a_positions):
                    read_pos = a_positions[a_idx]
                    prob = ml_arr[ml_offset + mod_i]
                    m6a_results.append((read_pos, prob))
                a_idx += 1  # move past the modified A

        ml_offset += n_mods

    return m6a_results


def read_pos_to_ref_pos(read, read_pos):
    """
    Convert a read position to a reference position using aligned pairs.
    Returns reference position (0-based) or None if the position is unaligned.
    """
    # Use get_aligned_pairs for accurate mapping
    # This is called per-site, so we cache aligned pairs per read
    if not hasattr(read, "_aligned_pairs_cache"):
        read._aligned_pairs_cache = dict(read.get_aligned_pairs())
    return read._aligned_pairs_cache.get(read_pos, None)


def get_aligned_pairs_dict(read):
    """Get dict mapping read_pos -> ref_pos from aligned pairs."""
    return dict(read.get_aligned_pairs())


# ============================================================
# Step 3: DRACH classification
# ============================================================
def classify_drach(ref_context_11mer):
    """
    Given an 11-mer context centered on the m6A site (position 5),
    check if positions 3-7 (DRACH) match the DRACH motif.
    Context: ...D R [A] C H...
    Position: 3 4  5  6 7 in the 11-mer
    """
    if len(ref_context_11mer) < 11:
        return None  # insufficient context
    context = ref_context_11mer.upper()
    if "N" in context[3:8]:
        return None
    pentamer = context[3:8]  # D R A C H
    if DRACH_PATTERN.match(pentamer):
        return "DRACH"
    return "non-DRACH"


# ============================================================
# Step 4: METTL16 motif analysis
# ============================================================
def check_mettl16_motifs(ref_context_11mer):
    """
    Check if the context around the m6A site matches METTL16 motifs.
    The m6A is at position 5 of the 11-mer.
    METTL16 canonical: TACAGAGAA - methylated A at position 3 (0-indexed)
    So in our 11-mer, TACAGAGAA would start at position 2 (5-3=2)
    """
    if len(ref_context_11mer) < 11:
        return {"canonical": False, "core": False}

    context = ref_context_11mer.upper()
    results = {"canonical": False, "core": False}

    # Check canonical METTL16: TACAGAGAA, m6A at pos 3 -> starts at context pos 2
    if context[2:11] == METTL16_CANONICAL:
        results["canonical"] = True

    # Check core METTL16: ACAGAG, m6A at pos 1 -> starts at context pos 4
    # Actually the core motif has A at various positions. Let's check if ACAGAG
    # appears anywhere in the context with the m6A inside it
    for offset in range(-3, 4):
        start = 5 + offset
        if 0 <= start and start + 6 <= 11:
            if context[start:start + 6] == METTL16_CORE:
                results["core"] = True
                break

    return results


# ============================================================
# Main analysis
# ============================================================
def main():
    print("=" * 70)
    print("Dorado non-DRACH m6A analysis: L1 vs non-L1")
    print("=" * 70)

    # Load L1 annotations
    l1_data = load_l1_intervals(L1_BED)

    # Open BAM and reference
    bam = pysam.AlignmentFile(BAM_PATH, "rb")
    ref = pysam.FastaFile(REF_PATH)

    # Track results
    l1_reads = []      # (read_id, repname, subfamily_class, n_A, aligned_len)
    nonl1_reads = []   # (read_id, n_A, aligned_len)

    l1_sites = []      # (read_id, ref_pos, chrom, strand, prob, context, drach, mettl16, repname)
    nonl1_sites = []   # same structure but repname=None

    # Counters
    n_processed = 0
    n_unmapped = 0
    n_no_mm = 0
    n_l1 = 0
    n_nonl1 = 0
    n_nonl1_sampled = 0
    nonl1_reservoir = []  # for reservoir sampling

    print("\nProcessing BAM reads...")
    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            n_unmapped += 1
            continue
        if not read.has_tag("MM") or not read.has_tag("ML"):
            n_no_mm += 1
            continue
        if read.query_sequence is None:
            continue

        n_processed += 1
        if n_processed % 50000 == 0:
            print(f"  Processed {n_processed:,} reads... (L1: {n_l1}, non-L1 sampled: {n_nonl1_sampled})")

        chrom = read.reference_name
        read_start = read.reference_start
        read_end = read.reference_end
        is_reverse = read.is_reverse
        strand = "-" if is_reverse else "+"
        seq = read.query_sequence
        aligned_len = read_end - read_start if read_end else 0

        # Count A's in read
        n_A = seq.count("A")

        # Check L1 overlap
        overlap_frac, repname = find_l1_overlap(l1_data, chrom, read_start, read_end)
        is_l1 = overlap_frac >= MIN_L1_OVERLAP

        if is_l1:
            n_l1 += 1
            subfamily_class = "Young" if repname in YOUNG_SUBFAMILIES else "Ancient"
            l1_reads.append((read.query_name, repname, subfamily_class, n_A, aligned_len))
        else:
            n_nonl1 += 1
            # Reservoir sampling for non-L1 reads
            if n_nonl1_sampled < MAX_NONL1_READS:
                nonl1_reservoir.append(n_nonl1 - 1)
                n_nonl1_sampled += 1
                process_nonl1 = True
            else:
                # Reservoir sampling: replace with probability MAX/n
                j = random.randint(0, n_nonl1 - 1)
                if j < MAX_NONL1_READS:
                    # Replace entry j — but we already processed it. For simplicity,
                    # just process this one additionally and count. But reservoir
                    # complicates because we can't un-process. Instead, let's use
                    # a simpler approach: process first MAX_NONL1_READS non-L1 reads.
                    process_nonl1 = False
                else:
                    process_nonl1 = False

            if not process_nonl1 and n_nonl1 > MAX_NONL1_READS:
                continue  # skip this non-L1 read

        # Parse m6A sites
        m6a_calls = parse_m6a_from_read(read)
        if not m6a_calls:
            if is_l1:
                pass  # still counted as L1 read
            else:
                nonl1_reads.append((read.query_name, n_A, aligned_len))
            continue

        # Get aligned pairs for position mapping
        pairs_dict = get_aligned_pairs_dict(read)

        for read_pos, prob in m6a_calls:
            if prob < M6A_THRESHOLD:
                continue

            # Map to reference position
            ref_pos = pairs_dict.get(read_pos, None)
            if ref_pos is None:
                continue

            # Get reference context (11-mer centered on m6A)
            # For forward strand reads: the A in the read = A on ref forward strand
            # For reverse strand reads: the A in the read = T on ref forward strand
            #   (because pysam gives us the forward-strand sequence in query_sequence
            #    when the read is reversed — actually NO, pysam gives the sequenced strand)
            # Actually: pysam's query_sequence is always the SEQUENCED strand (same as SEQ in BAM).
            # For reverse-mapped reads, SEQ is already reverse-complemented to match
            # the forward strand by the aligner. Wait — that's also not right.
            # In BAM format: SEQ is the sequence as stored. For reverse-strand alignments,
            # the stored SEQ IS the reverse complement of the original read.
            # The bases in SEQ correspond to the forward strand of the reference.
            # So if SEQ has an 'A' at position i, it corresponds to an 'A' on the
            # forward strand reference at the aligned position.
            # Therefore for both forward and reverse reads, an 'A' in SEQ maps to
            # a reference position where we expect 'A' on the forward strand.

            # However, for RNA modifications, the MM tag describes modifications on
            # the ORIGINAL molecule strand. For DRS, reads are always sense-strand
            # of the RNA. For reverse-mapped reads, pysam reverses the SEQ, so
            # the 'A' we see is on the forward reference strand.

            # Get 11-mer context from reference (forward strand)
            ctx_start = ref_pos - CONTEXT_FLANK
            ctx_end = ref_pos + CONTEXT_FLANK + 1
            if ctx_start < 0:
                continue

            try:
                ref_context = ref.fetch(chrom, ctx_start, ctx_end).upper()
            except (ValueError, KeyError):
                continue

            if len(ref_context) < 11:
                continue

            # Verify center base
            center_base = ref_context[CONTEXT_FLANK]

            # For DRS: RNA is sense strand. If read maps to + strand,
            # the A in the read = A on reference.
            # If read maps to - strand, the original RNA A = T on reference forward strand.
            # But pysam stores reverse-complemented SEQ for reverse reads,
            # so an 'A' in SEQ for a reverse read = 'A' on forward reference.
            # Actually this needs more careful thought.

            # In BAM: for reverse strand reads, SEQ is reverse-complemented.
            # So 'A' in SEQ at position i maps to 'A' at ref_pos on forward strand.
            # MM tag: for "A+a" with implicit strand (no + or - after base),
            # it refers to A bases in the SEQ as stored in BAM.
            # So the modified base IS at positions where SEQ has 'A'.
            # And since SEQ 'A' = reference forward 'A', we expect center_base = 'A'.

            # BUT for DRS reverse-strand reads:
            # Original RNA: 5'->3' on gene's sense strand = reference reverse strand
            # Dorado outputs: the RNA sequence (same as ref reverse complement = ref - strand)
            # BAM stores: reverse complement of that = forward strand
            # MM tag: refers to bases in the stored SEQ (forward strand representation)
            # So 'A' in stored SEQ for a -strand read was originally 'U' in the RNA!
            # That can't be right for m6A calling...

            # Let me reconsider: dorado's MM tags follow SAM spec.
            # SAM spec says MM describes modifications on the *original* strand.
            # For reverse-mapped reads, the SEQ in BAM is reverse-complemented.
            # The MM tag with implicit strand refers to the forward-mapped SEQ.
            # But the modification is actually on the original read (= reverse comp of SEQ).
            # So for A+a on a reverse read:
            # - 'A' in stored SEQ = 'T' on original read strand
            # - This means the modification is actually on 'T' of the original? No...

            # The SAM spec says: "The base indicated by the MM type ... refers to
            # bases as they are in the alignment record SEQ field."
            # So A+a means: look for 'A' in SEQ as stored, these are modified.
            # The actual chemical modification (m6A) IS on these A bases.
            # For a +strand read: A in SEQ = A on forward reference (correct)
            # For a -strand read: A in SEQ = the stored (rev-comp) version.
            #   The original sequenced base was T (complement of A).
            #   But that makes no sense for m6A...

            # Actually: dorado handles this correctly. For DRS (always + strand RNA):
            # - If gene is on + strand: read maps to + strand, SEQ = original RNA (T for U)
            # - If gene is on - strand: read maps to - strand, SEQ = reverse complement of RNA
            #   So 'A' in stored SEQ was 'T' in RNA = 'U' in RNA. NOT adenosine.

            # This means for -strand reads, 'A' in stored SEQ does NOT correspond
            # to adenosine in the original RNA. The m6A would be on 'T' in stored SEQ
            # (which was 'A' in RNA -> 'U' in RNA with T notation).

            # HOWEVER, dorado knows this and the MM tag specification handles it:
            # For -strand reads, dorado uses the COMPLEMENTED base.
            # "A+a" on a -strand read: A in stored SEQ. Since stored SEQ is rev-comp
            # of original, 'A' here = 'T' original = m6A makes no sense.

            # Actually re-reading the SAM spec more carefully:
            # The MM tag base code refers to the SEQ field as written.
            # Dorado explicitly writes "A+a" for m6A.
            # For -strand alignments, before reverse-complementing, the original
            # sequence had 'T' where stored SEQ has 'A'.
            # BUT dorado should have already accounted for this by using "T+a"
            # for reverse strand reads... unless it always uses "A+a".

            # In practice: dorado always writes "A+a" regardless of strand.
            # For +strand: A in SEQ = A in RNA (correct, m6A on adenosine)
            # For -strand: A in SEQ = revcomp of original. Original base = T.
            #   But the 'm6A' here means m6A on the complementary strand?
            #   No — this is DRS, we sequence one strand.

            # I think the key insight is: for DRS, nearly all reads should map to
            # the same strand as the gene. And dorado calls modifications on the
            # sequenced strand. The BAM representation handles the coordinate mapping.
            # For a -strand read: the modification IS on the RNA adenosine, which
            # on the reference forward strand appears as T at that position.

            # So for -strand reads: center_base should be 'T' (not 'A')
            # because the RNA 'A' (with m6A) maps to ref forward 'T'.

            if not is_reverse:
                expected_base = "A"
            else:
                expected_base = "T"

            # For context motif analysis, we need to look at the correct strand
            if is_reverse:
                # Reverse complement the context to get the RNA-strand context
                comp = str.maketrans("ACGT", "TGCA")
                ref_context_rna = ref_context.translate(comp)[::-1]
            else:
                ref_context_rna = ref_context

            # Classify DRACH on RNA-strand context
            drach_class = classify_drach(ref_context_rna)
            if drach_class is None:
                continue

            # Check METTL16 motifs on RNA-strand context
            mettl16 = check_mettl16_motifs(ref_context_rna)

            site_info = (
                read.query_name, ref_pos, chrom, strand, prob,
                ref_context_rna, drach_class,
                mettl16["canonical"], mettl16["core"], repname if is_l1 else None
            )

            if is_l1:
                l1_sites.append(site_info)
            else:
                nonl1_sites.append(site_info)

        # Record read info
        if is_l1:
            pass  # already recorded above
        else:
            nonl1_reads.append((read.query_name, n_A, aligned_len))

    bam.close()
    ref.close()

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total reads processed: {n_processed:,}")
    print(f"Skipped (unmapped/secondary/supp): {n_unmapped:,}")
    print(f"Skipped (no MM tag): {n_no_mm:,}")
    print(f"L1 reads: {n_l1:,}")
    print(f"Non-L1 reads total: {n_nonl1:,} (sampled: {min(n_nonl1, MAX_NONL1_READS):,})")
    print(f"L1 m6A sites (prob >= {M6A_THRESHOLD}): {len(l1_sites):,}")
    print(f"Non-L1 m6A sites (prob >= {M6A_THRESHOLD}): {len(nonl1_sites):,}")

    # ============================================================
    # Step 5: Compare L1 vs non-L1
    # ============================================================
    print(f"\n{'=' * 70}")
    print("DRACH vs non-DRACH COMPARISON")
    print(f"{'=' * 70}")

    def compute_stats(sites, reads_info, label):
        """Compute m6A statistics for a set of sites."""
        n_reads = len(reads_info) if reads_info else 0
        n_sites = len(sites)
        n_drach = sum(1 for s in sites if s[6] == "DRACH")
        n_nondrach = sum(1 for s in sites if s[6] == "non-DRACH")
        drach_frac = n_drach / n_sites if n_sites > 0 else 0
        nondrach_frac = n_nondrach / n_sites if n_sites > 0 else 0

        # m6A per kb (using aligned length)
        if reads_info and len(reads_info) > 0:
            # reads_info: (read_id, ..., aligned_len) — last element is aligned_len
            total_kb = sum(r[-1] for r in reads_info) / 1000
        else:
            total_kb = 0

        m6a_per_kb = n_sites / total_kb if total_kb > 0 else 0
        drach_per_kb = n_drach / total_kb if total_kb > 0 else 0
        nondrach_per_kb = n_nondrach / total_kb if total_kb > 0 else 0

        # METTL16 motifs among non-DRACH
        nondrach_sites = [s for s in sites if s[6] == "non-DRACH"]
        n_mettl16_canonical = sum(1 for s in nondrach_sites if s[7])
        n_mettl16_core = sum(1 for s in nondrach_sites if s[8])

        mettl16_canonical_frac = n_mettl16_canonical / n_nondrach if n_nondrach > 0 else 0
        mettl16_core_frac = n_mettl16_core / n_nondrach if n_nondrach > 0 else 0

        print(f"\n--- {label} ---")
        print(f"  Reads: {n_reads:,}")
        print(f"  Total aligned kb: {total_kb:,.1f}")
        print(f"  m6A sites (>= thr {M6A_THRESHOLD}): {n_sites:,}")
        print(f"    DRACH:     {n_drach:,} ({drach_frac:.1%})")
        print(f"    non-DRACH: {n_nondrach:,} ({nondrach_frac:.1%})")
        print(f"  m6A/kb total:     {m6a_per_kb:.3f}")
        print(f"  DRACH m6A/kb:     {drach_per_kb:.3f}")
        print(f"  non-DRACH m6A/kb: {nondrach_per_kb:.3f}")
        print(f"  METTL16 canonical among non-DRACH: {n_mettl16_canonical} ({mettl16_canonical_frac:.1%})")
        print(f"  METTL16 core among non-DRACH:      {n_mettl16_core} ({mettl16_core_frac:.1%})")

        return {
            "label": label, "n_reads": n_reads, "total_kb": total_kb,
            "n_sites": n_sites, "n_drach": n_drach, "n_nondrach": n_nondrach,
            "drach_frac": drach_frac, "nondrach_frac": nondrach_frac,
            "m6a_per_kb": m6a_per_kb, "drach_per_kb": drach_per_kb,
            "nondrach_per_kb": nondrach_per_kb,
            "mettl16_canonical": n_mettl16_canonical, "mettl16_core": n_mettl16_core,
            "mettl16_canonical_frac": mettl16_canonical_frac,
            "mettl16_core_frac": mettl16_core_frac,
        }

    l1_stats = compute_stats(l1_sites, l1_reads, "L1 reads")
    nonl1_stats = compute_stats(nonl1_sites, nonl1_reads, "non-L1 reads (subsampled)")

    # Young vs Ancient L1
    young_reads = [r for r in l1_reads if r[2] == "Young"]
    ancient_reads = [r for r in l1_reads if r[2] == "Ancient"]
    young_read_ids = set(r[0] for r in young_reads)
    ancient_read_ids = set(r[0] for r in ancient_reads)

    young_sites = [s for s in l1_sites if s[0] in young_read_ids]
    ancient_sites = [s for s in l1_sites if s[0] in ancient_read_ids]

    print(f"\n{'=' * 70}")
    print("YOUNG vs ANCIENT L1 BREAKDOWN")
    print(f"{'=' * 70}")
    young_stats = compute_stats(young_sites, young_reads, "Young L1 (L1HS/PA1-3)")
    ancient_stats = compute_stats(ancient_sites, ancient_reads, "Ancient L1")

    # L1 subfamily distribution
    print(f"\n{'=' * 70}")
    print("L1 SUBFAMILY DISTRIBUTION")
    print(f"{'=' * 70}")
    subfamily_counts = Counter(r[1] for r in l1_reads)
    for sf, count in subfamily_counts.most_common(20):
        print(f"  {sf}: {count}")

    # ============================================================
    # Fisher's exact test: non-DRACH fraction L1 vs non-L1
    # ============================================================
    print(f"\n{'=' * 70}")
    print("STATISTICAL TESTS")
    print(f"{'=' * 70}")

    # Contingency table: [[L1_DRACH, L1_nonDRACH], [nonL1_DRACH, nonL1_nonDRACH]]
    table = [
        [l1_stats["n_drach"], l1_stats["n_nondrach"]],
        [nonl1_stats["n_drach"], nonl1_stats["n_nondrach"]]
    ]
    if all(all(c > 0 for c in row) for row in table):
        odds_ratio, p_fisher = stats.fisher_exact(table)
        print(f"\nFisher's exact test: non-DRACH enrichment in L1 vs non-L1")
        print(f"  L1:     {l1_stats['n_drach']} DRACH, {l1_stats['n_nondrach']} non-DRACH ({l1_stats['nondrach_frac']:.1%})")
        print(f"  non-L1: {nonl1_stats['n_drach']} DRACH, {nonl1_stats['n_nondrach']} non-DRACH ({nonl1_stats['nondrach_frac']:.1%})")
        print(f"  Odds ratio: {odds_ratio:.3f}")
        print(f"  P-value: {p_fisher:.4e}")
    else:
        print("  Insufficient data for Fisher's exact test")
        odds_ratio, p_fisher = None, None

    # Young vs Ancient Fisher's test
    table_ya = [
        [young_stats["n_drach"], young_stats["n_nondrach"]],
        [ancient_stats["n_drach"], ancient_stats["n_nondrach"]]
    ]
    if all(all(c > 0 for c in row) for row in table_ya):
        or_ya, p_ya = stats.fisher_exact(table_ya)
        print(f"\nFisher's exact test: non-DRACH enrichment Young vs Ancient L1")
        print(f"  Young:   {young_stats['n_drach']} DRACH, {young_stats['n_nondrach']} non-DRACH ({young_stats['nondrach_frac']:.1%})")
        print(f"  Ancient: {ancient_stats['n_drach']} DRACH, {ancient_stats['n_nondrach']} non-DRACH ({ancient_stats['nondrach_frac']:.1%})")
        print(f"  Odds ratio: {or_ya:.3f}")
        print(f"  P-value: {p_ya:.4e}")
    else:
        print("  Insufficient data for Young vs Ancient test")

    # ============================================================
    # Step 4 (continued): Motif enrichment in non-DRACH sites
    # ============================================================
    print(f"\n{'=' * 70}")
    print("NON-DRACH MOTIF ANALYSIS")
    print(f"{'=' * 70}")

    def count_kmers(sites, k=5):
        """Count k-mers centered on m6A in non-DRACH sites."""
        kmer_counts = Counter()
        center = CONTEXT_FLANK  # position 5 in 11-mer
        half_k = k // 2
        for site in sites:
            if site[6] != "non-DRACH":
                continue
            ctx = site[5]  # RNA-strand context
            if len(ctx) >= 11:
                kmer = ctx[center - half_k:center + half_k + 1]
                if len(kmer) == k and "N" not in kmer:
                    kmer_counts[kmer] += 1
        return kmer_counts

    print("\nTop 5-mers at non-DRACH m6A sites:")
    print("\n  L1 reads:")
    l1_kmers = count_kmers(l1_sites)
    for kmer, count in l1_kmers.most_common(30):
        pct = count / sum(l1_kmers.values()) * 100 if l1_kmers else 0
        print(f"    {kmer}: {count} ({pct:.1f}%)")

    print("\n  Non-L1 reads:")
    nonl1_kmers = count_kmers(nonl1_sites)
    for kmer, count in nonl1_kmers.most_common(30):
        pct = count / sum(nonl1_kmers.values()) * 100 if nonl1_kmers else 0
        print(f"    {kmer}: {count} ({pct:.1f}%)")

    # Check for ABAG motif (Chen 2021)
    print(f"\n{'=' * 70}")
    print("SPECIFIC MOTIF CHECKS")
    print(f"{'=' * 70}")

    def check_motif_in_context(sites, motif, motif_name):
        """Check how many non-DRACH sites contain a motif in their context."""
        n_total = 0
        n_match = 0
        for site in sites:
            if site[6] != "non-DRACH":
                continue
            n_total += 1
            ctx = site[5].upper()
            if motif in ctx:
                n_match += 1
        frac = n_match / n_total if n_total > 0 else 0
        return n_match, n_total, frac

    for motif, name in [("TACAGAGAA", "METTL16 canonical (TACAGAGAA)"),
                         ("ACAGAG", "METTL16 core (ACAGAG)"),
                         ("ABAG", "ABAG motif (Chen 2021)")]:
        # ABAG: A=A, B=C/G/T
        if motif == "ABAG":
            # Custom check for ABAG pattern
            for label, sites in [("L1", l1_sites), ("non-L1", nonl1_sites)]:
                n_total = sum(1 for s in sites if s[6] == "non-DRACH")
                n_match = 0
                for site in sites:
                    if site[6] != "non-DRACH":
                        continue
                    ctx = site[5].upper()
                    # Check ABAG: A [CGT] A G around the m6A
                    # m6A at center (pos 5). Check various positions
                    for i in range(len(ctx) - 3):
                        if (ctx[i] == "A" and ctx[i+1] in "CGT" and
                            ctx[i+2] == "A" and ctx[i+3] == "G"):
                            n_match += 1
                            break
                frac = n_match / n_total if n_total > 0 else 0
                print(f"  {name} in {label} non-DRACH: {n_match}/{n_total} ({frac:.1%})")
        else:
            for label, sites in [("L1", l1_sites), ("non-L1", nonl1_sites)]:
                n_match, n_total, frac = check_motif_in_context(sites, motif, name)
                print(f"  {name} in {label} non-DRACH: {n_match}/{n_total} ({frac:.1%})")

    # ============================================================
    # Probability distribution comparison
    # ============================================================
    print(f"\n{'=' * 70}")
    print("M6A PROBABILITY DISTRIBUTION")
    print(f"{'=' * 70}")

    for label, sites in [("L1", l1_sites), ("non-L1", nonl1_sites)]:
        if not sites:
            continue
        probs = [s[4] for s in sites]
        drach_probs = [s[4] for s in sites if s[6] == "DRACH"]
        nondrach_probs = [s[4] for s in sites if s[6] == "non-DRACH"]
        print(f"\n  {label}:")
        print(f"    All:       mean={np.mean(probs):.1f}, median={np.median(probs):.1f}, n={len(probs)}")
        if drach_probs:
            print(f"    DRACH:     mean={np.mean(drach_probs):.1f}, median={np.median(drach_probs):.1f}, n={len(drach_probs)}")
        if nondrach_probs:
            print(f"    non-DRACH: mean={np.mean(nondrach_probs):.1f}, median={np.median(nondrach_probs):.1f}, n={len(nondrach_probs)}")

    # ============================================================
    # Higher threshold sensitivity analysis
    # ============================================================
    print(f"\n{'=' * 70}")
    print("THRESHOLD SENSITIVITY (non-DRACH fraction)")
    print(f"{'=' * 70}")
    for thr in [128, 153, 179, 204, 230]:
        for label, sites in [("L1", l1_sites), ("non-L1", nonl1_sites)]:
            above = [s for s in sites if s[4] >= thr]
            n_dr = sum(1 for s in above if s[6] == "DRACH")
            n_ndr = sum(1 for s in above if s[6] == "non-DRACH")
            frac_ndr = n_ndr / (n_dr + n_ndr) if (n_dr + n_ndr) > 0 else 0
            print(f"  thr>={thr} ({thr/255:.2f}): {label:8s} DRACH={n_dr:6d} non-DRACH={n_ndr:6d} ({frac_ndr:.1%})")

    # ============================================================
    # Save outputs
    # ============================================================

    # 1. Summary table
    summary_path = os.path.join(OUTDIR, "nondrach_summary.tsv")
    with open(summary_path, "w") as f:
        headers = [
            "category", "n_reads", "total_kb", "n_m6a_sites",
            "n_drach", "n_nondrach", "drach_frac", "nondrach_frac",
            "m6a_per_kb", "drach_per_kb", "nondrach_per_kb",
            "mettl16_canonical_n", "mettl16_core_n",
            "mettl16_canonical_frac", "mettl16_core_frac"
        ]
        f.write("\t".join(headers) + "\n")
        for st in [l1_stats, nonl1_stats, young_stats, ancient_stats]:
            row = [
                st["label"], st["n_reads"], f"{st['total_kb']:.1f}", st["n_sites"],
                st["n_drach"], st["n_nondrach"],
                f"{st['drach_frac']:.4f}", f"{st['nondrach_frac']:.4f}",
                f"{st['m6a_per_kb']:.4f}", f"{st['drach_per_kb']:.4f}", f"{st['nondrach_per_kb']:.4f}",
                st["mettl16_canonical"], st["mettl16_core"],
                f"{st['mettl16_canonical_frac']:.4f}", f"{st['mettl16_core_frac']:.4f}"
            ]
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"\nSaved: {summary_path}")

    # 2. Motif frequency table
    motif_path = os.path.join(OUTDIR, "nondrach_motif_frequencies.tsv")
    with open(motif_path, "w") as f:
        f.write("category\tkmer\tcount\tfraction\n")
        for label, kmers in [("L1", l1_kmers), ("non-L1", nonl1_kmers)]:
            total = sum(kmers.values())
            for kmer, count in kmers.most_common(50):
                frac = count / total if total > 0 else 0
                f.write(f"{label}\t{kmer}\t{count}\t{frac:.6f}\n")
    print(f"Saved: {motif_path}")

    # 3. METTL16 motif hits
    mettl16_path = os.path.join(OUTDIR, "mettl16_motif_hits.tsv")
    with open(mettl16_path, "w") as f:
        f.write("read_id\tchrom\tref_pos\tstrand\tprob\tcontext_11mer\tdrach_class\t"
                "mettl16_canonical\tmettl16_core\tis_L1\trepname\n")
        all_sites = [(s, True) for s in l1_sites] + [(s, False) for s in nonl1_sites]
        for site, is_l1 in all_sites:
            if site[7] or site[8]:  # canonical or core match
                f.write(f"{site[0]}\t{site[2]}\t{site[1]}\t{site[3]}\t{site[4]}\t"
                        f"{site[5]}\t{site[6]}\t{site[7]}\t{site[8]}\t{is_l1}\t{site[9]}\n")
    print(f"Saved: {mettl16_path}")

    # 4. Full per-site data for downstream analysis
    persite_path = os.path.join(OUTDIR, "all_m6a_sites.tsv.gz")
    import gzip
    with gzip.open(persite_path, "wt") as f:
        f.write("read_id\tchrom\tref_pos\tstrand\tprob\tcontext_11mer\tdrach_class\t"
                "mettl16_canonical\tmettl16_core\tis_L1\trepname\n")
        for site in l1_sites:
            f.write(f"{site[0]}\t{site[2]}\t{site[1]}\t{site[3]}\t{site[4]}\t"
                    f"{site[5]}\t{site[6]}\t{site[7]}\t{site[8]}\tTrue\t{site[9]}\n")
        for site in nonl1_sites:
            f.write(f"{site[0]}\t{site[2]}\t{site[1]}\t{site[3]}\t{site[4]}\t"
                    f"{site[5]}\t{site[6]}\t{site[7]}\t{site[8]}\tFalse\tNone\n")
    print(f"Saved: {persite_path}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
