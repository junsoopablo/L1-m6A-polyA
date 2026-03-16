#!/usr/bin/env python3
"""
Dorado RNA004 Per-Read m6A/kb Validation
=========================================
Validates MAFIA-detected L1 m6A enrichment (1.79x) using dorado basecaller
on RNA004 chemistry.

Metric: per-read m6A sites/kb (identical to MAFIA pipeline metric).

MM/ML parsing note:
  - Dorado MM tag contains three modification types on A: A+17596, A+69426, A+a
  - A+a is m6A. ML values are sequential: first N for 17596, next N for 69426, last N for a.
  - CRITICAL: For reverse-strand reads (flag=16), SEQ is reverse-complemented.
    The A+a skip values count T positions in SEQ (since original RNA A becomes T after RC).
    For forward reads, skip values count A positions in SEQ.

Usage:
  conda run -n research python dorado_m6a_per_read_validation.py
"""

import os
import re
import sys
import gzip
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left

import numpy as np
import pandas as pd
import pysam
from scipy import stats

# ============================================================
# Configuration
# ============================================================
BAM_PATH = "/blaze/junsoopablo/dorado_validation/HeLa_1_1_m6A/HeLa_1_1.dorado.m6A.sorted.bam"
REF_PATH = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta"
L1_BED = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/L1_TE_L1_family.bed"
TE_GTF = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/hg38_rmsk_TE.gtf"
HUMAN_GTF = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.gtf"

OUTDIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/"
              "topic_06_dorado_validation/dorado_m6a_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

M6A_PROB_THRESHOLD = 204  # >= 204/255 (80%)
L1_OVERLAP_FRAC = 0.10    # 10% overlap for stage 1
EXON_OVERLAP_MIN = 100    # >= 100bp exon overlap -> exclude

YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

# Subsample non-L1 reads to limit runtime
MAX_NONL1_READS = 20000

# DRACH motif for optional analysis
DRACH_RE = re.compile(r"^[AGT][AG]AC[ACT]$")

np.random.seed(42)


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
    """Find L1 overlap for a read. Returns (overlap_frac, best_repname, total_overlap_bp)."""
    if chrom not in l1_data:
        return 0.0, None, 0
    intervals = l1_data[chrom]
    read_len = read_end - read_start
    if read_len <= 0:
        return 0.0, None, 0

    left = bisect_left(intervals, (read_start,))
    start_idx = max(0, left - 1)
    best_overlap, best_repname = 0, None
    total_overlap = 0
    for i in range(start_idx, len(intervals)):
        istart, iend, repname = intervals[i]
        if istart >= read_end:
            break
        ov = min(read_end, iend) - max(read_start, istart)
        if ov > 0:
            total_overlap += ov
        if ov > best_overlap:
            best_overlap = ov
            best_repname = repname
    overlap_frac = total_overlap / read_len if read_len > 0 else 0.0
    return overlap_frac, best_repname, total_overlap


# ============================================================
# MM/ML parsing — CORRECT handling of reverse strand
# ============================================================
def parse_m6a_from_read(read):
    """
    Parse A+a modification from dorado MM/ML tags.

    CRITICAL: For reverse-strand reads, the stored SEQ is reverse-complemented.
    In MM tag format, A+a on the original strand means:
      - Forward read: count A positions in SEQ
      - Reverse read: count T positions in SEQ (original A -> RC -> T)

    Returns: dict {read_seq_position: ml_probability} for each called m6A site.
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
            # For reverse reads: original A is stored as T in SEQ
            if is_reverse:
                target_base = "T"
            else:
                target_base = "A"

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
# Reference context for DRACH analysis
# ============================================================
def get_ref_context(ref_fasta, chrom, ref_pos, is_reverse, flank=2):
    """Get (2*flank+1)-mer centered on ref_pos, oriented to RNA strand."""
    start = ref_pos - flank
    end = ref_pos + flank + 1
    if start < 0:
        return None
    try:
        ctx = ref_fasta.fetch(chrom, start, end).upper()
    except (ValueError, KeyError):
        return None
    if len(ctx) != 2 * flank + 1 or "N" in ctx:
        return None
    if is_reverse:
        comp = str.maketrans("ACGT", "TGCA")
        ctx = ctx.translate(comp)[::-1]
    return ctx


# ============================================================
# 2-stage L1 filtering
# ============================================================
def generate_exon_bed(gtf_path, out_bed):
    """Extract exon regions from Human.gtf -> 3-col BED."""
    if os.path.exists(out_bed):
        return
    print(f"  Generating exon BED from {gtf_path}...")
    with open(gtf_path) as gtf, open(out_bed, 'w') as out:
        for line in gtf:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 9 or fields[2] != 'exon':
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            out.write(f"{chrom}\t{start}\t{end}\n")


def generate_l1_strand_bed(gtf_path, out_bed):
    """Extract L1 TE regions with strand from TE GTF -> 6-col BED."""
    if os.path.exists(out_bed):
        return
    print(f"  Generating L1 strand BED from {gtf_path}...")
    with open(gtf_path) as gtf, open(out_bed, 'w') as out:
        for line in gtf:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 9 or 'family_id "L1"' not in fields[8]:
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            strand = fields[6] if len(fields) > 6 else '.'
            m = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid = m.group(1) if m else '.'
            out.write(f"{chrom}\t{start}\t{end}\t{tid}\t0\t{strand}\n")


def filter_l1_reads_2stage(bam_path, tmpdir):
    """
    2-stage L1 filtering matching MAFIA pipeline:
      Stage 1: >= 10% overlap with L1 regions
      Stage 2: exclude spliced (CIGAR N), exclude >= 100bp exon overlap, strand match

    Returns:
      l1_read_info: dict {read_id: {'subfamily': str, 'age_class': str}}
      ctrl_ids: set of non-L1 read IDs (subsampled)
      read_meta: dict {read_id: {'ref_len': int, 'is_reverse': bool, 'chrom': str, 'start': int, 'end': int}}
    """
    print("\n=== Stage 1+2 L1 Filtering ===")

    reads_bed = os.path.join(tmpdir, 'reads.bed')
    overlaps_tsv = os.path.join(tmpdir, 'overlaps.tsv')
    exon_bed = os.path.join(tmpdir, 'exons.bed')
    l1_strand_bed = os.path.join(tmpdir, 'l1_with_strand.bed')
    exon_overlaps_tsv = os.path.join(tmpdir, 'exon_overlaps.tsv')
    strand_overlaps_tsv = os.path.join(tmpdir, 'strand_overlaps.tsv')

    # Generate reference BEDs
    generate_exon_bed(HUMAN_GTF, exon_bed)
    generate_l1_strand_bed(TE_GTF, l1_strand_bed)

    # -- Scan BAM: collect read metadata --
    print("  Extracting read info from BAM...")
    read_meta = {}
    spliced_ids = set()
    n_total = 0

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            rid = read.query_name
            read_meta[rid] = {
                'ref_len': read.reference_length,
                'query_len': read.query_length,
                'is_reverse': read.is_reverse,
                'chrom': read.reference_name,
                'start': read.reference_start,
                'end': read.reference_end,
            }
            if read.cigartuples:
                for op, _ in read.cigartuples:
                    if op == 3:  # N = splice junction
                        spliced_ids.add(rid)
                        break
            n_total += 1
            if n_total % 500000 == 0:
                print(f"    ... {n_total:,} reads scanned")

    print(f"  Total primary mapped reads: {n_total:,}")
    print(f"  Spliced reads: {len(spliced_ids):,}")

    # -- Stage 1: bedtools intersect for 10% overlap --
    print("  Stage 1: bedtools intersect (10% overlap)...")
    subprocess.run(
        f"bedtools bamtobed -split -i {bam_path} > {reads_bed}",
        shell=True, check=True
    )
    subprocess.run(
        f"bedtools intersect -a {reads_bed} -b {L1_BED} -wo > {overlaps_tsv}",
        shell=True, check=True
    )

    # Compute overlap fraction per read
    overlap_sum = defaultdict(int)
    # Also track best-overlapping L1 subfamily
    best_l1_overlap = defaultdict(lambda: (0, None))  # rid -> (overlap_bp, repname)
    with open(overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            rid = fields[3]
            repname = fields[9]  # col 4 of L1 BED (repName)
            ov = int(fields[-1])
            overlap_sum[rid] += ov
            cur_best, cur_name = best_l1_overlap[rid]
            if ov > cur_best:
                best_l1_overlap[rid] = (ov, repname)

    stage1_l1 = set()
    l1_subfamily = {}
    for rid, total_ov in overlap_sum.items():
        if rid in read_meta:
            qlen = read_meta[rid]['query_len']
            if qlen and qlen > 0 and total_ov / qlen >= L1_OVERLAP_FRAC:
                stage1_l1.add(rid)
                _, repname = best_l1_overlap[rid]
                l1_subfamily[rid] = repname

    print(f"  Stage 1 L1 reads: {len(stage1_l1):,}")

    # -- Stage 2a: Exclude spliced --
    after_splice = stage1_l1 - spliced_ids
    print(f"  Stage 2a: -{len(stage1_l1) - len(after_splice):,} spliced -> {len(after_splice):,}")

    # -- Stage 2b: Exclude >= 100bp exon overlap --
    l1_reads_bed = os.path.join(tmpdir, 'l1_reads.bed')
    with open(reads_bed) as fin, open(l1_reads_bed, 'w') as fout:
        for line in fin:
            rid = line.strip().split('\t')[3]
            if rid in after_splice:
                fout.write(line)

    subprocess.run(
        f"bedtools intersect -a {l1_reads_bed} -b {exon_bed} -wo > {exon_overlaps_tsv}",
        shell=True, check=True
    )

    exon_overlap = defaultdict(int)
    with open(exon_overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            rid = fields[3]
            ov = int(fields[-1])
            if ov > exon_overlap[rid]:
                exon_overlap[rid] = ov

    exon_exclude = {rid for rid, ov in exon_overlap.items() if ov >= EXON_OVERLAP_MIN}
    after_exon = after_splice - exon_exclude
    print(f"  Stage 2b: -{len(exon_exclude):,} exon-overlapping -> {len(after_exon):,}")

    # -- Stage 2c: Strand match --
    subprocess.run(
        f"bedtools intersect -a {l1_reads_bed} -b {l1_strand_bed} -wo > {strand_overlaps_tsv}",
        shell=True, check=True
    )

    read_l1_strand = {}
    read_l1_overlap = defaultdict(int)
    with open(strand_overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            rid = fields[3]
            l1_strand = fields[11]  # strand col of l1_strand_bed
            ov = int(fields[-1])
            if ov > read_l1_overlap[rid]:
                read_l1_overlap[rid] = ov
                read_l1_strand[rid] = l1_strand

    strand_exclude = set()
    for rid in after_exon:
        if rid in read_l1_strand and rid in read_meta:
            read_strand = '-' if read_meta[rid]['is_reverse'] else '+'
            if read_strand != read_l1_strand[rid]:
                strand_exclude.add(rid)

    stage2_l1 = after_exon - strand_exclude
    print(f"  Stage 2c: -{len(strand_exclude):,} strand-mismatched -> {len(stage2_l1):,}")

    # -- Build L1 read info --
    l1_read_info = {}
    for rid in stage2_l1:
        subfam = l1_subfamily.get(rid, "Unknown")
        age_class = "Young" if subfam in YOUNG_SUBFAMILIES else "Ancient"
        l1_read_info[rid] = {'subfamily': subfam, 'age_class': age_class}

    # -- Control: no ANY L1 overlap --
    any_l1_overlap = set(overlap_sum.keys())
    ctrl_pool = set(read_meta.keys()) - any_l1_overlap
    print(f"  Control pool (no L1 overlap): {len(ctrl_pool):,}")

    # Subsample
    if len(ctrl_pool) > MAX_NONL1_READS:
        rng = np.random.default_rng(42)
        ctrl_ids = set(rng.choice(list(ctrl_pool), MAX_NONL1_READS, replace=False))
    else:
        ctrl_ids = ctrl_pool
    print(f"  Control subsampled: {len(ctrl_ids):,}")

    return l1_read_info, ctrl_ids, read_meta


# ============================================================
# Main per-read m6A/kb calculation
# ============================================================
def compute_per_read_m6a(bam_path, l1_read_info, ctrl_ids, read_meta, ref_path):
    """
    Iterate BAM once, parse m6A for L1 and control reads.
    Returns DataFrame with per-read m6A stats.
    """
    print("\n=== Computing per-read m6A/kb ===")

    target_ids = set(l1_read_info.keys()) | ctrl_ids
    print(f"  Target reads: {len(target_ids):,} (L1={len(l1_read_info):,}, Ctrl={len(ctrl_ids):,})")

    # Open reference for DRACH context
    ref_fasta = pysam.FastaFile(ref_path)

    records = []
    n_processed = 0
    n_found = 0
    n_total_reads = 0

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            n_total_reads += 1
            if n_total_reads % 500000 == 0:
                print(f"    ... {n_total_reads:,} reads scanned, {n_found:,}/{len(target_ids):,} found")

            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            rid = read.query_name
            if rid not in target_ids:
                continue

            n_found += 1
            ref_len = read.reference_length
            if ref_len is None or ref_len <= 0:
                continue

            aligned_kb = ref_len / 1000.0

            # Parse m6A sites
            m6a_map = parse_m6a_from_read(read)

            # Count high-confidence m6A
            n_m6a_total = len(m6a_map)
            n_m6a_high = sum(1 for p in m6a_map.values() if p >= M6A_PROB_THRESHOLD)
            m6a_per_kb = n_m6a_high / aligned_kb

            # DRACH analysis: map read positions to reference positions
            n_drach = 0
            n_nondrach = 0
            if m6a_map:
                pairs = read.get_aligned_pairs(matches_only=False)
                read_to_ref = {}
                for qpos, rpos in pairs:
                    if qpos is not None and rpos is not None:
                        read_to_ref[qpos] = rpos

                chrom = read.reference_name
                is_rev = read.is_reverse

                for read_pos, prob in m6a_map.items():
                    if prob < M6A_PROB_THRESHOLD:
                        continue
                    ref_pos = read_to_ref.get(read_pos)
                    if ref_pos is None:
                        continue
                    ctx = get_ref_context(ref_fasta, chrom, ref_pos, is_rev, flank=2)
                    if ctx and len(ctx) == 5:
                        if DRACH_RE.match(ctx):
                            n_drach += 1
                        else:
                            n_nondrach += 1

            drach_per_kb = n_drach / aligned_kb
            nondrach_per_kb = n_nondrach / aligned_kb

            # Classification
            if rid in l1_read_info:
                info = l1_read_info[rid]
                category = "L1"
                subfamily = info['subfamily']
                age_class = info['age_class']
            else:
                category = "Control"
                subfamily = "NA"
                age_class = "NA"

            records.append({
                'read_id': rid,
                'category': category,
                'subfamily': subfamily,
                'age_class': age_class,
                'ref_len': ref_len,
                'aligned_kb': aligned_kb,
                'n_m6a_called': n_m6a_total,
                'n_m6a_high': n_m6a_high,
                'm6a_per_kb': m6a_per_kb,
                'n_drach': n_drach,
                'n_nondrach': n_nondrach,
                'drach_per_kb': drach_per_kb,
                'nondrach_per_kb': nondrach_per_kb,
            })

            if n_found >= len(target_ids):
                break

    ref_fasta.close()

    print(f"  Processed {n_found:,} target reads out of {n_total_reads:,} total BAM reads")
    df = pd.DataFrame(records)
    return df


# ============================================================
# Analysis and summary
# ============================================================
def analyze_results(df):
    """Compute summary statistics and comparisons."""
    print("\n" + "=" * 70)
    print("DORADO RNA004 PER-READ m6A/kb VALIDATION RESULTS")
    print("=" * 70)

    # --- Basic counts ---
    l1_df = df[df['category'] == 'L1']
    ctrl_df = df[df['category'] == 'Control']
    young_df = l1_df[l1_df['age_class'] == 'Young']
    ancient_df = l1_df[l1_df['age_class'] == 'Ancient']

    print(f"\nRead counts:")
    print(f"  L1 total:  {len(l1_df):,}")
    print(f"    Young:   {len(young_df):,}")
    print(f"    Ancient: {len(ancient_df):,}")
    print(f"  Control:   {len(ctrl_df):,}")

    # --- m6A/kb summary ---
    print(f"\n{'='*70}")
    print("m6A/kb (threshold >= {}/255 = {:.0f}%)".format(M6A_PROB_THRESHOLD, M6A_PROB_THRESHOLD/255*100))
    print(f"{'='*70}")

    for label, sub_df in [("L1 (all)", l1_df), ("Young L1", young_df),
                           ("Ancient L1", ancient_df), ("Control", ctrl_df)]:
        vals = sub_df['m6a_per_kb']
        if len(vals) == 0:
            print(f"  {label:15s}: N=0")
            continue
        med = np.median(vals)
        mean = np.mean(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        print(f"  {label:15s}: N={len(vals):>6,}  median={med:.3f}  mean={mean:.3f}  IQR=[{q25:.3f}, {q75:.3f}]")

    # --- L1 vs Control comparison ---
    if len(l1_df) > 0 and len(ctrl_df) > 0:
        l1_med = np.median(l1_df['m6a_per_kb'])
        ctrl_med = np.median(ctrl_df['m6a_per_kb'])
        ratio = l1_med / ctrl_med if ctrl_med > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(l1_df['m6a_per_kb'], ctrl_df['m6a_per_kb'], alternative='two-sided')
        print(f"\n  L1 vs Control: {l1_med:.3f} vs {ctrl_med:.3f} = {ratio:.2f}x  (MWU P={pval:.2e})")

    # --- Young vs Ancient ---
    if len(young_df) > 0 and len(ancient_df) > 0:
        y_med = np.median(young_df['m6a_per_kb'])
        a_med = np.median(ancient_df['m6a_per_kb'])
        ya_ratio = y_med / a_med if a_med > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(young_df['m6a_per_kb'], ancient_df['m6a_per_kb'], alternative='two-sided')
        print(f"  Young vs Ancient: {y_med:.3f} vs {a_med:.3f} = {ya_ratio:.2f}x  (MWU P={pval:.2e})")

    # --- DRACH analysis ---
    print(f"\n{'='*70}")
    print("DRACH m6A/kb")
    print(f"{'='*70}")

    for label, sub_df in [("L1 (all)", l1_df), ("Young L1", young_df),
                           ("Ancient L1", ancient_df), ("Control", ctrl_df)]:
        vals = sub_df['drach_per_kb']
        if len(vals) == 0:
            continue
        med = np.median(vals)
        mean = np.mean(vals)
        print(f"  {label:15s}: median={med:.3f}  mean={mean:.3f}")

    if len(l1_df) > 0 and len(ctrl_df) > 0:
        l1_drach = np.median(l1_df['drach_per_kb'])
        ctrl_drach = np.median(ctrl_df['drach_per_kb'])
        drach_ratio = l1_drach / ctrl_drach if ctrl_drach > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(l1_df['drach_per_kb'], ctrl_df['drach_per_kb'], alternative='two-sided')
        print(f"\n  L1 vs Control DRACH: {l1_drach:.3f} vs {ctrl_drach:.3f} = {drach_ratio:.2f}x  (MWU P={pval:.2e})")

    # --- non-DRACH analysis ---
    print(f"\n{'='*70}")
    print("non-DRACH m6A/kb")
    print(f"{'='*70}")

    for label, sub_df in [("L1 (all)", l1_df), ("Young L1", young_df),
                           ("Ancient L1", ancient_df), ("Control", ctrl_df)]:
        vals = sub_df['nondrach_per_kb']
        if len(vals) == 0:
            continue
        med = np.median(vals)
        mean = np.mean(vals)
        print(f"  {label:15s}: median={med:.3f}  mean={mean:.3f}")

    if len(l1_df) > 0 and len(ctrl_df) > 0:
        l1_nd = np.median(l1_df['nondrach_per_kb'])
        ctrl_nd = np.median(ctrl_df['nondrach_per_kb'])
        nd_ratio = l1_nd / ctrl_nd if ctrl_nd > 0 else float('inf')
        stat, pval = stats.mannwhitneyu(l1_df['nondrach_per_kb'], ctrl_df['nondrach_per_kb'], alternative='two-sided')
        print(f"\n  L1 vs Control non-DRACH: {l1_nd:.3f} vs {ctrl_nd:.3f} = {nd_ratio:.2f}x  (MWU P={pval:.2e})")

    # --- DRACH fraction ---
    print(f"\n{'='*70}")
    print("DRACH fraction (of high-confidence m6A)")
    print(f"{'='*70}")
    for label, sub_df in [("L1 (all)", l1_df), ("Control", ctrl_df)]:
        total_drach = sub_df['n_drach'].sum()
        total_nondrach = sub_df['n_nondrach'].sum()
        total = total_drach + total_nondrach
        frac = total_drach / total if total > 0 else 0
        print(f"  {label:15s}: DRACH={total_drach:,}, non-DRACH={total_nondrach:,}, fraction={frac:.3f}")

    # --- MAFIA comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON WITH MAFIA (RNA002, threshold 0.80)")
    print(f"{'='*70}")
    print(f"{'Metric':<25s} {'MAFIA':>10s} {'Dorado':>10s}")
    print("-" * 50)

    mafia_vals = {
        'L1 m6A/kb': 2.891,
        'Ctrl m6A/kb': 1.618,
        'L1/Ctrl ratio': 1.79,
        'Young L1 m6A/kb': 4.16,
        'Ancient L1 m6A/kb': 2.69,
        'Young/Ancient ratio': 1.55,
    }

    dorado_vals = {}
    if len(l1_df) > 0:
        dorado_vals['L1 m6A/kb'] = np.median(l1_df['m6a_per_kb'])
    if len(ctrl_df) > 0:
        dorado_vals['Ctrl m6A/kb'] = np.median(ctrl_df['m6a_per_kb'])
    if 'L1 m6A/kb' in dorado_vals and 'Ctrl m6A/kb' in dorado_vals:
        dorado_vals['L1/Ctrl ratio'] = dorado_vals['L1 m6A/kb'] / dorado_vals['Ctrl m6A/kb'] if dorado_vals['Ctrl m6A/kb'] > 0 else float('inf')
    if len(young_df) > 0:
        dorado_vals['Young L1 m6A/kb'] = np.median(young_df['m6a_per_kb'])
    if len(ancient_df) > 0:
        dorado_vals['Ancient L1 m6A/kb'] = np.median(ancient_df['m6a_per_kb'])
    if 'Young L1 m6A/kb' in dorado_vals and 'Ancient L1 m6A/kb' in dorado_vals:
        dorado_vals['Young/Ancient ratio'] = dorado_vals['Young L1 m6A/kb'] / dorado_vals['Ancient L1 m6A/kb'] if dorado_vals['Ancient L1 m6A/kb'] > 0 else float('inf')

    for key in mafia_vals:
        m_val = mafia_vals[key]
        d_val = dorado_vals.get(key, float('nan'))
        print(f"  {key:<23s} {m_val:>10.3f} {d_val:>10.3f}")

    # --- Subfamily breakdown ---
    print(f"\n{'='*70}")
    print("L1 SUBFAMILY BREAKDOWN (top 10 by count)")
    print(f"{'='*70}")
    subfam_stats = l1_df.groupby('subfamily')['m6a_per_kb'].agg(['count', 'median', 'mean'])
    subfam_stats = subfam_stats.sort_values('count', ascending=False).head(10)
    print(f"{'Subfamily':<15s} {'N':>7s} {'Median':>10s} {'Mean':>10s}")
    print("-" * 45)
    for subfam, row in subfam_stats.iterrows():
        print(f"  {subfam:<13s} {int(row['count']):>7,} {row['median']:>10.3f} {row['mean']:>10.3f}")

    return dorado_vals


# ============================================================
# Main
# ============================================================
def main():
    print("Dorado RNA004 Per-Read m6A/kb Validation")
    print(f"BAM: {BAM_PATH}")
    print(f"Threshold: ML >= {M6A_PROB_THRESHOLD} ({M6A_PROB_THRESHOLD/255*100:.1f}%)")

    # Step 1: L1 filtering
    with tempfile.TemporaryDirectory(prefix='dorado_m6a_') as tmpdir:
        l1_read_info, ctrl_ids, read_meta = filter_l1_reads_2stage(BAM_PATH, tmpdir)

    # Step 2: Per-read m6A computation
    df = compute_per_read_m6a(BAM_PATH, l1_read_info, ctrl_ids, read_meta, REF_PATH)

    # Step 3: Save per-read data
    outfile_gz = OUTDIR / 'dorado_per_read_m6a.tsv.gz'
    df.to_csv(outfile_gz, sep='\t', index=False, compression='gzip')
    print(f"\nPer-read data saved: {outfile_gz}")

    # Step 4: Analysis
    dorado_vals = analyze_results(df)

    # Step 5: Save summary TSV
    summary_rows = []
    for cat in ['L1', 'Young', 'Ancient', 'Control']:
        if cat == 'L1':
            sub = df[df['category'] == 'L1']
        elif cat == 'Young':
            sub = df[(df['category'] == 'L1') & (df['age_class'] == 'Young')]
        elif cat == 'Ancient':
            sub = df[(df['category'] == 'L1') & (df['age_class'] == 'Ancient')]
        else:
            sub = df[df['category'] == 'Control']

        if len(sub) == 0:
            continue

        vals = sub['m6a_per_kb']
        drach_vals = sub['drach_per_kb']
        nondrach_vals = sub['nondrach_per_kb']

        summary_rows.append({
            'category': cat,
            'n_reads': len(sub),
            'm6a_per_kb_median': np.median(vals),
            'm6a_per_kb_mean': np.mean(vals),
            'm6a_per_kb_q25': np.percentile(vals, 25),
            'm6a_per_kb_q75': np.percentile(vals, 75),
            'drach_per_kb_median': np.median(drach_vals),
            'drach_per_kb_mean': np.mean(drach_vals),
            'nondrach_per_kb_median': np.median(nondrach_vals),
            'nondrach_per_kb_mean': np.mean(nondrach_vals),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTDIR / 'dorado_per_read_m6a_summary.tsv'
    summary_df.to_csv(summary_file, sep='\t', index=False, float_format='%.4f')
    print(f"Summary saved: {summary_file}")

    print("\nDone.")


if __name__ == '__main__':
    main()
