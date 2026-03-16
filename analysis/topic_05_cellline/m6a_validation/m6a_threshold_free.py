#!/usr/bin/env python3
"""
m6a_threshold_free.py

Threshold-free m6A analysis: use raw ML probabilities instead of binary cutoff.

For each DRACH motif site MAFIA evaluated, it assigns a probability (0-255).
Instead of counting sites >=128, we use:
  - Mean probability per motif site = sum(all probs) / n_DRACH_sites
  - Continuous "modification rate" (0-255 scale, or /255 for 0-1)

Comparisons:
  1. L1 vs Control (per-site mean probability)
  2. L1 body vs flanking (within-read, per-site mean probability)
  3. Threshold sweep at multiple cutoffs

chemodCode 21891 = m6A (VERIFIED)
"""

import pysam
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import os, sys

# === Configuration ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
REF_FASTA_PATH = f"{PROJECT}/reference/Human.fasta"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# DRACH motifs (18 total)
M6A_MOTIFS = sorted([
    'AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT',
    'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT',
    'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT',
])

def revcomp(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

M6A_MOTIFS_RC = [revcomp(m) for m in M6A_MOTIFS]

YOUNG = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

BASE_GROUPS = [
    'A549_4', 'A549_5', 'A549_6',
    'H9_2', 'H9_3', 'H9_4',
    'HeLa_1', 'HeLa_2', 'HeLa_3',
    'HepG2_5', 'HepG2_6',
    'HEYA8_1', 'HEYA8_2', 'HEYA8_3',
    'K562_4', 'K562_5', 'K562_6',
    'MCF7_2', 'MCF7_3', 'MCF7_4',
    'SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3',
]


# =====================================================================
# HELPERS
# =====================================================================

def get_aligned_blocks(read):
    """Get aligned reference blocks as list of (ref_start, ref_end). M/=/X only."""
    if read.cigartuples is None:
        return []
    blocks = []
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            blocks.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op in (2, 3):  # D, N
            ref_pos += length
    return blocks


def get_ref_seq_for_region(ref_fasta, chrom, blocks, region_start, region_end):
    """Get reference sequence for aligned blocks within a region."""
    parts = []
    for bs, be in blocks:
        s = max(bs, region_start)
        e = min(be, region_end)
        if s >= e:
            continue
        try:
            parts.append(ref_fasta.fetch(chrom, s, e).upper())
        except Exception:
            pass
    return ''.join(parts)


def get_full_aligned_ref_seq(ref_fasta, chrom, blocks):
    """Get full reference sequence for all aligned blocks."""
    parts = []
    total_bp = 0
    for bs, be in blocks:
        try:
            seq = ref_fasta.fetch(chrom, bs, be).upper()
            parts.append(seq)
            total_bp += (be - bs)
        except Exception:
            pass
    return ''.join(parts), total_bp


def count_motifs(seq, is_reverse):
    """Count m6A DRACH motif sites in sequence."""
    motifs = M6A_MOTIFS_RC if is_reverse else M6A_MOTIFS
    count = 0
    for motif in motifs:
        idx = 0
        while True:
            pos = seq.find(motif, idx)
            if pos == -1:
                break
            count += 1
            idx = pos + 1
    return count


def parse_all_m6a_with_probs(read):
    """
    Parse ALL m6A positions and their ML probabilities from MM/ML tags.
    NO threshold applied — returns every position MAFIA evaluated.
    chemodCode 21891 = m6A.
    Returns list of (read_relative_pos, probability_0_to_255).
    """
    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else \
                 read.get_tag('Mm') if read.has_tag('Mm') else None
        ml_tag = read.get_tag('ML') if read.has_tag('ML') else \
                 read.get_tag('Ml') if read.has_tag('Ml') else None
    except Exception:
        return []

    if mm_tag is None or ml_tag is None:
        return []

    ml_values = list(ml_tag)
    seq = read.query_sequence
    if seq is None:
        return []

    results = []
    ml_idx = 0

    for entry in mm_tag.rstrip(';').split(';'):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(',')
        mod_spec = parts[0]
        skips = [int(x) for x in parts[1:]] if len(parts) > 1 else []
        n_pos = len(skips)

        if n_pos <= 0:
            ml_idx += 0
            continue

        if '21891' in mod_spec:
            base_code = mod_spec[0]
            if base_code == 'N':
                base_positions = list(range(len(seq)))
            else:
                base_positions = [i for i, c in enumerate(seq) if c == base_code]

            bp_idx = 0
            for i, skip in enumerate(skips):
                bp_idx += skip
                if bp_idx < len(base_positions):
                    read_pos = base_positions[bp_idx]
                    prob = ml_values[ml_idx + i] if (ml_idx + i) < len(ml_values) else 0
                    results.append((read_pos, prob))
                bp_idx += 1

        ml_idx += n_pos

    return results


def read_pos_to_genomic(read, read_pos):
    """Convert read-relative position to genomic coordinate."""
    if read.cigartuples is None:
        return None
    query_idx = 0
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8):
            if query_idx <= read_pos < query_idx + length:
                return ref_pos + (read_pos - query_idx)
            query_idx += length
            ref_pos += length
        elif op == 1:
            if query_idx <= read_pos < query_idx + length:
                return None
            query_idx += length
        elif op in (2, 3):
            ref_pos += length
        elif op == 4:
            if query_idx <= read_pos < query_idx + length:
                return None
            query_idx += length
    return None


# =====================================================================
# MAIN
# =====================================================================
print("=" * 70)
print("THRESHOLD-FREE M6A ANALYSIS")
print("chemodCode 21891 = m6A (VERIFIED)")
print("=" * 70)

ref_fasta = pysam.FastaFile(REF_FASTA_PATH)

# =====================================================================
# PART 1: L1 vs Control — all probabilities, no threshold
# =====================================================================
print("\n" + "=" * 70)
print("PART 1: L1 vs Control — Mean probability per DRACH motif site")
print("=" * 70)

# Accumulators
l1_sum_prob = 0
l1_n_calls = 0
l1_n_motifs = 0
l1_n_reads = 0
l1_bp = 0
l1_prob_all = []

ctrl_sum_prob = 0
ctrl_n_calls = 0
ctrl_n_motifs = 0
ctrl_n_reads = 0
ctrl_bp = 0
ctrl_prob_all = []

thresholds = [0, 32, 64, 96, 128, 160, 192, 224]
l1_by_thresh = {t: 0 for t in thresholds}
ctrl_by_thresh = {t: 0 for t in thresholds}

for group in BASE_GROUPS:
    l1_bam_path = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
    ctrl_bam_path = f"{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam"

    for bam_path, is_l1 in [(l1_bam_path, True), (ctrl_bam_path, False)]:
        if not os.path.exists(bam_path):
            continue

        bam = pysam.AlignmentFile(bam_path, 'rb')
        for read in bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            blocks = get_aligned_blocks(read)
            if not blocks:
                continue

            ref_seq, aligned_bp = get_full_aligned_ref_seq(ref_fasta, read.reference_name, blocks)
            if aligned_bp < 50:
                continue

            n_motifs = count_motifs(ref_seq, read.is_reverse)

            # Get ALL m6A calls with probabilities
            m6a_calls = parse_all_m6a_with_probs(read)
            sum_prob = sum(p for _, p in m6a_calls)
            n_calls = len(m6a_calls)

            if is_l1:
                l1_sum_prob += sum_prob
                l1_n_calls += n_calls
                l1_n_motifs += n_motifs
                l1_n_reads += 1
                l1_bp += aligned_bp
                l1_prob_all.extend([p for _, p in m6a_calls])
                for t in thresholds:
                    l1_by_thresh[t] += sum(1 for _, p in m6a_calls if p >= t)
            else:
                ctrl_sum_prob += sum_prob
                ctrl_n_calls += n_calls
                ctrl_n_motifs += n_motifs
                ctrl_n_reads += 1
                ctrl_bp += aligned_bp
                ctrl_prob_all.extend([p for _, p in m6a_calls])
                for t in thresholds:
                    ctrl_by_thresh[t] += sum(1 for _, p in m6a_calls if p >= t)

        bam.close()

    print(f"  {group}: L1={l1_n_reads} reads, Ctrl={ctrl_n_reads} reads")

# Results
print(f"\n{'='*70}")
print(f"L1 vs CONTROL: Threshold-free results")
print(f"{'='*70}")

l1_mean_prob_per_call = l1_sum_prob / l1_n_calls if l1_n_calls > 0 else 0
ctrl_mean_prob_per_call = ctrl_sum_prob / ctrl_n_calls if ctrl_n_calls > 0 else 0
l1_mean_prob_per_motif = l1_sum_prob / l1_n_motifs if l1_n_motifs > 0 else 0
ctrl_mean_prob_per_motif = ctrl_sum_prob / ctrl_n_motifs if ctrl_n_motifs > 0 else 0

l1_cont_rate = (l1_sum_prob / 255) / l1_n_motifs if l1_n_motifs > 0 else 0
ctrl_cont_rate = (ctrl_sum_prob / 255) / ctrl_n_motifs if ctrl_n_motifs > 0 else 0

print(f"\n  {'Metric':<35s} {'L1':>12s} {'Control':>12s} {'Ratio':>8s}")
print(f"  {'-'*67}")
print(f"  {'N reads':<35s} {l1_n_reads:>12d} {ctrl_n_reads:>12d}")
print(f"  {'N MAFIA m6A calls (all probs)':<35s} {l1_n_calls:>12d} {ctrl_n_calls:>12d}")
print(f"  {'N DRACH motif sites':<35s} {l1_n_motifs:>12d} {ctrl_n_motifs:>12d}")
print(f"  {'Total aligned bp':<35s} {l1_bp:>12d} {ctrl_bp:>12d}")
print(f"  {'Mean prob per call (0-255)':<35s} {l1_mean_prob_per_call:>12.1f} {ctrl_mean_prob_per_call:>12.1f} {l1_mean_prob_per_call/ctrl_mean_prob_per_call:>7.3f}x")
print(f"  {'Sum(prob) / N_motifs (0-255)':<35s} {l1_mean_prob_per_motif:>12.2f} {ctrl_mean_prob_per_motif:>12.2f} {l1_mean_prob_per_motif/ctrl_mean_prob_per_motif:>7.3f}x")
print(f"  {'Continuous rate (sum(p/255)/motif)':<35s} {l1_cont_rate:>12.4f} {ctrl_cont_rate:>12.4f} {l1_cont_rate/ctrl_cont_rate:>7.3f}x")

l1_calls_per_motif = l1_n_calls / l1_n_motifs if l1_n_motifs > 0 else 0
ctrl_calls_per_motif = ctrl_n_calls / ctrl_n_motifs if ctrl_n_motifs > 0 else 0
print(f"  {'MAFIA calls / DRACH site':<35s} {l1_calls_per_motif:>12.4f} {ctrl_calls_per_motif:>12.4f} {l1_calls_per_motif/ctrl_calls_per_motif:>7.3f}x")

# Threshold sweep
print(f"\n  Threshold sweep: per-site rate at different cutoffs")
print(f"  {'Threshold':>10s} {'L1 rate':>10s} {'Ctrl rate':>10s} {'Ratio':>8s} {'L1 count':>10s} {'Ctrl count':>10s}")
print(f"  {'-'*58}")
for t in thresholds:
    l1_rate = l1_by_thresh[t] / l1_n_motifs if l1_n_motifs > 0 else 0
    ctrl_rate = ctrl_by_thresh[t] / ctrl_n_motifs if ctrl_n_motifs > 0 else 0
    ratio = l1_rate / ctrl_rate if ctrl_rate > 0 else 0
    print(f"  {t:>10d} {l1_rate:>10.4f} {ctrl_rate:>10.4f} {ratio:>7.3f}x {l1_by_thresh[t]:>10d} {ctrl_by_thresh[t]:>10d}")

# Distribution comparison
l1_arr = np.array(l1_prob_all)
ctrl_arr = np.array(ctrl_prob_all)
ks_stat, ks_p = stats.ks_2samp(l1_arr, ctrl_arr)
mw_stat, mw_p = stats.mannwhitneyu(l1_arr, ctrl_arr, alternative='two-sided')
print(f"\n  Distribution tests (all probabilities):")
print(f"  L1 mean={l1_arr.mean():.2f}, median={np.median(l1_arr):.0f}, Ctrl mean={ctrl_arr.mean():.2f}, median={np.median(ctrl_arr):.0f}")
print(f"  KS test: D={ks_stat:.4f}, p={ks_p:.2e}")
print(f"  Mann-Whitney: U={mw_stat:.0f}, p={mw_p:.2e}")


# =====================================================================
# PART 2: L1 Body vs Flanking — threshold-free
# =====================================================================
print(f"\n{'='*70}")
print("PART 2: L1 Body vs Flanking — Threshold-free")
print(f"{'='*70}")

body_sum_prob = 0
body_n_calls = 0
body_n_motifs = 0
body_bp_total = 0
body_probs = []
body_by_thresh = {t: 0 for t in thresholds}

flank_sum_prob = 0
flank_n_calls = 0
flank_n_motifs = 0
flank_bp_total = 0
flank_probs = []
flank_by_thresh = {t: 0 for t in thresholds}

n_valid_reads = 0

for group in BASE_GROUPS:
    summary_file = f"{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv"
    if not os.path.exists(summary_file):
        continue

    summary = pd.read_csv(summary_file, sep='\t')
    te_lookup = {}
    for _, row in summary.iterrows():
        te_lookup[row['read_id']] = (row['te_start'], row['te_end'], row['gene_id'])

    bam_path = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
    if not os.path.exists(bam_path):
        continue

    bam = pysam.AlignmentFile(bam_path, 'rb')
    gn = 0

    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.query_name not in te_lookup:
            continue

        te_start, te_end, subfamily = te_lookup[read.query_name]
        blocks = get_aligned_blocks(read)
        if not blocks:
            continue

        read_gstart = blocks[0][0]
        read_gend = blocks[-1][1]
        chrom = read.reference_name

        body_start = max(read_gstart, te_start)
        body_end = min(read_gend, te_end)
        if body_end <= body_start:
            continue

        # Get aligned bp in each region
        body_abp = 0
        flank_abp = 0
        for bs, be in blocks:
            ol_s = max(bs, body_start)
            ol_e = min(be, body_end)
            if ol_e > ol_s:
                body_abp += (ol_e - ol_s)
            if bs < body_start:
                flank_abp += (min(be, body_start) - bs)
            if be > body_end:
                flank_abp += (be - max(bs, body_end))

        if body_abp < 100 or flank_abp < 100:
            continue

        # Count motifs in each region
        body_seq = get_ref_seq_for_region(ref_fasta, chrom, blocks, body_start, body_end)
        flank_seq_left = get_ref_seq_for_region(ref_fasta, chrom, blocks, read_gstart, te_start)
        flank_seq_right = get_ref_seq_for_region(ref_fasta, chrom, blocks, te_end, read_gend)
        flank_seq = flank_seq_left + flank_seq_right

        nm_body = count_motifs(body_seq, read.is_reverse)
        nm_flank = count_motifs(flank_seq, read.is_reverse)
        if nm_body < 1 or nm_flank < 1:
            continue

        # Get ALL m6A calls with probabilities
        m6a_calls = parse_all_m6a_with_probs(read)

        bp_sum = 0
        bp_n = 0
        fp_sum = 0
        fp_n = 0
        bp_thresh = {t: 0 for t in thresholds}
        fp_thresh = {t: 0 for t in thresholds}

        for rp, prob in m6a_calls:
            gp = read_pos_to_genomic(read, rp)
            if gp is None:
                continue
            if body_start <= gp < body_end:
                bp_sum += prob
                bp_n += 1
                body_probs.append(prob)
                for t in thresholds:
                    if prob >= t:
                        bp_thresh[t] += 1
            else:
                fp_sum += prob
                fp_n += 1
                flank_probs.append(prob)
                for t in thresholds:
                    if prob >= t:
                        fp_thresh[t] += 1

        body_sum_prob += bp_sum
        body_n_calls += bp_n
        body_n_motifs += nm_body
        body_bp_total += body_abp
        flank_sum_prob += fp_sum
        flank_n_calls += fp_n
        flank_n_motifs += nm_flank
        flank_bp_total += flank_abp
        for t in thresholds:
            body_by_thresh[t] += bp_thresh[t]
            flank_by_thresh[t] += fp_thresh[t]

        n_valid_reads += 1
        gn += 1

    bam.close()
    print(f"  {group}: {gn} valid reads")

print(f"\nTotal valid reads: {n_valid_reads}")

# Results
b_mean_per_call = body_sum_prob / body_n_calls if body_n_calls > 0 else 0
f_mean_per_call = flank_sum_prob / flank_n_calls if flank_n_calls > 0 else 0
b_mean_per_motif = body_sum_prob / body_n_motifs if body_n_motifs > 0 else 0
f_mean_per_motif = flank_sum_prob / flank_n_motifs if flank_n_motifs > 0 else 0
b_cont_rate = (body_sum_prob / 255) / body_n_motifs if body_n_motifs > 0 else 0
f_cont_rate = (flank_sum_prob / 255) / flank_n_motifs if flank_n_motifs > 0 else 0
b_calls_per_motif = body_n_calls / body_n_motifs if body_n_motifs > 0 else 0
f_calls_per_motif = flank_n_calls / flank_n_motifs if flank_n_motifs > 0 else 0
b_motif_density = body_n_motifs / (body_bp_total / 1000) if body_bp_total > 0 else 0
f_motif_density = flank_n_motifs / (flank_bp_total / 1000) if flank_bp_total > 0 else 0

print(f"\n  {'Metric':<35s} {'L1 Body':>12s} {'Flanking':>12s} {'Ratio':>8s}")
print(f"  {'-'*67}")
print(f"  {'N MAFIA calls':<35s} {body_n_calls:>12d} {flank_n_calls:>12d}")
print(f"  {'N DRACH motif sites':<35s} {body_n_motifs:>12d} {flank_n_motifs:>12d}")
print(f"  {'Aligned bp':<35s} {body_bp_total:>12d} {flank_bp_total:>12d}")
print(f"  {'Motif density (/kb)':<35s} {b_motif_density:>12.2f} {f_motif_density:>12.2f} {b_motif_density/f_motif_density:>7.3f}x" if f_motif_density > 0 else "")
print(f"  {'MAFIA calls / DRACH site':<35s} {b_calls_per_motif:>12.4f} {f_calls_per_motif:>12.4f} {b_calls_per_motif/f_calls_per_motif:>7.3f}x" if f_calls_per_motif > 0 else "")
print(f"  {'Mean prob per call (0-255)':<35s} {b_mean_per_call:>12.1f} {f_mean_per_call:>12.1f} {b_mean_per_call/f_mean_per_call:>7.3f}x" if f_mean_per_call > 0 else "")
print(f"  {'Sum(prob)/motif (0-255 scale)':<35s} {b_mean_per_motif:>12.2f} {f_mean_per_motif:>12.2f} {b_mean_per_motif/f_mean_per_motif:>7.3f}x" if f_mean_per_motif > 0 else "")
print(f"  {'Continuous rate (sum(p/255)/motif)':<35s} {b_cont_rate:>12.4f} {f_cont_rate:>12.4f} {b_cont_rate/f_cont_rate:>7.3f}x" if f_cont_rate > 0 else "")

# Threshold sweep for body vs flanking
print(f"\n  Threshold sweep: body vs flanking per-site rate")
print(f"  {'Threshold':>10s} {'Body rate':>10s} {'Flank rate':>10s} {'Ratio':>8s}")
print(f"  {'-'*38}")
for t in thresholds:
    br = body_by_thresh[t] / body_n_motifs if body_n_motifs > 0 else 0
    fr = flank_by_thresh[t] / flank_n_motifs if flank_n_motifs > 0 else 0
    ratio = br / fr if fr > 0 else 0
    print(f"  {t:>10d} {br:>10.4f} {fr:>10.4f} {ratio:>7.3f}x")

# Distribution comparison
bp_arr = np.array(body_probs)
fp_arr = np.array(flank_probs)
if len(bp_arr) > 0 and len(fp_arr) > 0:
    ks_stat, ks_p = stats.ks_2samp(bp_arr, fp_arr)
    mw_stat, mw_p = stats.mannwhitneyu(bp_arr, fp_arr, alternative='two-sided')
    print(f"\n  Distribution tests (all probabilities):")
    print(f"  Body mean={bp_arr.mean():.2f}, median={np.median(bp_arr):.0f}")
    print(f"  Flank mean={fp_arr.mean():.2f}, median={np.median(fp_arr):.0f}")
    print(f"  KS test: D={ks_stat:.4f}, p={ks_p:.2e}")
    print(f"  Mann-Whitney: p={mw_p:.2e}")


ref_fasta.close()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"""
Key findings (threshold-free, m6A chemodCode=21891):
  L1 vs Control:
    Continuous rate = {l1_cont_rate:.4f} vs {ctrl_cont_rate:.4f} = {l1_cont_rate/ctrl_cont_rate:.3f}x
    Mean prob/call  = {l1_mean_prob_per_call:.1f} vs {ctrl_mean_prob_per_call:.1f}

  L1 Body vs Flanking:
    Continuous rate = {b_cont_rate:.4f} vs {f_cont_rate:.4f} = {b_cont_rate/f_cont_rate:.3f}x
    Motif density   = {b_motif_density:.2f} vs {f_motif_density:.2f} = {b_motif_density/f_motif_density:.3f}x
    Mean prob/call  = {b_mean_per_call:.1f} vs {f_mean_per_call:.1f}
""")

print("=== Done ===")
