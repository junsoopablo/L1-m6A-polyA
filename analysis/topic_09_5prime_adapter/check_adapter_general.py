#!/usr/bin/env python3
"""
Check 5' adapter (REL5) detection rate on general transcripts (not just L1).
Sample 5000 reads from HeLa_1 sorted BAM, stratified by read length.
If short transcripts show higher adapter rate → method works, L1 is just too long.
"""
import pysam
import numpy as np
from collections import defaultdict

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"

# REL5 adapter
REL5 = "AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT"
REL5_RC = REL5[::-1].translate(str.maketrans("ACGT", "TGCA"))

def revcomp(seq):
    return seq[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def has_adapter(seq, adapter, min_overlap=25, max_error_rate=0.29):
    """Check if adapter is at the end of the sequence (last 80bp)."""
    region = seq[-80:].upper() if len(seq) > 80 else seq.upper()
    best = 0
    for start in range(max(1, len(region) - len(adapter))):
        match_len = min(len(adapter), len(region) - start)
        if match_len < min_overlap:
            continue
        matches = sum(1 for i in range(match_len) if region[start+i] == adapter[i])
        score = matches / match_len
        if score > best:
            best = score
    return best

def check_with_cutadapt_logic(seq, adapter, min_overlap=20):
    """Sliding window search at both ends."""
    seq = seq.upper()
    best_score = 0
    best_loc = ""

    # Check last 100bp (expected location for DRS)
    for region, label in [(seq[-100:], "end"), (seq[:100], "start")]:
        for ad, ori in [(adapter, "fwd"), (revcomp(adapter), "rc")]:
            ad = ad.upper()
            for i in range(len(region)):
                ml = min(len(ad), len(region) - i)
                if ml < min_overlap:
                    continue
                m = sum(1 for j in range(ml) if region[i+j] == ad[j])
                s = m / ml
                if s > best_score:
                    best_score = s
                    best_loc = f"{label}_{ori}"
    return best_score, best_loc


# Use the sorted BAM from the main pipeline
# First find it
import subprocess
result = subprocess.run(
    ["find", f"{PROJECT}/results_group/HeLa_1", "-name", "*.sorted*.bam", "-type", "f"],
    capture_output=True, text=True
)
bam_files = result.stdout.strip().split("\n")
print(f"Found BAMs: {bam_files}")

# The L1 BAM has all mapped reads? No, it only has L1 reads.
# We need the full BAM. Check a_hg38_mapping_LRS
import os
# The main BAM might be on scratch
bam_path = None
for candidate in [
    f"/scratch1/junsoopablo/IsoTENT_002_L1/HeLa_1_1.sorted.bam",
    f"{PROJECT}/results_group/HeLa_1/a_hg38_mapping_LRS/HeLa_1_hg38_mapped.sorted_position.bam",
    f"{PROJECT}/data_bam/HeLa_1_1.sorted.bam",
]:
    if os.path.exists(candidate):
        bam_path = candidate
        break

if bam_path is None:
    # Search more broadly
    result = subprocess.run(
        ["find", PROJECT, "-name", "HeLa_1*sorted*.bam", "-type", "f"],
        capture_output=True, text=True
    )
    candidates = result.stdout.strip().split("\n")
    for c in candidates:
        if "L1" not in c and "control" not in c and "catB" not in c and os.path.exists(c):
            bam_path = c
            break

if bam_path is None:
    # Try the original FASTQ instead
    print("No full BAM found. Using FASTQ directly.")
    import gzip

    fastq_path = f"{PROJECT}/data_fastq/HeLa_1_1.fastq.gz"
    print(f"Reading {fastq_path}...")

    length_bins = defaultdict(list)  # bin → list of (score, loc)
    n_reads = 0
    n_sampled = 0
    MAX_SAMPLE = 10000

    with gzip.open(fastq_path, "rt") as fq:
        while n_sampled < MAX_SAMPLE:
            header = fq.readline().strip()
            if not header:
                break
            seq = fq.readline().strip()
            fq.readline()  # +
            fq.readline()  # qual
            n_reads += 1

            # Sample every Nth read
            if n_reads % 50 != 0:
                continue
            n_sampled += 1

            read_len = len(seq)

            # Bin by length
            if read_len < 500:
                bin_label = "<500"
            elif read_len < 1000:
                bin_label = "500-1K"
            elif read_len < 2000:
                bin_label = "1K-2K"
            elif read_len < 3000:
                bin_label = "2K-3K"
            else:
                bin_label = "≥3K"

            score, loc = check_with_cutadapt_logic(seq, REL5)
            length_bins[bin_label].append((score, loc, read_len))

    print(f"\nSampled {n_sampled} reads from {n_reads} total")
    print(f"\n{'Bin':<10} {'N':>6} {'≥0.7':>6} {'%':>6} {'≥0.8':>6} {'%':>6} {'MedRL':>6}")
    print("-" * 52)

    for bin_label in ["<500", "500-1K", "1K-2K", "2K-3K", "≥3K"]:
        if bin_label not in length_bins:
            continue
        data = length_bins[bin_label]
        n = len(data)
        scores = [d[0] for d in data]
        rls = [d[2] for d in data]
        n_70 = sum(1 for s in scores if s >= 0.7)
        n_80 = sum(1 for s in scores if s >= 0.8)
        print(f"{bin_label:<10} {n:>6} {n_70:>6} {n_70/n*100:>5.1f}% {n_80:>6} {n_80/n*100:>5.1f}% {int(np.median(rls)):>6}")

    # Location distribution for high-scoring reads
    high_reads = [(s, l, rl) for bin_data in length_bins.values()
                  for s, l, rl in bin_data if s >= 0.7]
    if high_reads:
        print(f"\nHigh-score reads (≥0.7): {len(high_reads)}")
        loc_counts = defaultdict(int)
        for s, l, rl in high_reads:
            loc_counts[l] += 1
        for loc, cnt in sorted(loc_counts.items(), key=lambda x: -x[1]):
            print(f"  {loc}: {cnt}")

        # Print a few examples
        print(f"\nExamples (first 5):")
        for s, l, rl in sorted(high_reads, key=lambda x: -x[0])[:5]:
            print(f"  score={s:.3f}, location={l}, read_len={rl}")
    else:
        print("\nNo high-score reads found!")

    # Overall score distribution
    all_scores = [s for bin_data in length_bins.values() for s, _, _ in bin_data]
    print(f"\nOverall score distribution:")
    print(f"  Mean: {np.mean(all_scores):.4f}")
    print(f"  Max: {np.max(all_scores):.4f}")
    for t in [0.5, 0.6, 0.7, 0.8]:
        n = sum(1 for s in all_scores if s >= t)
        print(f"  ≥{t}: {n}/{len(all_scores)} ({n/len(all_scores)*100:.2f}%)")

else:
    print(f"Using BAM: {bam_path}")
    # Process BAM
    length_bins = defaultdict(list)
    n_reads = 0
    MAX_SAMPLE = 10000

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            n_reads += 1
            if n_reads % 100 != 0:
                continue

            seq = read.query_sequence
            if seq is None:
                continue

            # Get original orientation
            if read.is_reverse:
                seq = revcomp(seq)

            read_len = len(seq)
            if read_len < 500:
                bin_label = "<500"
            elif read_len < 1000:
                bin_label = "500-1K"
            elif read_len < 2000:
                bin_label = "1K-2K"
            elif read_len < 3000:
                bin_label = "2K-3K"
            else:
                bin_label = "≥3K"

            score, loc = check_with_cutadapt_logic(seq, REL5)
            length_bins[bin_label].append((score, loc, read_len))

            if sum(len(v) for v in length_bins.values()) >= MAX_SAMPLE:
                break

    print(f"\nSampled from {n_reads} reads")
    print(f"\n{'Bin':<10} {'N':>6} {'≥0.7':>6} {'%':>6}")
    print("-" * 30)
    for bin_label in ["<500", "500-1K", "1K-2K", "2K-3K", "≥3K"]:
        if bin_label not in length_bins:
            continue
        data = length_bins[bin_label]
        n = len(data)
        n_70 = sum(1 for s, _, _ in data if s >= 0.7)
        print(f"{bin_label:<10} {n:>6} {n_70:>6} {n_70/n*100:>5.1f}%")
