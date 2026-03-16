#!/usr/bin/env python3
"""
Detect TERA-seq 5' adapter (REL5) in HeLa/HeLa-Ars L1 reads.
Data source: PRJNA842344 (Dar et al. eLife 2024), TERA-seq protocol.

The REL5 adapter is ligated to the 5' phosphorylated end of RNA.
If detected in a read → full-length (5' end intact).
If not detected → truncated/degraded (5' end missing).

Strategy:
  1. Extract L1 reads from BAM (original read orientation)
  2. Search for REL5 adapter at both ends using fuzzy matching
  3. Also check via cutadapt for robust detection
  4. Classify reads as full-length vs truncated
  5. Compare HeLa (mock) vs HeLa-Ars (arsenite)
"""

import pysam
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import subprocess
import tempfile
import gzip
import re
import sys

# ── Config ──
PROJECT = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
OUTDIR = PROJECT / "analysis/01_exploration/topic_09_5prime_adapter"
OUTDIR.mkdir(parents=True, exist_ok=True)

# REL5 adapter sequence (5TERA, 58nt)
REL5_ADAPTER = "AATGATACGGCGACCACCGAGATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT"
REL5_RC = REL5_ADAPTER[::-1].translate(str.maketrans("ACGT", "TGCA"))

GROUPS_HELA = ["HeLa_1", "HeLa_2", "HeLa_3"]
GROUPS_ARS = ["HeLa-Ars_1", "HeLa-Ars_2", "HeLa-Ars_3"]
ALL_GROUPS = GROUPS_HELA + GROUPS_ARS

# L1 summary for metadata
YOUNG_SUBFAMILIES = {"L1HS", "L1PA1", "L1PA2", "L1PA3"}

def revcomp(seq):
    return seq[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))

def fuzzy_match(seq, adapter, max_errors=0.29, min_overlap=31):
    """Simple fuzzy match: check if adapter prefix/suffix appears at read ends."""
    # Check beginning of seq for adapter (or end portion of adapter)
    best_score = 0
    best_pos = None
    best_end = None

    adapter_len = len(adapter)

    # Check first 100bp for adapter presence
    region_5 = seq[:100].upper()
    region_3 = seq[-100:].upper() if len(seq) > 100 else seq.upper()

    for region, end_label in [(region_5, "5prime"), (region_3, "3prime")]:
        for start in range(len(region)):
            # Compare adapter with region starting at 'start'
            match_len = min(adapter_len, len(region) - start)
            if match_len < min_overlap:
                continue

            matches = sum(1 for i in range(match_len)
                         if region[start + i] == adapter[i])
            score = matches / match_len

            if score > (1 - max_errors) and score > best_score:
                best_score = score
                best_pos = start
                best_end = end_label

    return best_score, best_pos, best_end

def extract_reads_and_detect(group):
    """Extract L1 reads and detect 5' adapter."""
    bam_path = PROJECT / f"results_group/{group}/d_LINE_quantification/{group}_L1_reads.bam"
    summary_path = PROJECT / f"results_group/{group}/g_summary/{group}_L1_summary.tsv"

    if not bam_path.exists():
        print(f"  WARNING: {bam_path} not found")
        return pd.DataFrame()

    # Load L1 summary for metadata
    summary = pd.read_csv(summary_path, sep="\t")
    summary_dict = {}
    for _, row in summary.iterrows():
        summary_dict[row["read_id"]] = {
            "read_length": row.get("read_length", 0),
            "polya_length": row.get("polya_length", np.nan),
            "qc_tag": row.get("qc_tag", ""),
            "gene_id": row.get("gene_id", ""),
            "class": row.get("class", ""),
            "TE_group": row.get("TE_group", ""),
            "dist_to_3prime": row.get("dist_to_3prime", np.nan),
        }

    results = []

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            read_id = read.query_name
            seq = read.query_sequence
            if seq is None:
                continue

            # Get the original read sequence (reverse complement if mapped to - strand)
            # In BAM, query_sequence is already in the stored orientation
            # For - strand reads, pysam gives the sequence as stored (RC of original)
            is_reverse = read.is_reverse

            # We need the ORIGINAL read sequence (as it came from the sequencer)
            # For ONT DRS mapped to - strand: BAM stores RC, so original = RC of BAM seq
            if is_reverse:
                original_seq = revcomp(seq)
            else:
                original_seq = seq

            read_len = len(seq)

            # Search for adapter at both ends, both orientations
            # In ONT DRS, 5' adapter should be at the END of the original read
            # (because DRS sequences 3'→5')

            # Check 1: adapter at END of original read (expected for DRS 3'→5')
            end_region = original_seq[-100:].upper() if read_len > 100 else original_seq.upper()
            score_end_fwd, pos_end_fwd, _ = fuzzy_match(end_region, REL5_ADAPTER, min_overlap=20)
            score_end_rc, pos_end_rc, _ = fuzzy_match(end_region, REL5_RC, min_overlap=20)

            # Check 2: adapter at START of original read
            start_region = original_seq[:100].upper()
            score_start_fwd, pos_start_fwd, _ = fuzzy_match(start_region, REL5_ADAPTER, min_overlap=20)
            score_start_rc, pos_start_rc, _ = fuzzy_match(start_region, REL5_RC, min_overlap=20)

            # Check 3: adapter in BAM orientation (for mapped reads)
            bam_end = seq[-100:].upper() if read_len > 100 else seq.upper()
            bam_start = seq[:100].upper()
            score_bam_start_fwd, _, _ = fuzzy_match(bam_start, REL5_ADAPTER, min_overlap=20)
            score_bam_start_rc, _, _ = fuzzy_match(bam_start, REL5_RC, min_overlap=20)
            score_bam_end_fwd, _, _ = fuzzy_match(bam_end, REL5_ADAPTER, min_overlap=20)
            score_bam_end_rc, _, _ = fuzzy_match(bam_end, REL5_RC, min_overlap=20)

            # Best score across all checks
            all_scores = {
                "orig_end_fwd": score_end_fwd,
                "orig_end_rc": score_end_rc,
                "orig_start_fwd": score_start_fwd,
                "orig_start_rc": score_start_rc,
                "bam_start_fwd": score_bam_start_fwd,
                "bam_start_rc": score_bam_start_rc,
                "bam_end_fwd": score_bam_end_fwd,
                "bam_end_rc": score_bam_end_rc,
            }

            best_location = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_location]

            # Get CIGAR soft-clip info
            cigar = read.cigartuples
            left_clip = cigar[0][1] if cigar and cigar[0][0] == 4 else 0
            right_clip = cigar[-1][1] if cigar and cigar[-1][0] == 4 else 0

            # Get metadata from summary
            meta = summary_dict.get(read_id, {})
            gene_id = meta.get("gene_id", "")
            is_young = gene_id.split("_")[0] in YOUNG_SUBFAMILIES if gene_id else False

            results.append({
                "read_id": read_id,
                "group": group,
                "condition": "HeLa" if "Ars" not in group else "HeLa-Ars",
                "read_length": read_len,
                "is_reverse": is_reverse,
                "left_softclip": left_clip,
                "right_softclip": right_clip,
                "best_adapter_score": best_score,
                "best_adapter_location": best_location,
                "polya_length": meta.get("polya_length", np.nan),
                "qc_tag": meta.get("qc_tag", ""),
                "gene_id": gene_id,
                "subfamily": gene_id.split("_")[0] if gene_id else "",
                "is_young": is_young,
                "te_group": meta.get("TE_group", ""),
                "dist_to_3prime": meta.get("dist_to_3prime", np.nan),
                # Store all individual scores for debugging
                "score_orig_end_fwd": score_end_fwd,
                "score_orig_start_fwd": score_start_fwd,
                "score_bam_start_fwd": score_bam_start_fwd,
                "score_bam_end_fwd": score_bam_end_fwd,
            })

    return pd.DataFrame(results)


def cutadapt_detection(group):
    """Use cutadapt for robust adapter detection."""
    bam_path = PROJECT / f"results_group/{group}/d_LINE_quantification/{group}_L1_reads.bam"
    if not bam_path.exists():
        return {}

    # Extract reads to temp FASTQ
    tmpfq = OUTDIR / f"{group}_l1_reads.fastq"
    with open(tmpfq, "w") as fout:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                seq = read.query_sequence
                qual = read.qual
                if seq is None:
                    continue
                # Write in original orientation
                if read.is_reverse:
                    seq = revcomp(seq)
                    qual = qual[::-1] if qual else "I" * len(seq)
                else:
                    qual = qual if qual else "I" * len(seq)
                fout.write(f"@{read.query_name}\n{seq}\n+\n{qual}\n")

    # Run cutadapt - try adapter at both ends
    # -g: 5' adapter (beginning of read)
    # -a: 3' adapter (end of read)
    results = {}

    for flag, label in [("-g", "5prime"), ("-a", "3prime")]:
        for adapter, ori in [(REL5_ADAPTER, "fwd"), (REL5_RC, "rc")]:
            key = f"{label}_{ori}"
            outfq = OUTDIR / f"{group}_cutadapt_{key}.fastq"

            cmd = [
                "cutadapt",
                flag, f"X{adapter}",
                "--overlap", "31",
                "--error-rate", "0.29",
                "--untrimmed-output", str(OUTDIR / f"{group}_untrimmed_{key}.fastq"),
                "-o", str(outfq),
                str(tmpfq)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                # Parse cutadapt output for number of trimmed reads
                for line in result.stderr.split("\n"):
                    if "Reads with adapters" in line:
                        # Extract count
                        parts = line.strip().split()
                        count = int(parts[-2].replace(",", "").replace("(", ""))
                        pct = parts[-1].strip("()")
                        results[key] = {"count": count, "pct": pct}
                        break
            except Exception as e:
                print(f"  cutadapt {key}: {e}")

    # Cleanup
    for f in OUTDIR.glob(f"{group}_cutadapt_*"):
        f.unlink(missing_ok=True)
    for f in OUTDIR.glob(f"{group}_untrimmed_*"):
        f.unlink(missing_ok=True)
    tmpfq.unlink(missing_ok=True)

    return results


# ── Main Analysis ──
print("=" * 60)
print("TERA-seq 5' Adapter (REL5) Detection in L1 Reads")
print("=" * 60)
print(f"Adapter: {REL5_ADAPTER}")
print(f"Rev-comp: {REL5_RC}")
print()

# Phase 1: Fuzzy matching
print("Phase 1: Fuzzy adapter matching in L1 reads")
print("-" * 50)

all_dfs = []
for group in ALL_GROUPS:
    print(f"  Processing {group}...")
    df = extract_reads_and_detect(group)
    if len(df) > 0:
        all_dfs.append(df)
        # Quick summary
        pass_df = df[df["qc_tag"] == "PASS"]
        n_high = (pass_df["best_adapter_score"] >= 0.7).sum()
        print(f"    {len(df)} reads, {len(pass_df)} PASS, "
              f"{n_high} with adapter score ≥0.7 ({n_high/max(1,len(pass_df))*100:.1f}%)")

if not all_dfs:
    print("ERROR: No data loaded!")
    sys.exit(1)

df_all = pd.concat(all_dfs, ignore_index=True)

# Save raw results
df_all.to_csv(OUTDIR / "adapter_detection_raw.tsv", sep="\t", index=False)
print(f"\nTotal: {len(df_all)} reads")

# Phase 2: Score distribution analysis
print("\nPhase 2: Adapter Score Distribution")
print("-" * 50)

pass_df = df_all[df_all["qc_tag"] == "PASS"].copy()
print(f"PASS reads: {len(pass_df)}")

# Score distribution
for threshold in [0.5, 0.6, 0.7, 0.75, 0.8]:
    n = (pass_df["best_adapter_score"] >= threshold).sum()
    pct = n / len(pass_df) * 100
    print(f"  Score ≥ {threshold}: {n} ({pct:.1f}%)")

# Where is adapter detected?
print("\nAdapter location distribution (score ≥ 0.7):")
high_score = pass_df[pass_df["best_adapter_score"] >= 0.7]
if len(high_score) > 0:
    loc_counts = high_score["best_adapter_location"].value_counts()
    for loc, count in loc_counts.items():
        print(f"  {loc}: {count} ({count/len(high_score)*100:.1f}%)")

# Phase 3: cutadapt validation
print("\nPhase 3: cutadapt Detection")
print("-" * 50)

cutadapt_results = {}
for group in ALL_GROUPS:
    print(f"  {group}...", end=" ")
    res = cutadapt_detection(group)
    cutadapt_results[group] = res
    if res:
        best_key = max(res, key=lambda k: res[k]["count"])
        print(f"Best: {best_key} = {res[best_key]['count']} ({res[best_key]['pct']})")
    else:
        print("no results")

# Phase 4: Compare HeLa vs HeLa-Ars
print("\n" + "=" * 60)
print("Phase 4: Full-length Analysis - HeLa vs HeLa-Ars")
print("=" * 60)

# Use best threshold from score distribution
# First, let's check what score cutoff separates signal from noise
scores = pass_df["best_adapter_score"].values
print(f"\nScore statistics:")
print(f"  Mean: {np.mean(scores):.3f}")
print(f"  Median: {np.median(scores):.3f}")
print(f"  P90: {np.percentile(scores, 90):.3f}")
print(f"  P95: {np.percentile(scores, 95):.3f}")
print(f"  P99: {np.percentile(scores, 99):.3f}")
print(f"  Max: {np.max(scores):.3f}")

# Score distribution by percentile
print(f"\nScore histogram:")
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
    n = ((scores >= lo) & (scores < hi)).sum()
    bar = "#" * int(n / len(scores) * 100)
    print(f"  [{lo:.1f}-{hi:.1f}): {n:5d} ({n/len(scores)*100:5.1f}%) {bar}")

# Use threshold 0.7 for classification
THRESHOLD = 0.7
pass_df["is_full_length"] = pass_df["best_adapter_score"] >= THRESHOLD

print(f"\nUsing threshold: {THRESHOLD}")
print(f"\n{'Condition':<12} {'Total':>6} {'Full-len':>8} {'Trunc':>6} {'FL%':>6}")
print("-" * 42)
for cond in ["HeLa", "HeLa-Ars"]:
    sub = pass_df[pass_df["condition"] == cond]
    fl = sub["is_full_length"].sum()
    tr = len(sub) - fl
    pct = fl / len(sub) * 100 if len(sub) > 0 else 0
    print(f"{cond:<12} {len(sub):>6} {fl:>8} {tr:>6} {pct:>5.1f}%")

# By age
print(f"\n{'Age':<10} {'Cond':<12} {'Total':>6} {'FL':>5} {'FL%':>6}")
print("-" * 45)
for age_label, age_filter in [("Young", True), ("Ancient", False)]:
    for cond in ["HeLa", "HeLa-Ars"]:
        sub = pass_df[(pass_df["condition"] == cond) & (pass_df["is_young"] == age_filter)]
        fl = sub["is_full_length"].sum()
        pct = fl / len(sub) * 100 if len(sub) > 0 else 0
        print(f"{age_label:<10} {cond:<12} {len(sub):>6} {fl:>5} {pct:>5.1f}%")

# Full-length reads: poly(A) comparison
print(f"\n{'Category':<25} {'N':>5} {'polyA_med':>10} {'RL_med':>8}")
print("-" * 52)
for cond in ["HeLa", "HeLa-Ars"]:
    for fl_label, fl_val in [("Full-length", True), ("Truncated", False)]:
        sub = pass_df[(pass_df["condition"] == cond) & (pass_df["is_full_length"] == fl_val)]
        if len(sub) > 0:
            polya_med = sub["polya_length"].median()
            rl_med = sub["read_length"].median()
            print(f"{cond} {fl_label:<14} {len(sub):>5} {polya_med:>10.1f} {rl_med:>8.0f}")

# Softclip analysis (adapter should create large softclips at one end)
print(f"\nSoftclip comparison (full-length vs truncated):")
for fl_label, fl_val in [("Full-length", True), ("Truncated", False)]:
    sub = pass_df[pass_df["is_full_length"] == fl_val]
    if len(sub) > 0:
        print(f"  {fl_label}: left_clip median={sub['left_softclip'].median():.0f}, "
              f"right_clip median={sub['right_softclip'].median():.0f}")

# Save processed results
pass_df.to_csv(OUTDIR / "adapter_detection_pass.tsv", sep="\t", index=False)
print(f"\nResults saved to {OUTDIR}/")
print("Done!")
