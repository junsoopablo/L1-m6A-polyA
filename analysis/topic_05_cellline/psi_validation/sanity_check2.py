#!/usr/bin/env python3
"""
Detailed comparison: Part3 cache psi_positions vs BAM MM/ML tag direct parsing.

The sanity check found cache has ~2x more psi calls than BAM.
Possible causes:
  A. Cache built with wrong threshold (e.g., all probs instead of ≥128)
  B. Cache built from different BAM
  C. Cache double-counts (N+ and N- for same positions?)
"""

import pysam
import pandas as pd
import numpy as np
import ast
import os

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
TOPICDIR = f"{PROJECT}/analysis/01_exploration/topic_05_cellline"

TEST_GROUP = "HeLa_1"

# =====================================================================
# Load cache
# =====================================================================
cache = pd.read_csv(f"{TOPICDIR}/part3_l1_per_read_cache/{TEST_GROUP}_l1_per_read.tsv", sep='\t')
print(f"Cache: {len(cache)} reads")

# =====================================================================
# For specific reads, compare cache vs BAM
# =====================================================================
bam_path = f"{RESULTS}/{TEST_GROUP}/h_mafia/{TEST_GROUP}.mAFiA.reads.bam"
bam = pysam.AlignmentFile(bam_path, 'rb')

# Pick first 10 reads from cache
test_reads = cache.head(10)

for _, row in test_reads.iterrows():
    read_id = row['read_id']
    cache_psi_str = str(row['psi_positions'])
    cache_psi_high = row['psi_sites_high']

    if cache_psi_str == '[]' or pd.isna(row['psi_positions']):
        cache_psi_positions = []
    else:
        cache_psi_positions = ast.literal_eval(cache_psi_str)

    cache_n = len(cache_psi_positions)

    # Find this read in BAM
    bam_psi_all = []
    bam_psi_high = []
    found = False

    for read in bam.fetch():
        if read.query_name == read_id and not read.is_secondary and not read.is_supplementary:
            found = True

            # Parse MM/ML
            try:
                mm_tag = read.get_tag('MM') if read.has_tag('MM') else None
                ml_tag = read.get_tag('ML') if read.has_tag('ML') else None
            except:
                mm_tag = ml_tag = None

            if mm_tag and ml_tag:
                ml_values = list(ml_tag)
                seq = read.query_sequence
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
                        continue

                    if '17802' in mod_spec:
                        base_code = mod_spec[0]
                        if base_code == 'N':
                            base_positions = list(range(len(seq)))
                        else:
                            base_positions = [i for i, c in enumerate(seq) if c == base_code]

                        bp_idx = 0
                        for i, skip in enumerate(skips):
                            bp_idx += skip
                            if bp_idx < len(base_positions):
                                rp = base_positions[bp_idx]
                                prob = ml_values[ml_idx + i] if (ml_idx + i) < len(ml_values) else 0
                                bam_psi_all.append((rp, prob))
                                if prob >= 128:
                                    bam_psi_high.append(rp)
                            bp_idx += 1

                    ml_idx += n_pos

                # Print MM tag structure for first read
                if _ == test_reads.index[0]:
                    print(f"\n  First read MM tag entries:")
                    for entry in mm_tag.rstrip(';').split(';'):
                        entry = entry.strip()
                        if not entry:
                            continue
                        parts = entry.split(',')
                        mod_spec = parts[0]
                        n_pos = len(parts) - 1
                        print(f"    {mod_spec}: {n_pos} positions")

            break

    match_high = "MATCH" if cache_n == len(bam_psi_high) else "MISMATCH"
    match_all = "MATCH" if cache_n == len(bam_psi_all) else ""

    print(f"\n  Read: {read_id[:40]}...")
    print(f"    Cache psi_positions: {cache_n} items, psi_sites_high: {cache_psi_high}")
    print(f"    BAM psi (≥128): {len(bam_psi_high)}, BAM psi (all): {len(bam_psi_all)}  → cache vs ≥128: {match_high}, cache vs all: {match_all}")

    if cache_n != len(bam_psi_high) and cache_n != len(bam_psi_all):
        # Show details
        print(f"    Cache positions: {sorted(cache_psi_positions)[:10]}...")
        print(f"    BAM high positions: {sorted(bam_psi_high)[:10]}...")
        bam_all_positions = sorted([p for p, _ in bam_psi_all])
        print(f"    BAM all positions: {bam_all_positions[:10]}...")
    elif cache_n == len(bam_psi_all):
        print(f"    *** Cache matches ALL BAM psi calls (including sub-threshold!) ***")

bam.close()

# =====================================================================
# Aggregate: cache total vs BAM total
# =====================================================================
print(f"\n{'='*70}")
print("AGGREGATE COMPARISON")
print(f"{'='*70}")

# Cache totals
def count_psi(s):
    if pd.isna(s) or s == '[]':
        return 0
    return len(ast.literal_eval(str(s)))

cache['n_psi_from_positions'] = cache['psi_positions'].apply(count_psi)
cache_total_from_pos = cache['n_psi_from_positions'].sum()
cache_total_from_high = cache['psi_sites_high'].sum()

print(f"  Cache n_psi (from psi_positions list): {cache_total_from_pos}")
print(f"  Cache n_psi (from psi_sites_high col): {cache_total_from_high}")
print(f"  Match: {cache_total_from_pos == cache_total_from_high}")

# BAM totals
bam = pysam.AlignmentFile(bam_path, 'rb')
bam_total_high = 0
bam_total_all = 0
bam_n_reads = 0

for read in bam.fetch():
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        continue

    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else None
        ml_tag = read.get_tag('ML') if read.has_tag('ML') else None
    except:
        mm_tag = ml_tag = None

    n_high = 0
    n_all = 0

    if mm_tag and ml_tag:
        ml_values = list(ml_tag)
        ml_idx = 0

        for entry in mm_tag.rstrip(';').split(';'):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(',')
            mod_spec = parts[0]
            n_pos = len(parts) - 1
            if n_pos <= 0:
                continue
            if '17802' in mod_spec:
                for i in range(n_pos):
                    if ml_idx + i < len(ml_values):
                        n_all += 1
                        if ml_values[ml_idx + i] >= 128:
                            n_high += 1
            ml_idx += n_pos

    bam_total_high += n_high
    bam_total_all += n_all
    bam_n_reads += 1

bam.close()

print(f"\n  BAM total psi (≥128): {bam_total_high}")
print(f"  BAM total psi (all): {bam_total_all}")
print(f"  BAM reads: {bam_n_reads}")

print(f"\n  Cache total / BAM ≥128: {cache_total_from_pos / bam_total_high:.3f}x")
print(f"  Cache total / BAM all:  {cache_total_from_pos / bam_total_all:.3f}x")

if abs(cache_total_from_pos - bam_total_all) < abs(cache_total_from_pos - bam_total_high):
    print(f"\n  *** CONCLUSION: Cache matches BAM ALL (no threshold), NOT BAM ≥128 ***")
    print(f"  *** Part3 cache was built WITHOUT applying the 128 threshold! ***")
else:
    print(f"\n  *** CONCLUSION: Cache is closer to BAM ≥128 ***")

print("\n=== Done ===")
