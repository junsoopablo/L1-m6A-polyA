#!/usr/bin/env python3
"""
Sanity check: Why did Part 3 show L1 psi/kb=5.17 vs Ctrl=3.60 (1.44x),
but validation shows 3.39 vs 3.36 (1.01x)?

Possible causes:
1. Sample swap (L1 BAM vs Control BAM)
2. Different denominator (read_length vs aligned_length)
3. Different psi counting (cache vs BAM)
4. Different read sets

Check each systematically.
"""

import pysam
import pandas as pd
import numpy as np
import os, ast

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
TOPICDIR = f"{PROJECT}/analysis/01_exploration/topic_05_cellline"

# Test with one representative group
TEST_GROUP = "HeLa_1"

print("=" * 70)
print(f"SANITY CHECK: {TEST_GROUP}")
print("=" * 70)

# =====================================================================
# 1. Check BAM files — are L1 reads actually in L1 regions?
# =====================================================================
print("\n--- CHECK 1: Are L1 BAM reads in L1 regions? ---")

l1_bam_path = f"{RESULTS}/{TEST_GROUP}/h_mafia/{TEST_GROUP}.mAFiA.reads.bam"
ctrl_bam_path = f"{RESULTS}/{TEST_GROUP}/i_control/mafia/{TEST_GROUP}.control.mAFiA.reads.bam"

# Load L1 summary for TE coordinates
summary = pd.read_csv(f"{RESULTS}/{TEST_GROUP}/g_summary/{TEST_GROUP}_L1_summary.tsv", sep='\t')
l1_read_ids = set(summary['read_id'])
print(f"  L1 summary: {len(l1_read_ids)} reads")

# Check L1 BAM
l1_bam = pysam.AlignmentFile(l1_bam_path, 'rb')
l1_bam_reads = set()
l1_sample_reads = []
n = 0
for read in l1_bam.fetch():
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        continue
    l1_bam_reads.add(read.query_name)
    if n < 5:
        l1_sample_reads.append({
            'name': read.query_name,
            'chrom': read.reference_name,
            'start': read.reference_start,
            'end': read.reference_end,
            'length': read.query_length,
        })
    n += 1
l1_bam.close()
print(f"  L1 BAM: {len(l1_bam_reads)} primary reads")

# How many L1 BAM reads are in L1 summary?
overlap = l1_bam_reads & l1_read_ids
print(f"  L1 BAM ∩ L1 summary: {len(overlap)} ({len(overlap)/len(l1_bam_reads)*100:.1f}%)")

# Check Control BAM
ctrl_bam = pysam.AlignmentFile(ctrl_bam_path, 'rb')
ctrl_bam_reads = set()
ctrl_sample_reads = []
n = 0
for read in ctrl_bam.fetch():
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        continue
    ctrl_bam_reads.add(read.query_name)
    if n < 5:
        ctrl_sample_reads.append({
            'name': read.query_name,
            'chrom': read.reference_name,
            'start': read.reference_start,
            'end': read.reference_end,
            'length': read.query_length,
        })
    n += 1
ctrl_bam.close()
print(f"  Control BAM: {len(ctrl_bam_reads)} primary reads")

# Any overlap between L1 BAM and Control BAM?
bam_overlap = l1_bam_reads & ctrl_bam_reads
print(f"  L1 BAM ∩ Control BAM: {len(bam_overlap)} (should be 0)")

# Any Control BAM reads in L1 summary?
ctrl_in_l1 = ctrl_bam_reads & l1_read_ids
print(f"  Control BAM ∩ L1 summary: {len(ctrl_in_l1)} (should be 0)")

print(f"\n  Sample L1 BAM reads:")
for r in l1_sample_reads:
    in_summary = "YES" if r['name'] in l1_read_ids else "NO"
    print(f"    {r['name'][:30]}... {r['chrom']}:{r['start']}-{r['end']} len={r['length']} in_summary={in_summary}")

print(f"\n  Sample Control BAM reads:")
for r in ctrl_sample_reads:
    in_l1 = "YES" if r['name'] in l1_read_ids else "NO"
    print(f"    {r['name'][:30]}... {r['chrom']}:{r['start']}-{r['end']} len={r['length']} in_L1={in_l1}")


# =====================================================================
# 2. Compare Part3 cache vs BAM direct
# =====================================================================
print("\n--- CHECK 2: Part3 cache vs BAM direct parsing ---")

# Load Part3 L1 cache
l1_cache_path = f"{TOPICDIR}/part3_l1_per_read_cache/{TEST_GROUP}_l1_per_read.tsv"
if os.path.exists(l1_cache_path):
    l1_cache = pd.read_csv(l1_cache_path, sep='\t')
    print(f"  L1 cache: {len(l1_cache)} reads")
    print(f"  L1 cache columns: {list(l1_cache.columns)}")

    # Cache psi/kb calculation
    def count_psi(pos_str):
        if pd.isna(pos_str) or pos_str == '[]':
            return 0
        return len(ast.literal_eval(str(pos_str)))

    l1_cache['n_psi'] = l1_cache['psi_positions'].apply(count_psi)
    l1_cache['n_m6a'] = l1_cache['m6a_positions'].apply(count_psi)

    # What is the denominator in cache?
    print(f"  Cache 'read_length' column exists: {'read_length' in l1_cache.columns}")
    if 'read_length' in l1_cache.columns:
        print(f"  Cache read_length: mean={l1_cache['read_length'].mean():.0f}, median={l1_cache['read_length'].median():.0f}")
        l1_cache['psi_per_kb'] = l1_cache['n_psi'] / (l1_cache['read_length'] / 1000)
        print(f"  Cache psi/kb (read_length denom): mean={l1_cache['psi_per_kb'].mean():.2f}")

    # Also try aligned_length if it exists
    if 'aligned_length' in l1_cache.columns:
        print(f"  Cache aligned_length: mean={l1_cache['aligned_length'].mean():.0f}")

    # Total psi / total kb
    total_psi = l1_cache['n_psi'].sum()
    if 'read_length' in l1_cache.columns:
        total_kb_rdlen = l1_cache['read_length'].sum() / 1000
        print(f"  Cache pooled psi/kb (read_length): {total_psi/total_kb_rdlen:.2f}")
else:
    print(f"  L1 cache not found!")
    l1_cache = None

# Load Part3 Control cache
ctrl_cache_path = f"{TOPICDIR}/part3_ctrl_per_read_cache/{TEST_GROUP}_ctrl_per_read.tsv"
if os.path.exists(ctrl_cache_path):
    ctrl_cache = pd.read_csv(ctrl_cache_path, sep='\t')
    print(f"\n  Control cache: {len(ctrl_cache)} reads")
    print(f"  Control cache columns: {list(ctrl_cache.columns)}")

    ctrl_cache['n_psi'] = ctrl_cache['psi_positions'].apply(count_psi)

    if 'read_length' in ctrl_cache.columns:
        print(f"  Cache read_length: mean={ctrl_cache['read_length'].mean():.0f}, median={ctrl_cache['read_length'].median():.0f}")
        total_psi_c = ctrl_cache['n_psi'].sum()
        total_kb_c = ctrl_cache['read_length'].sum() / 1000
        print(f"  Cache pooled psi/kb (read_length): {total_psi_c/total_kb_c:.2f}")
else:
    print(f"  Control cache not found!")
    ctrl_cache = None


# =====================================================================
# 3. Direct BAM comparison: read_length vs aligned_length
# =====================================================================
print("\n--- CHECK 3: read_length vs aligned_length from BAM ---")

def get_cigar_aligned_bp(read):
    """Sum of M/=/X blocks only."""
    if read.cigartuples is None:
        return 0
    return sum(length for op, length in read.cigartuples if op in (0, 7, 8))

# L1 BAM
l1_bam = pysam.AlignmentFile(l1_bam_path, 'rb')
l1_qlens = []
l1_alens = []
l1_ref_spans = []
l1_psi_counts = []
n = 0

for read in l1_bam.fetch():
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        continue

    qlen = read.query_length or 0
    alen = get_cigar_aligned_bp(read)
    ref_span = (read.reference_end - read.reference_start) if read.reference_end else 0

    # Count psi calls ≥128
    psi_count = 0
    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else None
        ml_tag = read.get_tag('ML') if read.has_tag('ML') else None
    except:
        mm_tag = ml_tag = None

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
                    if ml_idx + i < len(ml_values) and ml_values[ml_idx + i] >= 128:
                        psi_count += 1
            ml_idx += n_pos

    l1_qlens.append(qlen)
    l1_alens.append(alen)
    l1_ref_spans.append(ref_span)
    l1_psi_counts.append(psi_count)
    n += 1

l1_bam.close()

l1_qlens = np.array(l1_qlens)
l1_alens = np.array(l1_alens)
l1_ref_spans = np.array(l1_ref_spans)
l1_psi_counts = np.array(l1_psi_counts)

print(f"  L1 BAM ({n} reads):")
print(f"    query_length:   mean={l1_qlens.mean():.0f}, median={np.median(l1_qlens):.0f}")
print(f"    aligned_bp:     mean={l1_alens.mean():.0f}, median={np.median(l1_alens):.0f}")
print(f"    reference_span: mean={l1_ref_spans.mean():.0f}, median={np.median(l1_ref_spans):.0f}")
print(f"    q/a ratio:      mean={np.mean(l1_qlens/np.maximum(l1_alens, 1)):.3f}")
print(f"    Total psi (≥128): {l1_psi_counts.sum()}")
print(f"    psi/kb (query_length): {l1_psi_counts.sum() / (l1_qlens.sum()/1000):.2f}")
print(f"    psi/kb (aligned_bp):   {l1_psi_counts.sum() / (l1_alens.sum()/1000):.2f}")
print(f"    psi/kb (ref_span):     {l1_psi_counts.sum() / (l1_ref_spans.sum()/1000):.2f}")

# Control BAM
ctrl_bam = pysam.AlignmentFile(ctrl_bam_path, 'rb')
ctrl_qlens = []
ctrl_alens = []
ctrl_ref_spans = []
ctrl_psi_counts = []
n = 0

for read in ctrl_bam.fetch():
    if read.is_unmapped or read.is_secondary or read.is_supplementary:
        continue

    qlen = read.query_length or 0
    alen = get_cigar_aligned_bp(read)
    ref_span = (read.reference_end - read.reference_start) if read.reference_end else 0

    psi_count = 0
    try:
        mm_tag = read.get_tag('MM') if read.has_tag('MM') else None
        ml_tag = read.get_tag('ML') if read.has_tag('ML') else None
    except:
        mm_tag = ml_tag = None

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
                    if ml_idx + i < len(ml_values) and ml_values[ml_idx + i] >= 128:
                        psi_count += 1
            ml_idx += n_pos

    ctrl_qlens.append(qlen)
    ctrl_alens.append(alen)
    ctrl_ref_spans.append(ref_span)
    ctrl_psi_counts.append(psi_count)
    n += 1

ctrl_bam.close()

ctrl_qlens = np.array(ctrl_qlens)
ctrl_alens = np.array(ctrl_alens)
ctrl_ref_spans = np.array(ctrl_ref_spans)
ctrl_psi_counts = np.array(ctrl_psi_counts)

print(f"\n  Control BAM ({n} reads):")
print(f"    query_length:   mean={ctrl_qlens.mean():.0f}, median={np.median(ctrl_qlens):.0f}")
print(f"    aligned_bp:     mean={ctrl_alens.mean():.0f}, median={np.median(ctrl_alens):.0f}")
print(f"    reference_span: mean={ctrl_ref_spans.mean():.0f}, median={np.median(ctrl_ref_spans):.0f}")
print(f"    q/a ratio:      mean={np.mean(ctrl_qlens/np.maximum(ctrl_alens, 1)):.3f}")
print(f"    Total psi (≥128): {ctrl_psi_counts.sum()}")
print(f"    psi/kb (query_length): {ctrl_psi_counts.sum() / (ctrl_qlens.sum()/1000):.2f}")
print(f"    psi/kb (aligned_bp):   {ctrl_psi_counts.sum() / (ctrl_alens.sum()/1000):.2f}")
print(f"    psi/kb (ref_span):     {ctrl_psi_counts.sum() / (ctrl_ref_spans.sum()/1000):.2f}")

# =====================================================================
# 4. Splicing check: N in CIGAR for Control reads?
# =====================================================================
print("\n--- CHECK 4: Splicing in L1 vs Control BAMs ---")

for label, bam_path in [("L1", l1_bam_path), ("Control", ctrl_bam_path)]:
    bam = pysam.AlignmentFile(bam_path, 'rb')
    n_total = 0
    n_spliced = 0
    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        n_total += 1
        if read.cigartuples:
            has_N = any(op == 3 for op, _ in read.cigartuples)
            if has_N:
                n_spliced += 1
    bam.close()
    print(f"  {label}: {n_total} reads, {n_spliced} spliced ({n_spliced/n_total*100:.1f}%)")

print("\n=== Done ===")
