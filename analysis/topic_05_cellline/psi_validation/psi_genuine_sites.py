#!/usr/bin/env python3
"""
psi_genuine_sites.py

Two questions:

Q1: Are there genuine, reproducible psi sites on L1?
  - Find genomic positions with high-probability psi calls (≥192, ≥224)
  - Check reproducibility across replicates (same position, same high prob)
  - Compare L1 vs Control: reproducible high-confidence site density

Q2: Why isn't L1 motif density higher despite being AT-rich?
  - Compute base composition of L1 aligned regions vs Control
  - Compute expected motif density from base composition
  - Check which motifs are enriched/depleted in L1 vs Control
"""

import pysam
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import os

# === Configuration ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
REF_FASTA_PATH = f"{PROJECT}/reference/Human.fasta"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PSI_MOTIFS = sorted([
    'AGTGG', 'ATTTG', 'CATAA', 'CATCC', 'CCTCC', 'CTTTA',
    'GATGC', 'GGTCC', 'GGTGG', 'GTTCA', 'GTTCC', 'GTTCG',
    'GTTCT', 'TATAA', 'TGTAG', 'TGTGG',
])

def revcomp(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

PSI_MOTIFS_RC = [revcomp(m) for m in PSI_MOTIFS]

# Cell lines with ≥2 replicates for reproducibility
CL_GROUPS = {
    'A549': ['A549_4', 'A549_5', 'A549_6'],
    'H9': ['H9_2', 'H9_3', 'H9_4'],
    'HeLa': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    'HEYA8': ['HEYA8_1', 'HEYA8_2', 'HEYA8_3'],
    'K562': ['K562_4', 'K562_5', 'K562_6'],
    'MCF7': ['MCF7_2', 'MCF7_3', 'MCF7_4'],
    'SHSY5Y': ['SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3'],
    'HepG2': ['HepG2_5', 'HepG2_6'],
}


# =====================================================================
# HELPERS
# =====================================================================

def get_aligned_blocks(read):
    if read.cigartuples is None:
        return []
    blocks = []
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8):
            blocks.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op in (2, 3):
            ref_pos += length
    return blocks


def get_full_aligned_ref_seq(ref_fasta, chrom, blocks):
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


def count_motifs_by_type(seq, is_reverse):
    """Count each motif type separately."""
    motifs = PSI_MOTIFS_RC if is_reverse else PSI_MOTIFS
    labels = PSI_MOTIFS  # always use forward-strand labels
    counts = {}
    for motif, label in zip(motifs, labels):
        c = 0
        idx = 0
        while True:
            pos = seq.find(motif, idx)
            if pos == -1:
                break
            c += 1
            idx = pos + 1
        counts[label] = c
    return counts


def parse_all_psi_with_genomic_pos(read):
    """Parse ALL psi calls with genomic positions and probabilities."""
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

    read_positions = []
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
                    read_positions.append((rp, prob))
                bp_idx += 1

        ml_idx += n_pos

    # Convert to genomic positions
    results = []
    for rp, prob in read_positions:
        gp = _read_pos_to_genomic(read, rp)
        if gp is not None:
            results.append((read.reference_name, gp, prob))
    return results


def _read_pos_to_genomic(read, read_pos):
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
# Q2: Base composition & motif density
# =====================================================================
print("=" * 70)
print("Q2: WHY ISN'T L1 MOTIF DENSITY HIGHER?")
print("=" * 70)

ref_fasta = pysam.FastaFile(REF_FASTA_PATH)

# Sample a few groups for base composition
SAMPLE_GROUPS = ['HeLa_1', 'MCF7_2', 'K562_4', 'HepG2_5', 'A549_4']

l1_base_counts = Counter()
ctrl_base_counts = Counter()
l1_motif_counts = defaultdict(int)
ctrl_motif_counts = defaultdict(int)
l1_total_bp = 0
ctrl_total_bp = 0

for group in SAMPLE_GROUPS:
    for bam_type, is_l1 in [('h_mafia', True), ('i_control/mafia', False)]:
        suffix = '' if is_l1 else '.control'
        bam_path = f"{RESULTS}/{group}/{bam_type}/{group}{suffix}.mAFiA.reads.bam"
        if not os.path.exists(bam_path):
            continue

        bam = pysam.AlignmentFile(bam_path, 'rb')
        n = 0
        for read in bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            blocks = get_aligned_blocks(read)
            if not blocks:
                continue

            ref_seq, abp = get_full_aligned_ref_seq(ref_fasta, read.reference_name, blocks)
            if abp < 50:
                continue

            # Base composition
            bc = Counter(ref_seq)
            motif_c = count_motifs_by_type(ref_seq, read.is_reverse)

            if is_l1:
                l1_base_counts += bc
                for m, c in motif_c.items():
                    l1_motif_counts[m] += c
                l1_total_bp += abp
            else:
                ctrl_base_counts += bc
                for m, c in motif_c.items():
                    ctrl_motif_counts[m] += c
                ctrl_total_bp += abp

            n += 1
            if not is_l1 and n >= 2500:
                break

        bam.close()

    print(f"  {group} done")

# Base composition
print(f"\n--- Base Composition (aligned reference sequence) ---")
print(f"  {'Base':<6s} {'L1':>10s} {'L1 frac':>10s} {'Control':>10s} {'Ctrl frac':>10s}")
for base in 'ACGT':
    l1_c = l1_base_counts.get(base, 0)
    ctrl_c = ctrl_base_counts.get(base, 0)
    l1_f = l1_c / l1_total_bp if l1_total_bp > 0 else 0
    ctrl_f = ctrl_c / ctrl_total_bp if ctrl_total_bp > 0 else 0
    print(f"  {base:<6s} {l1_c:>10d} {l1_f:>10.4f} {ctrl_c:>10d} {ctrl_f:>10.4f}")

l1_at = (l1_base_counts.get('A', 0) + l1_base_counts.get('T', 0)) / l1_total_bp
ctrl_at = (ctrl_base_counts.get('A', 0) + ctrl_base_counts.get('T', 0)) / ctrl_total_bp
print(f"\n  AT content: L1 = {l1_at:.4f}, Control = {ctrl_at:.4f}")
print(f"  GC content: L1 = {1-l1_at:.4f}, Control = {1-ctrl_at:.4f}")

# Per-motif density comparison
print(f"\n--- Per-Motif Density (/kb) ---")
print(f"  {'Motif':<8s} {'Seq':<8s} {'L1/kb':>8s} {'Ctrl/kb':>8s} {'Ratio':>8s} {'AT%':>6s}")
print(f"  {'-'*46}")

l1_kb = l1_total_bp / 1000
ctrl_kb = ctrl_total_bp / 1000
total_l1_motifs = 0
total_ctrl_motifs = 0

motif_rows = []
for m in PSI_MOTIFS:
    l1_d = l1_motif_counts[m] / l1_kb
    ctrl_d = ctrl_motif_counts[m] / ctrl_kb
    ratio = l1_d / ctrl_d if ctrl_d > 0 else 0
    at_pct = sum(1 for c in m if c in 'AT') / len(m) * 100
    print(f"  {m:<8s} {m:<8s} {l1_d:>8.2f} {ctrl_d:>8.2f} {ratio:>7.2f}x {at_pct:>5.0f}%")
    total_l1_motifs += l1_motif_counts[m]
    total_ctrl_motifs += ctrl_motif_counts[m]
    motif_rows.append({'motif': m, 'l1_per_kb': l1_d, 'ctrl_per_kb': ctrl_d, 'ratio': ratio, 'at_pct': at_pct})

print(f"  {'TOTAL':<8s} {'':8s} {total_l1_motifs/l1_kb:>8.2f} {total_ctrl_motifs/ctrl_kb:>8.2f} {(total_l1_motifs/l1_kb)/(total_ctrl_motifs/ctrl_kb):>7.2f}x")

# Correlation: motif AT% vs L1/Ctrl ratio
at_pcts = [r['at_pct'] for r in motif_rows]
ratios = [r['ratio'] for r in motif_rows]
r_corr, p_corr = stats.pearsonr(at_pcts, ratios)
print(f"\n  Correlation motif AT% vs L1/Ctrl density ratio: r={r_corr:.3f}, p={p_corr:.3e}")
print(f"  AT-rich motifs (≥60% AT): L1 enriched. GC-rich motifs (<40% AT): L1 depleted")

# Count AT-rich vs GC-rich motifs
n_l1_higher = sum(1 for r in motif_rows if r['ratio'] > 1)
n_ctrl_higher = sum(1 for r in motif_rows if r['ratio'] < 1)
print(f"  Motifs with L1 > Ctrl: {n_l1_higher}/16, Ctrl > L1: {n_ctrl_higher}/16")


# =====================================================================
# Q1: Reproducible high-confidence psi sites
# =====================================================================
print(f"\n{'='*70}")
print("Q1: REPRODUCIBLE HIGH-CONFIDENCE PSI SITES IN L1")
print(f"{'='*70}")

# For each cell line, collect high-prob psi positions per replicate
# Then check: how many positions are called in ≥2 replicates?

HIGH_THRESHOLDS = [128, 192, 224]

for threshold in HIGH_THRESHOLDS:
    print(f"\n--- Threshold ≥ {threshold} ---")

    l1_total_reproducible = 0
    l1_total_unique_sites = 0
    l1_total_reads = 0
    ctrl_total_reproducible = 0
    ctrl_total_unique_sites = 0
    ctrl_total_reads = 0

    for cl, groups in CL_GROUPS.items():
        if len(groups) < 2:
            continue

        for is_l1 in [True, False]:
            # Collect sites per replicate
            rep_sites = {}  # rep -> set of (chrom, pos)
            rep_reads = {}

            for group in groups:
                if is_l1:
                    bam_path = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
                else:
                    bam_path = f"{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam"

                if not os.path.exists(bam_path):
                    continue

                sites = set()
                n_reads = 0
                bam = pysam.AlignmentFile(bam_path, 'rb')

                for read in bam.fetch():
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue

                    psi_calls = parse_all_psi_with_genomic_pos(read)
                    n_reads += 1

                    for chrom, gpos, prob in psi_calls:
                        if prob >= threshold:
                            sites.add((chrom, gpos))

                    if not is_l1 and n_reads >= 2500:
                        break

                bam.close()
                rep_sites[group] = sites
                rep_reads[group] = n_reads

            # Count reproducible sites (seen in ≥2 replicates)
            all_sites = set()
            for s in rep_sites.values():
                all_sites |= s

            reproducible = 0
            for site in all_sites:
                n_reps = sum(1 for s in rep_sites.values() if site in s)
                if n_reps >= 2:
                    reproducible += 1

            total_reads = sum(rep_reads.values())

            if is_l1:
                l1_total_reproducible += reproducible
                l1_total_unique_sites += len(all_sites)
                l1_total_reads += total_reads
            else:
                ctrl_total_reproducible += reproducible
                ctrl_total_unique_sites += len(all_sites)
                ctrl_total_reads += total_reads

        # Don't print per-CL to keep output clean

    l1_repro_frac = l1_total_reproducible / l1_total_unique_sites if l1_total_unique_sites > 0 else 0
    ctrl_repro_frac = ctrl_total_reproducible / ctrl_total_unique_sites if ctrl_total_unique_sites > 0 else 0
    l1_repro_per_read = l1_total_reproducible / l1_total_reads * 1000 if l1_total_reads > 0 else 0
    ctrl_repro_per_read = ctrl_total_reproducible / ctrl_total_reads * 1000 if ctrl_total_reads > 0 else 0

    print(f"  {'Metric':<35s} {'L1':>12s} {'Control':>12s}")
    print(f"  {'-'*59}")
    print(f"  {'Total reads':<35s} {l1_total_reads:>12d} {ctrl_total_reads:>12d}")
    print(f"  {'Unique sites (any rep)':<35s} {l1_total_unique_sites:>12d} {ctrl_total_unique_sites:>12d}")
    print(f"  {'Reproducible sites (≥2 reps)':<35s} {l1_total_reproducible:>12d} {ctrl_total_reproducible:>12d}")
    print(f"  {'Reproducible fraction':<35s} {l1_repro_frac:>12.4f} {ctrl_repro_frac:>12.4f}")
    print(f"  {'Reproducible per 1K reads':<35s} {l1_repro_per_read:>12.2f} {ctrl_repro_per_read:>12.2f}")

ref_fasta.close()
print("\n=== Done ===")
