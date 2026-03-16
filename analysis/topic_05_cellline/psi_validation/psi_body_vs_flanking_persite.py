#!/usr/bin/env python3
"""
psi_body_vs_flanking_persite.py

Validate L1 body vs flanking psi density (1.41x) using per-site modification rate.

Previous analysis (psi_density_body_vs_flanking.py) found L1 body 6.68 psi/kb vs flanking 4.73 psi/kb = 1.41x.
But L1 body is AT-rich → more psi motif sites → psi/kb naturally higher.

This script counts ACTUAL motif sites in L1 body vs flanking regions for each read,
then computes per-site modification rate = n_psi / n_motifs.

Approach:
  For each L1 read in the MAFIA BAM:
    1. Use CIGAR to get aligned reference sequence
    2. Map each aligned base to its genomic coordinate
    3. Split into L1 body (te_start..te_end) and flanking
    4. Count psi motif sites in each region
    5. Count psi calls (ML≥128) in each region
    6. Compare per-site rates: body vs flanking
"""

import pysam
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import os, sys, ast

# === Configuration ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
REF_FASTA_PATH = f"{PROJECT}/reference/Human.fasta"
TOPICDIR = f"{PROJECT}/analysis/01_exploration/topic_05_cellline"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# PSI motifs from MAFIA classifier (16 total)
PSI_MOTIFS = sorted([
    'AGTGG', 'ATTTG', 'CATAA', 'CATCC', 'CCTCC', 'CTTTA',
    'GATGC', 'GGTCC', 'GGTGG', 'GTTCA', 'GTTCC', 'GTTCG',
    'GTTCT', 'TATAA', 'TGTAG', 'TGTGG',
])

def revcomp(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

PSI_MOTIFS_RC = [revcomp(m) for m in PSI_MOTIFS]

PROB_THRESHOLD = 128

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

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}


# =====================================================================
# HELPERS
# =====================================================================

def get_aligned_blocks_with_coords(read):
    """
    Get aligned reference blocks as list of (ref_start, ref_end).
    Only M/=/X blocks (excluding introns/N).
    """
    if read.cigartuples is None:
        return []

    blocks = []
    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            blocks.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 3:  # N (intron)
            ref_pos += length
        elif op in (1, 4, 5):  # I, S, H
            pass

    return blocks


def count_motifs_in_region(ref_fasta, chrom, blocks, region_start, region_end, is_reverse):
    """
    Count psi motif sites within [region_start, region_end) that also fall within aligned blocks.
    Returns count of motif sites.
    """
    motifs = PSI_MOTIFS_RC if is_reverse else PSI_MOTIFS

    # Get reference sequence for aligned blocks that overlap the region
    seq_parts = []
    for bstart, bend in blocks:
        # Clip to region
        s = max(bstart, region_start)
        e = min(bend, region_end)
        if s >= e:
            continue
        try:
            seq = ref_fasta.fetch(chrom, s, e)
            seq_parts.append(seq.upper())
        except Exception:
            pass

    if not seq_parts:
        return 0, 0

    full_seq = ''.join(seq_parts)
    total_bp = len(full_seq)

    count = 0
    for motif in motifs:
        idx = 0
        while True:
            pos = full_seq.find(motif, idx)
            if pos == -1:
                break
            count += 1
            idx = pos + 1

    return count, total_bp


def parse_psi_positions_and_probs(read):
    """
    Parse MM/ML tags to get psi call positions (read-relative) and their probabilities.
    Returns list of (read_relative_pos, probability) for psi calls with prob >= threshold.
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

        if '17802' in mod_spec:
            # Determine the base code
            base_code = mod_spec[0]  # N, A, C, G, T

            # Find positions of this base in the sequence
            if base_code == 'N':
                base_positions = list(range(len(seq)))
            else:
                base_positions = [i for i, c in enumerate(seq) if c == base_code]

            # Walk through skips to find modified positions
            bp_idx = 0
            for i, skip in enumerate(skips):
                bp_idx += skip
                if bp_idx < len(base_positions):
                    read_pos = base_positions[bp_idx]
                    prob = ml_values[ml_idx + i] if (ml_idx + i) < len(ml_values) else 0
                    if prob >= PROB_THRESHOLD:
                        results.append(read_pos)
                bp_idx += 1

        ml_idx += n_pos

    return results


def read_pos_to_genomic(read, read_pos):
    """Convert read-relative position to genomic coordinate using CIGAR alignment."""
    if read.cigartuples is None:
        return None

    query_idx = 0
    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            if query_idx <= read_pos < query_idx + length:
                offset = read_pos - query_idx
                return ref_pos + offset
            query_idx += length
            ref_pos += length
        elif op == 1:  # I
            if query_idx <= read_pos < query_idx + length:
                return None  # In an insertion
            query_idx += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 3:  # N (intron)
            ref_pos += length
        elif op == 4:  # S
            if query_idx <= read_pos < query_idx + length:
                return None  # In soft clip
            query_idx += length
        elif op == 5:  # H
            pass

    return None


# =====================================================================
# MAIN
# =====================================================================
print("=" * 70)
print("PSI VALIDATION: L1 Body vs Flanking — Per-Site Rate")
print("=" * 70)

ref_fasta = pysam.FastaFile(REF_FASTA_PATH)

all_results = []

for group in BASE_GROUPS:
    # Load L1 summary for TE coordinates
    summary_file = f"{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv"
    if not os.path.exists(summary_file):
        print(f"  Skip {group}: no summary")
        continue

    summary = pd.read_csv(summary_file, sep='\t')
    # Build lookup: read_id -> (te_start, te_end, subfamily)
    te_lookup = {}
    for _, row in summary.iterrows():
        te_lookup[row['read_id']] = {
            'te_start': row['te_start'],
            'te_end': row['te_end'],
            'subfamily': row['gene_id'],
            'read_start': row['start'],
            'read_end': row['end'],
        }

    # Process MAFIA BAM
    bam_path = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
    if not os.path.exists(bam_path):
        print(f"  Skip {group}: no BAM")
        continue

    print(f"\n--- {group} ---")
    bam = pysam.AlignmentFile(bam_path, 'rb')
    n_processed = 0
    n_valid = 0

    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.query_name not in te_lookup:
            continue

        te_info = te_lookup[read.query_name]
        te_start = te_info['te_start']
        te_end = te_info['te_end']

        n_processed += 1

        # Get aligned blocks
        blocks = get_aligned_blocks_with_coords(read)
        if not blocks:
            continue

        # Read genomic extent from aligned blocks
        read_gstart = blocks[0][0]
        read_gend = blocks[-1][1]

        # L1 body overlap
        body_start = max(read_gstart, te_start)
        body_end = min(read_gend, te_end)
        if body_end <= body_start:
            continue

        # Flanking region(s) — portions of aligned blocks outside TE
        # Use aligned blocks extent, not reference_start/end (avoids intron inflation)
        total_aligned_bp = sum(e - s for s, e in blocks)
        body_aligned_bp = 0
        for bs, be in blocks:
            overlap_s = max(bs, body_start)
            overlap_e = min(be, body_end)
            if overlap_e > overlap_s:
                body_aligned_bp += (overlap_e - overlap_s)
        flank_aligned_bp = total_aligned_bp - body_aligned_bp

        # Need both regions with minimum 100bp
        if body_aligned_bp < 100 or flank_aligned_bp < 100:
            continue

        chrom = read.reference_name
        is_rev = read.is_reverse

        # Count motifs in L1 body vs flanking
        motifs_body, bp_body = count_motifs_in_region(
            ref_fasta, chrom, blocks, body_start, body_end, is_rev
        )
        # For flanking: everything outside TE boundary
        motifs_flank_left, bp_flank_left = count_motifs_in_region(
            ref_fasta, chrom, blocks, read_gstart, te_start, is_rev
        )
        motifs_flank_right, bp_flank_right = count_motifs_in_region(
            ref_fasta, chrom, blocks, te_end, read_gend, is_rev
        )
        motifs_flank = motifs_flank_left + motifs_flank_right

        if motifs_body < 1 or motifs_flank < 1:
            continue

        # Count psi calls in each region
        psi_read_positions = parse_psi_positions_and_probs(read)

        psi_body = 0
        psi_flank = 0
        for rp in psi_read_positions:
            gp = read_pos_to_genomic(read, rp)
            if gp is None:
                continue
            if body_start <= gp < body_end:
                psi_body += 1
            else:
                psi_flank += 1

        n_valid += 1
        all_results.append({
            'group': group,
            'read_id': read.query_name,
            'subfamily': te_info['subfamily'],
            'body_bp': bp_body,
            'flank_bp': bp_flank_left + bp_flank_right,
            'motifs_body': motifs_body,
            'motifs_flank': motifs_flank,
            'psi_body': psi_body,
            'psi_flank': psi_flank,
        })

    bam.close()
    print(f"  L1 reads in BAM: {n_processed}, valid (≥100bp body+flank, ≥1 motif each): {n_valid}")

ref_fasta.close()

# =====================================================================
# Aggregate
# =====================================================================
df = pd.DataFrame(all_results)
df['is_young'] = df['subfamily'].isin(YOUNG_SUBFAMILIES)

# Cell line from group
cl_map = {}
for g in BASE_GROUPS:
    cl = g.rsplit('_', 1)[0]
    cl_map[g] = cl
df['cell_line'] = df['group'].map(cl_map)

print(f"\n{'='*70}")
print(f"AGGREGATE RESULTS: {len(df)} reads")
print(f"{'='*70}")

# Per-site rate
tot_psi_body = df['psi_body'].sum()
tot_psi_flank = df['psi_flank'].sum()
tot_motifs_body = df['motifs_body'].sum()
tot_motifs_flank = df['motifs_flank'].sum()
tot_bp_body = df['body_bp'].sum()
tot_bp_flank = df['flank_bp'].sum()

rate_body = tot_psi_body / tot_motifs_body if tot_motifs_body > 0 else 0
rate_flank = tot_psi_flank / tot_motifs_flank if tot_motifs_flank > 0 else 0

motif_density_body = tot_motifs_body / (tot_bp_body / 1000)
motif_density_flank = tot_motifs_flank / (tot_bp_flank / 1000)

psi_per_kb_body = tot_psi_body / (tot_bp_body / 1000)
psi_per_kb_flank = tot_psi_flank / (tot_bp_flank / 1000)

print(f"\n  {'Metric':<25s} {'L1 Body':>12s} {'Flanking':>12s} {'Ratio':>8s}")
print(f"  {'-'*57}")
print(f"  {'Per-site rate':<25s} {rate_body:>11.4f} {rate_flank:>11.4f} {rate_body/rate_flank:>7.3f}x")
print(f"  {'Motif density (/kb)':<25s} {motif_density_body:>11.2f} {motif_density_flank:>11.2f} {motif_density_body/motif_density_flank:>7.3f}x")
print(f"  {'Psi/kb':<25s} {psi_per_kb_body:>11.2f} {psi_per_kb_flank:>11.2f} {psi_per_kb_body/psi_per_kb_flank:>7.3f}x")
print(f"  {'Total psi calls':<25s} {tot_psi_body:>12d} {tot_psi_flank:>12d}")
print(f"  {'Total motif sites':<25s} {tot_motifs_body:>12d} {tot_motifs_flank:>12d}")
print(f"  {'Total bp':<25s} {tot_bp_body:>12d} {tot_bp_flank:>12d}")

# Decomposition
print(f"\n  psi/kb ratio = {psi_per_kb_body/psi_per_kb_flank:.3f}x = "
      f"motif_density({motif_density_body/motif_density_flank:.3f}x) × "
      f"per_site_rate({rate_body/rate_flank:.3f}x)")

# Fisher exact test on per-site rate
table = [[tot_psi_body, tot_motifs_body - tot_psi_body],
         [tot_psi_flank, tot_motifs_flank - tot_psi_flank]]
OR, p_fisher = stats.fisher_exact(table)
print(f"\n  Fisher exact: OR={OR:.4f}, p={p_fisher:.2e}")

# Paired Wilcoxon on per-read per-site rates
df['rate_body'] = df['psi_body'] / df['motifs_body']
df['rate_flank'] = df['psi_flank'] / df['motifs_flank']
stat, p_wilcox = stats.wilcoxon(df['rate_body'] - df['rate_flank'], alternative='two-sided')
print(f"  Wilcoxon paired: stat={stat:.0f}, p={p_wilcox:.2e}")

# =====================================================================
# Per cell-line breakdown
# =====================================================================
print(f"\n{'='*70}")
print(f"PER CELL-LINE BREAKDOWN")
print(f"{'='*70}")

print(f"\n  {'CL':<10s} {'N':>6s} {'Rate_body':>10s} {'Rate_flank':>11s} {'Ratio':>7s} {'Motif_b/kb':>11s} {'Motif_f/kb':>11s} {'M_ratio':>8s}")
print(f"  {'-'*74}")

for cl in sorted(df['cell_line'].unique()):
    sub = df[df['cell_line'] == cl]
    pb = sub['psi_body'].sum()
    pf = sub['psi_flank'].sum()
    mb = sub['motifs_body'].sum()
    mf = sub['motifs_flank'].sum()
    bb = sub['body_bp'].sum()
    bf = sub['flank_bp'].sum()
    rb = pb / mb if mb > 0 else 0
    rf = pf / mf if mf > 0 else 0
    md_b = mb / (bb / 1000)
    md_f = mf / (bf / 1000)
    ratio = rb / rf if rf > 0 else 0
    m_ratio = md_b / md_f if md_f > 0 else 0
    print(f"  {cl:<10s} {len(sub):>6d} {rb:>10.4f} {rf:>10.4f} {ratio:>7.3f}x {md_b:>11.2f} {md_f:>11.2f} {m_ratio:>7.3f}x")

# =====================================================================
# Young vs Ancient breakdown
# =====================================================================
print(f"\n{'='*70}")
print(f"YOUNG vs ANCIENT L1 BODY vs FLANKING")
print(f"{'='*70}")

for label, mask in [('Young L1', df['is_young']), ('Ancient L1', ~df['is_young'])]:
    sub = df[mask]
    pb = sub['psi_body'].sum()
    pf = sub['psi_flank'].sum()
    mb = sub['motifs_body'].sum()
    mf = sub['motifs_flank'].sum()
    bb = sub['body_bp'].sum()
    bf = sub['flank_bp'].sum()
    rb = pb / mb if mb > 0 else 0
    rf = pf / mf if mf > 0 else 0
    md_b = mb / (bb / 1000)
    md_f = mf / (bf / 1000)
    pkb_b = pb / (bb / 1000) if bb > 0 else 0
    pkb_f = pf / (bf / 1000) if bf > 0 else 0

    print(f"\n  {label}: n={len(sub)}")
    if rf == 0 or mb == 0 or mf == 0 or bb == 0 or bf == 0:
        print(f"    Insufficient data, skipping")
        continue
    print(f"    Per-site rate:  body={rb:.4f}, flank={rf:.4f}, ratio={rb/rf:.3f}x")
    print(f"    Motif density:  body={md_b:.2f}/kb, flank={md_f:.2f}/kb, ratio={md_b/md_f:.3f}x")
    print(f"    Psi/kb:         body={pkb_b:.2f}, flank={pkb_f:.2f}, ratio={pkb_b/pkb_f:.3f}x")
    print(f"    Decomposition:  psi/kb({pkb_b/pkb_f:.3f}x) = motif({md_b/md_f:.3f}x) × rate({rb/rf:.3f}x)")

    if len(sub) > 20:
        table_ya = [[pb, mb - pb], [pf, mf - pf]]
        or_ya, p_ya = stats.fisher_exact(table_ya)
        print(f"    Fisher: OR={or_ya:.4f}, p={p_ya:.2e}")

# =====================================================================
# Save
# =====================================================================
out_file = os.path.join(OUT_DIR, 'psi_body_vs_flanking_persite.tsv')
df.to_csv(out_file, sep='\t', index=False)
print(f"\nSaved: {out_file}")

summary_file = os.path.join(OUT_DIR, 'psi_body_vs_flanking_persite_summary.tsv')
summary_rows = []
for label, mask in [('all', df.index == df.index), ('young', df['is_young']), ('ancient', ~df['is_young'])]:
    sub = df[mask] if isinstance(mask, pd.Series) else df
    pb = sub['psi_body'].sum()
    pf = sub['psi_flank'].sum()
    mb = sub['motifs_body'].sum()
    mf = sub['motifs_flank'].sum()
    bb = sub['body_bp'].sum()
    bf = sub['flank_bp'].sum()
    summary_rows.append({
        'category': label, 'n_reads': len(sub),
        'psi_body': pb, 'psi_flank': pf,
        'motifs_body': mb, 'motifs_flank': mf,
        'bp_body': bb, 'bp_flank': bf,
        'rate_body': pb / mb if mb > 0 else 0,
        'rate_flank': pf / mf if mf > 0 else 0,
        'motif_per_kb_body': mb / (bb / 1000),
        'motif_per_kb_flank': mf / (bf / 1000),
        'psi_per_kb_body': pb / (bb / 1000),
        'psi_per_kb_flank': pf / (bf / 1000),
    })
pd.DataFrame(summary_rows).to_csv(summary_file, sep='\t', index=False)
print(f"Saved: {summary_file}")

print("\n=== Done ===")
