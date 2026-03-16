#!/usr/bin/env python3
"""
Regenerate Part3 per-read cache with updated m6A threshold.

m6A threshold: 204/255 (0.80)  ← changed from 128/255 (0.50)
psi threshold: 128/255 (0.50)  ← unchanged

Parses MAFIA BAMs, counts m6A/psi sites per read, saves TSV cache.
Both N+ and N- MM entries are matched for chemodCode 21891(m6A) / 17802(psi).
"""
import os
import sys
import pysam
from collections import defaultdict

PROJECT = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1'
TOPIC = f'{PROJECT}/analysis/01_exploration/topic_05_cellline'

M6A_THRESHOLD = 204   # 0.80 probability
PSI_THRESHOLD = 128   # 0.50 probability

L1_CACHE_DIR = f'{TOPIC}/part3_l1_per_read_cache'
CTRL_CACHE_DIR = f'{TOPIC}/part3_ctrl_per_read_cache'

# All groups from existing cache files
L1_GROUPS = [
    'A549_4', 'A549_5', 'A549_6',
    'H9_2', 'H9_3', 'H9_4',
    'Hct116_3', 'Hct116_4',
    'HeLa_1', 'HeLa_2', 'HeLa_3',
    'HeLa-Ars_1', 'HeLa-Ars_2', 'HeLa-Ars_3',
    'HepG2_5', 'HepG2_6',
    'HEYA8_1', 'HEYA8_2', 'HEYA8_3',
    'K562_4', 'K562_5', 'K562_6',
    'MCF7_2', 'MCF7_3', 'MCF7_4',
    'MCF7-EV_1',
    'SHSY5Y_1', 'SHSY5Y_2', 'SHSY5Y_3',
]

# Ctrl groups (no MCF7-EV)
CTRL_GROUPS = [g for g in L1_GROUPS if g != 'MCF7-EV_1']


def parse_mafia_bam(bam_path, m6a_thr, psi_thr):
    """Parse MAFIA BAM, extract per-read m6A and psi data.

    Matches BOTH N+ and N- entries for chemodCode 21891(m6A) / 17802(psi).
    Returns list of dicts with: read_id, read_length, psi_sites_high,
    m6a_sites_high, psi_positions, m6a_positions.
    """
    if not os.path.exists(bam_path):
        return None

    bam = pysam.AlignmentFile(bam_path, 'rb')
    results = []

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        mm_tag = None
        ml_tag = None
        for tag_name in ['MM', 'Mm']:
            if read.has_tag(tag_name):
                mm_tag = read.get_tag(tag_name)
                break
        for tag_name in ['ML', 'Ml']:
            if read.has_tag(tag_name):
                ml_tag = read.get_tag(tag_name)
                break

        read_len = read.query_length  # full read length (incl soft-clips), matches old cache
        if read_len is None or read_len < 50:
            continue

        read_id = read.query_name
        seq = read.query_sequence

        if mm_tag is None or ml_tag is None or seq is None:
            results.append({
                'read_id': read_id, 'read_length': read_len,
                'psi_sites_high': 0, 'm6a_sites_high': 0,
                'psi_positions': '[]', 'm6a_positions': '[]',
            })
            continue

        ml_values = list(ml_tag)
        entries = mm_tag.rstrip(';').split(';')
        ml_offset = 0

        m6a_positions = []
        psi_positions = []

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(',')
            header = parts[0]
            skip_counts = [int(x) for x in parts[1:] if x]
            n_sites = len(skip_counts)
            if n_sites == 0:
                continue

            is_m6a = '21891' in header
            is_psi = '17802' in header
            site_mls = ml_values[ml_offset:ml_offset + n_sites]
            ml_offset += n_sites

            if not (is_m6a or is_psi):
                continue

            # Determine which base to track
            base_char = header[0]
            seq_len = len(seq)

            # Walk through sequence to find modification positions
            base_idx = -1
            skip_idx = 0
            if n_sites == 0:
                continue
            bases_to_skip = skip_counts[0]

            for read_pos in range(seq_len):
                base = seq[read_pos].upper()
                if base_char == 'N' or base == base_char:
                    if bases_to_skip == 0:
                        if skip_idx < n_sites:
                            ml_val = site_mls[skip_idx]
                            if is_m6a and ml_val >= m6a_thr:
                                m6a_positions.append(read_pos)
                            elif is_psi and ml_val >= psi_thr:
                                psi_positions.append(read_pos)
                            skip_idx += 1
                            if skip_idx < n_sites:
                                bases_to_skip = skip_counts[skip_idx]
                            else:
                                break
                    else:
                        bases_to_skip -= 1

        results.append({
            'read_id': read_id,
            'read_length': read_len,
            'psi_sites_high': len(psi_positions),
            'm6a_sites_high': len(m6a_positions),
            'psi_positions': str(psi_positions),
            'm6a_positions': str(m6a_positions),
        })

    bam.close()
    return results


def write_cache(results, out_path):
    """Write cache TSV."""
    with open(out_path, 'w') as f:
        f.write('read_id\tread_length\tpsi_sites_high\tm6a_sites_high\tpsi_positions\tm6a_positions\n')
        for r in results:
            f.write(f"{r['read_id']}\t{r['read_length']}\t{r['psi_sites_high']}\t"
                    f"{r['m6a_sites_high']}\t{r['psi_positions']}\t{r['m6a_positions']}\n")


# ── Main ──
print(f"Regenerating Part3 cache: m6A thr={M6A_THRESHOLD}/255 ({M6A_THRESHOLD/255:.2f}), "
      f"psi thr={PSI_THRESHOLD}/255 ({PSI_THRESHOLD/255:.2f})")
print(f"L1 groups: {len(L1_GROUPS)}, Ctrl groups: {len(CTRL_GROUPS)}\n")

n_done = 0
n_total = len(L1_GROUPS) + len(CTRL_GROUPS)

# L1 caches
for group in L1_GROUPS:
    n_done += 1
    bam_path = f'{PROJECT}/results_group/{group}/h_mafia/{group}.mAFiA.reads.bam'
    out_path = f'{L1_CACHE_DIR}/{group}_l1_per_read.tsv'
    print(f"[{n_done}/{n_total}] L1 {group} ... ", end='', flush=True)

    results = parse_mafia_bam(bam_path, M6A_THRESHOLD, PSI_THRESHOLD)
    if results is None:
        print("BAM not found, SKIP")
        continue

    write_cache(results, out_path)
    n_reads = len(results)
    m6a_total = sum(r['m6a_sites_high'] for r in results)
    psi_total = sum(r['psi_sites_high'] for r in results)
    print(f"{n_reads:,} reads, m6A={m6a_total:,}, psi={psi_total:,}")

# Ctrl caches
for group in CTRL_GROUPS:
    n_done += 1
    bam_path = f'{PROJECT}/results_group/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam'
    out_path = f'{CTRL_CACHE_DIR}/{group}_ctrl_per_read.tsv'
    print(f"[{n_done}/{n_total}] Ctrl {group} ... ", end='', flush=True)

    results = parse_mafia_bam(bam_path, M6A_THRESHOLD, PSI_THRESHOLD)
    if results is None:
        print("BAM not found, SKIP")
        continue

    write_cache(results, out_path)
    n_reads = len(results)
    m6a_total = sum(r['m6a_sites_high'] for r in results)
    psi_total = sum(r['psi_sites_high'] for r in results)
    print(f"{n_reads:,} reads, m6A={m6a_total:,}, psi={psi_total:,}")

print(f"\nDone. All caches regenerated with m6A threshold {M6A_THRESHOLD}/255.")
