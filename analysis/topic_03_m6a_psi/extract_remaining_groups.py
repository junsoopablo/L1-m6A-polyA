#!/usr/bin/env python3
"""
Extract per-read MAFIA data for cell lines not yet processed.
Already done: HeLa(3), HeLa-Ars(3), MCF7(3), A549(3), K562(3)
Need: H9(3), Hct116(2), HepG2(2), HEYA8(3), MCF7-EV(1), SHSY5Y(3)
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path

PROJECT_DIR = Path("/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1")
RESULTS_GROUP_DIR = PROJECT_DIR / "results_group"
OUTPUT_DIR = PROJECT_DIR / "analysis/01_exploration/topic_03_m6a_psi"
PROB_THRESHOLD = 128

GROUPS_TO_EXTRACT = [
    "H9_2", "H9_3", "H9_4",
    "Hct116_3", "Hct116_4",
    "HepG2_5", "HepG2_6",
    "HEYA8_1", "HEYA8_2", "HEYA8_3",
    "MCF7-EV_1",
    "SHSY5Y_1", "SHSY5Y_2", "SHSY5Y_3",
]


def parse_mm_ml_tags(mm_tag, ml_tag):
    result = {'m6A': [], 'psi': []}
    if mm_tag is None or ml_tag is None:
        return result
    mod_blocks = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    for block in mod_blocks:
        if not block:
            continue
        parts = block.split(',')
        mod_type = parts[0]
        if 'N+17802' in mod_type or 'A+a' in mod_type or 'A+m' in mod_type:
            mod_key = 'm6A'
        elif 'N+21891' in mod_type or 'U+p' in mod_type or 'T+p' in mod_type:
            mod_key = 'psi'
        else:
            ml_idx += len(parts) - 1
            continue
        current_pos = 0
        for pos_str in parts[1:]:
            if pos_str:
                delta = int(pos_str)
                current_pos += delta
                if ml_idx < len(ml_tag):
                    prob = ml_tag[ml_idx]
                    result[mod_key].append((current_pos, prob))
                ml_idx += 1
    return result


for group in GROUPS_TO_EXTRACT:
    out_path = OUTPUT_DIR / f"{group}_mafia_per_read.tsv"
    if out_path.exists():
        print(f"SKIP {group}: already exists")
        continue

    print(f"\nProcessing {group}...")

    # Load L1 read IDs
    l1_file = RESULTS_GROUP_DIR / group / "g_summary" / f"{group}_L1_summary.tsv"
    if not l1_file.exists():
        print(f"  L1 summary not found: {l1_file}")
        continue
    l1_df = pd.read_csv(l1_file, sep='\t')
    l1_df = l1_df[l1_df['qc_tag'] == 'PASS']
    l1_read_ids = set(l1_df['read_id'].unique())
    print(f"  L1 reads: {len(l1_read_ids):,}")

    # Read MAFIA BAM
    bam_path = RESULTS_GROUP_DIR / group / "h_mafia" / f"{group}.mAFiA.reads.bam"
    if not bam_path.exists():
        print(f"  BAM not found: {bam_path}")
        continue

    records = []
    n_processed = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam:
            if read.query_name not in l1_read_ids:
                continue
            mm_tag = read.get_tag("MM") if read.has_tag("MM") else None
            ml_tag = read.get_tag("ML") if read.has_tag("ML") else None
            mods = parse_mm_ml_tags(mm_tag, ml_tag)

            m6a_all = mods['m6A']
            m6a_high = [(p, prob) for p, prob in m6a_all if prob >= PROB_THRESHOLD]
            psi_all = mods['psi']
            psi_high = [(p, prob) for p, prob in psi_all if prob >= PROB_THRESHOLD]
            read_len = read.query_length if read.query_length else 0

            records.append({
                'read_id': read.query_name,
                'read_length': read_len,
                'chr': read.reference_name,
                'start': read.reference_start,
                'end': read.reference_end,
                'm6a_sites_total': len(m6a_all),
                'm6a_sites_high': len(m6a_high),
                'm6a_probs': [prob for _, prob in m6a_all] if m6a_all else [],
                'm6a_mean_prob': np.mean([prob for _, prob in m6a_all]) if m6a_all else np.nan,
                'm6a_max_prob': max([prob for _, prob in m6a_all]) if m6a_all else 0,
                'm6a_positions': [p for p, _ in m6a_all] if m6a_all else [],
                'psi_sites_total': len(psi_all),
                'psi_sites_high': len(psi_high),
                'psi_probs': [prob for _, prob in psi_all] if psi_all else [],
                'psi_mean_prob': np.mean([prob for _, prob in psi_all]) if psi_all else np.nan,
                'psi_max_prob': max([prob for _, prob in psi_all]) if psi_all else 0,
                'psi_positions': [p for p, _ in psi_all] if psi_all else [],
            })
            n_processed += 1
            if n_processed % 500 == 0:
                print(f"  {n_processed:,} reads processed...")

    df = pd.DataFrame(records)
    df['group'] = group
    df.to_csv(out_path, sep='\t', index=False)
    print(f"  Done: {len(df):,} reads → {out_path.name}")

print("\nAll extractions complete!")
