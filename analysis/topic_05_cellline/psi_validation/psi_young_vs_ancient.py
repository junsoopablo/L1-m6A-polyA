#!/usr/bin/env python3
"""
psi_young_vs_ancient.py

Per-site psi modification rate: Young L1 vs Ancient L1 vs Control.
Uses CIGAR-aware motif counting (excluding introns).

Key question: Is young L1's higher psi/kb due to higher per-site rate
or just more motif sites (different sequence composition)?
"""

import pysam
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import os, glob

PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_FASTA_PATH = f"{PROJECT}/reference/Human.fasta"

YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

PSI_MOTIFS = sorted([
    'AGTGG', 'ATTTG', 'CATAA', 'CATCC', 'CCTCC', 'CTTTA',
    'GATGC', 'GGTCC', 'GGTGG', 'GTTCA', 'GTTCC', 'GTTCG',
    'GTTCT', 'TATAA', 'TGTAG', 'TGTGG',
])
PSI_MOTIFS_RC = [m.translate(str.maketrans('ACGT', 'TGCA'))[::-1] for m in PSI_MOTIFS]

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


def load_l1_age_map(group):
    """Load read_id → 'young'/'ancient' mapping from L1 summary."""
    summary_path = f"{RESULTS}/{group}/g_summary/{group}_L1_summary.tsv"
    if not os.path.exists(summary_path):
        return {}
    df = pd.read_csv(summary_path, sep='\t', usecols=['read_id', 'gene_id'])
    age_map = {}
    for _, row in df.iterrows():
        subfam = str(row['gene_id']).split('_dup')[0] if '_dup' in str(row['gene_id']) else str(row['gene_id'])
        age_map[row['read_id']] = 'young' if subfam in YOUNG_SUBFAMILIES else 'ancient'
    return age_map


def extract_psi_probs(read):
    """Extract psi ML probabilities from MM/ML tags."""
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
    psi_probs = []
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
            psi_probs.extend(ml_values[ml_idx:ml_idx + n_pos])
        ml_idx += n_pos
    return psi_probs


def get_aligned_ref_seq(read, ref_fasta):
    """Get reference sequence for aligned blocks only (excluding introns)."""
    chrom = read.reference_name
    if chrom is None or read.cigartuples is None:
        return "", 0
    segments = []
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            segments.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 3:  # N (intron)
            ref_pos += length
        # I, S, H: no ref movement
    total_seq = ""
    total_len = 0
    for s, e in segments:
        try:
            seq = ref_fasta.fetch(chrom, s, e)
            total_seq += seq
            total_len += (e - s)
        except Exception:
            pass
    return total_seq.upper(), total_len


def count_motifs(seq, is_reverse):
    """Count psi motif sites on the correct strand."""
    motifs = PSI_MOTIFS_RC if is_reverse else PSI_MOTIFS
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


def main():
    print("=" * 70)
    print("PSI VALIDATION: Young L1 vs Ancient L1 vs Control")
    print("=" * 70)

    ref_fasta = pysam.FastaFile(REF_FASTA_PATH)

    # Accumulators: {category: {'n_reads', 'n_psi', 'n_motifs', 'aligned_bp', 'prob_hist'}}
    cats = ['young_L1', 'ancient_L1', 'control']
    data = {c: {'n_reads': 0, 'n_psi': 0, 'n_motifs': 0, 'aligned_bp': 0,
                 'prob_hist': np.zeros(256, dtype=np.int64),
                 'per_read': []}  # (n_psi_high, n_motifs, aligned_len)
            for c in cats}

    group_rows = []

    for group in BASE_GROUPS:
        l1_bam_path = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
        ctrl_bam_path = f"{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam"

        if not os.path.exists(l1_bam_path) or not os.path.exists(ctrl_bam_path):
            print(f"  {group}: BAM not found, skipping")
            continue

        print(f"\n--- {group} ---")

        # Load age classification
        age_map = load_l1_age_map(group)
        if not age_map:
            print(f"  WARNING: No L1 summary for {group}")
            continue

        # Process L1 BAM
        bam = pysam.AlignmentFile(l1_bam_path, 'rb')
        g_young = {'n': 0, 'psi': 0, 'motifs': 0, 'bp': 0}
        g_ancient = {'n': 0, 'psi': 0, 'motifs': 0, 'bp': 0}

        for read in bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            rid = read.query_name
            age = age_map.get(rid)
            if age is None:
                continue  # read not in L1 summary

            probs = extract_psi_probs(read)
            n_psi_high = sum(1 for p in probs if p >= PROB_THRESHOLD)
            ref_seq, aligned_len = get_aligned_ref_seq(read, ref_fasta)
            n_motifs = count_motifs(ref_seq, read.is_reverse) if aligned_len > 50 else 0

            cat = 'young_L1' if age == 'young' else 'ancient_L1'
            data[cat]['n_reads'] += 1
            data[cat]['n_psi'] += n_psi_high
            data[cat]['n_motifs'] += n_motifs
            data[cat]['aligned_bp'] += aligned_len
            for p in probs:
                if 0 <= p <= 255:
                    data[cat]['prob_hist'][int(p)] += 1
            data[cat]['per_read'].append((n_psi_high, n_motifs, aligned_len))

            g = g_young if age == 'young' else g_ancient
            g['n'] += 1
            g['psi'] += n_psi_high
            g['motifs'] += n_motifs
            g['bp'] += aligned_len

        bam.close()

        # Process Control BAM
        ctrl_bam = pysam.AlignmentFile(ctrl_bam_path, 'rb')
        g_ctrl = {'n': 0, 'psi': 0, 'motifs': 0, 'bp': 0}

        for read in ctrl_bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            probs = extract_psi_probs(read)
            n_psi_high = sum(1 for p in probs if p >= PROB_THRESHOLD)
            ref_seq, aligned_len = get_aligned_ref_seq(read, ref_fasta)
            n_motifs = count_motifs(ref_seq, read.is_reverse) if aligned_len > 50 else 0

            data['control']['n_reads'] += 1
            data['control']['n_psi'] += n_psi_high
            data['control']['n_motifs'] += n_motifs
            data['control']['aligned_bp'] += aligned_len
            for p in probs:
                if 0 <= p <= 255:
                    data['control']['prob_hist'][int(p)] += 1
            data['control']['per_read'].append((n_psi_high, n_motifs, aligned_len))

            g_ctrl['n'] += 1
            g_ctrl['psi'] += n_psi_high
            g_ctrl['motifs'] += n_motifs
            g_ctrl['bp'] += aligned_len

        ctrl_bam.close()

        # Per-group summary
        for label, g in [('young', g_young), ('ancient', g_ancient), ('ctrl', g_ctrl)]:
            rate = g['psi'] / g['motifs'] if g['motifs'] > 0 else 0
            psi_kb = g['psi'] / (g['bp'] / 1000) if g['bp'] > 0 else 0
            motif_kb = g['motifs'] / (g['bp'] / 1000) if g['bp'] > 0 else 0
            group_rows.append({
                'group': group, 'category': label,
                'n_reads': g['n'], 'n_psi': g['psi'], 'n_motifs': g['motifs'],
                'aligned_bp': g['bp'], 'per_site_rate': rate,
                'psi_per_kb': psi_kb, 'motif_per_kb': motif_kb,
            })

        # Print summary
        for label, g in [('young', g_young), ('ancient', g_ancient), ('ctrl', g_ctrl)]:
            rate = g['psi'] / g['motifs'] if g['motifs'] > 0 else 0
            psi_kb = g['psi'] / (g['bp'] / 1000) if g['bp'] > 0 else 0
            motif_kb = g['motifs'] / (g['bp'] / 1000) if g['bp'] > 0 else 0
            print(f"  {label:>8s}: {g['n']:>6,} reads, rate={rate:.4f}, "
                  f"psi/kb={psi_kb:.2f}, motif/kb={motif_kb:.1f}")

    ref_fasta.close()

    # ==================================================================
    # AGGREGATE RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("AGGREGATE: Young L1 vs Ancient L1 vs Control")
    print("=" * 70)

    print(f"\n{'Category':>12s}  {'Reads':>8s}  {'Psi':>8s}  {'Motifs':>10s}  "
          f"{'Rate':>8s}  {'Psi/kb':>8s}  {'Motif/kb':>9s}")

    agg = {}
    for cat in cats:
        d = data[cat]
        rate = d['n_psi'] / d['n_motifs'] if d['n_motifs'] > 0 else 0
        psi_kb = d['n_psi'] / (d['aligned_bp'] / 1000) if d['aligned_bp'] > 0 else 0
        motif_kb = d['n_motifs'] / (d['aligned_bp'] / 1000) if d['aligned_bp'] > 0 else 0
        agg[cat] = {'rate': rate, 'psi_kb': psi_kb, 'motif_kb': motif_kb}
        print(f"  {cat:>12s}  {d['n_reads']:>8,}  {d['n_psi']:>8,}  {d['n_motifs']:>10,}  "
              f"{rate:>8.4f}  {psi_kb:>8.3f}  {motif_kb:>9.2f}")

    # Comparisons
    print(f"\n--- Pairwise comparisons (per-site rate) ---")
    pairs = [('young_L1', 'ancient_L1'), ('young_L1', 'control'), ('ancient_L1', 'control')]
    for a, b in pairs:
        da, db = data[a], data[b]
        ratio = agg[a]['rate'] / agg[b]['rate'] if agg[b]['rate'] > 0 else float('inf')
        # Fisher exact
        a_nonpsi = da['n_motifs'] - da['n_psi']
        b_nonpsi = db['n_motifs'] - db['n_psi']
        or_val, p_val = stats.fisher_exact([[da['n_psi'], a_nonpsi],
                                             [db['n_psi'], b_nonpsi]])
        print(f"  {a} vs {b}:")
        print(f"    Rate: {agg[a]['rate']:.4f} vs {agg[b]['rate']:.4f} = {ratio:.4f}x")
        print(f"    Fisher OR={or_val:.4f}, p={p_val:.2e}")

    # Motif density comparison
    print(f"\n--- Motif density comparison ---")
    for a, b in pairs:
        ratio = agg[a]['motif_kb'] / agg[b]['motif_kb'] if agg[b]['motif_kb'] > 0 else 0
        print(f"  {a} vs {b}: {agg[a]['motif_kb']:.2f} vs {agg[b]['motif_kb']:.2f} = {ratio:.4f}x")

    # psi/kb decomposition
    print(f"\n--- psi/kb decomposition ---")
    for a, b in pairs:
        psi_ratio = agg[a]['psi_kb'] / agg[b]['psi_kb'] if agg[b]['psi_kb'] > 0 else 0
        motif_ratio = agg[a]['motif_kb'] / agg[b]['motif_kb'] if agg[b]['motif_kb'] > 0 else 0
        rate_ratio = agg[a]['rate'] / agg[b]['rate'] if agg[b]['rate'] > 0 else 0
        print(f"  {a} vs {b}:")
        print(f"    psi/kb ratio = {psi_ratio:.4f} = motif({motif_ratio:.4f}) × rate({rate_ratio:.4f})")

    # ML probability comparison
    print(f"\n--- ML probability (mean) ---")
    for cat in cats:
        hist = data[cat]['prob_hist']
        total = hist.sum()
        if total > 0:
            expanded = np.repeat(np.arange(256), hist.astype(int))
            mean_p = np.mean(expanded)
            frac_above = hist[PROB_THRESHOLD:].sum() / total
            print(f"  {cat:>12s}: mean={mean_p:.1f}/255, frac≥128={frac_above:.4f}, total={total:,}")

    # Read-length controlled per-site rate
    print(f"\n--- Read-length controlled per-site rate ---")
    len_bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000)]
    print(f"  {'Bin':>12s}  {'young':>8s}  {'ancient':>8s}  {'ctrl':>8s}  "
          f"{'y/a':>6s}  {'y/c':>6s}  {'a/c':>6s}")
    for lo, hi in len_bins:
        rates = {}
        for cat in cats:
            arr = np.array(data[cat]['per_read'])
            if len(arr) == 0:
                rates[cat] = 0
                continue
            mask = (arr[:, 2] >= lo) & (arr[:, 2] < hi) & (arr[:, 1] > 0)
            sub = arr[mask]
            if len(sub) < 10:
                rates[cat] = 0
                continue
            rates[cat] = sub[:, 0].sum() / sub[:, 1].sum()

        ya = rates['young_L1'] / rates['ancient_L1'] if rates['ancient_L1'] > 0 else 0
        yc = rates['young_L1'] / rates['control'] if rates['control'] > 0 else 0
        ac = rates['ancient_L1'] / rates['control'] if rates['control'] > 0 else 0
        print(f"  {lo}-{hi}bp  {rates['young_L1']:>8.4f}  {rates['ancient_L1']:>8.4f}  "
              f"{rates['control']:>8.4f}  {ya:>6.3f}  {yc:>6.3f}  {ac:>6.3f}")

    # Save
    df_groups = pd.DataFrame(group_rows)
    out_path = os.path.join(OUT_DIR, "psi_young_vs_ancient_per_group.tsv")
    df_groups.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    summary_rows = []
    for cat in cats:
        d = data[cat]
        summary_rows.append({
            'category': cat,
            'n_reads': d['n_reads'],
            'n_psi_high': d['n_psi'],
            'n_motif_sites': d['n_motifs'],
            'aligned_bp': d['aligned_bp'],
            'per_site_rate': agg[cat]['rate'],
            'psi_per_kb': agg[cat]['psi_kb'],
            'motif_per_kb': agg[cat]['motif_kb'],
        })
    df_summary = pd.DataFrame(summary_rows)
    out_path2 = os.path.join(OUT_DIR, "psi_young_vs_ancient_summary.tsv")
    df_summary.to_csv(out_path2, sep='\t', index=False)
    print(f"Saved: {out_path2}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
