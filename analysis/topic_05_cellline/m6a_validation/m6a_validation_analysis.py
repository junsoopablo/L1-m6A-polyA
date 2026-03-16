#!/usr/bin/env python3
"""
m6a_validation_analysis.py

Validate L1 m6A enrichment: real modification vs motif frequency artifact?

Core question: After chemodCode swap fix, Part3 reports L1 m6A/kb (5.17) > Control (3.60) = 1.44x.
Is this because L1 genuinely has higher per-site m6A rate, or because L1 sequence
composition leads to more DRACH motif sites, inflating m6A/kb at constant per-site rate?

Analysis 1: ML probability distribution comparison
  - Extract raw ML probabilities for m6A calls (chemodCode 21891) from MAFIA BAMs
  - Compare L1 vs Control distributions

Analysis 2: Per-read motif-based null model
  - Count actual DRACH motif sites in each read's aligned region (CIGAR-aware)
  - Calculate per-site modification rate = n_m6a / n_DRACH_sites
  - Compare L1 vs Control per-site rates
  - Null model: expected m6a/kb if per-site rate were identical
"""

import pysam
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import os, glob, sys

# === Configuration ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
RESULTS = f"{PROJECT}/results_group"
REF_FASTA_PATH = f"{PROJECT}/reference/Human.fasta"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# DRACH motifs from MAFIA classifier (18 total)
M6A_MOTIFS = sorted([
    'AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT',
    'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT',
    'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT',
])

def revcomp(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

M6A_MOTIFS_RC = [revcomp(m) for m in M6A_MOTIFS]

PROB_THRESHOLD = 128  # 50% = 128/255

# Base groups (exclude variants like HeLa-Ars, MCF7-EV)
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

def extract_m6a_probs(read):
    """Extract all ML probability values for m6A (chemodCode 21891) from a BAM read."""
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
    m6a_probs = []
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

        if '21891' in mod_spec:
            probs = ml_values[ml_idx:ml_idx + n_pos]
            m6a_probs.extend(probs)

        ml_idx += n_pos

    return m6a_probs


def get_aligned_ref_seq(read, ref_fasta):
    """
    Get reference sequence for aligned blocks only (excluding introns/N blocks).
    Returns (concatenated_ref_seq, total_aligned_bp).
    """
    chrom = read.reference_name
    if chrom is None or read.cigartuples is None:
        return "", 0

    segments = []
    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X: consumes both ref and query
            segments.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op == 1:  # I: insertion (query only)
            pass
        elif op == 2:  # D: deletion (ref only)
            ref_pos += length
        elif op == 3:  # N: skipped/intron (ref only) — EXCLUDE
            ref_pos += length
        elif op in (4, 5):  # S, H: clipping
            pass

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


def count_motifs_in_seq(seq, is_reverse):
    """Count m6A DRACH motif sites in a sequence on the correct strand."""
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


def process_bam(bam_path, ref_fasta, label=""):
    """
    Process a MAFIA BAM file.
    Returns:
        prob_hist: np.array(256) — histogram of all m6A ML probability values
        per_read: list of (n_m6a_high, n_m6a_all, n_motif_sites, aligned_len)
    """
    prob_hist = np.zeros(256, dtype=np.int64)
    per_read = []

    if not os.path.exists(bam_path):
        print(f"  WARNING: {bam_path} not found")
        return prob_hist, per_read

    bam = pysam.AlignmentFile(bam_path, 'rb')
    n = 0

    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        # ML probabilities
        probs = extract_m6a_probs(read)
        for p in probs:
            if 0 <= p <= 255:
                prob_hist[int(p)] += 1
        n_m6a_all = len(probs)
        n_m6a_high = sum(1 for p in probs if p >= PROB_THRESHOLD)

        # Motif sites in aligned blocks (CIGAR-aware, excluding introns)
        ref_seq, aligned_len = get_aligned_ref_seq(read, ref_fasta)

        n_motifs = 0
        if aligned_len > 50:
            n_motifs = count_motifs_in_seq(ref_seq, read.is_reverse)

        per_read.append((n_m6a_high, n_m6a_all, n_motifs, aligned_len))
        n += 1

    bam.close()
    return prob_hist, per_read


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("M6A VALIDATION: Real Modification vs Motif Frequency Artifact")
    print("chemodCode 21891 = m6A (VERIFIED)")
    print("=" * 70)

    ref_fasta = pysam.FastaFile(REF_FASTA_PATH)

    # Aggregate data
    l1_prob_hist = np.zeros(256, dtype=np.int64)
    ctrl_prob_hist = np.zeros(256, dtype=np.int64)
    l1_per_read_all = []
    ctrl_per_read_all = []

    # Per-group summaries
    group_results = []

    for group in BASE_GROUPS:
        l1_bam = f"{RESULTS}/{group}/h_mafia/{group}.mAFiA.reads.bam"
        ctrl_bam = f"{RESULTS}/{group}/i_control/mafia/{group}.control.mAFiA.reads.bam"

        if not os.path.exists(l1_bam):
            print(f"  {group}: L1 BAM not found, skipping")
            continue
        if not os.path.exists(ctrl_bam):
            print(f"  {group}: Control BAM not found, skipping")
            continue

        print(f"\n--- {group} ---")

        # L1
        print(f"  Processing L1 BAM...")
        l1_hist, l1_reads = process_bam(l1_bam, ref_fasta, f"{group}_L1")
        l1_prob_hist += l1_hist
        l1_per_read_all.extend(l1_reads)

        # Control
        print(f"  Processing Control BAM...")
        ctrl_hist, ctrl_reads = process_bam(ctrl_bam, ref_fasta, f"{group}_Ctrl")
        ctrl_prob_hist += ctrl_hist
        ctrl_per_read_all.extend(ctrl_reads)

        # Per-group quick stats
        l1_n = len(l1_reads)
        ctrl_n = len(ctrl_reads)

        l1_m6a_high = sum(r[0] for r in l1_reads)
        l1_motifs = sum(r[2] for r in l1_reads)
        l1_alen = sum(r[3] for r in l1_reads)

        ctrl_m6a_high = sum(r[0] for r in ctrl_reads)
        ctrl_motifs = sum(r[2] for r in ctrl_reads)
        ctrl_alen = sum(r[3] for r in ctrl_reads)

        l1_rate = l1_m6a_high / l1_motifs if l1_motifs > 0 else 0
        ctrl_rate = ctrl_m6a_high / ctrl_motifs if ctrl_motifs > 0 else 0
        l1_m6a_kb = l1_m6a_high / (l1_alen / 1000) if l1_alen > 0 else 0
        ctrl_m6a_kb = ctrl_m6a_high / (ctrl_alen / 1000) if ctrl_alen > 0 else 0
        l1_motif_kb = l1_motifs / (l1_alen / 1000) if l1_alen > 0 else 0
        ctrl_motif_kb = ctrl_motifs / (ctrl_alen / 1000) if ctrl_alen > 0 else 0

        print(f"  L1:   {l1_n:,} reads, {l1_m6a_high:,} m6A, {l1_motifs:,} motif sites, "
              f"rate={l1_rate:.4f}, m6A/kb={l1_m6a_kb:.2f}, motif/kb={l1_motif_kb:.1f}")
        print(f"  Ctrl: {ctrl_n:,} reads, {ctrl_m6a_high:,} m6A, {ctrl_motifs:,} motif sites, "
              f"rate={ctrl_rate:.4f}, m6A/kb={ctrl_m6a_kb:.2f}, motif/kb={ctrl_motif_kb:.1f}")

        group_results.append({
            'group': group,
            'l1_reads': l1_n, 'ctrl_reads': ctrl_n,
            'l1_m6a_high': l1_m6a_high, 'ctrl_m6a_high': ctrl_m6a_high,
            'l1_motif_sites': l1_motifs, 'ctrl_motif_sites': ctrl_motifs,
            'l1_aligned_bp': l1_alen, 'ctrl_aligned_bp': ctrl_alen,
            'l1_per_site_rate': l1_rate, 'ctrl_per_site_rate': ctrl_rate,
            'l1_m6a_per_kb': l1_m6a_kb, 'ctrl_m6a_per_kb': ctrl_m6a_kb,
            'l1_motif_per_kb': l1_motif_kb, 'ctrl_motif_per_kb': ctrl_motif_kb,
        })

    ref_fasta.close()

    # ==================================================================
    # ANALYSIS 1: ML Probability Distribution
    # ==================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ML Probability Distribution (L1 vs Control)")
    print("=" * 70)

    l1_total_calls = l1_prob_hist.sum()
    ctrl_total_calls = ctrl_prob_hist.sum()
    print(f"\nTotal m6A modification calls: L1={l1_total_calls:,}, Control={ctrl_total_calls:,}")

    # Mean and median probabilities
    l1_probs_expanded = np.repeat(np.arange(256), l1_prob_hist.astype(int))
    ctrl_probs_expanded = np.repeat(np.arange(256), ctrl_prob_hist.astype(int))

    l1_mean = np.mean(l1_probs_expanded) if len(l1_probs_expanded) > 0 else 0
    ctrl_mean = np.mean(ctrl_probs_expanded) if len(ctrl_probs_expanded) > 0 else 0
    l1_median = np.median(l1_probs_expanded) if len(l1_probs_expanded) > 0 else 0
    ctrl_median = np.median(ctrl_probs_expanded) if len(ctrl_probs_expanded) > 0 else 0

    print(f"  L1 mean prob:   {l1_mean:.1f}/255 ({l1_mean/255*100:.1f}%)")
    print(f"  Ctrl mean prob: {ctrl_mean:.1f}/255 ({ctrl_mean/255*100:.1f}%)")
    print(f"  L1 median prob:   {l1_median:.0f}/255 ({l1_median/255*100:.1f}%)")
    print(f"  Ctrl median prob: {ctrl_median:.0f}/255 ({ctrl_median/255*100:.1f}%)")

    # Fraction above threshold
    l1_above = l1_prob_hist[PROB_THRESHOLD:].sum()
    ctrl_above = ctrl_prob_hist[PROB_THRESHOLD:].sum()
    l1_frac_above = l1_above / l1_total_calls if l1_total_calls > 0 else 0
    ctrl_frac_above = ctrl_above / ctrl_total_calls if ctrl_total_calls > 0 else 0
    print(f"  L1 frac >= threshold:   {l1_frac_above:.4f} ({l1_above:,}/{l1_total_calls:,})")
    print(f"  Ctrl frac >= threshold: {ctrl_frac_above:.4f} ({ctrl_above:,}/{ctrl_total_calls:,})")

    # High-confidence calls (>=200/255 = ~78%)
    l1_high_conf = l1_prob_hist[200:].sum()
    ctrl_high_conf = ctrl_prob_hist[200:].sum()
    l1_frac_hc = l1_high_conf / l1_total_calls if l1_total_calls > 0 else 0
    ctrl_frac_hc = ctrl_high_conf / ctrl_total_calls if ctrl_total_calls > 0 else 0
    print(f"  L1 frac >= 200 (high conf):   {l1_frac_hc:.4f} ({l1_high_conf:,})")
    print(f"  Ctrl frac >= 200 (high conf): {ctrl_frac_hc:.4f} ({ctrl_high_conf:,})")

    # KS test on probability distributions
    l1_cdf = np.cumsum(l1_prob_hist) / l1_total_calls if l1_total_calls > 0 else np.zeros(256)
    ctrl_cdf = np.cumsum(ctrl_prob_hist) / ctrl_total_calls if ctrl_total_calls > 0 else np.zeros(256)
    ks_stat = np.max(np.abs(l1_cdf - ctrl_cdf))

    # Approximate KS p-value
    n_eff = (l1_total_calls * ctrl_total_calls) / (l1_total_calls + ctrl_total_calls) if (l1_total_calls + ctrl_total_calls) > 0 else 0
    ks_lambda = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * ks_stat if n_eff > 0 else 0
    ks_pvalue = 2 * np.exp(-2 * ks_lambda**2) if ks_lambda > 0 else 1.0

    print(f"\n  KS test (distribution comparison):")
    print(f"    D={ks_stat:.4f}, approx p={ks_pvalue:.2e}")
    if ks_stat < 0.02:
        print(f"    -> Distributions are nearly IDENTICAL (D < 0.02)")
    elif ks_stat < 0.05:
        print(f"    -> Distributions are very similar (D < 0.05)")
    else:
        print(f"    -> Distributions show meaningful difference (D >= 0.05)")

    # ==================================================================
    # ANALYSIS 2: Per-Site Modification Rate (Null Model)
    # ==================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Per-Site Modification Rate & Null Model")
    print("=" * 70)

    # Convert to arrays for vectorized analysis
    l1_arr = np.array(l1_per_read_all)  # (n_m6a_high, n_m6a_all, n_motif_sites, aligned_len)
    ctrl_arr = np.array(ctrl_per_read_all)

    # Filter reads with >=1 motif site
    l1_valid = l1_arr[l1_arr[:, 2] > 0]
    ctrl_valid = ctrl_arr[ctrl_arr[:, 2] > 0]

    print(f"\nReads with >=1 motif site: L1={len(l1_valid):,}/{len(l1_arr):,}, "
          f"Ctrl={len(ctrl_valid):,}/{len(ctrl_arr):,}")

    # Aggregate per-site rate
    l1_total_m6a = l1_valid[:, 0].sum()
    l1_total_motifs = l1_valid[:, 2].sum()
    l1_total_bp = l1_valid[:, 3].sum()
    ctrl_total_m6a = ctrl_valid[:, 0].sum()
    ctrl_total_motifs = ctrl_valid[:, 2].sum()
    ctrl_total_bp = ctrl_valid[:, 3].sum()

    l1_agg_rate = l1_total_m6a / l1_total_motifs
    ctrl_agg_rate = ctrl_total_m6a / ctrl_total_motifs
    rate_ratio = l1_agg_rate / ctrl_agg_rate if ctrl_agg_rate > 0 else float('inf')

    print(f"\n--- Aggregate per-site modification rate ---")
    print(f"  L1:   {l1_total_m6a:,} m6A / {l1_total_motifs:,} DRACH sites = {l1_agg_rate:.5f} ({l1_agg_rate*100:.3f}%)")
    print(f"  Ctrl: {ctrl_total_m6a:,} m6A / {ctrl_total_motifs:,} DRACH sites = {ctrl_agg_rate:.5f} ({ctrl_agg_rate*100:.3f}%)")
    print(f"  Rate ratio (L1/Ctrl): {rate_ratio:.4f}")

    # Fisher exact test on per-site rate
    l1_non_m6a = l1_total_motifs - l1_total_m6a
    ctrl_non_m6a = ctrl_total_motifs - ctrl_total_m6a
    or_fisher, p_fisher = stats.fisher_exact([[l1_total_m6a, l1_non_m6a],
                                               [ctrl_total_m6a, ctrl_non_m6a]])
    print(f"  Fisher exact: OR={or_fisher:.4f}, p={p_fisher:.2e}")

    # Motif density comparison
    l1_motif_kb = l1_total_motifs / (l1_total_bp / 1000)
    ctrl_motif_kb = ctrl_total_motifs / (ctrl_total_bp / 1000)
    motif_ratio = l1_motif_kb / ctrl_motif_kb

    l1_m6a_kb = l1_total_m6a / (l1_total_bp / 1000)
    ctrl_m6a_kb = ctrl_total_m6a / (ctrl_total_bp / 1000)
    m6a_ratio = l1_m6a_kb / ctrl_m6a_kb

    print(f"\n--- Decomposition of m6A/kb enrichment ---")
    print(f"  L1 DRACH motif sites/kb:  {l1_motif_kb:.2f}")
    print(f"  Ctrl DRACH motif sites/kb: {ctrl_motif_kb:.2f}")
    print(f"  Motif density ratio (L1/Ctrl): {motif_ratio:.4f}")
    print(f"")
    print(f"  L1 m6A/kb:  {l1_m6a_kb:.3f}")
    print(f"  Ctrl m6A/kb: {ctrl_m6a_kb:.3f}")
    print(f"  Observed m6A/kb ratio: {m6a_ratio:.4f}")
    print(f"")
    print(f"  Decomposition:")
    print(f"    m6A/kb ratio = motif_density_ratio x per_site_rate_ratio")
    print(f"    {m6a_ratio:.4f}     = {motif_ratio:.4f}           x {rate_ratio:.4f}")
    print(f"")
    expected_m6a_ratio_from_motifs = motif_ratio * 1.0
    excess = m6a_ratio / expected_m6a_ratio_from_motifs
    print(f"  If per-site rate were IDENTICAL:")
    print(f"    Expected m6A/kb ratio from motif density alone: {expected_m6a_ratio_from_motifs:.4f}")
    print(f"    Observed / Expected = {excess:.4f} ({(excess-1)*100:+.1f}% excess)")

    # Null model
    null_l1_m6a_kb = ctrl_agg_rate * l1_motif_kb
    null_ctrl_m6a_kb = ctrl_agg_rate * ctrl_motif_kb
    print(f"\n--- Null model (per-site rate = Control rate for both) ---")
    print(f"  Null L1 m6A/kb:  {null_l1_m6a_kb:.3f}")
    print(f"  Null Ctrl m6A/kb: {null_ctrl_m6a_kb:.3f}")
    print(f"  Null ratio: {null_l1_m6a_kb/null_ctrl_m6a_kb:.4f}")
    print(f"  Observed L1 m6A/kb: {l1_m6a_kb:.3f}")
    print(f"  Observed excess over null: {(l1_m6a_kb - null_l1_m6a_kb):.3f} m6A/kb "
          f"({(l1_m6a_kb/null_l1_m6a_kb - 1)*100:+.1f}%)")

    # Per-read distribution of per-site rates
    l1_per_read_rates = l1_valid[:, 0] / l1_valid[:, 2]
    ctrl_per_read_rates = ctrl_valid[:, 0] / ctrl_valid[:, 2]

    l1_rate_median = np.median(l1_per_read_rates)
    ctrl_rate_median = np.median(ctrl_per_read_rates)

    print(f"\n--- Per-read rate distribution ---")
    print(f"  L1 median per-read rate:   {l1_rate_median:.5f} ({l1_rate_median*100:.3f}%)")
    print(f"  Ctrl median per-read rate: {ctrl_rate_median:.5f} ({ctrl_rate_median*100:.3f}%)")

    mwu_stat, mwu_p = stats.mannwhitneyu(l1_per_read_rates, ctrl_per_read_rates, alternative='two-sided')
    print(f"  Mann-Whitney U: U={mwu_stat:.0f}, p={mwu_p:.2e}")

    # Read-length matched comparison
    print(f"\n--- Read-length controlled per-site rate ---")
    l1_lens = l1_valid[:, 3]
    ctrl_lens = ctrl_valid[:, 3]

    len_bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 100000)]
    print(f"  {'Bin':>12s}  {'L1_rate':>10s}  {'Ctrl_rate':>10s}  {'Ratio':>8s}  {'L1_n':>8s}  {'Ctrl_n':>8s}")
    for lo, hi in len_bins:
        l1_mask = (l1_lens >= lo) & (l1_lens < hi)
        ctrl_mask = (ctrl_lens >= lo) & (ctrl_lens < hi)
        l1_sub = l1_valid[l1_mask]
        ctrl_sub = ctrl_valid[ctrl_mask]
        if len(l1_sub) < 10 or len(ctrl_sub) < 10:
            continue
        l1_r = l1_sub[:, 0].sum() / l1_sub[:, 2].sum()
        ctrl_r = ctrl_sub[:, 0].sum() / ctrl_sub[:, 2].sum()
        ratio = l1_r / ctrl_r if ctrl_r > 0 else float('inf')
        print(f"  {lo}-{hi}bp  {l1_r:.5f}  {ctrl_r:.5f}  {ratio:.4f}  "
              f"{len(l1_sub):>8,}  {len(ctrl_sub):>8,}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
1. ML Probability Distribution:
   L1 mean={l1_mean:.1f}, Ctrl mean={ctrl_mean:.1f} (KS D={ks_stat:.4f})
   -> Signal confidence is {'similar' if ks_stat < 0.05 else 'different'} between L1 and Control.

2. Per-Site Modification Rate:
   L1 = {l1_agg_rate*100:.3f}% per DRACH motif site
   Ctrl = {ctrl_agg_rate*100:.3f}% per DRACH motif site
   Ratio = {rate_ratio:.4f} (Fisher p={p_fisher:.2e})

3. m6A/kb Enrichment Decomposition:
   Observed m6A/kb ratio (L1/Ctrl) = {m6a_ratio:.4f}
   = Motif density ratio ({motif_ratio:.4f}) x Per-site rate ratio ({rate_ratio:.4f})
   Motif density explains {(motif_ratio-1)/(m6a_ratio-1)*100:.0f}% of the enrichment (if m6a_ratio != 1).
   Per-site rate excess: {(rate_ratio-1)*100:+.1f}%

4. Null Model (if per-site rate were identical):
   Expected L1 m6A/kb = {null_l1_m6a_kb:.3f} (observed: {l1_m6a_kb:.3f})
   Excess over null = {(l1_m6a_kb/null_l1_m6a_kb - 1)*100:+.1f}%

5. Cross-validation with sites.bed motif enrichment:
   sites.bed m6A OR = 1.29 (L1 > Ctrl, 18/18 motifs enriched in L1)
   Per-site rate ratio here = {rate_ratio:.4f}
""")

    # ==================================================================
    # Save results
    # ==================================================================
    # Per-group results
    df_groups = pd.DataFrame(group_results)
    out_groups = os.path.join(OUT_DIR, "m6a_validation_per_group.tsv")
    df_groups.to_csv(out_groups, sep='\t', index=False)
    print(f"Saved: {out_groups}")

    # Probability histograms
    df_hist = pd.DataFrame({
        'prob_value': np.arange(256),
        'l1_count': l1_prob_hist,
        'ctrl_count': ctrl_prob_hist,
    })
    out_hist = os.path.join(OUT_DIR, "m6a_ml_probability_histogram.tsv")
    df_hist.to_csv(out_hist, sep='\t', index=False)
    print(f"Saved: {out_hist}")

    # Summary stats
    summary = {
        'l1_total_reads': len(l1_arr),
        'ctrl_total_reads': len(ctrl_arr),
        'l1_total_m6a_calls': int(l1_total_calls),
        'ctrl_total_m6a_calls': int(ctrl_total_calls),
        'l1_ml_mean': l1_mean,
        'ctrl_ml_mean': ctrl_mean,
        'l1_ml_median': float(l1_median),
        'ctrl_ml_median': float(ctrl_median),
        'ks_D': ks_stat,
        'l1_per_site_rate': l1_agg_rate,
        'ctrl_per_site_rate': ctrl_agg_rate,
        'per_site_rate_ratio': rate_ratio,
        'fisher_OR': or_fisher,
        'fisher_p': p_fisher,
        'l1_motif_per_kb': l1_motif_kb,
        'ctrl_motif_per_kb': ctrl_motif_kb,
        'motif_density_ratio': motif_ratio,
        'l1_m6a_per_kb': l1_m6a_kb,
        'ctrl_m6a_per_kb': ctrl_m6a_kb,
        'observed_m6a_ratio': m6a_ratio,
        'null_l1_m6a_kb': null_l1_m6a_kb,
        'excess_over_null_pct': (l1_m6a_kb / null_l1_m6a_kb - 1) * 100 if null_l1_m6a_kb > 0 else 0,
    }
    df_summary = pd.DataFrame([summary])
    out_summary = os.path.join(OUT_DIR, "m6a_validation_summary.tsv")
    df_summary.to_csv(out_summary, sep='\t', index=False)
    print(f"Saved: {out_summary}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
