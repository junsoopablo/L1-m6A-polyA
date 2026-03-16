#!/usr/bin/env python3
"""
Dorado RNA004 Pseudouridine Validation
=======================================
Validates MAFIA-detected L1 psi enrichment using independent method (dorado basecaller)
and independent chemistry (RNA004).

Steps:
  2. L1 filtering — Stage 1 (10% overlap) + Stage 2 (no splice, exon exclusion, strand match)
  3. Parse dorado MM/ML tags for pseU (T+17802)
  4. Compare L1 vs Control psi/kb; compare with MAFIA results

Usage:
  conda run -n research python dorado_validation.py

Requires: pysam, pandas, numpy, scipy, matplotlib
External: bedtools (on PATH)
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path('/blaze/junsoopablo/dorado_validation')
REF_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference')
ANALYSIS_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation')
PART3_CACHE = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline')

L1_BED = REF_DIR / 'L1_TE_L1_family.bed'
HUMAN_GTF = REF_DIR / 'Human.gtf'
TE_GTF = REF_DIR / 'hg38_rmsk_TE.gtf'

PSI_PROB_THRESHOLD = 128  # out of 255 (50%)
L1_OVERLAP_FRAC = 0.10    # 10% of read length must overlap L1
EXON_OVERLAP_MIN = 100    # ≥100bp exon overlap → exclude

SAMPLES = {
    'HeLa_1_1': {
        'bam': BASE_DIR / 'HeLa_1_1' / 'HeLa_1_1.dorado.sorted.bam',
        'mafia_l1_cache': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
        'mafia_ctrl_cache': ['HeLa_1', 'HeLa_2', 'HeLa_3'],
    },
    'HCT116_1_1': {
        'bam': BASE_DIR / 'HCT116_1_1' / 'HCT116_1_1.dorado.sorted.bam',
        'mafia_l1_cache': ['Hct116_3', 'Hct116_4'],
        'mafia_ctrl_cache': ['Hct116_3', 'Hct116_4'],
    },
    'HEK293T_1_1': {
        'bam': BASE_DIR / 'HEK293T_1_1' / 'HEK293T_1_1.dorado.sorted.bam',
        'mafia_l1_cache': [],
        'mafia_ctrl_cache': [],
    },
}


# ─── Helper: Generate BED files ──────────────────────────────────────────────

def generate_exon_bed(gtf_path: Path, out_bed: Path):
    """Extract exon regions from Human.gtf → 3-col BED."""
    if out_bed.exists():
        return
    print(f"  Generating exon BED from {gtf_path.name}...")
    with open(gtf_path) as gtf, open(out_bed, 'w') as out:
        for line in gtf:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 9 or fields[2] != 'exon':
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            out.write(f"{chrom}\t{start}\t{end}\n")


def generate_l1_bed_with_strand(gtf_path: Path, out_bed: Path):
    """Extract L1 TE regions with strand from TE GTF → 6-col BED."""
    if out_bed.exists():
        return
    print(f"  Generating L1 BED with strand from {gtf_path.name}...")
    with open(gtf_path) as gtf, open(out_bed, 'w') as out:
        for line in gtf:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 9 or 'family_id "L1"' not in fields[8]:
                continue
            chrom = fields[0]
            start = max(int(fields[3]) - 1, 0)
            end = int(fields[4])
            strand = fields[6] if len(fields) > 6 else '.'
            m = re.search(r'transcript_id "([^"]+)"', fields[8])
            tid = m.group(1) if m else '.'
            m2 = re.search(r'gene_id "([^"]+)"', fields[8])
            gid = m2.group(1) if m2 else '.'
            out.write(f"{chrom}\t{start}\t{end}\t{tid}\t{gid}\t{strand}\n")


# ─── Step 2: L1 Filtering (2-stage) ──────────────────────────────────────────

def filter_l1_reads_2stage(bam_path: Path, tmpdir: Path):
    """
    2-stage L1 filtering matching MAFIA pipeline:
      Stage 1: ≥10% overlap with L1 regions
      Stage 2: exclude spliced (CIGAR N), exclude ≥100bp exon overlap, strand match

    Returns:
      stage1_l1: set of read IDs (stage 1 only)
      stage2_l1: set of read IDs (stage 1 + stage 2)
      ctrl_ids: set of non-L1 read IDs
      read_lengths: dict of read_id → length
      read_strands: dict of read_id → strand (+/-)
      spliced_ids: set of spliced read IDs
    """
    print(f"  Filtering L1 reads from {bam_path.name}...")

    reads_bed = tmpdir / 'reads.bed'
    overlaps_tsv = tmpdir / 'overlaps.tsv'
    exon_bed = tmpdir / 'exons.bed'
    l1_strand_bed = tmpdir / 'l1_with_strand.bed'
    exon_overlaps_tsv = tmpdir / 'exon_overlaps.tsv'
    strand_overlaps_tsv = tmpdir / 'strand_overlaps.tsv'

    # Generate reference BEDs
    generate_exon_bed(HUMAN_GTF, exon_bed)
    generate_l1_bed_with_strand(TE_GTF, l1_strand_bed)

    # ── Scan BAM: read lengths, strands, spliced detection ──
    print("    Extracting read info from BAM...")
    read_lengths = {}
    read_strands = {}
    spliced_ids = set()

    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            rid = read.query_name
            read_lengths[rid] = read.query_length
            read_strands[rid] = '-' if read.is_reverse else '+'
            # Check for splice junctions (CIGAR operation 3 = N)
            if read.cigartuples:
                for op, _ in read.cigartuples:
                    if op == 3:  # N = skip region
                        spliced_ids.add(rid)
                        break

    print(f"    Found {len(read_lengths):,} primary mapped reads, "
          f"{len(spliced_ids):,} spliced")

    # ── Stage 1: 10% overlap ──
    print("    Stage 1: bedtools intersect (10% overlap)...")
    subprocess.run(
        f"bedtools bamtobed -split -i {bam_path} > {reads_bed}",
        shell=True, check=True
    )
    subprocess.run(
        f"bedtools intersect -a {reads_bed} -b {L1_BED} -wo > {overlaps_tsv}",
        shell=True, check=True
    )

    overlap_sum = defaultdict(int)
    with open(overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            overlap_sum[fields[3]] += int(fields[-1])

    stage1_l1 = set()
    for rid, ov in overlap_sum.items():
        if rid in read_lengths and read_lengths[rid] > 0:
            if ov / read_lengths[rid] >= L1_OVERLAP_FRAC:
                stage1_l1.add(rid)

    print(f"    Stage 1: {len(stage1_l1):,} L1 reads")

    # ── Stage 2a: Exclude spliced ──
    after_splice = stage1_l1 - spliced_ids
    n_splice_removed = len(stage1_l1) - len(after_splice)
    print(f"    Stage 2a: -{n_splice_removed:,} spliced → {len(after_splice):,}")

    # ── Stage 2b: Exclude ≥100bp exon overlap ──
    # Create BED of only stage1 L1 reads for exon intersection
    l1_reads_bed = tmpdir / 'l1_reads.bed'
    with open(reads_bed) as fin, open(l1_reads_bed, 'w') as fout:
        for line in fin:
            rid = line.strip().split('\t')[3]
            if rid in after_splice:
                fout.write(line)

    subprocess.run(
        f"bedtools intersect -a {l1_reads_bed} -b {exon_bed} -wo > {exon_overlaps_tsv}",
        shell=True, check=True
    )

    exon_overlap = defaultdict(int)
    with open(exon_overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            rid = fields[3]
            ov = int(fields[-1])
            if ov > exon_overlap[rid]:
                exon_overlap[rid] = ov

    exon_exclude = {rid for rid, ov in exon_overlap.items() if ov >= EXON_OVERLAP_MIN}
    after_exon = after_splice - exon_exclude
    print(f"    Stage 2b: -{len(exon_exclude):,} exon-overlapping → {len(after_exon):,}")

    # ── Stage 2c: Strand match with L1 ──
    # Intersect L1 reads with L1 BED (with strand) to check strand match
    subprocess.run(
        f"bedtools intersect -a {l1_reads_bed} -b {l1_strand_bed} -wo > {strand_overlaps_tsv}",
        shell=True, check=True
    )

    # For each read, find the best-overlapping L1 element and its strand
    read_l1_strand = {}  # rid → L1 strand (from best overlap)
    read_l1_overlap = defaultdict(int)
    with open(strand_overlaps_tsv) as f:
        for line in f:
            fields = line.strip().split('\t')
            rid = fields[3]
            l1_strand = fields[11]  # 6th col of l1_strand_bed (strand)
            ov = int(fields[-1])
            if ov > read_l1_overlap[rid]:
                read_l1_overlap[rid] = ov
                read_l1_strand[rid] = l1_strand

    strand_exclude = set()
    for rid in after_exon:
        if rid in read_l1_strand and rid in read_strands:
            if read_strands[rid] != read_l1_strand[rid]:
                strand_exclude.add(rid)

    stage2_l1 = after_exon - strand_exclude
    print(f"    Stage 2c: -{len(strand_exclude):,} strand-mismatched → {len(stage2_l1):,}")

    # ── Control: ANY L1 overlap excluded (matching MAFIA pipeline) ──
    # reads that have ANY overlap with L1 (even 1bp) are excluded
    any_l1_overlap = set(overlap_sum.keys())  # all reads with any L1 overlap
    ctrl_strict = set(read_lengths.keys()) - any_l1_overlap
    print(f"    Control (strict, no ANY L1 overlap): {len(ctrl_strict):,}")

    # Subsample to CTRL_MAX_READS (matching MAFIA pipeline)
    CTRL_MAX_READS = 3000
    rng_ctrl = np.random.default_rng(42)
    if len(ctrl_strict) > CTRL_MAX_READS:
        ctrl_subsample = set(rng_ctrl.choice(list(ctrl_strict), CTRL_MAX_READS, replace=False))
    else:
        ctrl_subsample = ctrl_strict
    print(f"    Control subsampled: {len(ctrl_subsample):,}")

    # Also keep full ctrl for comparison
    ctrl_full = set(read_lengths.keys()) - stage1_l1

    print(f"    Final: Stage2 L1={len(stage2_l1):,}, "
          f"Ctrl_strict={len(ctrl_subsample):,}, Ctrl_full={len(ctrl_full):,}")

    return stage1_l1, stage2_l1, ctrl_subsample, ctrl_full, read_lengths, read_strands, spliced_ids


# ─── Step 3: Parse Dorado MM/ML Tags ─────────────────────────────────────────

def parse_dorado_psi(bam_path: Path, read_ids: set, read_lengths: dict):
    """
    Parse dorado pseU modifications from MM/ML tags.
    Dorado uses T+17802. for pseudouridine on RNA004.

    Returns DataFrame with columns: read_id, read_length, psi_sites_high, psi_per_kb
    """
    results = []
    n_with_mm = 0
    n_no_mm = 0

    seen = set()
    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.query_name not in read_ids or read.query_name in seen:
                continue
            seen.add(read.query_name)

            rlen = read_lengths.get(read.query_name, read.query_length)
            mm_tag = None
            ml_tag = None

            try:
                mm_tag = read.get_tag('MM')
            except KeyError:
                pass
            try:
                mm_tag = mm_tag or read.get_tag('Mm')
            except KeyError:
                pass
            try:
                ml_tag = read.get_tag('ML')
            except KeyError:
                pass
            try:
                ml_tag = ml_tag or read.get_tag('Ml')
            except KeyError:
                pass

            if mm_tag is None or ml_tag is None:
                n_no_mm += 1
                results.append({
                    'read_id': read.query_name,
                    'read_length': rlen,
                    'psi_sites_high': 0,
                    'psi_per_kb': 0.0,
                })
                continue

            n_with_mm += 1
            psi_probs = extract_mod_probs(mm_tag, ml_tag, 'T', '17802')
            psi_high = sum(1 for p in psi_probs if p >= PSI_PROB_THRESHOLD)

            results.append({
                'read_id': read.query_name,
                'read_length': rlen,
                'psi_sites_high': psi_high,
                'psi_per_kb': (psi_high / rlen * 1000) if rlen > 0 else 0.0,
            })

    print(f"    Reads with MM tag: {n_with_mm:,}, without: {n_no_mm:,}")
    return pd.DataFrame(results)


def extract_mod_probs(mm_tag: str, ml_tag, base: str, mod_code: str):
    """
    Extract modification probabilities for a specific base+mod_code from MM/ML tags.
    MM format: "BASE+CODE.,delta1,delta2,...;BASE+CODE.,delta1,...;"
    ML format: array of uint8 probabilities, concatenated for all mods in MM order.
    """
    ml_array = list(ml_tag)
    mm_str = mm_tag.rstrip(';')
    blocks = mm_str.split(';')

    ml_offset = 0
    target_probs = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        parts = block.split(',')
        header = parts[0].rstrip('.?')  # Strip SAM spec trailing '.' or '?'
        deltas = parts[1:] if len(parts) > 1 else []
        n_sites = len(deltas)

        match_pos = f"{base}+{mod_code}"
        match_neg = f"{base}-{mod_code}"

        if header == match_pos or header == match_neg:
            target_probs = ml_array[ml_offset:ml_offset + n_sites]

        ml_offset += n_sites

    return target_probs


# ─── Step 4: Compare & Report ─────────────────────────────────────────────────

def compute_stats(l1_psi_kb, ctrl_psi_kb, label=""):
    """Compute MWU, Welch t-test, and permutation test for mean difference."""
    # MWU: subsample control if too large (>100K) for speed
    if len(ctrl_psi_kb) > 100_000:
        rng_sub = np.random.default_rng(123)
        ctrl_sub_idx = rng_sub.choice(len(ctrl_psi_kb), 100_000, replace=False)
        ctrl_for_mwu = ctrl_psi_kb.iloc[ctrl_sub_idx]
    else:
        ctrl_for_mwu = ctrl_psi_kb
    u_stat, u_pval = stats.mannwhitneyu(l1_psi_kb, ctrl_for_mwu, alternative='two-sided')

    # Welch t-test (works on full data, fast)
    t_stat, t_pval = stats.ttest_ind(l1_psi_kb, ctrl_psi_kb, equal_var=False)

    # Permutation test for mean difference (subsample both to max 50K each)
    observed_diff = l1_psi_kb.mean() - ctrl_psi_kb.mean()
    rng = np.random.default_rng(42)
    max_n = 50_000
    l1_vals = l1_psi_kb.values
    ctrl_vals = ctrl_psi_kb.values
    if len(l1_vals) > max_n:
        l1_vals = rng.choice(l1_vals, max_n, replace=False)
    if len(ctrl_vals) > max_n:
        ctrl_vals = rng.choice(ctrl_vals, max_n, replace=False)

    combined = np.concatenate([l1_vals, ctrl_vals])
    n_l1 = len(l1_vals)
    n_perm = 10000
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        perm_diffs[i] = perm[:n_l1].mean() - perm[n_l1:].mean()
    perm_pval = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()

    return {
        'mwu_stat': u_stat, 'mwu_pval': u_pval,
        'welch_t': t_stat, 'welch_pval': t_pval,
        'perm_mean_diff': observed_diff, 'perm_pval': perm_pval,
    }


def load_mafia_cache(sample_keys: list, data_type: str):
    """Load MAFIA per-read data from part3 cache."""
    dfs = []
    for key in sample_keys:
        path = PART3_CACHE / f'part3_{data_type}_per_read_cache' / f'{key}_{data_type}_per_read.tsv'
        if path.exists():
            df = pd.read_csv(path, sep='\t')
            df['sample'] = key
            dfs.append(df)
        else:
            print(f"  WARNING: MAFIA cache not found: {path}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def run_analysis():
    """Main analysis: filter, parse, compare."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    summary_rows = []

    for sample_name, config in SAMPLES.items():
        bam_path = config['bam']
        print(f"\n{'='*60}")
        print(f"Processing {sample_name}")
        print(f"{'='*60}")

        if not bam_path.exists():
            print(f"  SKIPPING: BAM not found at {bam_path}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Step 2: 2-stage L1 filtering
            stage1_l1, stage2_l1, ctrl_strict, ctrl_full, read_lengths, _, _ = \
                filter_l1_reads_2stage(bam_path, tmpdir)

            # Step 3: Parse psi
            print("  Parsing Control (strict, subsampled) psi...")
            ctrl_df = parse_dorado_psi(bam_path, ctrl_strict, read_lengths)
            ctrl_df['category'] = 'Control'

            print("  Parsing L1 psi (stage1)...")
            l1_s1_df = parse_dorado_psi(bam_path, stage1_l1, read_lengths)
            l1_s1_df['category'] = 'L1_stage1'

            print("  Parsing L1 psi (stage2)...")
            l1_s2_df = parse_dorado_psi(bam_path, stage2_l1, read_lengths)
            l1_s2_df['category'] = 'L1_stage2'

        # Save per-read data
        outdir = ANALYSIS_DIR / sample_name
        os.makedirs(outdir, exist_ok=True)
        l1_s1_df.to_csv(outdir / f'{sample_name}_l1_stage1_per_read.tsv', sep='\t', index=False)
        l1_s2_df.to_csv(outdir / f'{sample_name}_l1_stage2_per_read.tsv', sep='\t', index=False)
        ctrl_df.to_csv(outdir / f'{sample_name}_ctrl_per_read.tsv', sep='\t', index=False)

        # Step 4: Statistics for both stages
        for stage_label, l1_df in [('stage1', l1_s1_df), ('stage2', l1_s2_df)]:
            l1_psi_kb = l1_df['psi_per_kb']
            ctrl_psi_kb = ctrl_df['psi_per_kb']

            stat_results = compute_stats(l1_psi_kb, ctrl_psi_kb, stage_label)

            row = {
                'sample': sample_name,
                'method': f'dorado_{stage_label}',
                'chemistry': 'RNA004',
                'l1_n': len(l1_df),
                'ctrl_n': len(ctrl_df),
                'l1_psi_kb_median': l1_psi_kb.median(),
                'l1_psi_kb_mean': l1_psi_kb.mean(),
                'ctrl_psi_kb_median': ctrl_psi_kb.median(),
                'ctrl_psi_kb_mean': ctrl_psi_kb.mean(),
                'median_ratio': l1_psi_kb.median() / ctrl_psi_kb.median() if ctrl_psi_kb.median() > 0 else np.nan,
                'mean_ratio': l1_psi_kb.mean() / ctrl_psi_kb.mean() if ctrl_psi_kb.mean() > 0 else np.nan,
                'mwu_pval': stat_results['mwu_pval'],
                'welch_t': stat_results['welch_t'],
                'welch_pval': stat_results['welch_pval'],
                'perm_mean_diff': stat_results['perm_mean_diff'],
                'perm_pval': stat_results['perm_pval'],
                'l1_frac_with_psi': (l1_df['psi_sites_high'] > 0).mean(),
                'ctrl_frac_with_psi': (ctrl_df['psi_sites_high'] > 0).mean(),
                'l1_rdlen_median': l1_df['read_length'].median(),
                'ctrl_rdlen_median': ctrl_df['read_length'].median(),
            }
            summary_rows.append(row)

            print(f"\n  === {stage_label.upper()} ===")
            print(f"    L1:  n={row['l1_n']:,}, median={row['l1_psi_kb_median']:.2f}, "
                  f"mean={row['l1_psi_kb_mean']:.2f} psi/kb, rdLen={row['l1_rdlen_median']:.0f}")
            print(f"    Ctrl: n={row['ctrl_n']:,}, median={row['ctrl_psi_kb_median']:.2f}, "
                  f"mean={row['ctrl_psi_kb_mean']:.2f} psi/kb, rdLen={row['ctrl_rdlen_median']:.0f}")
            print(f"    Median ratio: {row['median_ratio']:.3f}, Mean ratio: {row['mean_ratio']:.3f}")
            print(f"    MWU p={row['mwu_pval']:.2e}, Welch t={row['welch_t']:.2f} p={row['welch_pval']:.2e}")
            print(f"    Perm mean diff={row['perm_mean_diff']:.3f}, perm p={row['perm_pval']:.4f}")
            print(f"    frac psi+: L1={row['l1_frac_with_psi']:.1%}, Ctrl={row['ctrl_frac_with_psi']:.1%}")

        # Load MAFIA comparison
        mafia_l1 = load_mafia_cache(config['mafia_l1_cache'], 'l1')
        mafia_ctrl = load_mafia_cache(config['mafia_ctrl_cache'], 'ctrl')

        if not mafia_l1.empty and not mafia_ctrl.empty:
            mafia_l1_psi_kb = mafia_l1['psi_sites_high'] / mafia_l1['read_length'] * 1000
            mafia_ctrl_psi_kb = mafia_ctrl['psi_sites_high'] / mafia_ctrl['read_length'] * 1000

            mafia_stats = compute_stats(mafia_l1_psi_kb, mafia_ctrl_psi_kb, 'MAFIA')

            mafia_row = {
                'sample': sample_name,
                'method': 'MAFIA',
                'chemistry': 'r9.4.1',
                'l1_n': len(mafia_l1),
                'ctrl_n': len(mafia_ctrl),
                'l1_psi_kb_median': mafia_l1_psi_kb.median(),
                'l1_psi_kb_mean': mafia_l1_psi_kb.mean(),
                'ctrl_psi_kb_median': mafia_ctrl_psi_kb.median(),
                'ctrl_psi_kb_mean': mafia_ctrl_psi_kb.mean(),
                'median_ratio': mafia_l1_psi_kb.median() / mafia_ctrl_psi_kb.median() if mafia_ctrl_psi_kb.median() > 0 else np.nan,
                'mean_ratio': mafia_l1_psi_kb.mean() / mafia_ctrl_psi_kb.mean() if mafia_ctrl_psi_kb.mean() > 0 else np.nan,
                'mwu_pval': mafia_stats['mwu_pval'],
                'welch_t': mafia_stats['welch_t'],
                'welch_pval': mafia_stats['welch_pval'],
                'perm_mean_diff': mafia_stats['perm_mean_diff'],
                'perm_pval': mafia_stats['perm_pval'],
                'l1_frac_with_psi': (mafia_l1['psi_sites_high'] > 0).mean(),
                'ctrl_frac_with_psi': (mafia_ctrl['psi_sites_high'] > 0).mean(),
                'l1_rdlen_median': mafia_l1['read_length'].median(),
                'ctrl_rdlen_median': mafia_ctrl['read_length'].median(),
            }
            summary_rows.append(mafia_row)

            print(f"\n  === MAFIA (r9.4.1, {len(config['mafia_l1_cache'])} reps) ===")
            print(f"    L1:  n={mafia_row['l1_n']:,}, median={mafia_row['l1_psi_kb_median']:.2f}, "
                  f"mean={mafia_row['l1_psi_kb_mean']:.2f}, rdLen={mafia_row['l1_rdlen_median']:.0f}")
            print(f"    Ctrl: n={mafia_row['ctrl_n']:,}, median={mafia_row['ctrl_psi_kb_median']:.2f}, "
                  f"mean={mafia_row['ctrl_psi_kb_mean']:.2f}, rdLen={mafia_row['ctrl_rdlen_median']:.0f}")
            print(f"    Median ratio: {mafia_row['median_ratio']:.3f}, Mean ratio: {mafia_row['mean_ratio']:.3f}")
            print(f"    MWU p={mafia_row['mwu_pval']:.2e}, Welch p={mafia_row['welch_pval']:.2e}")
            print(f"    Perm mean diff={mafia_row['perm_mean_diff']:.3f}, perm p={mafia_row['perm_pval']:.4f}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = ANALYSIS_DIR / 'validation_summary.tsv'
    summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.4f')
    print(f"\nSummary saved to {summary_path}")

    if summary_rows:
        generate_figure(summary_df)

    return summary_df


def generate_figure(summary_df: pd.DataFrame):
    """Generate validation comparison figure: 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for sample in summary_df['sample'].unique():
        sub = summary_df[summary_df['sample'] == sample]

        # ─── Panel A: Median psi/kb comparison ──────────────────────────
        ax = axes[0, 0]
        methods = sub['method'].tolist()
        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width/2, sub['l1_psi_kb_median'], width, label='L1', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, sub['ctrl_psi_kb_median'], width, label='Control', color='#3498db', alpha=0.8)

        for i, (_, row) in enumerate(sub.iterrows()):
            ymax = max(row['l1_psi_kb_median'], row['ctrl_psi_kb_median'])
            pval = row['mwu_pval']
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            ax.text(i, ymax + 0.15, stars, ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Median psi/kb')
        ax.set_title(f'{sample}: Median psi/kb (MWU test)')
        ax.legend(fontsize=9)

        # ─── Panel B: Mean psi/kb comparison ────────────────────────────
        ax = axes[0, 1]
        ax.bar(x - width/2, sub['l1_psi_kb_mean'], width, label='L1', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, sub['ctrl_psi_kb_mean'], width, label='Control', color='#3498db', alpha=0.8)

        for i, (_, row) in enumerate(sub.iterrows()):
            ymax = max(row['l1_psi_kb_mean'], row['ctrl_psi_kb_mean'])
            pval = row['welch_pval']
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            ax.text(i, ymax + 0.3, stars, ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Mean psi/kb')
        ax.set_title(f'{sample}: Mean psi/kb (Welch t-test)')
        ax.legend(fontsize=9)

        # ─── Panel C: Ratio comparison ──────────────────────────────────
        ax = axes[1, 0]
        x2 = np.arange(len(methods))
        ax.bar(x2 - width/2, sub['median_ratio'], width, label='Median ratio', color='#9b59b6', alpha=0.8)
        ax.bar(x2 + width/2, sub['mean_ratio'], width, label='Mean ratio', color='#f39c12', alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x2)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('L1 / Control ratio')
        ax.set_title(f'{sample}: L1/Ctrl psi/kb ratio')
        ax.legend(fontsize=9)

        # ─── Panel D: Read length comparison ────────────────────────────
        ax = axes[1, 1]
        ax.bar(x - width/2, sub['l1_rdlen_median'], width, label='L1', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, sub['ctrl_rdlen_median'], width, label='Control', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Median read length (bp)')
        ax.set_title(f'{sample}: Read length (confound check)')
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = ANALYSIS_DIR / 'dorado_validation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    print("Dorado RNA004 Pseudouridine Validation (2-stage filtering)")
    print("=" * 60)
    run_analysis()
