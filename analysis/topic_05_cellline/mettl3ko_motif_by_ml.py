#!/usr/bin/env python3
"""METTL3 KO: m6A site sequence motif analysis by ML probability bin.

Hypothesis: MAFIA assigns higher ML scores to canonical DRACH-motif m6A sites
(METTL3-dependent) and lower ML to non-canonical motifs (potentially METTL16-
dependent). If L1 has a different motif composition than mRNA, this could explain
why thr=204 eliminates the METTL3 KO effect on L1 but not on control mRNA.

For each sample (WT × 3, KO × 3):
  - Parse MAFIA BAMs for L1 reads and control (mRNA) reads
  - Extract m6A sites with ML values
  - Map to reference coordinates via CIGAR
  - Extract 11-mer sequence context centered on m6A
  - Stratify by ML bin and compute DRACH fraction, METTL16 motif, top 5-mers
  - Compute m6A/kb by ML bin for WT vs KO comparison
"""

import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths
# =============================================================================
VAULT = Path('/vault/external-datasets/2026/PRJEB40872_HEK293T_METTL3KO_xPore')
MAFIA_DIR = VAULT / 'mafia_guppy'
L1_FILTER = VAULT / 'l1_filter_guppy'
REF_FASTA = '/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta'

OUTDIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_05_cellline/mettl3ko_motif_analysis')
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    'WT_rep1': 'WT', 'WT_rep2': 'WT', 'WT_rep3': 'WT',
    'KO_rep1': 'KO', 'KO_rep2': 'KO', 'KO_rep3': 'KO',
}

# ML bins
ML_BINS = [
    ('bin1_128-159', 128, 159),
    ('bin2_160-191', 160, 191),
    ('bin3_192-223', 192, 223),
    ('bin4_224-255', 224, 255),
]
# Threshold comparison
THR_BINS = [
    ('below_204', 128, 203),
    ('at_or_above_204', 204, 255),
]

COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')


def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]


# =============================================================================
# CIGAR-based query-to-reference mapping
# =============================================================================
def query_to_ref_positions(read):
    """Build query_pos -> ref_pos mapping from CIGAR."""
    mapping = {}
    ref_pos = read.reference_start
    query_pos = 0
    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X: consumes both query and ref
            for i in range(length):
                mapping[query_pos] = ref_pos
                query_pos += 1
                ref_pos += 1
        elif op == 1:  # I: consumes query only
            for i in range(length):
                mapping[query_pos] = None
                query_pos += 1
        elif op in (2, 3):  # D, N: consumes ref only
            ref_pos += length
        elif op == 4:  # S: soft clip, consumes query only
            for i in range(length):
                mapping[query_pos] = None
                query_pos += 1
        elif op == 5:  # H: hard clip, consumes neither
            pass
    return mapping


# =============================================================================
# Parse MM/ML tags for m6A sites
# =============================================================================
def parse_mm_ml_m6a(mm_tag, ml_tag):
    """Parse MM/ML tags to get m6A site query positions + ML values.
    Returns list of (query_pos, ml_value).
    """
    if mm_tag is None or ml_tag is None:
        return []

    ml_list = list(ml_tag)
    entries = mm_tag.rstrip(';').split(';')
    ml_idx = 0
    m6a_sites = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(',')
        base_mod = parts[0]
        skip_counts = parts[1:]

        is_m6a = '21891' in base_mod or 'A+a' in base_mod or 'A+m' in base_mod

        if not is_m6a:
            ml_idx += len(skip_counts)
            continue

        # Walk through query positions using delta encoding
        query_pos = -1
        for skip_str in skip_counts:
            if not skip_str:
                continue
            skip = int(skip_str)
            query_pos += skip + 1
            if ml_idx < len(ml_list):
                m6a_sites.append((query_pos, ml_list[ml_idx]))
                ml_idx += 1
            else:
                ml_idx += 1

    return m6a_sites


# =============================================================================
# Motif checks
# =============================================================================
def is_drach(kmer_11):
    """Check if 11-mer centered on m6A (index 5) has DRACH motif at positions [3:8].
    D=A/G/T, R=A/G, A=m6A, C=C, H=A/C/T
    """
    if len(kmer_11) < 8:
        return False
    return (kmer_11[3] in 'AGT' and
            kmer_11[4] in 'AG' and
            kmer_11[5] == 'A' and
            kmer_11[6] == 'C' and
            kmer_11[7] in 'ACT')


def is_mettl16_motif(kmer_11):
    """Check METTL16-like motif: the methylated A in U6 context UACAGAG.
    Around the m6A (index 5): positions [2:9] = UACAGAG.
    In DNA: [2:9] = TACAGAG
    Simplified: check if pos 4=C, pos 5=A(m6A), pos 6=G (the 'CAG' core).
    More stringent: check ACAG at [4,5,6,7] where A at index 4, C at 5 -- wait.

    Actually METTL16 methylates the A in UACAGAG where reading 5'->3':
    U-A-C-[m6A]-G-A-G. So around the m6A:
    - pos -3 = T (U in RNA)
    - pos -2 = A
    - pos -1 = C
    - pos  0 = A (m6A) = index 5
    - pos +1 = G
    - pos +2 = A
    - pos +3 = G
    Check: kmer_11[2]='T', [3]='A', [4]='C', [5]='A', [6]='G', [7]='A', [8]='G'
    """
    if len(kmer_11) < 9:
        return False
    return (kmer_11[2] == 'T' and
            kmer_11[3] == 'A' and
            kmer_11[4] == 'C' and
            kmer_11[5] == 'A' and
            kmer_11[6] == 'G' and
            kmer_11[7] == 'A' and
            kmer_11[8] == 'G')


def is_cag_core(kmer_11):
    """Relaxed METTL16 check: just CAG at [-1, 0, +1].
    kmer_11[4]='C', [5]='A'(m6A), [6]='G'
    """
    if len(kmer_11) < 7:
        return False
    return (kmer_11[4] == 'C' and
            kmer_11[5] == 'A' and
            kmer_11[6] == 'G')


# =============================================================================
# Process one BAM file
# =============================================================================
def process_bam(bam_path, biotype, sample, condition, fa, read_filter=None):
    """Process a MAFIA BAM and extract m6A site-level records.

    Args:
        bam_path: path to mAFiA.reads.bam
        biotype: 'L1' or 'Ctrl'
        sample: sample name
        condition: 'WT' or 'KO'
        fa: pysam.FastaFile for reference
        read_filter: set of read IDs to include (None = include all)

    Returns:
        site_records: list of dicts (one per m6A site)
        read_records: list of dicts (one per read, for m6A/kb calculation)
    """
    if not Path(bam_path).exists():
        print(f"  WARNING: {bam_path} not found")
        return [], []

    bam = pysam.AlignmentFile(bam_path, 'rb')
    site_records = []
    read_records = []

    for read in bam:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.cigartuples is None:
            continue
        if read_filter is not None and read.query_name not in read_filter:
            continue

        chrom = read.reference_name
        rlen = read.query_alignment_length or read.infer_query_length() or 0
        if rlen < 100:
            continue
        is_reverse = read.is_reverse

        # Get MM/ML tags
        mm_tag = ml_tag = None
        for t in ['MM', 'Mm']:
            if read.has_tag(t):
                mm_tag = read.get_tag(t)
                break
        for t in ['ML', 'Ml']:
            if read.has_tag(t):
                ml_tag = read.get_tag(t)
                break

        m6a_sites = parse_mm_ml_m6a(mm_tag, ml_tag)
        if not m6a_sites:
            # Still record the read for denominator (0 sites, has length)
            read_records.append({
                'read_id': read.query_name,
                'sample': sample,
                'condition': condition,
                'biotype': biotype,
                'read_length': rlen,
                'n_m6a_all': 0,
                'n_m6a_128': 0,
                'n_m6a_204': 0,
            })
            continue

        # Build query->ref mapping
        q2r = query_to_ref_positions(read)

        # Track per-read site counts by ML threshold
        n_all = 0
        n_128 = 0
        n_204 = 0

        for qpos, ml_val in m6a_sites:
            rpos = q2r.get(qpos)
            if rpos is None:
                continue  # insertion position

            # Only consider ML >= 128 (minimum threshold)
            if ml_val < 128:
                continue

            n_all += 1
            if ml_val >= 128:
                n_128 += 1
            if ml_val >= 204:
                n_204 += 1

            # Extract 11-mer centered on m6A reference position
            # pos-5 to pos+5 (inclusive) = 11 bases
            chrom_len = fa.get_reference_length(chrom)
            start = rpos - 5
            end = rpos + 6  # exclusive
            if start < 0 or end > chrom_len:
                continue

            seq = fa.fetch(chrom, start, end).upper()
            if len(seq) != 11:
                continue

            # Reverse complement if read is on minus strand
            if is_reverse:
                seq = reverse_complement(seq)

            # Verify center is A
            if seq[5] != 'A':
                continue  # unexpected, skip

            # Extract 5-mer centered on m6A (positions 3-7)
            fivemer = seq[3:8]
            if 'N' in fivemer:
                continue

            drach = is_drach(seq)
            mettl16_full = is_mettl16_motif(seq)
            cag_core = is_cag_core(seq)

            site_records.append({
                'read_id': read.query_name,
                'sample': sample,
                'condition': condition,
                'biotype': biotype,
                'chrom': chrom,
                'ref_pos': rpos,
                'ml_value': ml_val,
                'is_reverse': is_reverse,
                'kmer_11': seq,
                'fivemer': fivemer,
                'is_drach': drach,
                'is_mettl16_full': mettl16_full,
                'is_cag_core': cag_core,
            })

        read_records.append({
            'read_id': read.query_name,
            'sample': sample,
            'condition': condition,
            'biotype': biotype,
            'read_length': rlen,
            'n_m6a_all': n_all,
            'n_m6a_128': n_128,
            'n_m6a_204': n_204,
        })

    bam.close()
    return site_records, read_records


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("METTL3 KO: m6A Motif Analysis by ML Probability Bin")
    print("=" * 80)

    # Open reference genome
    fa = pysam.FastaFile(REF_FASTA)
    print(f"\nReference: {REF_FASTA}")
    print(f"Chromosomes: {len(fa.references)}")

    all_site_records = []
    all_read_records = []

    for sample, condition in SAMPLES.items():
        print(f"\nProcessing {sample} ({condition})...")

        # L1 reads: from L1 MAFIA BAM (all reads in this BAM are L1)
        l1_bam = str(MAFIA_DIR / sample / 'mAFiA.reads.bam')
        sites, reads = process_bam(l1_bam, 'L1', sample, condition, fa)
        print(f"  L1:   {len(reads)} reads, {len(sites)} m6A sites (ML>=128)")
        all_site_records.extend(sites)
        all_read_records.extend(reads)

        # Control reads: from ctrl MAFIA BAM
        ctrl_bam = str(MAFIA_DIR / f'{sample}_ctrl' / 'mAFiA.reads.bam')
        sites, reads = process_bam(ctrl_bam, 'Ctrl', sample, condition, fa)
        print(f"  Ctrl: {len(reads)} reads, {len(sites)} m6A sites (ML>=128)")
        all_site_records.extend(sites)
        all_read_records.extend(reads)

    fa.close()

    if not all_site_records:
        print("ERROR: No m6A sites collected")
        return

    sites_df = pd.DataFrame(all_site_records)
    reads_df = pd.DataFrame(all_read_records)
    print(f"\nTotal: {len(sites_df):,} m6A sites from {len(reads_df):,} reads")
    print(f"  L1:   {(sites_df['biotype']=='L1').sum():,} sites, "
          f"{(reads_df['biotype']=='L1').sum():,} reads")
    print(f"  Ctrl: {(sites_df['biotype']=='Ctrl').sum():,} sites, "
          f"{(reads_df['biotype']=='Ctrl').sum():,} reads")

    # =================================================================
    # 1. DRACH motif fraction by ML bin
    # =================================================================
    print("\n" + "=" * 80)
    print("1. DRACH MOTIF FRACTION BY ML BIN")
    print("=" * 80)

    summary_rows = []

    for bin_name, ml_lo, ml_hi in ML_BINS + THR_BINS:
        bin_df = sites_df[(sites_df['ml_value'] >= ml_lo) & (sites_df['ml_value'] <= ml_hi)]
        for biotype in ['L1', 'Ctrl']:
            for cond in ['WT', 'KO']:
                sub = bin_df[(bin_df['biotype'] == biotype) & (bin_df['condition'] == cond)]
                n = len(sub)
                if n == 0:
                    continue
                drach_frac = sub['is_drach'].mean()
                mettl16_frac = sub['is_mettl16_full'].mean()
                cag_frac = sub['is_cag_core'].mean()
                summary_rows.append({
                    'ml_bin': bin_name,
                    'biotype': biotype,
                    'condition': cond,
                    'n_sites': n,
                    'drach_frac': round(drach_frac, 4),
                    'mettl16_frac': round(mettl16_frac, 4),
                    'cag_core_frac': round(cag_frac, 4),
                    'non_drach_frac': round(1 - drach_frac, 4),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / 'motif_summary.tsv', sep='\t', index=False)

    # Print nicely
    print(f"\n{'ML Bin':<20} {'Biotype':<6} {'Cond':<4} {'N sites':>8} "
          f"{'DRACH%':>8} {'METTL16%':>9} {'CAG%':>7}")
    print("-" * 75)
    for _, row in summary_df.iterrows():
        print(f"{row['ml_bin']:<20} {row['biotype']:<6} {row['condition']:<4} "
              f"{row['n_sites']:>8} {row['drach_frac']*100:>7.1f}% "
              f"{row['mettl16_frac']*100:>8.2f}% {row['cag_core_frac']*100:>6.1f}%")

    # =================================================================
    # 2. DRACH fraction: L1 vs Ctrl comparison (pooled across conditions)
    # =================================================================
    print("\n" + "=" * 80)
    print("2. DRACH FRACTION: L1 vs Ctrl (pooled WT+KO)")
    print("=" * 80)

    for bin_name, ml_lo, ml_hi in ML_BINS + THR_BINS:
        bin_df = sites_df[(sites_df['ml_value'] >= ml_lo) & (sites_df['ml_value'] <= ml_hi)]
        l1_sub = bin_df[bin_df['biotype'] == 'L1']
        ctrl_sub = bin_df[bin_df['biotype'] == 'Ctrl']
        if len(l1_sub) == 0 or len(ctrl_sub) == 0:
            continue
        l1_drach = l1_sub['is_drach'].mean()
        ctrl_drach = ctrl_sub['is_drach'].mean()
        # Fisher's exact test for DRACH vs non-DRACH
        a = l1_sub['is_drach'].sum()
        b = len(l1_sub) - a
        c = ctrl_sub['is_drach'].sum()
        d = len(ctrl_sub) - c
        _, pval = stats.fisher_exact([[a, b], [c, d]])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        print(f"  {bin_name:<20}: L1 {l1_drach*100:.1f}% (n={len(l1_sub)}) vs "
              f"Ctrl {ctrl_drach*100:.1f}% (n={len(ctrl_sub)})  "
              f"P={pval:.2e} {sig}")

    # =================================================================
    # 3. Position-specific nucleotide frequencies (motif logo data)
    # =================================================================
    print("\n" + "=" * 80)
    print("3. POSITION-SPECIFIC NUCLEOTIDE FREQUENCIES")
    print("=" * 80)

    freq_rows = []
    for biotype in ['L1', 'Ctrl']:
        for bin_name, ml_lo, ml_hi in THR_BINS:
            sub = sites_df[(sites_df['biotype'] == biotype) &
                           (sites_df['ml_value'] >= ml_lo) &
                           (sites_df['ml_value'] <= ml_hi)]
            if len(sub) == 0:
                continue
            print(f"\n  {biotype} / {bin_name} (n={len(sub)}):")
            print(f"  {'Pos':>4} {'A%':>7} {'C%':>7} {'G%':>7} {'T%':>7}")
            for pos_idx in range(11):
                rel_pos = pos_idx - 5  # -5 to +5
                bases = sub['kmer_11'].str[pos_idx]
                total = len(bases)
                counts = bases.value_counts()
                freqs = {}
                for b in 'ACGT':
                    freqs[b] = counts.get(b, 0) / total if total > 0 else 0
                print(f"  {rel_pos:>+4d} {freqs['A']*100:>6.1f}% {freqs['C']*100:>6.1f}% "
                      f"{freqs['G']*100:>6.1f}% {freqs['T']*100:>6.1f}%")
                freq_rows.append({
                    'biotype': biotype,
                    'ml_bin': bin_name,
                    'position': rel_pos,
                    'A_freq': round(freqs['A'], 4),
                    'C_freq': round(freqs['C'], 4),
                    'G_freq': round(freqs['G'], 4),
                    'T_freq': round(freqs['T'], 4),
                    'n_sites': total,
                })

    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(OUTDIR / 'motif_frequencies.tsv', sep='\t', index=False)

    # =================================================================
    # 4. Top 5-mers by biotype and ML bin
    # =================================================================
    print("\n" + "=" * 80)
    print("4. TOP 10 FIVE-MERS BY BIOTYPE × ML BIN")
    print("=" * 80)

    fivemer_rows = []
    for biotype in ['L1', 'Ctrl']:
        for bin_name, ml_lo, ml_hi in THR_BINS:
            sub = sites_df[(sites_df['biotype'] == biotype) &
                           (sites_df['ml_value'] >= ml_lo) &
                           (sites_df['ml_value'] <= ml_hi)]
            if len(sub) == 0:
                continue
            counts = sub['fivemer'].value_counts()
            total = len(sub)
            print(f"\n  {biotype} / {bin_name} (n={total}):")
            for i, (kmer, cnt) in enumerate(counts.head(10).items()):
                frac = cnt / total
                drach_flag = '(DRACH)' if is_drach('NN' + 'N' + kmer + 'NNN') else ''
                # More precise check using the 5-mer directly
                d_ok = kmer[0] in 'AGT'
                r_ok = kmer[1] in 'AG'
                a_ok = kmer[2] == 'A'
                c_ok = kmer[3] == 'C'
                h_ok = kmer[4] in 'ACT'
                is_dr = d_ok and r_ok and a_ok and c_ok and h_ok
                drach_flag = ' *DRACH*' if is_dr else ''
                print(f"    {i+1:>2}. {kmer}  {cnt:>5} ({frac*100:.1f}%){drach_flag}")
                fivemer_rows.append({
                    'biotype': biotype,
                    'ml_bin': bin_name,
                    'rank': i + 1,
                    'fivemer': kmer,
                    'count': cnt,
                    'fraction': round(frac, 4),
                    'is_drach': is_dr,
                })

    fivemer_df = pd.DataFrame(fivemer_rows)
    fivemer_df.to_csv(OUTDIR / 'top_5mers.tsv', sep='\t', index=False)

    # =================================================================
    # 5. METTL3 KO effect by ML bin (m6A/kb)
    # =================================================================
    print("\n" + "=" * 80)
    print("5. METTL3 KO EFFECT ON m6A/kb BY ML BIN")
    print("=" * 80)

    # For each ML threshold, compute per-read m6A/kb
    for bin_name, ml_lo, ml_hi in ML_BINS + THR_BINS:
        print(f"\n  --- {bin_name} (ML {ml_lo}-{ml_hi}) ---")
        # Count sites per read in this ML range
        bin_sites = sites_df[(sites_df['ml_value'] >= ml_lo) & (sites_df['ml_value'] <= ml_hi)]
        site_counts = bin_sites.groupby(['read_id', 'biotype', 'condition']).size().reset_index(name='n_sites')

        for biotype in ['L1', 'Ctrl']:
            # All reads of this biotype (including those with 0 sites in this bin)
            bio_reads = reads_df[reads_df['biotype'] == biotype].copy()
            # Merge site counts
            merged = bio_reads.merge(
                site_counts[site_counts['biotype'] == biotype][['read_id', 'n_sites']],
                on='read_id', how='left'
            )
            merged['n_sites'] = merged['n_sites'].fillna(0)
            merged['m6a_per_kb'] = merged['n_sites'] / (merged['read_length'] / 1000)

            wt = merged[merged['condition'] == 'WT']['m6a_per_kb']
            ko = merged[merged['condition'] == 'KO']['m6a_per_kb']

            if len(wt) == 0 or len(ko) == 0:
                continue

            wt_med = wt.median()
            ko_med = ko.median()
            wt_mean = wt.mean()
            ko_mean = ko.mean()
            fc = ko_med / wt_med if wt_med > 0 else float('inf')
            _, pval = stats.mannwhitneyu(wt, ko, alternative='two-sided')
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

            print(f"  {biotype:<5}: WT median={wt_med:.3f} mean={wt_mean:.3f} (n={len(wt)}) | "
                  f"KO median={ko_med:.3f} mean={ko_mean:.3f} (n={len(ko)}) | "
                  f"FC={fc:.3f} P={pval:.2e} {sig}")

    # =================================================================
    # 6. DRACH fraction trend across ML bins (is higher ML = more DRACH?)
    # =================================================================
    print("\n" + "=" * 80)
    print("6. DRACH FRACTION TREND ACROSS ML BINS")
    print("=" * 80)

    for biotype in ['L1', 'Ctrl']:
        bio_sites = sites_df[sites_df['biotype'] == biotype]
        print(f"\n  {biotype}:")
        ml_vals = []
        drach_vals = []
        for bin_name, ml_lo, ml_hi in ML_BINS:
            sub = bio_sites[(bio_sites['ml_value'] >= ml_lo) & (bio_sites['ml_value'] <= ml_hi)]
            if len(sub) == 0:
                continue
            frac = sub['is_drach'].mean()
            cag = sub['is_cag_core'].mean()
            print(f"    {bin_name}: DRACH={frac*100:.1f}%  CAG_core={cag*100:.1f}%  (n={len(sub)})")
            ml_vals.append((ml_lo + ml_hi) / 2)
            drach_vals.append(frac)

        if len(ml_vals) >= 3:
            rho, pval = stats.spearmanr(ml_vals, drach_vals)
            print(f"    Spearman rho={rho:.3f}, P={pval:.3f} (ML vs DRACH fraction)")

    # =================================================================
    # 7. WT vs KO DRACH fraction (does KO reduce DRACH sites more?)
    # =================================================================
    print("\n" + "=" * 80)
    print("7. WT vs KO: DRACH FRACTION COMPARISON")
    print("=" * 80)

    for biotype in ['L1', 'Ctrl']:
        print(f"\n  {biotype}:")
        for bin_name, ml_lo, ml_hi in ML_BINS + THR_BINS:
            sub = sites_df[(sites_df['biotype'] == biotype) &
                           (sites_df['ml_value'] >= ml_lo) &
                           (sites_df['ml_value'] <= ml_hi)]
            wt_sub = sub[sub['condition'] == 'WT']
            ko_sub = sub[sub['condition'] == 'KO']
            if len(wt_sub) < 10 or len(ko_sub) < 10:
                continue
            wt_drach = wt_sub['is_drach'].mean()
            ko_drach = ko_sub['is_drach'].mean()
            # Fisher's exact
            a = wt_sub['is_drach'].sum()
            b = len(wt_sub) - a
            c = ko_sub['is_drach'].sum()
            d = len(ko_sub) - c
            _, pval = stats.fisher_exact([[a, b], [c, d]])
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            print(f"    {bin_name:<20}: WT {wt_drach*100:.1f}% (n={len(wt_sub)}) vs "
                  f"KO {ko_drach*100:.1f}% (n={len(ko_sub)})  P={pval:.2e} {sig}")

    # =================================================================
    # 8. Key question: Does L1's motif composition differ from Ctrl?
    # =================================================================
    print("\n" + "=" * 80)
    print("8. KEY QUESTION: L1 vs Ctrl MOTIF COMPOSITION")
    print("=" * 80)

    # Overall (all ML >= 128)
    l1_all = sites_df[sites_df['biotype'] == 'L1']
    ctrl_all = sites_df[sites_df['biotype'] == 'Ctrl']
    print(f"\n  Overall (ML >= 128):")
    print(f"    L1:   DRACH={l1_all['is_drach'].mean()*100:.1f}%, "
          f"METTL16={l1_all['is_mettl16_full'].mean()*100:.2f}%, "
          f"CAG={l1_all['is_cag_core'].mean()*100:.1f}%  (n={len(l1_all)})")
    print(f"    Ctrl: DRACH={ctrl_all['is_drach'].mean()*100:.1f}%, "
          f"METTL16={ctrl_all['is_mettl16_full'].mean()*100:.2f}%, "
          f"CAG={ctrl_all['is_cag_core'].mean()*100:.1f}%  (n={len(ctrl_all)})")

    # High ML only (>= 204)
    l1_hi = sites_df[(sites_df['biotype'] == 'L1') & (sites_df['ml_value'] >= 204)]
    ctrl_hi = sites_df[(sites_df['biotype'] == 'Ctrl') & (sites_df['ml_value'] >= 204)]
    print(f"\n  High ML (>= 204):")
    if len(l1_hi) > 0 and len(ctrl_hi) > 0:
        print(f"    L1:   DRACH={l1_hi['is_drach'].mean()*100:.1f}%, "
              f"CAG={l1_hi['is_cag_core'].mean()*100:.1f}%  (n={len(l1_hi)})")
        print(f"    Ctrl: DRACH={ctrl_hi['is_drach'].mean()*100:.1f}%, "
              f"CAG={ctrl_hi['is_cag_core'].mean()*100:.1f}%  (n={len(ctrl_hi)})")

    # Low ML only (128-203)
    l1_lo = sites_df[(sites_df['biotype'] == 'L1') &
                     (sites_df['ml_value'] >= 128) & (sites_df['ml_value'] <= 203)]
    ctrl_lo = sites_df[(sites_df['biotype'] == 'Ctrl') &
                       (sites_df['ml_value'] >= 128) & (sites_df['ml_value'] <= 203)]
    print(f"\n  Low ML (128-203):")
    if len(l1_lo) > 0 and len(ctrl_lo) > 0:
        print(f"    L1:   DRACH={l1_lo['is_drach'].mean()*100:.1f}%, "
              f"CAG={l1_lo['is_cag_core'].mean()*100:.1f}%  (n={len(l1_lo)})")
        print(f"    Ctrl: DRACH={ctrl_lo['is_drach'].mean()*100:.1f}%, "
              f"CAG={ctrl_lo['is_cag_core'].mean()*100:.1f}%  (n={len(ctrl_lo)})")

    # =================================================================
    # 9. Summary interpretation
    # =================================================================
    print("\n" + "=" * 80)
    print("9. SUMMARY")
    print("=" * 80)

    # DRACH fraction at low vs high ML for each biotype
    for biotype in ['L1', 'Ctrl']:
        lo = sites_df[(sites_df['biotype'] == biotype) &
                      (sites_df['ml_value'] >= 128) & (sites_df['ml_value'] <= 203)]
        hi = sites_df[(sites_df['biotype'] == biotype) &
                      (sites_df['ml_value'] >= 204)]
        if len(lo) > 0 and len(hi) > 0:
            lo_drach = lo['is_drach'].mean()
            hi_drach = hi['is_drach'].mean()
            print(f"  {biotype}: Low-ML DRACH={lo_drach*100:.1f}% → High-ML DRACH={hi_drach*100:.1f}% "
                  f"(Δ={hi_drach*100-lo_drach*100:+.1f}pp)")

    print("\n  Hypothesis test: Higher ML = more DRACH?")
    all_sites = sites_df.copy()
    all_sites['ml_high'] = (all_sites['ml_value'] >= 204).astype(int)
    for biotype in ['L1', 'Ctrl']:
        sub = all_sites[all_sites['biotype'] == biotype]
        # Point-biserial correlation between ML value and DRACH
        if len(sub) > 10:
            rho, pval = stats.pointbiserialr(sub['is_drach'].astype(int), sub['ml_value'])
            print(f"    {biotype}: ML vs DRACH point-biserial r={rho:.3f}, P={pval:.2e}")

    print(f"\n  Output files saved to: {OUTDIR}")
    print("  - motif_summary.tsv")
    print("  - motif_frequencies.tsv")
    print("  - top_5mers.tsv")
    print("\nDone.")


if __name__ == '__main__':
    main()
