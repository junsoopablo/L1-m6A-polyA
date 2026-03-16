#!/usr/bin/env python3
"""
Quick check: dorado pseU calls — are they at MAFIA motif sites?
Compare psi/kb when restricted to MAFIA's 16 PSI motifs vs all-T context.
"""
import pysam
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

BAM = Path('/blaze/junsoopablo/dorado_validation/HeLa_1_1/HeLa_1_1.dorado.sorted.bam')
REF = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta')
ANALYSIS_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation')

PSI_PROB_THRESHOLD = 128

# MAFIA's 16 psi motifs (5-mer, position 2 = T that gets modified)
MAFIA_PSI_MOTIFS = {
    'GTTCA', 'GTTCC', 'GTTCG', 'GTTCT',
    'AGTGG', 'GGTGG', 'TGTGG',
    'TGTAG',
    'GGTCC',
    'CATAA', 'TATAA',
    'CATCC',
    'CTTTA',
    'ATTTG',
    'GATGC',
    'CCTCC',
}

# Load pre-computed L1/Control read IDs from stage2 analysis
L1_READS = ANALYSIS_DIR / 'HeLa_1_1' / 'HeLa_1_1_l1_stage2_per_read.tsv'
CTRL_READS = ANALYSIS_DIR / 'HeLa_1_1' / 'HeLa_1_1_ctrl_per_read.tsv'


def parse_mm_positions(mm_tag, seq, base='T', mod_code='17802'):
    """
    Parse MM tag to get query positions of modified bases.
    Returns list of (query_pos, delta_index) for the target modification.
    """
    mm_str = mm_tag.rstrip(';')
    blocks = mm_str.split(';')

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parts = block.split(',')
        header = parts[0].rstrip('.?')
        if header != f"{base}+{mod_code}" and header != f"{base}-{mod_code}":
            continue

        deltas = [int(d) for d in parts[1:]] if len(parts) > 1 else []

        # Find all T positions in query sequence
        t_positions = [i for i, c in enumerate(seq) if c == base]

        # Apply deltas to get modified T positions
        mod_positions = []
        t_idx = 0
        for delta in deltas:
            t_idx += delta
            if t_idx < len(t_positions):
                mod_positions.append(t_positions[t_idx])
            t_idx += 1  # skip the modified base itself

        return mod_positions

    return []


def query_pos_to_ref_pos(read, query_pos):
    """Convert query position to reference position using alignment pairs."""
    pairs = read.get_aligned_pairs(matches_only=False)
    for qpos, rpos in pairs:
        if qpos == query_pos:
            return rpos
    return None


def get_ref_5mer(ref_fasta, chrom, pos):
    """Get 5-mer context at position (pos is 0-based, T at position 2 of 5-mer)."""
    start = pos - 2
    end = pos + 3
    if start < 0:
        return None
    try:
        seq = ref_fasta.fetch(chrom, start, end).upper()
        if len(seq) == 5:
            return seq
    except (ValueError, KeyError):
        pass
    return None


def analyze_reads(bam_path, ref_path, read_ids, label, max_reads=5000):
    """
    For a set of reads, analyze dorado psi calls and their motif context.
    """
    ref = pysam.FastaFile(str(ref_path))
    results = []
    motif_counter = Counter()
    n_processed = 0

    rng = np.random.default_rng(42)
    read_ids_list = list(read_ids)
    if len(read_ids_list) > max_reads:
        read_ids_list = list(rng.choice(read_ids_list, max_reads, replace=False))
    read_ids_sub = set(read_ids_list)

    seen = set()
    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.query_name not in read_ids_sub or read.query_name in seen:
                continue
            seen.add(read.query_name)
            n_processed += 1

            if n_processed % 1000 == 0:
                print(f"    {label}: {n_processed:,} reads processed...")

            seq = read.query_sequence
            if not seq:
                continue
            rlen = len(seq)

            # Get MM/ML tags
            try:
                mm_tag = read.get_tag('MM')
            except KeyError:
                try:
                    mm_tag = read.get_tag('Mm')
                except KeyError:
                    continue
            try:
                ml_tag = list(read.get_tag('ML'))
            except KeyError:
                try:
                    ml_tag = list(read.get_tag('Ml'))
                except KeyError:
                    continue

            # Parse modification positions and probabilities
            mod_positions = parse_mm_positions(mm_tag, seq)
            psi_probs = extract_mod_probs_only(mm_tag, ml_tag)

            if len(mod_positions) != len(psi_probs):
                continue

            # For each high-confidence psi site, check motif context
            psi_all = 0
            psi_at_mafia_motif = 0
            psi_outside_motif = 0

            # Count T bases at MAFIA motifs (denominator)
            t_positions_all = [i for i, c in enumerate(seq) if c == 'T']
            n_t_total = len(t_positions_all)

            for qpos, prob in zip(mod_positions, psi_probs):
                if prob < PSI_PROB_THRESHOLD:
                    continue

                psi_all += 1

                # Get reference position and 5-mer context
                rpos = query_pos_to_ref_pos(read, qpos)
                if rpos is None:
                    continue

                chrom = read.reference_name
                fivemer = get_ref_5mer(ref, chrom, rpos)
                if fivemer is None:
                    continue

                motif_counter[fivemer] += 1

                if fivemer in MAFIA_PSI_MOTIFS:
                    psi_at_mafia_motif += 1
                else:
                    psi_outside_motif += 1

            results.append({
                'read_id': read.query_name,
                'read_length': rlen,
                'n_t_total': n_t_total,
                'psi_all': psi_all,
                'psi_at_mafia_motif': psi_at_mafia_motif,
                'psi_outside_motif': psi_outside_motif,
                'psi_all_per_kb': psi_all / rlen * 1000 if rlen > 0 else 0,
                'psi_motif_per_kb': psi_at_mafia_motif / rlen * 1000 if rlen > 0 else 0,
            })

    ref.close()
    return pd.DataFrame(results), motif_counter


def extract_mod_probs_only(mm_tag, ml_tag):
    """Extract just T+17802 probabilities from MM/ML."""
    ml_array = list(ml_tag)
    mm_str = mm_tag.rstrip(';')
    blocks = mm_str.split(';')
    ml_offset = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parts = block.split(',')
        header = parts[0].rstrip('.?')
        n_sites = len(parts) - 1 if len(parts) > 1 else 0

        if header in ('T+17802', 'T-17802'):
            return ml_array[ml_offset:ml_offset + n_sites]
        ml_offset += n_sites

    return []


def main():
    print("=== Dorado pseU Motif Context Analysis ===\n")

    # Load read IDs
    l1_df = pd.read_csv(L1_READS, sep='\t')
    ctrl_df = pd.read_csv(CTRL_READS, sep='\t')
    l1_ids = set(l1_df['read_id'])
    ctrl_ids = set(ctrl_df['read_id'])

    print(f"L1 reads: {len(l1_ids):,}, Control reads: {len(ctrl_ids):,}\n")

    # Analyze L1
    print("  Analyzing L1 reads...")
    l1_results, l1_motifs = analyze_reads(BAM, REF, l1_ids, 'L1', max_reads=5000)

    print("  Analyzing Control reads...")
    ctrl_results, ctrl_motifs = analyze_reads(BAM, REF, ctrl_ids, 'Ctrl', max_reads=3000)

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for label, df, motifs in [('L1', l1_results, l1_motifs), ('Control', ctrl_results, ctrl_motifs)]:
        n = len(df)
        total_psi = df['psi_all'].sum()
        motif_psi = df['psi_at_mafia_motif'].sum()
        outside_psi = df['psi_outside_motif'].sum()
        frac_at_motif = motif_psi / total_psi if total_psi > 0 else 0

        print(f"\n  {label} (n={n:,}):")
        print(f"    Total psi calls (≥128): {total_psi:,}")
        print(f"    At MAFIA motif: {motif_psi:,} ({frac_at_motif:.1%})")
        print(f"    Outside motif: {outside_psi:,} ({1-frac_at_motif:.1%})")
        print(f"    psi_all/kb:   median={df['psi_all_per_kb'].median():.2f}, mean={df['psi_all_per_kb'].mean():.2f}")
        print(f"    psi_motif/kb: median={df['psi_motif_per_kb'].median():.2f}, mean={df['psi_motif_per_kb'].mean():.2f}")

        # Top 10 motifs
        print(f"    Top 10 5-mers at psi sites:")
        for motif, count in motifs.most_common(10):
            in_mafia = " ← MAFIA" if motif in MAFIA_PSI_MOTIFS else ""
            print(f"      {motif}: {count:,} ({count/total_psi*100:.1f}%){in_mafia}")

    # ─── Direct comparison: motif-restricted psi/kb ───────────────────────
    print(f"\n{'='*60}")
    print("MOTIF-RESTRICTED COMPARISON")
    print(f"{'='*60}")

    from scipy import stats

    l1_motif_kb = l1_results['psi_motif_per_kb']
    ctrl_motif_kb = ctrl_results['psi_motif_per_kb']
    l1_all_kb = l1_results['psi_all_per_kb']
    ctrl_all_kb = ctrl_results['psi_all_per_kb']

    for metric, l1_v, ctrl_v in [
        ('All T (dorado default)', l1_all_kb, ctrl_all_kb),
        ('MAFIA 16 motifs only', l1_motif_kb, ctrl_motif_kb),
    ]:
        t_stat, t_pval = stats.ttest_ind(l1_v, ctrl_v, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(l1_v, ctrl_v, alternative='two-sided')
        ratio_med = l1_v.median() / ctrl_v.median() if ctrl_v.median() > 0 else float('nan')
        ratio_mean = l1_v.mean() / ctrl_v.mean() if ctrl_v.mean() > 0 else float('nan')

        print(f"\n  {metric}:")
        print(f"    L1:   median={l1_v.median():.3f}, mean={l1_v.mean():.3f}")
        print(f"    Ctrl: median={ctrl_v.median():.3f}, mean={ctrl_v.mean():.3f}")
        print(f"    Median ratio={ratio_med:.3f}, Mean ratio={ratio_mean:.3f}")
        print(f"    MWU p={u_pval:.2e}, Welch t={t_stat:.2f} p={t_pval:.2e}")

    # Save results
    l1_results.to_csv(ANALYSIS_DIR / 'HeLa_1_1' / 'motif_check_l1.tsv', sep='\t', index=False)
    ctrl_results.to_csv(ANALYSIS_DIR / 'HeLa_1_1' / 'motif_check_ctrl.tsv', sep='\t', index=False)
    print(f"\nResults saved to {ANALYSIS_DIR / 'HeLa_1_1'}")


if __name__ == '__main__':
    main()
