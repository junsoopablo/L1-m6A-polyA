#!/usr/bin/env python3
"""
Dorado psi analysis RESTRICTED to MAFIA's 16 motif sites.

For each read:
1. Map ALL T positions to reference → check 5-mer context
2. Classify each T as "motif" (MAFIA 16) or "non-motif"
3. For motif T positions: get dorado probability from ML tag
4. Compute: modification rate = psi_calls / n_motif_T (not per kb)
5. Compare probability distributions: motif vs non-motif T positions

This answers: at positions where MAFIA would evaluate psi, does dorado
also show L1 > Control enrichment?
"""
import pysam
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────────────────

BAM = Path('/blaze/junsoopablo/dorado_validation/HeLa_1_1/HeLa_1_1.dorado.sorted.bam')
REF = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/reference/Human.fasta')
ANALYSIS_DIR = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/analysis/01_exploration/topic_06_dorado_validation')

L1_READS = ANALYSIS_DIR / 'HeLa_1_1' / 'HeLa_1_1_l1_stage2_per_read.tsv'
CTRL_READS = ANALYSIS_DIR / 'HeLa_1_1' / 'HeLa_1_1_ctrl_per_read.tsv'

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


def parse_mm_all_probs(mm_tag, ml_tag, seq):
    """
    Parse MM/ML to get (query_position, probability) for ALL T+17802 sites.
    Returns dict: {query_pos: prob} for every T evaluated by dorado.
    """
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
            deltas = [int(d) for d in parts[1:]] if n_sites > 0 else []
            probs = ml_array[ml_offset:ml_offset + n_sites]

            # Find all T positions
            t_positions = [i for i, c in enumerate(seq) if c == 'T']

            # Map deltas to query positions
            result = {}
            t_idx = 0
            for i, delta in enumerate(deltas):
                t_idx += delta
                if t_idx < len(t_positions) and i < len(probs):
                    result[t_positions[t_idx]] = probs[i]
                t_idx += 1

            return result

        ml_offset += n_sites

    return {}


def analyze_motif_restricted(bam_path, ref_path, read_ids, label, max_reads=3000):
    """
    For each read, classify ALL T positions by motif context,
    then compare dorado probabilities at motif vs non-motif sites.
    """
    ref = pysam.FastaFile(str(ref_path))
    rng = np.random.default_rng(42)
    read_ids_list = list(read_ids)
    if len(read_ids_list) > max_reads:
        read_ids_list = list(rng.choice(read_ids_list, max_reads, replace=False))
    read_ids_sub = set(read_ids_list)

    per_read = []
    all_motif_probs = []      # all probabilities at MAFIA motif T positions
    all_nonmotif_probs = []   # all probabilities at non-motif T positions
    motif_site_counts = Counter()  # per-motif call counts (≥128)
    motif_total_counts = Counter() # per-motif total eligible T positions

    seen = set()
    n_processed = 0

    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.query_name not in read_ids_sub or read.query_name in seen:
                continue
            seen.add(read.query_name)
            n_processed += 1

            if n_processed % 500 == 0:
                print(f"    {label}: {n_processed:,} reads...")

            seq = read.query_sequence
            if not seq:
                continue
            rlen = len(seq)

            # Get MM/ML
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

            # Get ALL T+17802 probabilities
            t_probs = parse_mm_all_probs(mm_tag, ml_tag, seq)
            if not t_probs:
                continue

            # Build query→ref mapping (dict for O(1) lookup)
            pairs = read.get_aligned_pairs(matches_only=False)
            q2r = {}
            for qpos, rpos in pairs:
                if qpos is not None and rpos is not None:
                    q2r[qpos] = rpos

            chrom = read.reference_name

            # Classify each evaluated T by motif context
            n_motif_t = 0
            n_nonmotif_t = 0
            n_motif_psi = 0      # ≥128 at motif
            n_nonmotif_psi = 0   # ≥128 at non-motif
            read_motif_probs = []
            read_nonmotif_probs = []

            for qpos, prob in t_probs.items():
                rpos = q2r.get(qpos)
                if rpos is None:
                    continue

                # Get 5-mer context from reference
                start = rpos - 2
                end = rpos + 3
                if start < 0:
                    continue
                try:
                    fivemer = ref.fetch(chrom, start, end).upper()
                except (ValueError, KeyError):
                    continue
                if len(fivemer) != 5:
                    continue

                if fivemer in MAFIA_PSI_MOTIFS:
                    n_motif_t += 1
                    read_motif_probs.append(prob)
                    if prob >= 128:
                        n_motif_psi += 1
                        motif_site_counts[fivemer] += 1
                    motif_total_counts[fivemer] += 1
                else:
                    n_nonmotif_t += 1
                    read_nonmotif_probs.append(prob)
                    if prob >= 128:
                        n_nonmotif_psi += 1

            all_motif_probs.extend(read_motif_probs)
            all_nonmotif_probs.extend(read_nonmotif_probs)

            # Per-read metrics
            motif_rate = n_motif_psi / n_motif_t if n_motif_t > 0 else np.nan
            nonmotif_rate = n_nonmotif_psi / n_nonmotif_t if n_nonmotif_t > 0 else np.nan
            motif_mean_prob = np.mean(read_motif_probs) if read_motif_probs else np.nan

            per_read.append({
                'read_id': read.query_name,
                'read_length': rlen,
                'n_t_evaluated': len(t_probs),
                'n_motif_t': n_motif_t,
                'n_nonmotif_t': n_nonmotif_t,
                'n_motif_psi': n_motif_psi,
                'n_nonmotif_psi': n_nonmotif_psi,
                'motif_mod_rate': motif_rate,
                'nonmotif_mod_rate': nonmotif_rate,
                'motif_psi_per_kb': n_motif_psi / rlen * 1000 if rlen > 0 else 0,
                'nonmotif_psi_per_kb': n_nonmotif_psi / rlen * 1000 if rlen > 0 else 0,
                'motif_mean_prob': motif_mean_prob,
                'motif_t_per_kb': n_motif_t / rlen * 1000 if rlen > 0 else 0,
            })

    ref.close()

    df = pd.DataFrame(per_read)
    return df, np.array(all_motif_probs), np.array(all_nonmotif_probs), motif_site_counts, motif_total_counts


def main():
    print("=" * 70)
    print("DORADO PSI — MAFIA MOTIF-RESTRICTED ANALYSIS")
    print("=" * 70)

    # Load read IDs
    l1_df = pd.read_csv(L1_READS, sep='\t')
    ctrl_df = pd.read_csv(CTRL_READS, sep='\t')
    l1_ids = set(l1_df['read_id'])
    ctrl_ids = set(ctrl_df['read_id'])
    print(f"\nL1 stage2 reads: {len(l1_ids):,}, Control reads: {len(ctrl_ids):,}")

    # ─── Analyze ──────────────────────────────────────────────────────────
    print("\n  Analyzing L1 reads...")
    l1_res, l1_motif_probs, l1_nonmotif_probs, l1_mcounts, l1_mtotals = \
        analyze_motif_restricted(BAM, REF, l1_ids, 'L1', max_reads=3000)

    print("\n  Analyzing Control reads...")
    ctrl_res, ctrl_motif_probs, ctrl_nonmotif_probs, ctrl_mcounts, ctrl_mtotals = \
        analyze_motif_restricted(BAM, REF, ctrl_ids, 'Ctrl', max_reads=3000)

    # ─── 1. Motif T density: how many MAFIA motif T positions per kb? ────
    print(f"\n{'='*70}")
    print("1. MAFIA MOTIF T DENSITY (substrate availability)")
    print(f"{'='*70}")

    for label, df in [('L1', l1_res), ('Ctrl', ctrl_res)]:
        valid = df.dropna(subset=['motif_mod_rate'])
        print(f"\n  {label} (n={len(valid):,}):")
        print(f"    Motif T/kb:    median={valid['motif_t_per_kb'].median():.2f}, "
              f"mean={valid['motif_t_per_kb'].mean():.2f}")
        print(f"    Motif T/read:  median={valid['n_motif_t'].median():.0f}, "
              f"mean={valid['n_motif_t'].mean():.1f}")
        print(f"    Total T eval:  median={valid['n_t_evaluated'].median():.0f}")

    u, p = stats.mannwhitneyu(
        l1_res['motif_t_per_kb'].dropna(),
        ctrl_res['motif_t_per_kb'].dropna(),
        alternative='two-sided'
    )
    print(f"\n  L1 vs Ctrl motif T/kb: MWU p={p:.2e}")

    # ─── 2. Modification rate at motif sites ─────────────────────────────
    print(f"\n{'='*70}")
    print("2. MODIFICATION RATE AT MAFIA MOTIF SITES")
    print(f"   (= psi calls ≥128 / eligible motif T positions)")
    print(f"{'='*70}")

    for label, df in [('L1', l1_res), ('Ctrl', ctrl_res)]:
        valid = df.dropna(subset=['motif_mod_rate'])
        print(f"\n  {label} (n={len(valid):,} reads with ≥1 motif T):")
        print(f"    Motif mod rate:    median={valid['motif_mod_rate'].median():.4f}, "
              f"mean={valid['motif_mod_rate'].mean():.4f}")
        print(f"    Non-motif mod rate: median={valid['nonmotif_mod_rate'].dropna().median():.4f}, "
              f"mean={valid['nonmotif_mod_rate'].dropna().mean():.4f}")
        print(f"    Motif psi/kb:      median={valid['motif_psi_per_kb'].median():.3f}, "
              f"mean={valid['motif_psi_per_kb'].mean():.3f}")

    l1_valid = l1_res.dropna(subset=['motif_mod_rate'])
    ctrl_valid = ctrl_res.dropna(subset=['motif_mod_rate'])

    u, p = stats.mannwhitneyu(
        l1_valid['motif_mod_rate'], ctrl_valid['motif_mod_rate'],
        alternative='two-sided'
    )
    ratio_mean = l1_valid['motif_mod_rate'].mean() / ctrl_valid['motif_mod_rate'].mean() \
        if ctrl_valid['motif_mod_rate'].mean() > 0 else float('nan')
    print(f"\n  L1 vs Ctrl motif mod rate:")
    print(f"    Mean ratio = {ratio_mean:.3f}")
    print(f"    MWU p = {p:.2e}")

    # ─── 3. Probability distribution at motif vs non-motif sites ─────────
    print(f"\n{'='*70}")
    print("3. PROBABILITY DISTRIBUTION: MOTIF vs NON-MOTIF T POSITIONS")
    print(f"{'='*70}")

    for label, mprobs, nmprobs in [
        ('L1', l1_motif_probs, l1_nonmotif_probs),
        ('Ctrl', ctrl_motif_probs, ctrl_nonmotif_probs),
    ]:
        print(f"\n  {label}:")
        print(f"    Motif T sites:     n={len(mprobs):,}")
        print(f"      mean prob={mprobs.mean():.1f}, median={np.median(mprobs):.0f}")
        print(f"      ≥128: {(mprobs >= 128).sum():,} ({(mprobs >= 128).mean()*100:.2f}%)")
        print(f"      ≥64:  {(mprobs >= 64).sum():,} ({(mprobs >= 64).mean()*100:.2f}%)")
        print(f"      ≥32:  {(mprobs >= 32).sum():,} ({(mprobs >= 32).mean()*100:.2f}%)")
        print(f"    Non-motif T sites: n={len(nmprobs):,}")
        print(f"      mean prob={nmprobs.mean():.1f}, median={np.median(nmprobs):.0f}")
        print(f"      ≥128: {(nmprobs >= 128).sum():,} ({(nmprobs >= 128).mean()*100:.2f}%)")
        print(f"      ≥64:  {(nmprobs >= 64).sum():,} ({(nmprobs >= 64).mean()*100:.2f}%)")
        print(f"      ≥32:  {(nmprobs >= 32).sum():,} ({(nmprobs >= 32).mean()*100:.2f}%)")

    # Cross-comparison: L1 motif vs Ctrl motif
    if len(l1_motif_probs) > 0 and len(ctrl_motif_probs) > 0:
        u, p = stats.mannwhitneyu(l1_motif_probs, ctrl_motif_probs, alternative='two-sided')
        print(f"\n  L1 vs Ctrl at motif sites:")
        print(f"    Mean prob: L1={l1_motif_probs.mean():.2f}, Ctrl={ctrl_motif_probs.mean():.2f}")
        print(f"    MWU p={p:.2e}")

    # ─── 4. Per-motif breakdown ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("4. PER-MOTIF MODIFICATION RATE (site counts / eligible T)")
    print(f"{'='*70}")

    print(f"\n  {'Motif':<8} {'L1_calls':>9} {'L1_total':>9} {'L1_rate':>8} "
          f"{'Ctrl_calls':>10} {'Ctrl_total':>10} {'Ctrl_rate':>9} {'Ratio':>7}")
    print(f"  {'-'*75}")

    all_motifs = sorted(MAFIA_PSI_MOTIFS)
    for motif in all_motifs:
        l1_c = l1_mcounts.get(motif, 0)
        l1_t = l1_mtotals.get(motif, 0)
        ctrl_c = ctrl_mcounts.get(motif, 0)
        ctrl_t = ctrl_mtotals.get(motif, 0)
        l1_r = l1_c / l1_t if l1_t > 0 else 0
        ctrl_r = ctrl_c / ctrl_t if ctrl_t > 0 else 0
        ratio = l1_r / ctrl_r if ctrl_r > 0 else float('nan')
        print(f"  {motif:<8} {l1_c:>9} {l1_t:>9} {l1_r:>8.4f} "
              f"{ctrl_c:>10} {ctrl_t:>10} {ctrl_r:>9.4f} {ratio:>7.3f}")

    # Totals
    l1_total_c = sum(l1_mcounts.values())
    l1_total_t = sum(l1_mtotals.values())
    ctrl_total_c = sum(ctrl_mcounts.values())
    ctrl_total_t = sum(ctrl_mtotals.values())
    l1_total_r = l1_total_c / l1_total_t if l1_total_t > 0 else 0
    ctrl_total_r = ctrl_total_c / ctrl_total_t if ctrl_total_t > 0 else 0
    total_ratio = l1_total_r / ctrl_total_r if ctrl_total_r > 0 else float('nan')
    print(f"  {'TOTAL':<8} {l1_total_c:>9} {l1_total_t:>9} {l1_total_r:>8.4f} "
          f"{ctrl_total_c:>10} {ctrl_total_t:>10} {ctrl_total_r:>9.4f} {total_ratio:>7.3f}")

    # ─── 5. Threshold sweep ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("5. THRESHOLD SWEEP: motif-restricted psi/kb at different cutoffs")
    print(f"{'='*70}")

    for thresh in [32, 64, 96, 128, 192]:
        # Recompute per-read with this threshold
        l1_vals = []
        for _, row in l1_res.iterrows():
            # We only have ≥128 counts in the dataframe, need raw probs
            pass

        # Use site-level data instead
        l1_frac = (l1_motif_probs >= thresh).mean() if len(l1_motif_probs) > 0 else 0
        ctrl_frac = (ctrl_motif_probs >= thresh).mean() if len(ctrl_motif_probs) > 0 else 0
        ratio = l1_frac / ctrl_frac if ctrl_frac > 0 else float('nan')
        print(f"  Threshold ≥{thresh:>3}: L1 motif call rate={l1_frac:.4f}, "
              f"Ctrl={ctrl_frac:.4f}, ratio={ratio:.3f}")

    # ─── 6. Compare with MAFIA ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("6. COMPARISON WITH MAFIA")
    print(f"{'='*70}")

    print(f"""
  Method         Context         L1 psi/kb   Ctrl psi/kb   Ratio
  ─────────────────────────────────────────────────────────────────
  MAFIA r9.4.1   16 motifs only  5.17        3.60          1.44x  ***
  Dorado RNA004  All T           2.33        2.86          0.82x  (opposite)
  Dorado RNA004  16 motifs only  {l1_valid['motif_psi_per_kb'].mean():.2f}        {ctrl_valid['motif_psi_per_kb'].mean():.2f}          {l1_valid['motif_psi_per_kb'].mean()/ctrl_valid['motif_psi_per_kb'].mean():.2f}x
  Dorado RNA004  Motif mod rate  {l1_valid['motif_mod_rate'].mean():.4f}      {ctrl_valid['motif_mod_rate'].mean():.4f}        {ratio_mean:.3f}x
    """)

    # ─── Save ─────────────────────────────────────────────────────────────
    outdir = ANALYSIS_DIR / 'HeLa_1_1'
    l1_res.to_csv(outdir / 'motif_restricted_l1.tsv', sep='\t', index=False)
    ctrl_res.to_csv(outdir / 'motif_restricted_ctrl.tsv', sep='\t', index=False)

    # Save probability arrays
    np.savez(outdir / 'motif_probs.npz',
             l1_motif=l1_motif_probs, l1_nonmotif=l1_nonmotif_probs,
             ctrl_motif=ctrl_motif_probs, ctrl_nonmotif=ctrl_nonmotif_probs)

    print(f"Results saved to {outdir}")


if __name__ == '__main__':
    main()
