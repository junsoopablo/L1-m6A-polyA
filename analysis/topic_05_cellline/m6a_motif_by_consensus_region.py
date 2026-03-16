#!/usr/bin/env python3
"""
m6a_motif_by_consensus_region.py

Analyze DRACH motif density and composition across L1 consensus regions
(5'UTR, ORF1, inter-ORF, ORF2, 3'UTR) to explain why ORF1 region has
higher m6A modification.

Approach:
1. Fetch L1HS consensus (L1.2, M80343) from GenBank with annotated ORF boundaries
2. Count and map all 18 DRACH motifs across the consensus
3. Calculate per-region DRACH density (motifs/kb) and A/T content
4. Identify which high-rate motifs (TAACT, GGACT, etc.) are enriched in ORF1
5. Repeat for ancient L1 subfamilies using representative full-length elements

L1 region boundaries (from M80343 annotation, 1-based):
  5'UTR:     1-910
  ORF1:      911-1927
  Inter-ORF: 1928-1990
  ORF2:      1991-5818
  3'UTR:     5819-6050

DRACH = [AGT][AG]AC[ACT] = 18 possible pentamer motifs
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
PROJECT = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
CUSTOM_REF = "/qbio/junsoopablo/02_Projects/05_IsoTENT/reference/custom_LINE_reference.fasta"
OUT_DIR = os.path.join(PROJECT, "analysis/01_exploration/topic_05_cellline/m6a_motif_by_consensus_region")
os.makedirs(OUT_DIR, exist_ok=True)

# L1 region boundaries (1-based, from M80343 GenBank annotation)
# These are standard L1HS/L1PA boundaries
REGIONS_1BASED = {
    "5'UTR":     (1, 910),
    "ORF1":      (911, 1927),
    "Inter-ORF": (1928, 1990),
    "ORF2":      (1991, 5818),
    "3'UTR":     (5819, None),  # None = to end of sequence
}

# DRACH motifs: D=[AGT], R=[AG], A, C, H=[ACT]
D_BASES = ['A', 'G', 'T']
R_BASES = ['A', 'G']
H_BASES = ['A', 'C', 'T']

ALL_DRACH = sorted([f"{d}{r}AC{h}" for d in D_BASES for r in R_BASES for h in H_BASES])
assert len(ALL_DRACH) == 18, f"Expected 18 DRACH motifs, got {len(ALL_DRACH)}"

# Top motifs by per-site m6A rate (from previous motif_enrichment analysis)
TOP_MOTIFS_BY_RATE = {
    'TAACT': 0.469,  # 46.9%
    'GGACT': 0.435,  # 43.5%
    'TAACA': 0.371,  # 37.1%
    'GAACT': 0.342,  # 34.2%
    'GGACA': 0.326,  # 32.6%
}

# Young L1 subfamilies
YOUNG_SUBFAMILIES = {'L1HS', 'L1PA1', 'L1PA2', 'L1PA3'}

# Ancient L1 subfamilies to analyze
ANCIENT_SUBFAMILIES = {
    'L1PA': ['L1PA4', 'L1PA5', 'L1PA6', 'L1PA7', 'L1PA8'],
    'L1M':  ['L1M1', 'L1M2', 'L1M3', 'L1MA1', 'L1MA2', 'L1MA3', 'L1MA4'],
    'L1MC': ['L1MC1', 'L1MC2', 'L1MC3', 'L1MC4', 'L1MC5'],
    'L1ME': ['L1ME1', 'L1ME2', 'L1ME3', 'L1ME4', 'L1ME4a', 'L1ME5'],
}


def revcomp(seq):
    """Reverse complement"""
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def find_drach_positions(seq, strand='+'):
    """
    Find all DRACH motif positions in a sequence.
    Returns list of (position, motif) tuples.
    Position is 0-based, refers to the 'A' in xRACx (the central A).
    """
    seq = seq.upper()
    results = []

    for motif in ALL_DRACH:
        # Forward strand
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            # Position of the central 'A' (3rd base, index 2 within motif)
            a_pos = idx + 2
            results.append((a_pos, motif, '+'))
            start = idx + 1

        # Reverse complement: find revcomp motif in forward sequence
        rc_motif = revcomp(motif)
        start = 0
        while True:
            idx = seq.find(rc_motif, start)
            if idx == -1:
                break
            # Central A on reverse strand maps to position idx+2 on forward
            a_pos = idx + 2
            results.append((a_pos, motif, '-'))
            start = idx + 1

    return sorted(results, key=lambda x: x[0])


def count_motifs_in_region(positions, start, end):
    """Count motifs in a region [start, end) (0-based)."""
    motif_counts = Counter()
    for pos, motif, strand in positions:
        if start <= pos < end:
            motif_counts[motif] += 1
    return motif_counts


def compute_base_composition(seq):
    """Compute base composition of a sequence."""
    seq = seq.upper()
    total = len(seq)
    if total == 0:
        return {}
    counts = Counter(seq)
    return {
        'A': counts.get('A', 0),
        'T': counts.get('T', 0),
        'G': counts.get('G', 0),
        'C': counts.get('C', 0),
        'AT_frac': (counts.get('A', 0) + counts.get('T', 0)) / total,
        'GC_frac': (counts.get('G', 0) + counts.get('C', 0)) / total,
        'length': total,
    }


def analyze_sequence(seq, seq_name, region_boundaries=None):
    """
    Analyze DRACH motifs across a sequence with defined regions.

    Args:
        seq: DNA sequence string
        seq_name: Name of the sequence
        region_boundaries: dict of region_name -> (start_1based, end_1based)

    Returns:
        DataFrame with per-region statistics
    """
    seq = seq.upper()
    seq_len = len(seq)

    if region_boundaries is None:
        region_boundaries = REGIONS_1BASED

    # Find all DRACH positions (both strands)
    all_positions = find_drach_positions(seq)

    # Also find forward-only positions (DRS reads are sense-strand)
    fwd_positions = [(p, m, s) for p, m, s in all_positions if s == '+']

    results = []

    for region_name, (start_1b, end_1b) in region_boundaries.items():
        # Convert to 0-based
        start_0b = start_1b - 1
        end_0b = (end_1b if end_1b is not None else seq_len)

        # Ensure bounds
        start_0b = max(0, start_0b)
        end_0b = min(seq_len, end_0b)

        region_seq = seq[start_0b:end_0b]
        region_len = len(region_seq)

        if region_len == 0:
            continue

        # Base composition
        comp = compute_base_composition(region_seq)

        # Count motifs (forward strand only for DRS context)
        fwd_counts = count_motifs_in_region(fwd_positions, start_0b, end_0b)
        total_fwd = sum(fwd_counts.values())

        # Count motifs (both strands)
        both_counts = count_motifs_in_region(all_positions, start_0b, end_0b)
        total_both = sum(both_counts.values())

        row = {
            'sequence': seq_name,
            'region': region_name,
            'start_1based': start_1b,
            'end_1based': end_1b if end_1b else seq_len,
            'length_bp': region_len,
            'AT_frac': comp['AT_frac'],
            'GC_frac': comp['GC_frac'],
            'total_drach_fwd': total_fwd,
            'drach_per_kb_fwd': total_fwd / (region_len / 1000),
            'total_drach_both': total_both,
            'drach_per_kb_both': total_both / (region_len / 1000),
        }

        # Per-motif counts (forward strand)
        for motif in ALL_DRACH:
            row[f'fwd_{motif}'] = fwd_counts.get(motif, 0)
            row[f'fwd_{motif}_per_kb'] = fwd_counts.get(motif, 0) / (region_len / 1000)

        # Top motifs count
        top_count = sum(fwd_counts.get(m, 0) for m in TOP_MOTIFS_BY_RATE)
        row['top_5_motif_count'] = top_count
        row['top_5_motif_frac'] = top_count / total_fwd if total_fwd > 0 else 0

        results.append(row)

    return pd.DataFrame(results)


def sliding_window_drach(seq, window_size=100, step=10):
    """
    Compute DRACH density in sliding windows along sequence.
    Returns array of (center_position, density_per_kb, at_frac).
    """
    seq = seq.upper()
    seq_len = len(seq)
    fwd_positions = find_drach_positions(seq)
    fwd_positions = [(p, m, s) for p, m, s in fwd_positions if s == '+']

    results = []
    for start in range(0, seq_len - window_size + 1, step):
        end = start + window_size
        center = (start + end) / 2

        # Count motifs in window
        count = sum(1 for p, m, s in fwd_positions if start <= p < end)
        density = count / (window_size / 1000)

        # AT fraction
        window_seq = seq[start:end]
        at_frac = (window_seq.count('A') + window_seq.count('T')) / len(window_seq)

        results.append((center, density, at_frac))

    return results


def fetch_genbank_sequence(accession):
    """Fetch sequence from GenBank."""
    from Bio import Entrez, SeqIO
    Entrez.email = 'analysis@example.com'
    handle = Entrez.efetch(db='nucleotide', id=accession, rettype='fasta', retmode='text')
    record = SeqIO.read(handle, 'fasta')
    handle.close()
    return str(record.seq), record.description


def load_custom_ref_sequences(fasta_path, subfamily_prefix_list, min_length=5500, max_count=50):
    """
    Load sequences from custom reference for specific subfamilies.
    Only keep near-full-length sequences.
    """
    from Bio import SeqIO

    seqs = defaultdict(list)
    for rec in SeqIO.parse(fasta_path, 'fasta'):
        name = rec.id
        subfamily = name.split('_')[0]
        if subfamily in subfamily_prefix_list and len(rec.seq) >= min_length:
            seqs[subfamily].append(str(rec.seq))

    # Limit to max_count per subfamily (take longest)
    for sf in seqs:
        seqs[sf] = sorted(seqs[sf], key=len, reverse=True)[:max_count]

    return seqs


def build_consensus_profile(sequences, seq_len_target=6050):
    """
    Build a DRACH density profile from multiple sequences.
    Since sequences may vary in length, normalize positions to [0, 1].
    Returns per-region statistics as if we had a single consensus.
    """
    # For each sequence, compute per-region DRACH stats
    # Use fractional positions: 5'UTR ~0-0.15, ORF1 ~0.15-0.32, ORF2 ~0.33-0.96, 3'UTR ~0.96-1.0

    all_results = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)

        # Scale region boundaries to this sequence's length
        scale = seq_len / seq_len_target

        scaled_regions = {}
        for rname, (s, e) in REGIONS_1BASED.items():
            s_scaled = max(1, int(s * scale))
            e_scaled = int(e * scale) if e is not None else seq_len
            e_scaled = min(seq_len, e_scaled)
            scaled_regions[rname] = (s_scaled, e_scaled)

        df = analyze_sequence(seq, f"seq_{i}", scaled_regions)
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Average per region
    summary = combined.groupby('region').agg({
        'length_bp': 'mean',
        'AT_frac': 'mean',
        'GC_frac': 'mean',
        'total_drach_fwd': 'mean',
        'drach_per_kb_fwd': 'mean',
        'total_drach_both': 'mean',
        'drach_per_kb_both': 'mean',
        'top_5_motif_count': 'mean',
        'top_5_motif_frac': 'mean',
        **{f'fwd_{m}': 'mean' for m in ALL_DRACH},
        **{f'fwd_{m}_per_kb': 'mean' for m in ALL_DRACH},
    }).reset_index()

    summary['n_sequences'] = combined.groupby('region').size().values

    return summary


def main():
    print("=" * 80)
    print("DRACH Motif Density by L1 Consensus Region")
    print("=" * 80)

    # =========================================================================
    # 1. Fetch L1HS consensus (L1.2, M80343 with annotated ORFs)
    # =========================================================================
    print("\n[1] Fetching L1HS consensus (M80343 / L1.2)...")
    try:
        l1hs_seq, l1hs_desc = fetch_genbank_sequence('M80343')
        print(f"  Fetched: {l1hs_desc[:80]}")
        print(f"  Length: {len(l1hs_seq)} bp")
    except Exception as e:
        print(f"  GenBank fetch failed: {e}")
        print("  Using longest L1HS from custom reference instead...")
        from Bio import SeqIO
        l1hs_records = []
        for rec in SeqIO.parse(CUSTOM_REF, 'fasta'):
            if rec.id.startswith('L1HS_'):
                l1hs_records.append((len(rec.seq), str(rec.seq), rec.id))
        l1hs_records.sort(reverse=True)
        l1hs_seq = l1hs_records[0][1]
        l1hs_desc = l1hs_records[0][2]
        print(f"  Using: {l1hs_desc}, Length: {len(l1hs_seq)} bp")

    # Update 3'UTR end to actual sequence length
    regions = dict(REGIONS_1BASED)
    regions["3'UTR"] = (5819, len(l1hs_seq))

    # =========================================================================
    # 2. Analyze L1HS consensus
    # =========================================================================
    print("\n[2] Analyzing DRACH motifs across L1HS consensus regions...")

    df_l1hs = analyze_sequence(l1hs_seq, "L1HS_consensus", regions)

    print("\n  --- L1HS Consensus DRACH Summary (forward strand) ---")
    print(f"  {'Region':<12} {'Length':>6} {'AT%':>6} {'DRACH':>6} {'DRACH/kb':>9} {'Top5':>5} {'Top5%':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*5} {'-'*6}")
    for _, row in df_l1hs.iterrows():
        print(f"  {row['region']:<12} {row['length_bp']:>6} {row['AT_frac']:>6.1%} "
              f"{row['total_drach_fwd']:>6.0f} {row['drach_per_kb_fwd']:>9.2f} "
              f"{row['top_5_motif_count']:>5.0f} {row['top_5_motif_frac']:>6.1%}")

    # Full sequence stats
    full_positions = find_drach_positions(l1hs_seq)
    fwd_full = [(p, m, s) for p, m, s in full_positions if s == '+']
    print(f"\n  Full L1HS: {len(fwd_full)} DRACH sites (fwd), "
          f"{len(fwd_full)/(len(l1hs_seq)/1000):.2f}/kb")

    # =========================================================================
    # 3. Per-motif breakdown by region
    # =========================================================================
    print("\n[3] Per-motif breakdown by region (forward strand)...")

    motif_data = []
    for _, row in df_l1hs.iterrows():
        for motif in ALL_DRACH:
            rate_info = TOP_MOTIFS_BY_RATE.get(motif, None)
            motif_data.append({
                'region': row['region'],
                'motif': motif,
                'count': row[f'fwd_{motif}'],
                'per_kb': row[f'fwd_{motif}_per_kb'],
                'is_top5': motif in TOP_MOTIFS_BY_RATE,
                'm6a_rate': rate_info if rate_info else 0,
            })

    df_motif = pd.DataFrame(motif_data)

    # Print per-region motif composition
    print("\n  Top 5 high-rate DRACH motifs by region:")
    print(f"  {'Motif':<8} {'m6A%':>5} ", end='')
    for rname in ["5'UTR", "ORF1", "Inter-ORF", "ORF2", "3'UTR"]:
        print(f"{'|'+rname:>12}", end='')
    print()
    print(f"  {'-'*8} {'-'*5} " + " ".join([f"{'|'+'-'*10:>12}"] * 5))

    for motif in sorted(TOP_MOTIFS_BY_RATE.keys(), key=lambda m: TOP_MOTIFS_BY_RATE[m], reverse=True):
        rate = TOP_MOTIFS_BY_RATE[motif]
        print(f"  {motif:<8} {rate:>5.1%} ", end='')
        for rname in ["5'UTR", "ORF1", "Inter-ORF", "ORF2", "3'UTR"]:
            sub = df_motif[(df_motif['region'] == rname) & (df_motif['motif'] == motif)]
            if len(sub) > 0:
                cnt = sub.iloc[0]['count']
                pkb = sub.iloc[0]['per_kb']
                print(f"|{cnt:>4.0f} ({pkb:>4.1f})", end='')
            else:
                print(f"|{'N/A':>10}", end='')
        print()

    # All 18 motifs
    print("\n  All 18 DRACH motifs - density per kb by region:")
    print(f"  {'Motif':<8}", end='')
    for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
        print(f"  {rname:>10}", end='')
    print(f"  {'ORF1/ORF2':>10}")
    print(f"  {'-'*8}" + "  ".join([f"{'-'*10}"] * 5))

    for motif in ALL_DRACH:
        print(f"  {motif:<8}", end='')
        orf1_val = 0
        orf2_val = 0
        for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
            sub = df_motif[(df_motif['region'] == rname) & (df_motif['motif'] == motif)]
            if len(sub) > 0:
                pkb = sub.iloc[0]['per_kb']
                print(f"  {pkb:>10.2f}", end='')
                if rname == 'ORF1':
                    orf1_val = pkb
                elif rname == 'ORF2':
                    orf2_val = pkb
            else:
                print(f"  {'N/A':>10}", end='')
        # ORF1/ORF2 ratio
        ratio = orf1_val / orf2_val if orf2_val > 0 else float('inf')
        print(f"  {ratio:>10.2f}")

    # =========================================================================
    # 4. Weighted DRACH density (by per-site m6A rate)
    # =========================================================================
    print("\n[4] Rate-weighted DRACH analysis...")
    print("  (Weight each motif by its per-site m6A modification rate)")

    # Per-site rates from motif enrichment analysis (L1 rates)
    # Using the rates from motif_enrichment_l1_vs_control.py
    L1_MOTIF_RATES = {
        'AAACA': 0.193, 'AAACC': 0.112, 'AAACT': 0.282,
        'AGACA': 0.219, 'AGACC': 0.162, 'AGACT': 0.334,
        'GAACA': 0.268, 'GAACC': 0.179, 'GAACT': 0.342,
        'GGACA': 0.326, 'GGACC': 0.240, 'GGACT': 0.435,
        'TAACA': 0.371, 'TAACC': 0.163, 'TAACT': 0.469,
        'TGACA': 0.088, 'TGACC': 0.065, 'TGACT': 0.141,
    }

    print(f"\n  {'Region':<12} {'DRACH/kb':>9} {'Weighted/kb':>12} {'Predicted m6A/kb':>16} {'Avg rate':>9}")
    print(f"  {'-'*12} {'-'*9} {'-'*12} {'-'*16} {'-'*9}")

    for _, row in df_l1hs.iterrows():
        region = row['region']
        region_len_kb = row['length_bp'] / 1000

        # Unweighted density
        total_drach = row['drach_per_kb_fwd']

        # Rate-weighted: sum(count_i * rate_i) / length_kb
        weighted_sum = 0
        total_count = 0
        for motif in ALL_DRACH:
            cnt = row[f'fwd_{motif}']
            rate = L1_MOTIF_RATES.get(motif, 0.2)
            weighted_sum += cnt * rate
            total_count += cnt

        weighted_density = weighted_sum / region_len_kb if region_len_kb > 0 else 0
        predicted_m6a = weighted_density  # weighted density IS predicted m6A/kb
        avg_rate = weighted_sum / total_count if total_count > 0 else 0

        print(f"  {region:<12} {total_drach:>9.2f} {weighted_density:>12.2f} "
              f"{predicted_m6a:>16.2f} {avg_rate:>9.1%}")

    # =========================================================================
    # 5. Sliding window analysis
    # =========================================================================
    print("\n[5] Sliding window DRACH density (100bp window, 10bp step)...")

    sw_results = sliding_window_drach(l1hs_seq, window_size=100, step=10)

    # Save sliding window data
    sw_df = pd.DataFrame(sw_results, columns=['position', 'drach_per_kb', 'at_frac'])
    sw_df.to_csv(os.path.join(OUT_DIR, 'l1hs_sliding_window_drach.tsv'), sep='\t', index=False)
    print(f"  Saved sliding window data: {len(sw_df)} windows")

    # Regional summary from sliding windows
    for rname, (s, e) in regions.items():
        e_actual = e if e else len(l1hs_seq)
        mask = (sw_df['position'] >= s) & (sw_df['position'] <= e_actual)
        sub = sw_df[mask]
        if len(sub) > 0:
            print(f"  {rname:<12}: mean={sub['drach_per_kb'].mean():.1f}/kb, "
                  f"max={sub['drach_per_kb'].max():.1f}/kb, "
                  f"AT={sub['at_frac'].mean():.1%}")

    # =========================================================================
    # 6. A/T content analysis
    # =========================================================================
    print("\n[6] Base composition by region...")
    print(f"  {'Region':<12} {'Length':>6} {'A%':>5} {'T%':>5} {'G%':>5} {'C%':>5} {'AT%':>6} {'GC%':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")

    for rname, (s, e) in regions.items():
        e_actual = e if e else len(l1hs_seq)
        region_seq = l1hs_seq[s-1:e_actual].upper()
        comp = compute_base_composition(region_seq)
        rlen = len(region_seq)
        print(f"  {rname:<12} {rlen:>6} "
              f"{comp['A']/rlen:>5.1%} {comp['T']/rlen:>5.1%} "
              f"{comp['G']/rlen:>5.1%} {comp['C']/rlen:>5.1%} "
              f"{comp['AT_frac']:>6.1%} {comp['GC_frac']:>6.1%}")

    # =========================================================================
    # 7. Expected vs observed DRACH by region
    # =========================================================================
    print("\n[7] Expected vs Observed DRACH density...")
    print("  (Expected based on base composition, assuming independence)")

    print(f"\n  {'Region':<12} {'Obs/kb':>7} {'Exp/kb':>7} {'O/E':>6} {'AT%':>6}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")

    for _, row in df_l1hs.iterrows():
        region = row['region']
        s, e = regions[region]
        e_actual = e if e else len(l1hs_seq)
        region_seq = l1hs_seq[s-1:e_actual].upper()
        rlen = len(region_seq)

        comp = Counter(region_seq)
        total = rlen

        # Expected DRACH frequency under independence
        # DRACH = [AGT][AG]AC[ACT]
        # P(DRACH) = P(D)*P(R)*P(A)*P(C)*P(H) at any position
        pA = comp.get('A', 0) / total
        pT = comp.get('T', 0) / total
        pG = comp.get('G', 0) / total
        pC = comp.get('C', 0) / total

        pD = pA + pG + pT  # D = [AGT]
        pR = pA + pG       # R = [AG]
        pH = pA + pC + pT  # H = [ACT]

        expected_per_pos = pD * pR * pA * pC * pH
        expected_per_kb = expected_per_pos * 1000

        obs_per_kb = row['drach_per_kb_fwd']
        oe_ratio = obs_per_kb / expected_per_kb if expected_per_kb > 0 else float('inf')

        print(f"  {region:<12} {obs_per_kb:>7.2f} {expected_per_kb:>7.2f} {oe_ratio:>6.2f} {row['AT_frac']:>6.1%}")

    # =========================================================================
    # 8. Ancient L1 analysis
    # =========================================================================
    print("\n[8] Ancient L1 subfamily analysis...")
    print("  Loading full-length sequences from custom reference...")

    from Bio import SeqIO

    # Load all subfamily sequences
    all_subfamilies = list(YOUNG_SUBFAMILIES)
    for group_sfs in ANCIENT_SUBFAMILIES.values():
        all_subfamilies.extend(group_sfs)

    sf_seqs = load_custom_ref_sequences(CUSTOM_REF, all_subfamilies, min_length=5500, max_count=30)

    print(f"\n  Full-length sequences loaded (>5500bp, max 30 per subfamily):")
    for sf in sorted(sf_seqs.keys()):
        if sf_seqs[sf]:
            lengths = [len(s) for s in sf_seqs[sf]]
            print(f"    {sf:<10}: {len(sf_seqs[sf]):>3} seqs, "
                  f"median length {np.median(lengths):.0f}bp")

    # Analyze each subfamily group
    subfamily_results = []

    # Young L1 (L1HS, L1PA1-3)
    for sf_name in sorted(YOUNG_SUBFAMILIES):
        if sf_name in sf_seqs and sf_seqs[sf_name]:
            summary = build_consensus_profile(sf_seqs[sf_name], seq_len_target=6050)
            if len(summary) > 0:
                summary['subfamily'] = sf_name
                summary['age_group'] = 'young'
                subfamily_results.append(summary)

    # Ancient L1 groups
    for group_name, sf_list in ANCIENT_SUBFAMILIES.items():
        # Pool sequences from all subfamilies in this group
        group_seqs = []
        for sf in sf_list:
            if sf in sf_seqs:
                group_seqs.extend(sf_seqs[sf])

        if group_seqs:
            # Also analyze individual subfamilies with enough sequences
            for sf in sf_list:
                if sf in sf_seqs and len(sf_seqs[sf]) >= 5:
                    summary = build_consensus_profile(sf_seqs[sf], seq_len_target=6050)
                    if len(summary) > 0:
                        summary['subfamily'] = sf
                        summary['age_group'] = group_name
                        subfamily_results.append(summary)

    if subfamily_results:
        df_all_sf = pd.concat(subfamily_results, ignore_index=True)

        # Print summary per subfamily
        print("\n  DRACH/kb by region for each subfamily (fwd strand):")
        print(f"  {'Subfamily':<10} {'Age':>6}", end='')
        for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
            print(f"  {rname:>10}", end='')
        print(f"  {'ORF1/ORF2':>10} {'nSeqs':>6}")
        print(f"  {'-'*10} {'-'*6}" + "  ".join([f"{'-'*10}"] * 5) + f"  {'-'*6}")

        for sf in df_all_sf['subfamily'].unique():
            sub = df_all_sf[df_all_sf['subfamily'] == sf]
            age = sub['age_group'].iloc[0]
            nseqs = sub['n_sequences'].iloc[0] if 'n_sequences' in sub.columns else 0

            print(f"  {sf:<10} {age:>6}", end='')
            orf1_d = 0
            orf2_d = 0
            for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
                r = sub[sub['region'] == rname]
                if len(r) > 0:
                    d = r.iloc[0]['drach_per_kb_fwd']
                    print(f"  {d:>10.2f}", end='')
                    if rname == 'ORF1':
                        orf1_d = d
                    elif rname == 'ORF2':
                        orf2_d = d
                else:
                    print(f"  {'N/A':>10}", end='')

            ratio = orf1_d / orf2_d if orf2_d > 0 else float('inf')
            print(f"  {ratio:>10.2f} {nseqs:>6.0f}")

        # AT% by region for each subfamily
        print("\n  AT% by region for each subfamily:")
        print(f"  {'Subfamily':<10} {'Age':>6}", end='')
        for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
            print(f"  {rname:>10}", end='')
        print()
        print(f"  {'-'*10} {'-'*6}" + "  ".join([f"{'-'*10}"] * 4))

        for sf in df_all_sf['subfamily'].unique():
            sub = df_all_sf[df_all_sf['subfamily'] == sf]
            age = sub['age_group'].iloc[0]

            print(f"  {sf:<10} {age:>6}", end='')
            for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]:
                r = sub[sub['region'] == rname]
                if len(r) > 0:
                    at = r.iloc[0]['AT_frac']
                    print(f"  {at:>10.1%}", end='')
                else:
                    print(f"  {'N/A':>10}", end='')
            print()

    # =========================================================================
    # 9. Rate-weighted comparison: ORF1 vs ORF2 predicted m6A
    # =========================================================================
    print("\n[9] Predicted m6A/kb by region (rate-weighted) for L1HS consensus...")

    print(f"\n  {'Region':<12} {'DRACH/kb':>9} {'Predicted m6A/kb':>16} {'Effective rate':>14}")
    print(f"  {'-'*12} {'-'*9} {'-'*16} {'-'*14}")

    for _, row in df_l1hs.iterrows():
        region = row['region']
        region_len_kb = row['length_bp'] / 1000

        weighted_sum = 0
        total_count = 0
        for motif in ALL_DRACH:
            cnt = row[f'fwd_{motif}']
            rate = L1_MOTIF_RATES.get(motif, 0.2)
            weighted_sum += cnt * rate
            total_count += cnt

        predicted = weighted_sum / region_len_kb if region_len_kb > 0 else 0
        eff_rate = weighted_sum / total_count if total_count > 0 else 0

        print(f"  {region:<12} {row['drach_per_kb_fwd']:>9.2f} {predicted:>16.2f} {eff_rate:>14.1%}")

    # =========================================================================
    # 10. Motif composition shift: ORF1 vs ORF2
    # =========================================================================
    print("\n[10] Motif composition shift: ORF1 vs ORF2...")

    orf1_row = df_l1hs[df_l1hs['region'] == 'ORF1'].iloc[0]
    orf2_row = df_l1hs[df_l1hs['region'] == 'ORF2'].iloc[0]

    orf1_total = orf1_row['total_drach_fwd']
    orf2_total = orf2_row['total_drach_fwd']

    print(f"\n  {'Motif':<8} {'Rate':>6} {'ORF1 cnt':>9} {'ORF1 frac':>10} "
          f"{'ORF2 cnt':>9} {'ORF2 frac':>10} {'Frac shift':>11}")
    print(f"  {'-'*8} {'-'*6} {'-'*9} {'-'*10} {'-'*9} {'-'*10} {'-'*11}")

    composition_shifts = []
    for motif in ALL_DRACH:
        rate = L1_MOTIF_RATES.get(motif, 0)
        orf1_cnt = orf1_row[f'fwd_{motif}']
        orf2_cnt = orf2_row[f'fwd_{motif}']
        orf1_frac = orf1_cnt / orf1_total if orf1_total > 0 else 0
        orf2_frac = orf2_cnt / orf2_total if orf2_total > 0 else 0
        shift = orf1_frac - orf2_frac

        composition_shifts.append({
            'motif': motif, 'rate': rate,
            'orf1_cnt': orf1_cnt, 'orf1_frac': orf1_frac,
            'orf2_cnt': orf2_cnt, 'orf2_frac': orf2_frac,
            'shift': shift,
        })

        print(f"  {motif:<8} {rate:>6.1%} {orf1_cnt:>9.0f} {orf1_frac:>10.1%} "
              f"{orf2_cnt:>9.0f} {orf2_frac:>10.1%} {shift:>+11.1%}")

    shifts_df = pd.DataFrame(composition_shifts)

    # Correlation between motif rate and ORF1 enrichment
    from scipy import stats
    r, p = stats.pearsonr(shifts_df['rate'], shifts_df['shift'])
    print(f"\n  Correlation between motif m6A rate and ORF1-ORF2 frac shift: r={r:.3f}, p={p:.3g}")

    # =========================================================================
    # 11. Save all results
    # =========================================================================
    print("\n[11] Saving results...")

    # Main region summary
    df_l1hs.to_csv(os.path.join(OUT_DIR, 'l1hs_consensus_region_summary.tsv'),
                    sep='\t', index=False)

    # Per-motif by region
    df_motif.to_csv(os.path.join(OUT_DIR, 'l1hs_motif_by_region.tsv'),
                    sep='\t', index=False)

    # Composition shift
    shifts_df.to_csv(os.path.join(OUT_DIR, 'orf1_vs_orf2_motif_shift.tsv'),
                     sep='\t', index=False)

    # Subfamily results
    if subfamily_results:
        df_all_sf.to_csv(os.path.join(OUT_DIR, 'subfamily_region_summary.tsv'),
                        sep='\t', index=False)

    # =========================================================================
    # 12. Generate figures
    # =========================================================================
    print("\n[12] Generating figures...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 2, 2]})

    # --- Panel A: Sliding window DRACH density ---
    ax = axes[0]
    ax.plot(sw_df['position'], sw_df['drach_per_kb'], color='#2166ac', linewidth=0.8, alpha=0.7)

    # Smoothed line
    window = 20
    smoothed = sw_df['drach_per_kb'].rolling(window, center=True).mean()
    ax.plot(sw_df['position'], smoothed, color='#d73027', linewidth=2, label='Smoothed (200bp)')

    # Region boundaries
    region_colors = {
        "5'UTR": '#fee090',
        "ORF1": '#fc8d59',
        "Inter-ORF": '#eeeeee',
        "ORF2": '#91bfdb',
        "3'UTR": '#d9ef8b',
    }

    ymin, ymax = ax.get_ylim()
    for rname, (s, e) in regions.items():
        e_actual = e if e else len(l1hs_seq)
        ax.axvspan(s, e_actual, alpha=0.15, color=region_colors.get(rname, '#cccccc'))
        ax.text((s + e_actual) / 2, ymax * 0.95, rname, ha='center', va='top', fontsize=9, fontweight='bold')

    ax.set_xlabel('Position in L1HS consensus (bp)')
    ax.set_ylabel('DRACH motifs per kb')
    ax.set_title('A. DRACH Motif Density Across L1HS Consensus (100bp sliding window)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(l1hs_seq))

    # --- Panel B: AT content sliding window ---
    ax = axes[1]
    ax.plot(sw_df['position'], sw_df['at_frac'] * 100, color='#636363', linewidth=0.8, alpha=0.7)
    smoothed_at = (sw_df['at_frac'] * 100).rolling(window, center=True).mean()
    ax.plot(sw_df['position'], smoothed_at, color='#252525', linewidth=2)

    for rname, (s, e) in regions.items():
        e_actual = e if e else len(l1hs_seq)
        ax.axvspan(s, e_actual, alpha=0.15, color=region_colors.get(rname, '#cccccc'))

    ax.set_xlabel('Position in L1HS consensus (bp)')
    ax.set_ylabel('A+T content (%)')
    ax.set_title('B. Base Composition (AT%) Across L1HS Consensus')
    ax.set_xlim(0, len(l1hs_seq))
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # --- Panel C: DRACH/kb by region bar chart ---
    ax = axes[2]

    region_order = ["5'UTR", "ORF1", "Inter-ORF", "ORF2", "3'UTR"]
    x = np.arange(len(region_order))

    drach_vals = []
    at_vals = []
    for rname in region_order:
        r = df_l1hs[df_l1hs['region'] == rname]
        if len(r) > 0:
            drach_vals.append(r.iloc[0]['drach_per_kb_fwd'])
            at_vals.append(r.iloc[0]['AT_frac'] * 100)
        else:
            drach_vals.append(0)
            at_vals.append(0)

    bars = ax.bar(x, drach_vals, color=[region_colors[r] for r in region_order],
                  edgecolor='black', linewidth=0.8)

    # Add value labels
    for i, (v, at) in enumerate(zip(drach_vals, at_vals)):
        ax.text(i, v + 0.3, f'{v:.1f}/kb\n({at:.0f}% AT)', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(region_order)
    ax.set_ylabel('DRACH motifs per kb (fwd strand)')
    ax.set_title('C. DRACH Density by L1HS Region')
    ax.set_ylim(0, max(drach_vals) * 1.3)

    # --- Panel D: Top 5 motif composition by region ---
    ax = axes[3]

    top5 = sorted(TOP_MOTIFS_BY_RATE.keys(), key=lambda m: TOP_MOTIFS_BY_RATE[m], reverse=True)

    # Stacked bar
    bottom = np.zeros(len(region_order))
    colors_top5 = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']

    for i_motif, motif in enumerate(top5):
        vals = []
        for rname in region_order:
            sub = df_motif[(df_motif['region'] == rname) & (df_motif['motif'] == motif)]
            if len(sub) > 0:
                vals.append(sub.iloc[0]['per_kb'])
            else:
                vals.append(0)

        ax.bar(x, vals, bottom=bottom, label=f"{motif} ({TOP_MOTIFS_BY_RATE[motif]:.0%})",
               color=colors_top5[i_motif], edgecolor='black', linewidth=0.5)
        bottom += vals

    # Other motifs
    other_vals = []
    for j, rname in enumerate(region_order):
        total = drach_vals[j]
        other_vals.append(total - bottom[j])

    ax.bar(x, other_vals, bottom=bottom, label='Other 13 DRACH',
           color='#999999', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(region_order)
    ax.set_ylabel('Motif density per kb (fwd strand)')
    ax.set_title('D. DRACH Motif Composition by Region (colored by m6A rate)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'l1hs_drach_by_region.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Figure 2: Subfamily comparison ---
    if subfamily_results:
        # Get subfamilies with all 4 main regions
        valid_sfs = []
        for sf in df_all_sf['subfamily'].unique():
            sub = df_all_sf[df_all_sf['subfamily'] == sf]
            if all(rname in sub['region'].values for rname in ["5'UTR", "ORF1", "ORF2", "3'UTR"]):
                valid_sfs.append(sf)

        if valid_sfs:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

            # Panel A: DRACH/kb by region for each subfamily
            ax = axes2[0]
            x_sf = np.arange(len(region_order[:-1]))  # exclude Inter-ORF for cleaner plot
            main_regions = ["5'UTR", "ORF1", "ORF2", "3'UTR"]
            width = 0.8 / len(valid_sfs)

            for i, sf in enumerate(valid_sfs[:8]):  # max 8 for readability
                sub = df_all_sf[df_all_sf['subfamily'] == sf]
                vals = []
                for rname in main_regions:
                    r = sub[sub['region'] == rname]
                    vals.append(r.iloc[0]['drach_per_kb_fwd'] if len(r) > 0 else 0)

                offset = (i - len(valid_sfs[:8]) / 2 + 0.5) * width
                ax.bar(np.arange(len(main_regions)) + offset, vals, width,
                       label=sf, alpha=0.8)

            ax.set_xticks(np.arange(len(main_regions)))
            ax.set_xticklabels(main_regions)
            ax.set_ylabel('DRACH/kb (fwd strand)')
            ax.set_title('A. DRACH Density by Region Across Subfamilies')
            ax.legend(fontsize=7, ncol=2)

            # Panel B: ORF1/ORF2 ratio by subfamily
            ax = axes2[1]
            sf_ratios = []
            sf_names = []
            sf_colors = []

            for sf in valid_sfs:
                sub = df_all_sf[df_all_sf['subfamily'] == sf]
                orf1 = sub[sub['region'] == 'ORF1']['drach_per_kb_fwd'].values
                orf2 = sub[sub['region'] == 'ORF2']['drach_per_kb_fwd'].values
                if len(orf1) > 0 and len(orf2) > 0 and orf2[0] > 0:
                    sf_ratios.append(orf1[0] / orf2[0])
                    sf_names.append(sf)
                    age = sub['age_group'].iloc[0]
                    if age == 'young':
                        sf_colors.append('#d73027')
                    elif age == 'L1PA':
                        sf_colors.append('#fc8d59')
                    elif age == 'L1M':
                        sf_colors.append('#91bfdb')
                    elif age == 'L1MC':
                        sf_colors.append('#4575b4')
                    elif age == 'L1ME':
                        sf_colors.append('#313695')
                    else:
                        sf_colors.append('#999999')

            ax.barh(range(len(sf_names)), sf_ratios, color=sf_colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(sf_names)))
            ax.set_yticklabels(sf_names, fontsize=8)
            ax.set_xlabel('ORF1/ORF2 DRACH density ratio')
            ax.set_title('B. ORF1/ORF2 DRACH Ratio by Subfamily')
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

            # Add legend for age groups
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d73027', label='Young'),
                Patch(facecolor='#fc8d59', label='L1PA (intermediate)'),
                Patch(facecolor='#91bfdb', label='L1M'),
                Patch(facecolor='#4575b4', label='L1MC'),
                Patch(facecolor='#313695', label='L1ME'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

            plt.tight_layout()
            fig2_path = os.path.join(OUT_DIR, 'subfamily_drach_by_region.png')
            plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fig2_path}")

    # =========================================================================
    # 13. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    orf1 = df_l1hs[df_l1hs['region'] == 'ORF1'].iloc[0]
    orf2 = df_l1hs[df_l1hs['region'] == 'ORF2'].iloc[0]
    utr5 = df_l1hs[df_l1hs['region'] == "5'UTR"].iloc[0]
    utr3 = df_l1hs[df_l1hs['region'] == "3'UTR"].iloc[0]

    print(f"\n  L1HS Consensus (M80343, {len(l1hs_seq)}bp):")
    print(f"  Total DRACH sites (fwd): {len(fwd_full)}")
    print(f"  Overall density: {len(fwd_full)/(len(l1hs_seq)/1000):.2f}/kb")

    print(f"\n  Region comparison:")
    print(f"    5'UTR:  {utr5['drach_per_kb_fwd']:.2f}/kb (AT={utr5['AT_frac']:.1%})")
    print(f"    ORF1:   {orf1['drach_per_kb_fwd']:.2f}/kb (AT={orf1['AT_frac']:.1%})")
    print(f"    ORF2:   {orf2['drach_per_kb_fwd']:.2f}/kb (AT={orf2['AT_frac']:.1%})")
    print(f"    3'UTR:  {utr3['drach_per_kb_fwd']:.2f}/kb (AT={utr3['AT_frac']:.1%})")

    orf1_orf2_ratio = orf1['drach_per_kb_fwd'] / orf2['drach_per_kb_fwd'] if orf2['drach_per_kb_fwd'] > 0 else float('inf')
    print(f"\n  ORF1/ORF2 DRACH density ratio: {orf1_orf2_ratio:.2f}x")
    print(f"  ORF1/ORF2 AT% ratio: {orf1['AT_frac']/orf2['AT_frac']:.2f}x")

    # Rate-weighted summary
    for region_name, row in [("ORF1", orf1), ("ORF2", orf2)]:
        ws = sum(row[f'fwd_{m}'] * L1_MOTIF_RATES.get(m, 0.2) for m in ALL_DRACH)
        tc = sum(row[f'fwd_{m}'] for m in ALL_DRACH)
        eff = ws / tc if tc > 0 else 0
        pred = ws / (row['length_bp'] / 1000)
        print(f"  {region_name} effective rate: {eff:.1%}, predicted m6A/kb: {pred:.2f}")

    print(f"\n  Top 5 high-rate motifs fraction:")
    print(f"    ORF1: {orf1['top_5_motif_frac']:.1%} of DRACH sites")
    print(f"    ORF2: {orf2['top_5_motif_frac']:.1%} of DRACH sites")

    print(f"\n  Output directory: {OUT_DIR}")
    print("  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        fsize = os.path.getsize(fpath) / 1024
        print(f"    {f} ({fsize:.1f} KB)")

    print("\nDone!")


if __name__ == '__main__':
    main()
