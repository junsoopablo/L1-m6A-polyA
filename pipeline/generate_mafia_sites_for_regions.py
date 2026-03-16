#!/usr/bin/env python3
"""
Generate m6A and psi sites for specific genomic regions.
Based on psi-co-mAFiA generate_sites but limited to input BED regions.
"""

import argparse
import os
from Bio import SeqIO
from Bio.Seq import Seq

# m6A motifs: DRACH (D=A/G/U, R=A/G, A, C, H=A/C/U)
# In DNA: [AGT][AG]AC[ACT]
M6A_MOTIFS = set()
for d in ['A', 'G', 'T']:
    for r in ['A', 'G']:
        for h in ['A', 'C', 'T']:
            M6A_MOTIFS.add(f"{d}{r}AC{h}")

# psi motifs: 16 high-frequency Ψ 5-mers from psi-co-mAFiA classifier models
# Source: psi-co-mAFiA/models/psi-co-mAFiA/psi/*.pkl
# DNA equivalents (position 2 = U/T that gets modified)
PSI_MOTIFS = {
    'GTTCA', 'GTTCC', 'GTTCG', 'GTTCT',  # GUUCN (TRUB1 target)
    'AGTGG', 'GGTGG', 'TGTGG',            # xGUGG
    'TGTAG',                                # UGUAG
    'GGTCC',                                # GGUCC
    'CATAA', 'TATAA',                       # xAUAA
    'CATCC',                                # CAUCC
    'CTTTA',                                # CUUUA
    'ATTTG',                                # AUUUG
    'GATGC',                                # GAUGC
    'CCTCC',                                # CCUCC
}


def load_regions(bed_file, strand_specific=False):
    """Load BED regions with optional strand info."""
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                # Get strand from column 4 (after bedtools merge -c 6 -o distinct)
                # or column 6 for standard BED6
                strand = '.'
                if strand_specific:
                    if len(parts) >= 4:
                        strand = parts[3] if parts[3] in ['+', '-'] else parts[5] if len(parts) >= 6 else '.'
                regions.append((chrom, start, end, strand))
    return regions


def get_sequence(ref_dict, chrom, start, end):
    """Get sequence from reference."""
    if chrom not in ref_dict:
        return None
    seq = ref_dict[chrom].seq[start:end]
    return str(seq).upper()


def find_sites_in_region(seq, region_start, chrom, strand='+'):
    """Find m6A and psi sites in a sequence region."""
    sites = []

    for i in range(len(seq) - 4):
        kmer = seq[i:i+5]

        # Check m6A (A is at position 2, 0-indexed)
        if kmer in M6A_MOTIFS:
            pos = region_start + i + 2  # Position of A in DRACH
            sites.append({
                'chrom': chrom,
                'start': pos,
                'end': pos + 1,
                'name': 'm6A',
                'strand': strand,
                'ref5mer': kmer
            })

        # Check psi (U/T is at position 2)
        if kmer in PSI_MOTIFS:
            pos = region_start + i + 2  # Position of U/T
            sites.append({
                'chrom': chrom,
                'start': pos,
                'end': pos + 1,
                'name': 'psi',
                'strand': strand,
                'ref5mer': kmer
            })

    return sites


def reverse_complement(seq):
    """Get reverse complement of a sequence."""
    return str(Seq(seq).reverse_complement())


def main():
    parser = argparse.ArgumentParser(description="Generate m6A/psi sites for specific regions")
    parser.add_argument("--ref_file", required=True, help="Reference FASTA file")
    parser.add_argument("--regions_bed", required=True, help="BED file with regions to analyze")
    parser.add_argument("--out_file", required=True, help="Output BED file")
    parser.add_argument("--flank", type=int, default=100, help="Flanking region to add (default: 100)")
    parser.add_argument("--strand_specific", action="store_true",
                        help="Only generate sites matching the region's strand")
    args = parser.parse_args()

    print(f"Loading reference: {args.ref_file}")
    ref_dict = SeqIO.to_dict(SeqIO.parse(args.ref_file, "fasta"))
    print(f"Loaded {len(ref_dict)} contigs")

    print(f"Loading regions: {args.regions_bed}")
    regions = load_regions(args.regions_bed, args.strand_specific)
    print(f"Loaded {len(regions)} regions")
    if args.strand_specific:
        plus_count = sum(1 for r in regions if r[3] == '+')
        minus_count = sum(1 for r in regions if r[3] == '-')
        print(f"  + strand: {plus_count}, - strand: {minus_count}")

    all_sites = []
    seen = set()  # To avoid duplicates

    for chrom, start, end, region_strand in regions:
        # Add flanking regions
        start_flanked = max(0, start - args.flank)
        end_flanked = end + args.flank

        # Get forward strand sequence
        seq_fwd = get_sequence(ref_dict, chrom, start_flanked, end_flanked)
        if seq_fwd is None:
            continue

        # Determine which strands to analyze
        if args.strand_specific and region_strand in ['+', '-']:
            strands_to_analyze = [region_strand]
        else:
            strands_to_analyze = ['+', '-']

        for strand in strands_to_analyze:
            if strand == '+':
                # Find sites on forward strand
                fwd_sites = find_sites_in_region(seq_fwd, start_flanked, chrom, '+')
                for site in fwd_sites:
                    key = (site['chrom'], site['start'], site['name'], site['strand'])
                    if key not in seen:
                        seen.add(key)
                        all_sites.append(site)
            else:
                # Find sites on reverse strand
                seq_rev = reverse_complement(seq_fwd)
                rev_sites = find_sites_in_region(seq_rev, start_flanked, chrom, '-')
                # Adjust positions for reverse strand
                for site in rev_sites:
                    # Convert position back to forward strand coordinates
                    orig_pos = site['start']
                    new_pos = end_flanked - (orig_pos - start_flanked) - 1
                    site['start'] = new_pos
                    site['end'] = new_pos + 1
                    key = (site['chrom'], site['start'], site['name'], site['strand'])
                    if key not in seen:
                        seen.add(key)
                        all_sites.append(site)

    # Sort sites
    all_sites.sort(key=lambda x: (x['chrom'], x['start']))

    print(f"Found {len(all_sites)} sites")
    m6a_count = sum(1 for s in all_sites if s['name'] == 'm6A')
    psi_count = sum(1 for s in all_sites if s['name'] == 'psi')
    print(f"  m6A: {m6a_count}")
    print(f"  psi: {psi_count}")
    if args.strand_specific:
        plus_sites = sum(1 for s in all_sites if s['strand'] == '+')
        minus_sites = sum(1 for s in all_sites if s['strand'] == '-')
        print(f"  + strand sites: {plus_sites}, - strand sites: {minus_sites}")

    # Write output
    with open(args.out_file, 'w') as f:
        f.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tref5mer\n")
        for site in all_sites:
            f.write(f"{site['chrom']}\t{site['start']}\t{site['end']}\t{site['name']}\t.\t{site['strand']}\t{site['ref5mer']}\n")

    print(f"Output written to: {args.out_file}")


if __name__ == "__main__":
    main()
