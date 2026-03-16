#!/usr/bin/env python3
"""
Validate the 100bp protein-coding exon overlap threshold used in Stage 2 L1 filter.

Approach:
1. Recompute exon overlap for ALL stage 1 L1 reads (from BAM + exons.bed)
2. Show overlap distribution — is there a natural breakpoint?
3. For PASS reads (overlap <100bp), check gradient of arsenite Δpoly(A)
4. Threshold sweep: what fraction of reads pass at each threshold?
"""
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT = Path('/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1')
RESULTS = PROJECT / 'results'
OUTDIR = PROJECT / 'analysis/01_exploration/topic_07_catB_nc_exonic/exon_overlap_validation'
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMTOOLS = 'samtools'
BEDTOOLS = 'bedtools'

SAMPLES = {
    'HeLa': ['HeLa_1_1', 'HeLa_2_1', 'HeLa_3_1'],
    'HeLa-Ars': ['HeLa-Ars_1_1', 'HeLa-Ars_2_1', 'HeLa-Ars_3_1'],
}


def get_stage1_read_ids(sample):
    """Get all stage 1 L1 read IDs."""
    path = RESULTS / sample / 'b_l1_te_filter' / f'{sample}_L1_readIDs.txt'
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def extract_bed_blocks_from_bam(sample, read_ids):
    """Extract BED blocks from BAM for given read IDs, using CIGAR."""
    bam = RESULTS / sample / 'a_hg38_mapping_LRS' / f'{sample}_hg38_mapped.sorted_position.bam'

    # Write read IDs to temp file for filtering
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for rid in read_ids:
            f.write(rid + '\n')
        rid_file = f.name

    bed_entries = []
    try:
        # Extract reads from BAM
        cmd = f'{SAMTOOLS} view {bam} | grep -F -f {rid_file}'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

        for line in proc.stdout.strip().split('\n'):
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 6:
                continue
            qname = fields[0]
            flag = int(fields[1])
            chrom = fields[2]
            pos = int(fields[3]) - 1  # 0-based
            cigar = fields[5]

            if chrom == '*' or cigar == '*':
                continue

            # Parse CIGAR to get aligned blocks
            import re
            blocks = []
            ref_pos = pos
            for length, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar):
                length = int(length)
                if op in ('M', '=', 'X'):
                    blocks.append((chrom, ref_pos, ref_pos + length, qname))
                    ref_pos += length
                elif op in ('D', 'N'):
                    ref_pos += length
                # I, S, H, P don't consume reference

            bed_entries.extend(blocks)
    finally:
        os.unlink(rid_file)

    return bed_entries


def compute_exon_overlap(bed_entries, exons_bed_path):
    """Compute max exon overlap per read using bedtools intersect."""
    # Write BED entries to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        for chrom, start, end, name in bed_entries:
            f.write(f'{chrom}\t{start}\t{end}\t{name}\n')
        reads_bed = f.name

    overlap_per_read = {}
    try:
        # Sort BED
        sorted_bed = reads_bed + '.sorted'
        subprocess.run(f'sort -k1,1 -k2,2n {reads_bed} > {sorted_bed}',
                       shell=True, check=True)

        cmd = f'{BEDTOOLS} intersect -a {sorted_bed} -b {exons_bed_path} -wo'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        for line in proc.stdout.strip().split('\n'):
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 8:
                continue
            read_id = fields[3]
            overlap_len = int(fields[7])
            current = overlap_per_read.get(read_id, 0)
            if overlap_len > current:
                overlap_per_read[read_id] = overlap_len
    finally:
        os.unlink(reads_bed)
        if os.path.exists(reads_bed + '.sorted'):
            os.unlink(reads_bed + '.sorted')

    return overlap_per_read


def main():
    all_results = []

    for condition, samples in SAMPLES.items():
        for sample in samples:
            print(f'\n=== Processing {sample} ===')

            # Get stage 1 read IDs
            stage1_ids = get_stage1_read_ids(sample)
            print(f'  Stage 1 reads: {len(stage1_ids)}')

            # Get PASS read IDs from L1_reads.tsv
            l1_tsv = RESULTS / sample / 'd_LINE_quantification' / f'{sample}_L1_reads.tsv'
            pass_ids = set()
            if l1_tsv.exists():
                with open(l1_tsv) as f:
                    next(f)  # skip header
                    for line in f:
                        pass_ids.add(line.split('\t')[0])
            print(f'  PASS reads: {len(pass_ids)}')
            print(f'  Rejected at stage 2: {len(stage1_ids) - len(pass_ids)}')

            # Extract BED blocks from BAM for ALL stage 1 reads
            print(f'  Extracting BED blocks from BAM...')
            bed_entries = extract_bed_blocks_from_bam(sample, stage1_ids)
            print(f'  BED blocks: {len(bed_entries)}')

            # Compute exon overlap
            exons_bed = RESULTS / sample / 'd_LINE_quantification' / 'exons.bed'
            print(f'  Computing exon overlaps...')
            overlap_per_read = compute_exon_overlap(bed_entries, exons_bed)

            # Store results
            for rid in stage1_ids:
                ov = overlap_per_read.get(rid, 0)
                is_pass = rid in pass_ids
                all_results.append({
                    'read_id': rid,
                    'sample': sample,
                    'condition': condition,
                    'exon_overlap': ov,
                    'is_pass': is_pass,
                })

            # Quick summary
            overlaps = [overlap_per_read.get(rid, 0) for rid in stage1_ids]
            pass_overlaps = [overlap_per_read.get(rid, 0) for rid in pass_ids]
            rejected_ids = stage1_ids - pass_ids
            rejected_overlaps = [overlap_per_read.get(rid, 0) for rid in rejected_ids]

            print(f'  PASS overlap: median={np.median(pass_overlaps):.0f}, '
                  f'max={max(pass_overlaps) if pass_overlaps else 0}')
            if rejected_overlaps:
                print(f'  Rejected overlap: median={np.median(rejected_overlaps):.0f}, '
                      f'max={max(rejected_overlaps)}')

    # Save full results
    outfile = OUTDIR / 'exon_overlap_all_stage1.tsv'
    with open(outfile, 'w') as f:
        f.write('read_id\tsample\tcondition\texon_overlap\tis_pass\n')
        for r in all_results:
            f.write(f"{r['read_id']}\t{r['sample']}\t{r['condition']}\t{r['exon_overlap']}\t{r['is_pass']}\n")
    print(f'\nSaved: {outfile}')
    print(f'Total reads: {len(all_results)}')

    # === Analysis ===
    overlaps_all = np.array([r['exon_overlap'] for r in all_results])
    is_pass = np.array([r['is_pass'] for r in all_results])
    conditions = np.array([r['condition'] for r in all_results])

    print('\n' + '='*60)
    print('OVERLAP DISTRIBUTION (ALL STAGE 1 READS)')
    print('='*60)

    bins = [0, 1, 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 5000, 100000]
    labels = ['0', '1-9', '10-24', '25-49', '50-74', '75-99',
              '100-149', '150-199', '200-299', '300-499', '500-999', '1000-4999', '5000+']

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (overlaps_all >= lo) & (overlaps_all < hi)
        n = mask.sum()
        n_pass = (mask & is_pass).sum()
        n_rej = (mask & ~is_pass).sum()
        pct = 100.0 * n / len(overlaps_all)
        print(f'  {labels[i]:>10s}: {n:6d} ({pct:5.1f}%)  [PASS: {n_pass}, Rejected: {n_rej}]')

    # Threshold sweep
    print('\n' + '='*60)
    print('THRESHOLD SWEEP')
    print('='*60)
    print(f'{"Threshold":>10s}  {"N pass":>8s}  {"N reject":>8s}  {"% pass":>8s}  {"Lost vs 100":>12s}')

    n_total = len(overlaps_all)
    n_pass_100 = (overlaps_all < 100).sum()

    for thr in [25, 50, 75, 100, 125, 150, 200, 300, 500]:
        n_pass_thr = (overlaps_all < thr).sum()
        pct = 100.0 * n_pass_thr / n_total
        delta = n_pass_thr - n_pass_100
        print(f'  {thr:>8d}  {n_pass_thr:>8d}  {n_total - n_pass_thr:>8d}  {pct:>7.1f}%  {delta:>+10d}')

    # For PASS reads, check gradient within 0-99bp
    print('\n' + '='*60)
    print('WITHIN-PASS OVERLAP GRADIENT')
    print('='*60)

    pass_mask = is_pass
    pass_overlaps = overlaps_all[pass_mask]

    pct_0 = 100.0 * (pass_overlaps == 0).sum() / pass_mask.sum()
    pct_lt10 = 100.0 * (pass_overlaps < 10).sum() / pass_mask.sum()
    pct_lt50 = 100.0 * (pass_overlaps < 50).sum() / pass_mask.sum()
    print(f'  PASS reads with overlap = 0bp: {(pass_overlaps == 0).sum()} ({pct_0:.1f}%)')
    print(f'  PASS reads with overlap < 10bp: {(pass_overlaps < 10).sum()} ({pct_lt10:.1f}%)')
    print(f'  PASS reads with overlap < 50bp: {(pass_overlaps < 50).sum()} ({pct_lt50:.1f}%)')
    print(f'  PASS reads with overlap 50-99bp: {((pass_overlaps >= 50) & (pass_overlaps < 100)).sum()}')


if __name__ == '__main__':
    main()
