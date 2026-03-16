#!/usr/bin/env python3
"""
Create summary table for housekeeping gene poly(A) analysis.

Combines nanopolish poly(A) measurements with ninetails classification
for housekeeping gene reads.
"""

import argparse
import gzip
import pandas as pd
from pathlib import Path


def load_gene_regions(bed_file):
    """Load housekeeping gene regions from BED file."""
    regions = {}
    with open(bed_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom, start, end, gene = parts[0], int(parts[1]), int(parts[2]), parts[3]
                regions[gene] = (chrom, start, end)
    return regions


def find_gene(chrom, pos, regions):
    """Find which gene a position belongs to."""
    for gene, (g_chr, g_start, g_end) in regions.items():
        if chrom == g_chr and g_start <= pos <= g_end:
            return gene
    return None


def main():
    parser = argparse.ArgumentParser(description='Create HK summary table')
    parser.add_argument('--group', required=True, help='Group name')
    parser.add_argument('--nanopolish', required=True, help='Nanopolish polya TSV (gzipped)')
    parser.add_argument('--ninetails-dir', required=True, help='Ninetails output directory')
    parser.add_argument('--bed', required=True, help='Housekeeping gene BED file')
    parser.add_argument('--output', required=True, help='Output summary TSV')
    args = parser.parse_args()

    ninetails_dir = Path(args.ninetails_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load gene regions
    regions = load_gene_regions(args.bed)

    # Load ninetails classes if available
    class_dict = {}
    class_files = list(ninetails_dir.glob("*_HK_read_classes.tsv"))
    if class_files:
        class_df = pd.read_csv(class_files[0], sep='\t')
        class_dict = dict(zip(class_df['readname'], class_df['class']))

    # Load nanopolish polya data
    records = []
    try:
        with gzip.open(args.nanopolish, 'rt') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue

                read_id = fields[0]
                chrom = fields[1]
                pos = int(fields[2])
                polya_length = float(fields[8])
                qc_tag = fields[9]

                gene = find_gene(chrom, pos, regions)
                if gene and qc_tag == 'PASS':
                    read_class = class_dict.get(read_id, 'unknown')
                    records.append({
                        'read_id': read_id,
                        'gene': gene,
                        'chrom': chrom,
                        'position': pos,
                        'polya_length': polya_length,
                        'qc_tag': qc_tag,
                        'class': read_class,
                        'group': args.group,
                    })
    except (EOFError, gzip.BadGzipFile):
        # Empty or invalid gzip file
        pass

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output, sep='\t', index=False)
        print(f"Created HK summary with {len(df)} reads for group {args.group}")
    else:
        # Create empty file with header
        with open(output, 'w') as f:
            f.write("read_id\tgene\tchrom\tposition\tpolya_length\tqc_tag\tclass\tgroup\n")
        print(f"No HK reads found for group {args.group}")


if __name__ == '__main__':
    main()
