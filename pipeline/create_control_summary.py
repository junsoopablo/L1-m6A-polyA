#!/usr/bin/env python3
"""
Create summary table for control (non-L1) poly(A) analysis.

Combines nanopolish poly(A) measurements with ninetails classification
for non-L1 control reads.
"""

import argparse
import gzip
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Create control summary table')
    parser.add_argument('--group', required=True, help='Group name')
    parser.add_argument('--nanopolish', required=True, help='Nanopolish polya TSV (gzipped)')
    parser.add_argument('--ninetails-dir', required=True, help='Ninetails output directory')
    parser.add_argument('--output', required=True, help='Output summary TSV')
    args = parser.parse_args()

    ninetails_dir = Path(args.ninetails_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load ninetails classes if available
    class_dict = {}
    class_files = list(ninetails_dir.glob("*_control_read_classes.tsv"))
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

                if qc_tag == 'PASS':
                    read_class = class_dict.get(read_id, 'unknown')
                    records.append({
                        'read_id': read_id,
                        'chrom': chrom,
                        'position': pos,
                        'polya_length': polya_length,
                        'qc_tag': qc_tag,
                        'class': read_class,
                        'group': args.group,
                        'type': 'control',
                    })
    except (EOFError, gzip.BadGzipFile):
        pass

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output, sep='\t', index=False)
        print(f"Created control summary with {len(df)} reads for group {args.group}")
    else:
        with open(output, 'w') as f:
            f.write("read_id\tchrom\tposition\tpolya_length\tqc_tag\tclass\tgroup\ttype\n")
        print(f"No control reads found for group {args.group}")


if __name__ == '__main__':
    main()
