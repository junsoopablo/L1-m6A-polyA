#!/usr/bin/env python3
"""
Reclassify ninetails read_classes.tsv based on nonadenosine_residues.tsv.
- If a read has at least one est_nonA_pos >= 30: keep as "decorated"
- If a read is "decorated" but all est_nonA_pos < 30: change to "3UTR"
"""

import os
import glob
import pandas as pd
import argparse


def reclassify_group(group_dir, output_dir=None):
    """Reclassify reads for a single group."""

    # Find the most recent files
    read_classes_pattern = os.path.join(group_dir, "f_ninetails", "*_read_classes.tsv")
    nonA_pattern = os.path.join(group_dir, "f_ninetails", "*_nonadenosine_residues.tsv")

    read_classes_files = glob.glob(read_classes_pattern)
    nonA_files = glob.glob(nonA_pattern)

    if not read_classes_files or not nonA_files:
        print(f"Skipping {group_dir}: missing files")
        return None

    read_classes_path = sorted(read_classes_files)[-1]
    nonA_path = sorted(nonA_files)[-1]

    # Read files
    classes_df = pd.read_csv(read_classes_path, sep='\t')
    nonA_df = pd.read_csv(nonA_path, sep='\t')

    # Find reads with at least one est_nonA_pos >= 30
    decorated_reads = set(nonA_df[nonA_df['est_nonA_pos'] >= 30]['readname'].unique())

    # Reclassify
    def reclassify(row):
        if row['class'] == 'decorated':
            if row['readname'] in decorated_reads:
                return 'decorated'
            else:
                return '3UTR'
        return row['class']

    classes_df['class'] = classes_df.apply(reclassify, axis=1)

    # Output path
    if output_dir is None:
        output_dir = os.path.join(group_dir, "f_ninetails")

    os.makedirs(output_dir, exist_ok=True)

    # Get group name from path
    group_name = os.path.basename(group_dir)
    output_path = os.path.join(output_dir, f"{group_name}_read_classes_reclassified.tsv")

    classes_df.to_csv(output_path, sep='\t', index=False)

    # Print stats
    print(f"\n=== {group_name} ===")
    print(f"Original file: {os.path.basename(read_classes_path)}")
    print(f"Output: {output_path}")
    print(f"\nClass distribution:")
    print(classes_df['class'].value_counts().to_string())

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reclassify ninetails results")
    parser.add_argument("--results-group-dir",
                        default="/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1/results_group",
                        help="Path to results_group directory")
    parser.add_argument("--group", help="Process specific group only")
    args = parser.parse_args()

    if args.group:
        group_dir = os.path.join(args.results_group_dir, args.group)
        if os.path.isdir(group_dir):
            reclassify_group(group_dir)
        else:
            print(f"Group directory not found: {group_dir}")
    else:
        # Process all groups
        for group_name in sorted(os.listdir(args.results_group_dir)):
            group_dir = os.path.join(args.results_group_dir, group_name)
            if os.path.isdir(group_dir):
                ninetails_dir = os.path.join(group_dir, "f_ninetails")
                if os.path.isdir(ninetails_dir):
                    reclassify_group(group_dir)


if __name__ == "__main__":
    main()
