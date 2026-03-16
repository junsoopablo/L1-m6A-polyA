#!/usr/bin/env python3
"""
Collect statistics for each sample and group using pre-calculated files.
"""

import os
import glob
import gzip
import pandas as pd
import yaml
from pathlib import Path

# Paths
BASE_DIR = "/qbio/junsoopablo/02_Projects/05_IsoTENT_002_L1"
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_GROUP_DIR = os.path.join(BASE_DIR, "results_group")
L1_BASECALL_DIR = os.path.join(BASE_DIR, "l1_basecall")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def read_bam_summary(sample):
    """Read fastq read count from bam_summary.txt"""
    path = os.path.join(RESULTS_DIR, sample, "a_hg38_mapping_LRS", sample, "bam_summary.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            for line in f:
                if "Number of reads" in line:
                    return int(line.strip().split('\t')[1])
    except:
        pass
    return None


def read_l1_counts(sample):
    """Read mapped_reads and l1_reads from L1_counts.tsv"""
    path = os.path.join(RESULTS_DIR, sample, "b_l1_te_filter", f"{sample}_L1_counts.tsv")
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path, sep='\t')
        return df['mapped_reads'].iloc[0], df['l1_reads'].iloc[0]
    except:
        return None, None


def count_l1_pass_reads(sample):
    """Count lines in L1_pass_readIDs.txt"""
    path = os.path.join(RESULTS_DIR, sample, "d_LINE_quantification", f"{sample}_L1_pass_readIDs.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except:
        return None


def count_guppy_reads(group):
    """Count reads from sequencing_summary.txt (lines - 1 for header)"""
    path = os.path.join(L1_BASECALL_DIR, f"{group}.sequencing_summary.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1  # subtract header
    except:
        return None


def get_nanopolish_stats(group):
    """Get PASS count and average polya_length from nanopolish output."""
    path = os.path.join(RESULTS_GROUP_DIR, group, "e_nanopolish", f"{group}.nanopolish.polya.tsv.gz")
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path, sep='\t', compression='gzip')
        pass_df = df[df['qc_tag'] == 'PASS']
        pass_count = len(pass_df)
        avg_polya = pass_df['polya_length'].mean() if pass_count > 0 else None
        return pass_count, avg_polya
    except:
        return None, None


def get_ninetails_stats(group):
    """Get decorated/non-decorated stats from ninetails output files."""
    # Decorated: from nonadenosine_residues.tsv where est_nonA_pos >= 30 (unique readnames)
    nonA_pattern = os.path.join(RESULTS_GROUP_DIR, group, "f_ninetails", "*_nonadenosine_residues.tsv")
    nonA_files = glob.glob(nonA_pattern)

    decorated_count, decorated_avg_polya = None, None
    if nonA_files:
        try:
            path = sorted(nonA_files)[-1]
            df = pd.read_csv(path, sep='\t')
            decorated = df[df['est_nonA_pos'] >= 30]
            # Deduplicate by readname for count and polya_length average
            decorated_unique = decorated.drop_duplicates(subset='readname')
            decorated_count = len(decorated_unique)
            decorated_avg_polya = decorated_unique['polya_length'].mean() if decorated_count > 0 else None
        except:
            pass

    # Non-decorated: from read_classes.tsv where qc_tag='PASS' and class != 'decorated'
    classes_pattern = os.path.join(RESULTS_GROUP_DIR, group, "f_ninetails", "*_read_classes.tsv")
    classes_files = glob.glob(classes_pattern)

    non_decorated_count, non_decorated_avg_polya = None, None
    if classes_files:
        try:
            path = sorted(classes_files)[-1]
            df = pd.read_csv(path, sep='\t')
            non_decorated = df[(df['qc_tag'] == 'PASS') & (df['class'] != 'decorated')]
            non_decorated_count = len(non_decorated)
            non_decorated_avg_polya = non_decorated['polya_length'].mean() if non_decorated_count > 0 else None
        except:
            pass

    return decorated_count, decorated_avg_polya, non_decorated_count, non_decorated_avg_polya


def sample_to_group(sample):
    """Derive group name from sample name by removing last _N suffix."""
    # e.g., A549_4_1 -> A549_4, H9_2_2 -> H9_2, HeLa-Ars_3_1 -> HeLa-Ars_3
    import re
    match = re.match(r'^(.+)_\d+$', sample)
    if match:
        return match.group(1)
    return sample


def main():
    config = load_config()
    samples = config.get('samples', [])

    results = []

    for sample in samples:
        group = sample_to_group(sample)

        row = {
            'sample': sample,
            'group': group,
        }

        # 3. FASTQ read count from bam_summary.txt
        row['fastq_reads'] = read_bam_summary(sample)

        # 4 & 5. hg38 mapped and L1 reads from L1_counts.tsv
        mapped, l1 = read_l1_counts(sample)
        row['hg38_mapped_reads'] = mapped
        row['l1_reads'] = l1

        # 6. L1 filter passed read count
        row['l1_pass_reads'] = count_l1_pass_reads(sample)

        # 7. Guppy basecalled reads (group level)
        row['guppy_fastq_reads'] = count_guppy_reads(group)

        # 8 & 9. Nanopolish stats (group level)
        pass_count, avg_polya = get_nanopolish_stats(group)
        row['nanopolish_pass_reads'] = pass_count
        row['nanopolish_avg_polya_length'] = round(avg_polya, 2) if avg_polya else None

        # 10. Ninetails stats (group level)
        decorated_count, decorated_avg_polya, non_decorated_count, non_decorated_avg_polya = get_ninetails_stats(group)
        row['ninetails_decorated_reads'] = decorated_count
        row['ninetails_decorated_avg_polya'] = round(decorated_avg_polya, 2) if decorated_avg_polya else None
        row['ninetails_non_decorated_reads'] = non_decorated_count
        row['ninetails_non_decorated_avg_polya'] = round(non_decorated_avg_polya, 2) if non_decorated_avg_polya else None

        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sample-level TSV
    sample_cols = ['sample', 'fastq_reads', 'hg38_mapped_reads', 'l1_reads', 'l1_pass_reads']
    sample_df = df[sample_cols]

    # Group-level TSV (deduplicated)
    group_cols = ['group', 'guppy_fastq_reads', 'nanopolish_pass_reads', 'nanopolish_avg_polya_length',
                  'ninetails_decorated_reads', 'ninetails_decorated_avg_polya',
                  'ninetails_non_decorated_reads', 'ninetails_non_decorated_avg_polya']
    group_df = df[group_cols].drop_duplicates()

    # Save to TSV
    os.makedirs(os.path.join(BASE_DIR, "analysis"), exist_ok=True)

    sample_output = os.path.join(BASE_DIR, "analysis", "sample_stats.tsv")
    sample_df.to_csv(sample_output, sep='\t', index=False)
    print(f"Saved to {sample_output}")

    group_output = os.path.join(BASE_DIR, "analysis", "group_stats.tsv")
    group_df.to_csv(group_output, sep='\t', index=False)
    print(f"Saved to {group_output}")

    # Print to stdout
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\n=== Sample Stats ===")
    print(sample_df.to_string(index=False))
    print("\n=== Group Stats ===")
    print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
