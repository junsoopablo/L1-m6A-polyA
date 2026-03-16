#!/usr/bin/env python3
"""
Pipeline validation script for IsoTENT L1 analysis.
Checks for:
1. Missing expected output files
2. Files that need regeneration (upstream newer than downstream)
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Color codes for terminal output
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'


def get_mtime(path):
    """Get modification time of a file."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def check_file_exists(path, allow_empty=False):
    """Check if file exists and is non-empty (unless allow_empty=True)."""
    if not os.path.isfile(path):
        return False
    if allow_empty:
        return True
    return os.path.getsize(path) > 0


def is_marker_file(filename):
    """Check if file is a marker file (allowed to be empty)."""
    return filename.endswith('.done')


def load_config(config_path):
    """Load config.yaml and return samples and groups."""
    if HAS_YAML:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        samples = config.get('samples', [])
    else:
        # Fallback: parse samples list from yaml without yaml module
        samples = []
        in_samples = False
        with open(config_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('samples:'):
                    in_samples = True
                    continue
                if in_samples:
                    if line.startswith('  - ') and not line.strip().startswith('#'):
                        sample = line.strip()[2:].strip()  # Remove "- " prefix
                        samples.append(sample)
                    elif not line.startswith('  ') and not line.startswith('#') and line.strip():
                        # End of samples list
                        break

    # Build groups from samples (sample format: GROUP_REPLICATE)
    groups = set()
    for sample in samples:
        # Extract group name (everything before the last underscore + number)
        parts = sample.rsplit('_', 1)
        if len(parts) == 2:
            group = parts[0]
            groups.add(group)

    return samples, sorted(groups)


def format_time(timestamp):
    """Format timestamp for display."""
    if timestamp is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def validate_pipeline(results_dir, results_group_dir, config_path=None, verbose=False):
    """
    Validate pipeline outputs and check file timestamps.
    Returns lists of missing files and files needing regeneration.
    """
    missing_files = []
    needs_regen = []

    # Get expected samples and groups from config
    if config_path and os.path.exists(config_path):
        samples, groups = load_config(config_path)
        print(f"\n{BOLD}=== Pipeline Validation (from config.yaml) ==={RESET}")
    else:
        # Fallback to directory scan
        samples = [d for d in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, d))
                   and not d.startswith('@') and not d.endswith('.tsv')]
        groups = [d for d in os.listdir(results_group_dir)
                  if os.path.isdir(os.path.join(results_group_dir, d))
                  and not d.startswith('@')]
        print(f"\n{BOLD}=== Pipeline Validation (directory scan) ==={RESET}")

    print(f"Samples: {len(samples)}, Groups: {len(groups)}\n")

    # Define expected files per sample with dependencies
    sample_files = {
        'a_hg38_mapping_LRS': {
            'files': [
                '{sample}_hg38_mapped.sorted_position.bam',
                '{sample}_hg38_mapped.sorted_position.bam.bai',
                '{sample}_hg38_counts.tsv',
            ],
            'depends_on': ['data_fastq/{sample}.fastq.gz'],
        },
        'b_l1_te_filter': {
            'files': [
                '{sample}_L1_readIDs.txt',
                '{sample}_L1_counts.tsv',
            ],
            'depends_on': ['a_hg38_mapping_LRS/{sample}_hg38_mapped.sorted_position.bam'],
        },
        'd_LINE_quantification': {
            'files': [
                '{sample}_L1_reads.tsv',
                '{sample}_L1_reads.bam',
                '{sample}_L1_reads.bam.bai',
            ],
            'depends_on': ['b_l1_te_filter/{sample}_L1_readIDs.txt'],
        },
    }

    # Define expected files per group with dependencies
    group_files = {
        'e_nanopolish': {
            'files': [
                '{group}.nanopolish.polya.tsv.gz',
                '{group}.nanopolish.index.done',
            ],
            'depends_on': [],
        },
        'f_ninetails': {
            'files': [
                '{group}_read_classes_reclassified.tsv',
            ],
            'depends_on': ['e_nanopolish/{group}.nanopolish.polya.tsv.gz'],
        },
        'g_summary': {
            'files': [
                '{group}_L1_summary.tsv',
            ],
            'depends_on': ['f_ninetails/{group}_read_classes_reclassified.tsv'],
        },
        'h_mafia': {
            'files': [
                '{group}.mAFiA.reads.bam',
                '{group}.mAFiA.sites.bed',
            ],
            'depends_on': [],
        },
    }

    # Check sample-level files
    print(f"{BOLD}Checking sample-level files...{RESET}")
    for sample in sorted(samples):
        sample_dir = os.path.join(results_dir, sample)

        for stage, config in sample_files.items():
            stage_dir = os.path.join(sample_dir, stage)

            for file_pattern in config['files']:
                filename = file_pattern.format(sample=sample)
                filepath = os.path.join(stage_dir, filename)

                allow_empty = is_marker_file(filename)
                if not check_file_exists(filepath, allow_empty=allow_empty):
                    missing_files.append({
                        'type': 'sample',
                        'name': sample,
                        'stage': stage,
                        'file': filename,
                        'path': filepath,
                    })
                    if verbose:
                        print(f"  {RED}MISSING:{RESET} {filepath}")
                else:
                    # Check if upstream files are newer
                    file_mtime = get_mtime(filepath)
                    for dep_pattern in config['depends_on']:
                        if dep_pattern.startswith('data_fastq'):
                            dep_path = os.path.join(os.path.dirname(results_dir),
                                                   dep_pattern.format(sample=sample))
                        else:
                            dep_path = os.path.join(sample_dir,
                                                   dep_pattern.format(sample=sample))

                        dep_mtime = get_mtime(dep_path)
                        if dep_mtime and file_mtime and dep_mtime > file_mtime:
                            needs_regen.append({
                                'type': 'sample',
                                'name': sample,
                                'stage': stage,
                                'file': filename,
                                'path': filepath,
                                'file_time': file_mtime,
                                'upstream': dep_path,
                                'upstream_time': dep_mtime,
                            })
                            if verbose:
                                print(f"  {YELLOW}STALE:{RESET} {filepath}")
                                print(f"         (upstream {dep_path} is newer)")

    # Check group-level files
    print(f"\n{BOLD}Checking group-level files...{RESET}")
    for group in sorted(groups):
        group_dir = os.path.join(results_group_dir, group)

        for stage, config in group_files.items():
            stage_dir = os.path.join(group_dir, stage)

            for file_pattern in config['files']:
                filename = file_pattern.format(group=group)
                filepath = os.path.join(stage_dir, filename)

                allow_empty = is_marker_file(filename)
                if not check_file_exists(filepath, allow_empty=allow_empty):
                    missing_files.append({
                        'type': 'group',
                        'name': group,
                        'stage': stage,
                        'file': filename,
                        'path': filepath,
                    })
                    if verbose:
                        print(f"  {RED}MISSING:{RESET} {filepath}")
                else:
                    # Check if upstream files are newer
                    file_mtime = get_mtime(filepath)
                    for dep_pattern in config['depends_on']:
                        dep_path = os.path.join(group_dir, dep_pattern.format(group=group))
                        dep_mtime = get_mtime(dep_path)
                        if dep_mtime and file_mtime and dep_mtime > file_mtime:
                            needs_regen.append({
                                'type': 'group',
                                'name': group,
                                'stage': stage,
                                'file': filename,
                                'path': filepath,
                                'file_time': file_mtime,
                                'upstream': dep_path,
                                'upstream_time': dep_mtime,
                            })
                            if verbose:
                                print(f"  {YELLOW}STALE:{RESET} {filepath}")
                                print(f"         (upstream {dep_path} is newer)")

    return missing_files, needs_regen


def print_summary(missing_files, needs_regen):
    """Print summary of validation results."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}VALIDATION SUMMARY{RESET}")
    print(f"{'='*60}\n")

    if missing_files:
        print(f"{RED}{BOLD}MISSING FILES ({len(missing_files)}):{RESET}")
        for item in missing_files:
            print(f"  [{item['type']}] {item['name']}/{item['stage']}/{item['file']}")
        print()
    else:
        print(f"{GREEN}No missing files.{RESET}\n")

    if needs_regen:
        print(f"{YELLOW}{BOLD}FILES NEEDING REGENERATION ({len(needs_regen)}):{RESET}")
        for item in needs_regen:
            print(f"  [{item['type']}] {item['name']}/{item['stage']}/{item['file']}")
            print(f"    File time:     {format_time(item['file_time'])}")
            print(f"    Upstream time: {format_time(item['upstream_time'])} ({os.path.basename(item['upstream'])})")
        print()
    else:
        print(f"{GREEN}No stale files.{RESET}\n")

    return len(missing_files) == 0 and len(needs_regen) == 0


def generate_snakemake_targets(missing_files, needs_regen, results_dir, results_group_dir):
    """Generate snakemake command to regenerate missing/stale files."""
    targets = []

    for item in missing_files + needs_regen:
        targets.append(item['path'])

    if targets:
        print(f"\n{BOLD}Snakemake command to regenerate:{RESET}")
        # For files that need regeneration, we need to remove them first
        if needs_regen:
            print(f"\n# First, remove stale files:")
            for item in needs_regen:
                print(f"rm -f {item['path']}")

        print(f"\n# Then run snakemake:")
        print(f"snakemake --rerun-incomplete -j 4 \\")
        for target in targets[:5]:  # Show first 5
            print(f"  {target} \\")
        if len(targets) > 5:
            print(f"  # ... and {len(targets) - 5} more targets")

    return targets


def main():
    parser = argparse.ArgumentParser(description='Validate IsoTENT L1 pipeline outputs')
    parser.add_argument('--results-dir', default='results',
                        help='Path to results directory')
    parser.add_argument('--results-group-dir', default='results_group',
                        help='Path to results_group directory')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--show-targets', action='store_true',
                        help='Show snakemake targets to regenerate')
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    results_dir = os.path.join(project_dir, args.results_dir)
    results_group_dir = os.path.join(project_dir, args.results_group_dir)
    config_path = os.path.join(project_dir, args.config)

    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    if not os.path.isdir(results_group_dir):
        print(f"Error: Results group directory not found: {results_group_dir}")
        sys.exit(1)

    missing_files, needs_regen = validate_pipeline(
        results_dir, results_group_dir, config_path=config_path, verbose=args.verbose
    )

    all_ok = print_summary(missing_files, needs_regen)

    if args.show_targets and (missing_files or needs_regen):
        generate_snakemake_targets(missing_files, needs_regen, results_dir, results_group_dir)

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
