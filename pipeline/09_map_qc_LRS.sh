#!/bin/bash
set -euo pipefail

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}

# This script calls LongReadSum to generate a quality check analysis summary report of the BAM file following the read filter of the artifact-based filtering step 

# Set the following parameters in the command: 
# 1: Sample Name 


# Example command:   ./09_map_qc_LRS.sh sample_name active
# (active, inactive, or ORF2)


sample_name=$1
L1_ref_type=$2

cd $OUTPUT_DIR/$sample_name/d_LINE_quantification/$L1_ref_type/read_filter/

# Prefer LongReadSum from PATH; override with LONGREADSUM_CMD if needed.
BAM_INPUT=$OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/read_filter/${sample_name}_read_filter_passed.sorted_position.bam
LONGREADSUM_CMD=${LONGREADSUM_CMD:-LongReadSum}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}

out_dir="$sample_name"
mkdir -p "$out_dir"

read_count=$($SAMTOOLS_CMD view -c "$BAM_INPUT" 2>/dev/null || echo 0)
if [ "$read_count" -eq 0 ]; then
    printf "Number of reads\t0\n" > "$out_dir/bam_summary.txt"
    exit 0
fi
min_reads=${MIN_READS_FOR_LRS:-10}
if [ "$read_count" -lt "$min_reads" ]; then
    printf "Number of reads\t%s\n" "$read_count" > "$out_dir/bam_summary.txt"
    exit 0
fi

cmd="$LONGREADSUM_CMD bam -i \"$BAM_INPUT\" -o \"$sample_name\""
if ! eval "$cmd"; then
    echo "LongReadSum failed; writing minimal bam_summary.txt" >&2
    printf "Number of reads\t%s\n" "$read_count" > "$out_dir/bam_summary.txt"
fi
