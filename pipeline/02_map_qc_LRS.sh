#!/bin/bash

# This script writes a minimal QC summary using samtools (LongReadSum disabled).

# Set the following parameters in the command: 
# 1: Sample Name 

# Example command:   ./02_map_qc_LRS.sh sample_name


sample_name=$1

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}

cd "$OUTPUT_DIR/$sample_name/a_hg38_mapping_LRS/"

# LongReadSum is intentionally skipped; we only report total read count.
BAM_INPUT=${sample_name}_hg38_mapped.sorted_position.bam
LONGREADSUM_CMD=${LONGREADSUM_CMD:-LongReadSum}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
read -r -a SAMTOOLS_CMD_ARR <<< "$SAMTOOLS_CMD"

out_dir="$sample_name"
mkdir -p "$out_dir"

read_count=$("${SAMTOOLS_CMD_ARR[@]}" view -c "$BAM_INPUT" 2>/dev/null || echo 0)
printf "Number of reads\t%s\n" "$read_count" > "$out_dir/bam_summary.txt"
