#!/bin/bash
set -euo pipefail

# Minimal QC for LINE mapping
# Args: 1) sample_name

sample_name=$1

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}

cd "$OUTPUT_DIR/$sample_name/a_line_mapping"

out_dir="$sample_name"
mkdir -p "$out_dir"

bam_input=${sample_name}_line_mapped.sorted.bam
read_count=$($SAMTOOLS_CMD view -c "$bam_input" 2>/dev/null || echo 0)
printf "Number of reads\t%s\n" "$read_count" > "$out_dir/bam_summary.txt"
