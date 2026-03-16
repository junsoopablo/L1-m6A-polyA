#!/bin/bash
set -euo pipefail

# Map reads to LINE reference first, then emit LINE-mapped FASTQ.
# Args: 1) sample_name

sample_name=$1

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_LINE=${REF_LINE:-$ROOT_DIR/reference/custom_LINE_reference.fasta}
FASTQ_INPUT=${FASTQ_INPUT:-$OUTPUT_DIR/$sample_name/a_dataset/${sample_name}.fastq.gz}

# Resolve relative paths against project root (cd below would break them).
if [[ "$FASTQ_INPUT" != /* ]]; then
  FASTQ_INPUT="$ROOT_DIR/$FASTQ_INPUT"
fi
if [[ "$REF_LINE" != /* ]]; then
  REF_LINE="$ROOT_DIR/$REF_LINE"
fi

MINIMAP2_CMD=${MINIMAP2_CMD:-minimap2}
MINIMAP2_THREADS=${MINIMAP2_THREADS:-15}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
read -r -a SAMTOOLS_CMD_ARR <<< "$SAMTOOLS_CMD"

mkdir -p "$OUTPUT_DIR/$sample_name/a_line_mapping"
cd "$OUTPUT_DIR/$sample_name/a_line_mapping"

echo "Mapping reads to LINE reference..."

$MINIMAP2_CMD -ax map-ont -uf --secondary=no -t "$MINIMAP2_THREADS" "$REF_LINE" "$FASTQ_INPUT" \
  | "${SAMTOOLS_CMD_ARR[@]}" view -h -F 4 - \
  | awk 'BEGIN{OFS="\t"} /^@/ {print; next} $6 !~ /N/ {print}' \
  | "${SAMTOOLS_CMD_ARR[@]}" view -b - \
  | "${SAMTOOLS_CMD_ARR[@]}" sort -o ${sample_name}_line_mapped.sorted.bam
"${SAMTOOLS_CMD_ARR[@]}" index ${sample_name}_line_mapped.sorted.bam

# Export mapped reads as FASTQ for hg38 remap
"${SAMTOOLS_CMD_ARR[@]}" fastq ${sample_name}_line_mapped.sorted.bam | gzip -c > ${sample_name}_line_mapped.fastq.gz

echo "Finished mapping to LINE reference"
