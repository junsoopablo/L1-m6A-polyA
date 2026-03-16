#!/bin/bash

# This script maps all cDNA reads to the hg38 reference genome.

# Set the following parameters in the command: 
# 1: Sample Name 

sample_name=$1

# Example command:   ./01_map_hg38.sh sample_name


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

FASTQ_INPUT=${FASTQ_INPUT:-$OUTPUT_DIR/$sample_name/a_dataset/${sample_name}_cDNA.fasta}
REF_SPLICE=${REF_SPLICE:-$REF_DIR/gencode.v40.annotation.bed}
REF_GENOME38=${REF_GENOME38:-$REF_DIR/hg38.fa}

# Resolve relative paths against the project root so cd won't break inputs.
if [[ "$FASTQ_INPUT" != /* ]]; then
    FASTQ_INPUT="$ROOT_DIR/$FASTQ_INPUT"
fi
if [[ "$REF_SPLICE" != /* ]]; then
    REF_SPLICE="$ROOT_DIR/$REF_SPLICE"
fi
if [[ "$REF_GENOME38" != /* ]]; then
    REF_GENOME38="$ROOT_DIR/$REF_GENOME38"
fi

mkdir -p "$OUTPUT_DIR/$sample_name/a_hg38_mapping_LRS"
cd "$OUTPUT_DIR/$sample_name/a_hg38_mapping_LRS/"

MINIMAP2_CMD=${MINIMAP2_CMD:-minimap2}
MINIMAP2_THREADS=${MINIMAP2_THREADS:-15}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}

echo "Mapping reads to the hg38 reference genome..."

$MINIMAP2_CMD -ax splice --junc-bed "$REF_SPLICE" -uf --secondary=no -k14 -t "$MINIMAP2_THREADS" "$REF_GENOME38" "$FASTQ_INPUT" \
  | $SAMTOOLS_CMD view -b - \
  | $SAMTOOLS_CMD sort -o "${sample_name}_hg38_mapped.sorted_position.bam"
$SAMTOOLS_CMD index ${sample_name}_hg38_mapped.sorted_position.bam

echo "Finished mapping reads to the hg38 reference genome"
