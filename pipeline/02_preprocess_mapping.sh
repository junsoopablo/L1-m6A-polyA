#!/bin/bash
set -euo pipefail

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

# This script is part of the data preprocessing and maps the input reads to the Custom LINE Reference Library, removing reads that do not align to a LINE/L1 element 

# Set the following parameters in the command: 
# 1: Sample Name (no_spaces)

# Example command:   ./02_preprocess_mapping.sh sample_name

sample_name=$1       # this is the name of the sample you are analyzing 
echo "$sample_name"


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

cd "$OUTPUT_DIR/$sample_name/a_dataset"

MINIMAP2_CMD=${MINIMAP2_CMD:-minimap2}
SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
SEQTK_CMD=${SEQTK_CMD:-seqtk}

echo "Mapping to Custom LINE Reference Library ..."

REF_L1_mega="$REF_DIR/custom_LINE_reference.fasta"

$MINIMAP2_CMD -ax map-ont $REF_L1_mega $1_cDNA.fasta -t 24 \
    | $SAMTOOLS_CMD view -b -o $1_mapped_cDNA.bam -

read_ids=${1}_mapped_readIDs.txt
$SAMTOOLS_CMD view -F 4 "$1_mapped_cDNA.bam" | awk '{print $1}' | sort -u > "$read_ids"

if [ -s "$read_ids" ]; then
    cmd="$SEQTK_CMD subseq $1_cDNA.fasta $read_ids > $1_mapped_cDNA.fa"
    eval "$cmd"
else
    : > $1_mapped_cDNA.fa
fi

echo "Mapping to Custom LINE Reference Library complete!" 
