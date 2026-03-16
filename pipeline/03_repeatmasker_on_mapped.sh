#!/bin/bash
set -euo pipefail

# Run RepeatMasker on hg38-mapped reads and collect LINE/L1 read IDs.
sample_name=$1

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
REPEATMASKER_CMD=${REPEATMASKER_CMD:-RepeatMasker}
SEQTK_CMD=${SEQTK_CMD:-seqtk}

out_dir="$OUTPUT_DIR/$sample_name/b_repeatmasker_on_mapped"
mkdir -p "$out_dir"
cd "$out_dir"

input_bam="$OUTPUT_DIR/$sample_name/a_hg38_mapping_LRS/${sample_name}_hg38_mapped.sorted_position.bam"
mapped_fasta="${sample_name}_hg38_mapped.fasta"

echo "Extracting mapped reads to FASTA..."
$SAMTOOLS_CMD fasta -F 4 "$input_bam" > "$mapped_fasta"

rm_out="${mapped_fasta}.out"
if [ -s "$rm_out" ]; then
    echo "RepeatMasker output exists; skipping RepeatMasker run."
else
    echo "Running RepeatMasker on mapped reads..."
    command -v "$REPEATMASKER_CMD" >/dev/null 2>&1 || { echo "RepeatMasker not found in PATH"; exit 1; }
    REPEATMASKER_THREADS=${REPEATMASKER_THREADS:-6}
    $REPEATMASKER_CMD -pa "$REPEATMASKER_THREADS" -dir "$out_dir" -nolow -norna -species human -no_is -a -u -xsmall -xm "$mapped_fasta"
fi
l1_read_ids="${sample_name}_L1_readIDs.txt"
l1_fasta="${sample_name}_L1_reads.fasta"

echo "Collecting LINE/L1 read IDs..."
if [ -s "$rm_out" ]; then
    awk 'index($0, "LINE/L1") {print $5}' "$rm_out" | sort -u > "$l1_read_ids"
else
    : > "$l1_read_ids"
fi

if [ -s "$l1_read_ids" ]; then
    $SEQTK_CMD subseq "$mapped_fasta" "$l1_read_ids" > "$l1_fasta"
else
    : > "$l1_fasta"
fi

echo "RepeatMasker on mapped reads complete!"
