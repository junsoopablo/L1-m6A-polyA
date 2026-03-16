#!/bin/bash
set -euo pipefail

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

# This script utilizes RepeatMasker for LINE/L1 detection, post-processing of the output file, and artifact-based filtering for L1 quantification.

# Set the following parameters in the command: 
# 1: Sample Name 

# Example command:   ./03_L1_detection.sh sample_name



sample_name=$1


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

output_dir=$OUTPUT_DIR/$sample_name/b_repeat_masker_process/
input_file=$OUTPUT_DIR/$sample_name/a_dataset/${sample_name}_cDNA.fasta
input_fasta=$OUTPUT_DIR/$sample_name/a_dataset/${sample_name}_cDNA.fasta


echo "Running RepeatMasker..."
REPEATMASKER_CMD=${REPEATMASKER_CMD:-RepeatMasker}
SEQTK_CMD=${SEQTK_CMD:-seqtk}
command -v "$REPEATMASKER_CMD" >/dev/null 2>&1 || { echo "RepeatMasker not found in PATH"; exit 1; }
$REPEATMASKER_CMD -pa 6 -dir ${output_dir} -nolow -norna -species human -no_is -a -u -xsmall -xm ${input_file}

echo "Finished RepeatMasker!"



# Post-RepeatMasker Filtering by LINE/L1 class (no divergence filter)


cd "$OUTPUT_DIR/$sample_name/b_repeat_masker_process/"
RM_input_file=${sample_name}_cDNA.fasta.out
output_file="LINE_L1.out"

readIDs_lines=()

while read -r line; do
    if [[ $line == *"LINE/L1"* ]]; then
        fields=($line)
        echo "$line" >> "$output_file"
        readIDs_lines+=("${fields[4]}")
    fi
done < "$RM_input_file"



# Get ReadIDs and Generate the new FASTA file of these ReadIDs
read_id_file="LINE_L1_readIDs.txt"

echo "Gathering Final ReadIDs of LINE/L1 elements..."

for item in "${readIDs_lines[@]}"; do
    echo "$item" >> "$read_id_file"
done


if [ -s "$read_id_file" ]; then
    $SEQTK_CMD subseq $input_fasta $read_id_file > ${sample_name}_l1.fa
else
    : > ${sample_name}_l1.fa
fi

echo "Completed processing RepeatMasker output and generated a new FASTA file of LINE/L1 reads!"
