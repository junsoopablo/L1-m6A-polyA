#!/bin/bash

# This script prepares the downstream folders and input files for analysis 

# Set the following parameters in the command: 
# 1: Sample Name (no_spaces)
# 2: Full Path to the Sample Input File (/insert/full/path/to/input.fastq or input.fasta)
# 3: FASTQ or FASTA (Is the input file in the form of a FASTQ or FASTA file?)
# 4: RNA or DNA/cDNA (Is the input file RNA or already DNA/cDNA?)

# Example command if your input file is in the FASTA format and is from RNA sequencing:   ./01_preprocess_input.sh sample_name /path/to/input/fasta/file.fasta FASTA RNA






ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}

# 1: Sample Name 
sample_name=$1       # this is the name of the sample you are analyzing 
echo "$sample_name"

input_sample=$2       # this is the path to the fastq or fasta file you are analyzing 

# Folder Preparation 
mkdir -p "$OUTPUT_DIR/$sample_name"
cd "$OUTPUT_DIR/$sample_name"

mkdir -p "a_dataset"
mkdir -p "a_hg38_mapping_LRS"
mkdir -p "d_LINE_quantification"

echo "All folders have been made!"





# Input File Preparation 
# Input takes in one file
# If you have multiple fastq files (typical after basecalling with Nanopore data) combine together with the 'cat' command 
# No length filter is applied; all reads are retained.

cd "a_dataset"

if [ "$3" == "FASTQ" ]; then
    echo "Input file is:   FASTQ"
    echo "Converting to FASTA..."
    awk 'NR%4==1{printf ">%s\n", substr($0,2)} NR%4==2{print}' $input_sample > $1.fasta

    echo "Skipping length filter; keeping all reads..."
    cp $1.fasta $1_all.fasta
fi

if [ "$3" == "FASTA" ]; then
    echo "Input file is:  FASTA"
    echo "Skipping length filter; keeping all reads..."
    cp $input_sample $1_all.fasta
fi


# RNA to cDNA Conversion 
# If input file is RNA, convert to cDNA 

if [ "$4" == "RNA" ]; then
    echo "Input file is:  RNA"
    echo " Converting to cDNA..."
    perl -pe 'tr/uU/tT/ unless />/' <$1_all.fasta> $1_cDNA.fasta
fi

if [ "$4" == "DNA" ] || [ "$4" == "cDNA" ]; then
    echo "Input file is:  DNA/cDNA"
    mv $1_all.fasta $1_cDNA.fasta
fi
echo "Preprocessing complete!"
echo""
