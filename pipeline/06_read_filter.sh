#!/bin/bash
set -euo pipefail

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

# Artifact-Based Filtering Part 1:  Read Filter
# Generate new BAM file of the reads that have 90%  of the read itself within the defined reference L1 Loci and generate a bedgraph of this BAM file

# Set the following parameters in the command:
# 1: Sample Name
# 2: L1 Region Reference File Type 
# 3: Output Files

# Example command:   ./06_read_filter.sh sample_name active 
# (active, inactive, or ORF2)



sample_name=$1

L1_ref_type=$2     # change between: active, inactive, orf2  
echo "Read Filter on the $L1_ref_type Reference L1 Regions"

SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
BEDTOOLS_CMD=${BEDTOOLS_CMD:-bedtools}
SORTBED_CMD=${SORTBED_CMD:-sortBed}

sample_bam_input=$OUTPUT_DIR/${sample_name}/a_hg38_mapping_LRS/${sample_name}_hg38_mapped.sorted_position.bam


cd $OUTPUT_DIR/${sample_name}/d_LINE_quantification/

mkdir -p $L1_ref_type; cd $L1_ref_type

mkdir -p "read_filter"; cd "read_filter"

# Prepare L1 reference BED.
if [ "$L1_ref_type" = "all" ]; then
    combined_bed="$OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/all_filtered.bed"
    if [ ! -s "$combined_bed" ]; then
        cat "$REF_DIR/L1Base2_filtered/active_filtered.bed" \
            "$REF_DIR/L1Base2_filtered/inactive_filtered.bed" \
            "$REF_DIR/L1Base2_filtered/orf2_filtered.bed" \
            | sort -k1,1 -k2,2n > "$combined_bed"
    fi
    L1_ref_input="$combined_bed"
else
    L1_ref_input="$REF_DIR/L1Base2_filtered/${L1_ref_type}_filtered.bed"
fi


# Output Files
L1_regions_reads=${sample_name}_L1_regions_reads.bam
echo "Generating filtered BAM file with reads only located within the L1 reference regions..."
$SAMTOOLS_CMD view -b -h -L "$L1_ref_input" -o "$L1_regions_reads" "$sample_bam_input"


read_filter_bam=${sample_name}_read_filter_passed.bam
echo "Removing reads with less than 90% of the read maps to the L1 reference regions"
$BEDTOOLS_CMD intersect -a "$L1_regions_reads" -b "$L1_regions_reads" -f 0.9 > "$read_filter_bam"


sorted_read_filter_bam=${sample_name}_read_filter_passed.sorted_position.bam
echo "Sorting and Indexing the resulting Read Filter BAM file..."
$SAMTOOLS_CMD sort "$read_filter_bam" -o "$sorted_read_filter_bam"
$SAMTOOLS_CMD index "$sorted_read_filter_bam"
touch "${sorted_read_filter_bam}.bai"







# Generate the bedgraph for the new BAM file that passed the Read Filter 

bedgraph_output=${sample_name}"_bedgraph.bg"         
echo "Generating the bedgraph..." 
$BEDTOOLS_CMD genomecov -ibam "$sorted_read_filter_bam" -bga -split > "$bedgraph_output"


bedgraph_output_clean=${sample_name}"_bedgraph_clean.bg"  
echo "Cleaning the bedgraph..."
grep -v 'fix\|alt\|random\|[(]\|Un' $bedgraph_output > $bedgraph_output_clean

bedgraph_sort_output=${sample_name}"_bedgraph_sorted.bg"  
echo "Sorting the cleaned bedgraph..."
$SORTBED_CMD -i $bedgraph_output_clean > $bedgraph_sort_output
