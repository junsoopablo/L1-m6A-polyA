#!/bin/bash 
set -euo pipefail

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}
REF_DIR=${REF_DIR:-$ROOT_DIR/references}

# Artifact-Based Filtering Part 2: L1 Loci Filter

# Set the following parameters in the command:
# 1: Sample Name
# 2: L1 Region Reference File Type
# 3: Output Files


# Example command:   ./08_L1_loci_filter.sh sample_name active
# (active, inactive, or ORF2)


sample_name=$1
L1_ref_type=$2     # change between: active, inactive, orf2
echo "L1 coverage on the $L1_ref_type Reference L1 Regions"

SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
BEDTOOLS_CMD=${BEDTOOLS_CMD:-bedtools}

sorted_output_bam=$OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/read_filter/${sample_name}_read_filter_passed.sorted_position.bam
bedgraph_sort_output=$OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/read_filter/${sample_name}_bedgraph_sorted.bg

cd $OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/

mkdir -p "L1_loci_coverage"; cd "L1_loci_coverage"

# Prepare L1 reference BED.
if [ "$L1_ref_type" = "all" ]; then
    combined_bed="$OUTPUT_DIR/${sample_name}/d_LINE_quantification/${L1_ref_type}/all_filtered.bed"
    if [ ! -s "$combined_bed" ]; then
        cat "$REF_DIR/L1Base2_filtered/active_filtered.bed" \
            "$REF_DIR/L1Base2_filtered/inactive_filtered.bed" \
            "$REF_DIR/L1Base2_filtered/orf2_filtered.bed" \
            | sort -k1,1 -k2,2n > "$combined_bed"
    fi
    L1_ref_regions="$combined_bed"
else
    L1_ref_regions="$REF_DIR/L1Base2_filtered/${L1_ref_type}_filtered.bed"
fi

# Skip loci filtering and just compute mean coverage for all regions.
FINAL_COV_OUTPUT="${L1_ref_type}_coverage_all_regions.bed"
coverage_output_mean="raw_coverage_values_mean.txt"
$BEDTOOLS_CMD map -a "$L1_ref_regions" -b "$bedgraph_sort_output" -c 4 -o mean -null 0 > "$coverage_output_mean"
cp "$coverage_output_mean" "$FINAL_COV_OUTPUT"
echo "Skipped L1 loci filtering; generated coverage for all regions"
exit 0


###########
# Check 1 #
###########
# Checking the read start or end positions (taking into account strandness) and filtering regions with inconsistent read start positions 
echo "Checking the read start or end positions (taking into account strandness) and filtering regions with inconsistent read start positions"

OUTPUT_REGIONS_FILE="${sample_name}_${L1_ref_type}_consistent.bed"

# Set the output file paths
CONSISTENT_REGIONS_FILE="${sample_name}_consistent_passed_regions.bed"


# Create the output files (or clear their contents if they already exist)
> "$CONSISTENT_REGIONS_FILE"

# Read from the original reference regions file ($OG_REF_REGIONS) instead
while IFS=$'\t' read -r chrom start end name score strand; do
    TEMP_FILE=$(mktemp)
    $SAMTOOLS_CMD view -b "$sorted_output_bam" "$chrom:$start-$end" | $BEDTOOLS_CMD bamtobed -i - > "$TEMP_FILE"

    # Check if the temporary file is empty
    if [ ! -s "$TEMP_FILE" ]; then
        continue
    fi

    # Determine whether to check starting position or end position based on strand
    if [[ "$strand" == "+" ]]; then
        position_col=2
        position_label="start"
    else
        position_col=3
        position_label="end"
    fi

    # Extract the end positions of the reads within the region
    end_positions=$(awk -v pos_col="$position_col" '{print $pos_col}' "$TEMP_FILE")

    # Check if end positions of reads are consistent within 100 bps of each other
    consistent_count=$(echo "$end_positions" | awk -v diff_limit=100 '{
        prev_pos=0;
        count=0;
        for (i=1; i<=NF; i++) {
            if (prev_pos != 0 && ($i - prev_pos) > diff_limit) {
                exit 1;
            }
            prev_pos = $i;
            count++;
        }
    }')

    if [ -z "$consistent_count" ]; then
        # Output the region to the consistent regions file
        echo -e "$chrom\t$start\t$end\t$name\t$score\t$strand" >> "$CONSISTENT_REGIONS_FILE"
    fi

    # Remove the temporary file
    rm "$TEMP_FILE"

done < "$L1_ref_regions"

echo "Finished Check #1" 


###########
# Check 2 #
###########
echo "Checking if the starting position falls within the 1.5kb window between the average consistent starting position and the reference starting position (taking into account strandness)."


# Set the output file paths
OUTPUT_FILE_threshold="${sample_name}_threshold_passed_regions.bed"
UNDER1500_FILE="${sample_name}_regions_for_coverage.bed"


# Create the output files (or clear their contents if they already exist)
> "$OUTPUT_FILE_threshold"
> "$UNDER1500_FILE"

# Process positive strand regions
awk '$6 == "+" {print}' "$L1_ref_regions" | while IFS=$'\t' read -r chrom start end name score strand; do
    TEMP_FILE=$(mktemp)
    $SAMTOOLS_CMD view -b "$sorted_output_bam" "$chrom:$start-$end" | $BEDTOOLS_CMD bamtobed -i - > "$TEMP_FILE"

    # Check if the temporary file is empty
    if [ ! -s "$TEMP_FILE" ]; then
        continue
    fi

    position_col=2
    position_label="start"

    # Extract the positions of the reads within the region
    positions=$(cut -f "$position_col" "$TEMP_FILE")

    # Calculate the differences
    differences=()
    for pos in $positions; do
        diff=$((pos - start))
        differences+=("$diff")
    done

    # Find the mode of the differences
    mode_diff=$(printf '%s\n' "${differences[@]}" | awk '{a[$1]++}END{for(i in a){if(a[i]>max){max=a[i];n=i}}}END{print n}')

    # Check if the mode is negative and make it positive
    if [ $mode_diff -lt 0 ]; then
        mode_diff=$((-$mode_diff))
    fi

    # Check if the mode difference is below 1500
    if [ $mode_diff -lt 1500 ]; then
        echo "Region: ${chrom}_${start}_${end} (Strand: $strand)"
        echo "Mode Difference: $mode_diff"

        # Append the region to the under 1500 file
        echo -e "${chrom}\t${start}\t${end}\t${name}\t${score}\t${strand}" >> "$UNDER1500_FILE"
    fi

    # Append the extracted reads to the output file
    cat "$TEMP_FILE" >> "$OUTPUT_FILE_threshold"

    # Remove the temporary file
    rm "$TEMP_FILE"

done

# Process negative strand regions
# the input bam file for this part is under the variable:  $SORTED_OUTPUT_BAM
# the input reference regions are under $OG_REF_REGIONS


awk '$6 == "-" {print}' "$L1_ref_regions" | while IFS=$'\t' read -r chrom start end name score strand; do
    TEMP_FILE=$(mktemp)
    $SAMTOOLS_CMD view -b "$sorted_output_bam" "$chrom:$start-$end" | $BEDTOOLS_CMD bamtobed -i - > "$TEMP_FILE"

    # Check if the temporary file is empty
    if [ ! -s "$TEMP_FILE" ]; then
        continue
    fi

    position_col=3
    position_label="end"

    # Extract the positions of the reads within the region
    positions=$(cut -f "$position_col" "$TEMP_FILE")

    # Perform your analysis on the read positions here
    # Replace the following echo statements with your desired logic

    # Calculate the differences
    differences=()
    for pos in $positions; do
        diff=$((end - pos))
        differences+=("$diff")
    done

    # Find the mode of the differences
    mode_diff=$(printf '%s\n' "${differences[@]}" | awk '{a[$1]++}END{for(i in a){if(a[i]>max){max=a[i];n=i}}}END{print n}')

    # Check if the mode is negative and make it positive
    if [ $mode_diff -lt 0 ]; then
        mode_diff=$((-$mode_diff))
    fi

    # Check if the mode difference is below 1500
    if [ $mode_diff -lt 1500 ]; then
        echo "Region: ${chrom}_${start}_${end} (Strand: $strand)"
        echo "Mode Difference: $mode_diff"

        # Append the region to the under 1500 file
        echo -e "${chrom}\t${start}\t${end}\t${name}\t${score}\t${strand}" >> "$UNDER1500_FILE"
    fi

    # Append the extracted reads to the output file
    cat "$TEMP_FILE" >> "$OUTPUT_FILE_threshold"

    # Remove the temporary file
    rm "$TEMP_FILE"

done



# sort the regions to look over 
coverage_regions="${sample_name}_regions_for_coverage.sorted.bed"
$BEDTOOLS_CMD sort -i $UNDER1500_FILE > $coverage_regions



echo "Finished Check #2"




###########
# Check 3 #
###########
echo "Calculate coverage over the remaining L1 regions and filter those with less than 2 reads"


coverage_output_mean="raw_coverage_values_mean.txt"
$BEDTOOLS_CMD map -a $coverage_regions -b $bedgraph_sort_output -c 4 -o mean -null 0 > $coverage_output_mean
echo "calculated coverage; by MEAN"

# Filter regions with a value less than 3 in the last column
filtered_coverage_output_mean="filtered_coverage_values_mean.txt"
awk '$NF >= 3' $coverage_output_mean > $filtered_coverage_output_mean
echo "Filtered regions with less than 2 reads. "


# Replace regions with less than 2 reads with 0
FINAL_COV_OUTPUT="${L1_ref_type}_coverage_for_weighted_avg.bed"

if [[ -s $filtered_coverage_output_mean ]]; then
    awk 'NR==FNR{regions[$1,$2,$3]=$0; next} {if (($1,$2,$3) in regions) print regions[$1,$2,$3]; else print $0}' $filtered_coverage_output_mean $L1_ref_regions > $FINAL_COV_OUTPUT
else
    cp $L1_ref_regions $FINAL_COV_OUTPUT
fi

echo "Calculated the $L1_ref_type regions coverage values"
