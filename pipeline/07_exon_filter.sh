#!/bin/bash 

# LEGACY: Not used by the current Snakemake pipeline; kept for reference.

# L1 Loci Filter
# Exon Overlap Removal 

cd ../references

active_regions_ref=L1Base2_filtered/active_filtered.bed

exon_only_gencode=/path/to/gencode.bed/file



overlap_output=../L1Base2_filtered/overlap_removed/active_no_ORF2_no_INACTIVE.bed

bedtools intersect -a $active_regions_ref -b $exon_only_gencode -v > $overlap_output

# Example command:   ./07_exon_filter.sh sample_name
