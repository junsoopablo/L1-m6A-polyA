#!/bin/bash
set -euo pipefail

# Filter mapped reads by TE GTF (family_id "L1") and generate read IDs + FASTA.
sample_name=$1

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR}

SAMTOOLS_CMD=${SAMTOOLS_CMD:-samtools}
BEDTOOLS_CMD=${BEDTOOLS_CMD:-bedtools}
L1_TE_GTF=${L1_TE_GTF:-$ROOT_DIR/reference/hg38_rmsk_TE.gtf}
L1_TE_BED=${L1_TE_BED:-$ROOT_DIR/reference/L1_TE_L1_family.bed}
read -r -a SAMTOOLS_CMD_ARR <<< "$SAMTOOLS_CMD"

out_dir="$OUTPUT_DIR/$sample_name/b_l1_te_filter"
mkdir -p "$out_dir"
cd "$out_dir"

input_bam="$OUTPUT_DIR/$sample_name/a_hg38_mapping_LRS/${sample_name}_hg38_mapped.sorted_position.bam"
l1_read_ids="${sample_name}_L1_readIDs.txt"
l1_fasta="${sample_name}_L1_reads.fasta"
l1_te_bed="$L1_TE_BED"

echo "Filtering mapped reads by TE GTF (family_id \"L1\")..."
needs_build=1
if [ -s "$l1_te_bed" ]; then
    if awk 'NF < 5 {exit 1}' "$l1_te_bed"; then
        needs_build=0
    fi
fi

if [ "$needs_build" -eq 1 ]; then
    lock="${l1_te_bed}.lock"
    while ! mkdir "$lock" 2>/dev/null; do
        sleep 1
    done
    trap 'rmdir "$lock" 2>/dev/null || true' EXIT

    if [ -s "$l1_te_bed" ]; then
        if awk 'NF < 5 {exit 1}' "$l1_te_bed"; then
            needs_build=0
        fi
    fi

    if [ "$needs_build" -eq 1 ]; then
        tmp_bed="$(mktemp "${l1_te_bed}.tmp.XXXXXX")"
        awk -F '\t' 'BEGIN{OFS="\t"} $0 !~ /^#/ && $9 ~ /family_id "L1"/ {
            gene=""; transcript="";
            if (match($9, /gene_id "([^"]+)"/, g)) gene=g[1];
            if (match($9, /transcript_id "([^"]+)"/, t)) transcript=t[1];
            start=$4-1; if (start < 0) start=0;
            print $1, start, $5, gene, transcript;
        }' "$L1_TE_GTF" > "$tmp_bed"
        if [ ! -s "$tmp_bed" ]; then
            echo "Failed to build $l1_te_bed (empty output). Check L1_TE_GTF: $L1_TE_GTF" 1>&2
            rm -f "$tmp_bed"
            exit 1
        fi
        mv "$tmp_bed" "$l1_te_bed"
    fi

    rmdir "$lock" 2>/dev/null || true
    trap - EXIT
fi

read_len_tsv="${sample_name}_read_lengths.tsv"
reads_bed="${sample_name}_reads.bed"
overlaps_tsv="${sample_name}_l1_overlaps.tsv"

"${SAMTOOLS_CMD_ARR[@]}" view -F 4 "$input_bam" \
    | awk '{print $1"\t"length($10)}' \
    | sort -u > "$read_len_tsv"

$BEDTOOLS_CMD bamtobed -split -i "$input_bam" > "$reads_bed"
$BEDTOOLS_CMD intersect -a "$reads_bed" -b "$l1_te_bed" -wo > "$overlaps_tsv"

awk -v min_frac=0.10 '
    BEGIN { OFS="\t" }
    FNR==NR { len[$1]=$2; next }
    { id=$4; ov[id]+=$NF }
    END {
        for (id in len) {
            if (len[id] > 0 && (ov[id] / len[id]) >= min_frac) {
                print id
            }
        }
    }
' "$read_len_tsv" "$overlaps_tsv" | sort -u > "$l1_read_ids"

if [ -s "$l1_read_ids" ]; then
    "${SAMTOOLS_CMD_ARR[@]}" view -b -N "$l1_read_ids" "$input_bam" \
        | "${SAMTOOLS_CMD_ARR[@]}" fasta - > "$l1_fasta"
else
    : > "$l1_fasta"
    echo "No L1 read IDs found for ${sample_name}." 1>&2
fi

rm -f "$read_len_tsv" "$reads_bed" "$overlaps_tsv"

echo "L1 TE filtering complete!"
