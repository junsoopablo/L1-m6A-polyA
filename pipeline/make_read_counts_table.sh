#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$ROOT_DIR/config.yaml}
OUT=${OUT:-$ROOT_DIR/results/read_counts_summary.tsv}
NO_FASTQ=${NO_FASTQ:-0}

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" 1>&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  samples=("$@")
else
  mapfile -t samples < <(awk '
    $1=="samples:" {in_samples=1; next}
    in_samples && $1 ~ /^#/ {next}
    in_samples && $1 ~ /^-/ {
      line=$0
      sub(/^ *- */,"", line)
      sub(/#.*/,"", line)
      gsub(/^[ \t]+|[ \t]+$/, "", line)
      if (line!="") {print line}
      next
    }
    in_samples && $1 !~ /^-/ {exit}
  ' "$CONFIG")
fi

if [ "${#samples[@]}" -eq 0 ]; then
  echo "No samples found in $CONFIG" 1>&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
if [ "$NO_FASTQ" -eq 1 ]; then
  printf "sample\tl1_aligned_reads\thg38_aligned_reads\tl1_filtered_reads\n" > "$OUT"
else
  printf "sample\tfastq_reads\tl1_aligned_reads\thg38_aligned_reads\tl1_filtered_reads\n" > "$OUT"
fi

for s in "${samples[@]}"; do
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  fastq="$ROOT_DIR/data_fastq/${s}.fastq.gz"
  line_tsv="$ROOT_DIR/results/${s}/a_line_mapping/${s}_line_counts.tsv"
  hg38_qc="$ROOT_DIR/results/${s}/a_hg38_mapping_LRS/${s}/bam_summary.txt"
  l1_tsv="$ROOT_DIR/results/${s}/b_l1_te_filter/${s}_L1_counts.tsv"

  fastq_reads="NA"
  l1_aligned="NA"
  hg38_aligned="NA"
  l1_filtered="NA"

  if [ "$NO_FASTQ" -ne 1 ]; then
    if [ -s "$fastq" ]; then
      lines=$(zcat -f "$fastq" | wc -l)
      fastq_reads=$((lines / 4))
    else
      echo "Missing fastq: $fastq" 1>&2
    fi
  fi

  if [ -s "$line_tsv" ]; then
    l1_aligned=$(awk 'NR==2 {print $2}' "$line_tsv")
  else
    echo "Missing line_counts: $line_tsv" 1>&2
  fi

  if [ -s "$hg38_qc" ]; then
    hg38_aligned=$(awk 'NR==1 {print $NF}' "$hg38_qc")
  else
    echo "Missing hg38 bam_summary: $hg38_qc" 1>&2
  fi

  if [ -s "$l1_tsv" ]; then
    l1_filtered=$(awk 'NR==2 {print $3}' "$l1_tsv")
  else
    echo "Missing l1_counts: $l1_tsv" 1>&2
  fi

  if [ "$NO_FASTQ" -eq 1 ]; then
    printf "%s\t%s\t%s\t%s\n" "$s" "$l1_aligned" "$hg38_aligned" "$l1_filtered" >> "$OUT"
  else
    printf "%s\t%s\t%s\t%s\t%s\n" "$s" "$fastq_reads" "$l1_aligned" "$hg38_aligned" "$l1_filtered" >> "$OUT"
  fi
done

echo "Wrote: $OUT"
