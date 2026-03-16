#!/bin/bash
set -euo pipefail

# Convert fast5 -> pod5, subset by read_id list, then convert subset pod5 -> fast5.
# Usage: pod5_subset_fast5.sh <fast5_root> <read_ids.txt> <work_dir> [threads]

fast5_root=$1
read_ids=$2
work_dir=$3
threads=${4:-${POD5_THREADS:-8}}

if ! command -v pod5 >/dev/null 2>&1; then
  echo "pod5 command not found in PATH" 1>&2
  exit 1
fi

if [ ! -d "$fast5_root" ]; then
  echo "fast5_root not found: $fast5_root" 1>&2
  exit 1
fi
if [ ! -s "$read_ids" ]; then
  echo "read_ids file not found or empty: $read_ids" 1>&2
  exit 1
fi

mkdir -p "$work_dir"

pod5_all="$work_dir/all_reads.pod5"
pod5_subset="$work_dir/subset_reads.pod5"
fast5_out="$work_dir/fast5_subset"

if [ ! -s "$pod5_all" ]; then
pod5 convert fast5 --recursive -t "$threads" -o "$pod5_all" "$fast5_root"
fi

pod5 subset --read-id-list "$read_ids" -t "$threads" -o "$pod5_subset" "$pod5_all"
pod5 convert fast5 -o "$fast5_out" "$pod5_subset"

echo "Done"
echo "All pod5: $pod5_all"
echo "Subset pod5: $pod5_subset"
echo "Subset fast5 dir: $fast5_out"
