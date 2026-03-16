#!/bin/bash
set -euo pipefail

# Subset reads from pod5 by read_id list, then convert subset pod5 -> fast5.
# Usage: pod5_subset_from_pod5.sh <pod5_root> <read_ids.txt> <out_dir> [threads]

pod5_root=$1
read_ids=$2
out_dir=$3
threads=${4:-${POD5_THREADS:-8}}

if ! command -v pod5 >/dev/null 2>&1; then
  echo "pod5 command not found in PATH" 1>&2
  exit 1
fi

if [ ! -e "$pod5_root" ]; then
  echo "pod5_root not found: $pod5_root" 1>&2
  exit 1
fi
if [ ! -s "$read_ids" ]; then
  echo "read_ids file not found or empty: $read_ids" 1>&2
  exit 1
fi

work_dir="${out_dir}/.pod5_work"
mkdir -p "$work_dir" "$out_dir"

pod5_subset="$work_dir/subset_reads.pod5"

# Handle both file and directory inputs
# Resolve symlink to get the actual target path
resolved_path=$(readlink -f "$pod5_root")

if [ -f "$resolved_path" ] && [[ "$resolved_path" == *.pod5 ]]; then
  # Direct pod5 file specified (or symlink to pod5 file)
  pod5_files=("$resolved_path")
elif [ -d "$resolved_path" ]; then
  # Directory specified - find pod5 files (maxdepth 1 to avoid duplicates from subdirectories)
  mapfile -t pod5_files < <(find -L "$resolved_path" -maxdepth 1 -type f -name "*.pod5" | sort)
else
  echo "pod5_root is neither a .pod5 file nor a directory: $pod5_root (resolved: $resolved_path)" 1>&2
  exit 1
fi

if [ "${#pod5_files[@]}" -eq 0 ]; then
  echo "No .pod5 files found: $pod5_root" 1>&2
  exit 1
fi

pod5 filter -i "$read_ids" -t "$threads" -o "$pod5_subset" -M "${pod5_files[@]}"
pod5 convert to_fast5 -t "$threads" -o "$out_dir" "$pod5_subset"

echo "Done"
echo "Pod5 root: $pod5_root"
echo "Subset pod5: $pod5_subset"
echo "Subset fast5 dir: $out_dir"
