#!/usr/bin/env python3
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subset FAST5 files by read IDs and write multi-fast5 outputs."
    )
    parser.add_argument("--input-dir", required=True, help="Input FAST5 root directory")
    parser.add_argument("--read-ids", required=True, help="File with read IDs (one per line)")
    parser.add_argument("--output-dir", required=True, help="Output directory for multi-fast5 files")
    parser.add_argument("--prefix", required=True, help="Output filename prefix")
    parser.add_argument(
        "--max-reads-per-file",
        type=int,
        default=4000,
        help="Max reads per output multi-fast5 file",
    )
    return parser.parse_args()


def iter_fast5_files(root_dir):
    for base, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".fast5"):
                yield os.path.join(base, name)


def load_read_ids(path):
    read_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            read_id = line.strip().split()[0] if line.strip() else ""
            if read_id:
                read_ids.add(read_id)
    return read_ids


def main():
    args = parse_args()
    try:
        from ont_fast5_api.fast5_interface import get_fast5_file
        from ont_fast5_api.multi_fast5 import MultiFast5File
    except Exception as exc:
        sys.stderr.write(f"Failed to import ont_fast5_api: {exc}\n")
        sys.stderr.write("Ensure ont_fast5_api and h5py are available in the python environment.\n")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    requested_ids = load_read_ids(args.read_ids)
    remaining_ids = set(requested_ids)

    if args.max_reads_per_file <= 0:
        sys.stderr.write("--max-reads-per-file must be > 0\n")
        return 1

    out_index = 1
    out_count = 0
    out_handle = None

    def open_output():
        nonlocal out_index, out_count, out_handle
        if out_handle is not None:
            out_handle.close()
        out_path = os.path.join(args.output_dir, f"{args.prefix}.part{out_index}.fast5")
        out_index += 1
        out_count = 0
        out_handle = MultiFast5File(out_path, mode="w")
        return out_handle

    for fast5_path in iter_fast5_files(args.input_dir):
        if not remaining_ids:
            break
        try:
            with get_fast5_file(fast5_path, mode="r") as f5:
                for read in f5.get_reads():
                    if read.read_id in remaining_ids:
                        if out_handle is None or out_count >= args.max_reads_per_file:
                            open_output()
                        out_handle.add_existing_read(read)
                        out_count += 1
                        remaining_ids.discard(read.read_id)
        except Exception as exc:
            sys.stderr.write(f"Skipping {fast5_path}: {exc}\n")
            continue

    if out_handle is not None:
        out_handle.close()

    report_path = os.path.join(args.output_dir, f"{args.prefix}.subset_report.txt")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"requested\t{len(requested_ids)}\n")
        report.write(f"found\t{len(requested_ids) - len(remaining_ids)}\n")
        report.write(f"missing\t{len(remaining_ids)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
