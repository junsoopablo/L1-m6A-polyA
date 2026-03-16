#!/usr/bin/env python3
import argparse
import os
import sys
from multiprocessing import Pool

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def find_fast5_files(root, recursive):
    fast5_files = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".fast5"):
                    fast5_files.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root):
            if name.endswith(".fast5"):
                fast5_files.append(os.path.join(root, name))
    return fast5_files


def read_ids_for_file(path):
    # HDF5 file locking can fail on shared filesystems; disable via env.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    try:
        import h5py  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return (path, None, f"h5py import failed: {exc}")

    import h5py

    try:
        with h5py.File(path, "r") as f:
            ids = []
            for k in f.keys():
                if k.startswith("read_"):
                    ids.append(k[len("read_"):])
            return (path, ids, None)
    except Exception as exc:
        return (path, None, str(exc))


def main():
    parser = argparse.ArgumentParser(
        description="Build read_id -> fast5 path index from a directory of fast5 files"
    )
    parser.add_argument("-i", "--input", required=True, help="Input fast5 directory")
    parser.add_argument("-o", "--output", required=True, help="Output TSV (read_id<TAB>fast5_path)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Worker processes (default: 4)")
    args = parser.parse_args()

    fast5_files = find_fast5_files(args.input, args.recursive)
    if not fast5_files:
        print(f"No .fast5 files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    tmp_out = args.output + ".tmp"
    err_log = args.output + ".errors.log"

    total = len(fast5_files)
    with open(tmp_out, "w") as out_f, open(err_log, "w") as err_f:
        out_f.write("read_id\tfast5_path\n")
        with Pool(processes=args.threads) as pool:
            iterator = pool.imap_unordered(read_ids_for_file, fast5_files)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=total,
                    unit="file",
                    desc="Indexing fast5",
                    mininterval=1.0,
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            processed = 0
            for path, ids, err in iterator:
                processed += 1
                if err:
                    err_f.write(f"{path}\t{err}\n")
                    continue
                for rid in ids:
                    out_f.write(f"{rid}\t{path}\n")
                if tqdm is None and (processed % 100 == 0 or processed == total):
                    print(
                        f"Processed {processed}/{total} fast5 files",
                        file=sys.stderr,
                        flush=True,
                    )

    os.replace(tmp_out, args.output)
    print(f"Wrote: {args.output}")
    print(f"Errors (if any): {err_log}")


if __name__ == "__main__":
    main()
