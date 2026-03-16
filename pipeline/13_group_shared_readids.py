#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-read-tsv", required=True)
    p.add_argument("--shared-loci", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    loci = set()
    with open(args.shared_loci, "r") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            loci.add((fields[0], fields[1], fields[2], fields[3]))

    with open(args.input_read_tsv, "r") as f, open(args.output, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        required = ["read_id", "te_chr", "te_start", "te_end", "te_strand"]
        missing = [r for r in required if r not in idx]
        if missing:
            raise SystemExit(f"Missing columns {missing} in {args.input_read_tsv}")
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            try:
                key = (
                    fields[idx["te_chr"]],
                    fields[idx["te_start"]],
                    fields[idx["te_end"]],
                    fields[idx["te_strand"]],
                )
                read_id = fields[idx["read_id"]]
            except IndexError:
                continue
            if key in loci:
                out.write(read_id + "\n")

if __name__ == "__main__":
    main()
