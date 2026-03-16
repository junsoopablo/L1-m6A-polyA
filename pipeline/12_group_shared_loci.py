#!/usr/bin/env python3
import argparse
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--inputs", nargs="+", required=True)
    args = p.parse_args()

    locus_counts = Counter()

    for path in args.inputs:
        with open(path, "r") as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {name: i for i, name in enumerate(header)}
            required = ["te_chr", "te_start", "te_end", "te_strand"]
            missing = [r for r in required if r not in idx]
            if missing:
                raise SystemExit(f"Missing columns {missing} in {path}")
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
                except IndexError:
                    continue
                locus_counts[key] += 1

    with open(args.output, "w") as out:
        out.write("te_chr\tte_start\tte_end\tte_strand\tread_count\n")
        for (te_chr, te_start, te_end, te_strand), count in sorted(locus_counts.items()):
            if count >= args.min_count:
                out.write(f"{te_chr}\t{te_start}\t{te_end}\t{te_strand}\t{count}\n")

if __name__ == "__main__":
    main()
